import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights
import dotenv
import replicate
dotenv.load_dotenv()


# =========================
# Defaults
# =========================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEFAULT_CKPT = r"runs\ortho_finetune\checkpoint_best.pt"
DEFAULT_EXPLAIN_DIR = r"runs\ortho_explainer"  # concepts.jsonl, thresholds.json, meta.json, llm_output.jsonl
DEFAULT_SYSTEM_PROMPT = "You are an AI explainability agent. You have the task of giving an expanation on why did a neural network classifier output that specific class. For that, you have the output of the top 20 sparse autoencoder neurons, that have been traced back for interpretability, jointly with the activation score (higher the better). Your task is to answer in one sentence to the question \"why did you chose that class?\".\nDon't mention the access to the internal neural activations. Just describe the input characteristics and features that made the model select the given class with the given information.\n\nFor example, an acceptable output can be:\nPrediction: {predicted\\_class} with calibrated probability {p}; I arrived here by detecting the following human-interpretable concepts in the image and combining them with a learned rule set: {primary\\_concepts\\_detected} provided {primary\\_concept\\_contribution}% of the logit evidence, {secondary\\_concepts\\_detected} provided {secondary\\_concept\\_contribution}%, and low-level textures/edges at spatial frequencies {freq\\_range} contributed {texture\\_contribution}%; these were localized mainly in {key\\_regions} (covering {region\\_coverage}% of the object’s area), and ablating these regions drops the class logit by {ablation\\_drop}σ, confirming causality; the evidence was computed via concept-specific circuits {circuit\\_ids} whose behavior has been verified on a held-out causal dataset and passes sufficiency/necessity tests ({sufficiency\\_score}/{necessity\\_score}); counterfactual checks show that replacing {critical\\_attribute} with {counterfactual\\_variant} shifts the top prediction to {alt\\_class} at {alt\\_prob}, while changes to nuisance factors ({lighting}, {background}, {scale}, {pose}) leave the logit within ±{robust\\_margin}; nearest validated prototypes in training are {prototype\\_ids\\_or\\_links} with similarities {sim\\_scores}, and no single example dominated (influence scores ≤ {influence\\_cap}); I rejected close alternatives because {alt1} lacks {missing\\_concept\\_or\\_pattern} (margin {margin1}) and {alt2} exhibits {conflicting\\_cue} inconsistent with {detected\\_concept} (margin {margin2}); the input is in-distribution ({ood\\_score}), passes artifact checks ({artifact\\_checks}), and my uncertainty arises mainly from {ambiguity\\_source}; this explanation is faithful by integrated-gradients infidelity {infidelity} and deletion AUC {deletion\\_auc}, and it is reproducible: re-evaluations under {num\\_augmentations} stochastic augmentations varied the probability by ±{prob\\_std}; training provenance for the supporting concepts traces to {dataset\\_summaries} with balanced coverage across {relevant\\_factors}, and fairness checks indicate no reliance on spurious cues ({spurious\\_cue\\_test}); in plain terms: I see {short\\_human\\_summary\\_of\\_visual\\_evidence}, which is characteristic of a {predicted\\_class}, and removing those cues makes the prediction disappear.",
DEFAULT_LLM_MODEL = "openai/gpt-5"       # or "openai/gpt-5" if you have access
DEFAULT_TOP_CONCEPTS = 12
DEFAULT_SIDE = 224


# =========================
# User configuration
# =========================

# Set this to the image you want to explain (required)
USER_IMAGE = r"G:\My Drive\PythonProjects\OrthogonalityExplainability\runs\ortho_explainer\thumbs\feature_00002_img0.webp"

# Optional overrides (defaults point to your repo artifacts)
USER_CKPT = DEFAULT_CKPT
USER_EXPLAIN_DIR = DEFAULT_EXPLAIN_DIR
USER_TOP_K = DEFAULT_TOP_CONCEPTS
USER_NO_LLM = False  # set to False to enable one-sentence LLM summary
USER_SYSTEM_PROMPT_FILE = r"runs\ortho_explainer\system_prompt.txt"
USER_LLM_MODEL = DEFAULT_LLM_MODEL


# =========================
# Model bits (match ortho)
# =========================

def build_model(num_classes: int) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def pick_target_conv(model: nn.Module) -> nn.Conv2d:
    return model.layer3[-1].conv2

def get_bn_after_target(model: nn.Module) -> nn.BatchNorm2d:
    return model.layer3[-1].bn2

def build_transforms(image_size: int = DEFAULT_SIDE):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# =========================
# Hooks
# =========================

class CachedActivation:
    def __init__(self, module: nn.Module):
        self.cache = None
        self.h = module.register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        self.cache = out.detach()

    def pop(self):
        x = self.cache
        self.cache = None
        return x

    def remove(self):
        self.h.remove()


# =========================
# IO helpers
# =========================

def load_ckpt(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = int(ckpt.get("num_classes", -1))
    if num_classes <= 0:
        raise RuntimeError("num_classes missing from checkpoint.")
    model = build_model(num_classes=num_classes).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[warn] load_state: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    return model, ckpt

def load_explainer_artifacts(explain_dir: str) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]], Dict[int, str], Dict[str, Any]]:
    """
    Returns:
      thresholds: {concept_id -> threshold}
      concepts: {concept_id -> concept_obj}
      label_decoder: {int -> name} (if present)
      meta dict
    """
    explain_dir = Path(explain_dir)
    # thresholds
    with open(explain_dir / "thresholds.json", "r", encoding="utf-8") as f:
        thresholds_blob = json.load(f)
    thresholds = thresholds_blob.get("concept_thresholds", {})
    # concepts
    concepts: Dict[str, Dict[str, Any]] = {}
    with open(explain_dir / "concepts.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = obj["concept_id"]
            concepts[cid] = obj
    # label decoder (optional)
    meta = {}
    label_decoder: Dict[int, str] = {}
    meta_path = explain_dir / "meta.json"
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        names = meta.get("class_names", None)
        if names:
            # Either list indexed by class id, or mapping
            if isinstance(names, list):
                label_decoder = {i: str(n) for i, n in enumerate(names)}
            elif isinstance(names, dict):
                label_decoder = {int(k): str(v) for k, v in names.items()}
    return thresholds, concepts, label_decoder, meta

def load_llm_summaries(explain_dir: str) -> Dict[str, str]:
    """
    Reads ortho_summary_to_llm.py outputs: llm_output.jsonl with lines:
      {"concept_id": "...", "neuron_id": <int>, "answer": "<SHORT ANSWER>"}
    """
    path = Path(explain_dir) / "llm_output.jsonl"
    summaries: Dict[str, str] = {}
    if not path.is_file():
        return summaries
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                cid = obj.get("concept_id")
                ans = obj.get("answer")
                if cid and ans:
                    summaries[cid] = ans
            except Exception:
                continue
    return summaries


# =========================
# Core: run one image
# =========================

@torch.no_grad()
def run_one_image(
    pil_or_path: str,
    model: nn.Module,
    bn_after: nn.BatchNorm2d,
    thresholds: Dict[str, float],
    concepts: Dict[str, Dict[str, Any]],
    label_decoder: Dict[int, str],
    image_size: int = DEFAULT_SIDE,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Any]:
    """
    Returns a dict containing:
      - predicted_class (int), predicted_name (str), prob (float), top5 [(id,prob,name)]
      - z_max: np.ndarray (P,)
      - active_concepts: List[(concept_id, channel_index, z, threshold, delta_predclass, pmi_predclass)]
    """
    # Prepare image
    if isinstance(pil_or_path, (str, Path)):
        pil = Image.open(str(pil_or_path)).convert("L")
    else:
        pil = pil_or_path.convert("L")
    tf = build_transforms(image_size)
    x = tf(pil).unsqueeze(0).to(device)  # (1,3,224,224)

    # Forward and cache bn2
    cache = CachedActivation(bn_after)
    logits = model(x)
    A = cache.pop()  # (1, P, H, W) pre-residual, post-BN
    cache.remove()
    if A is None:
        raise RuntimeError("bn2 hook returned None.")

    # bn2 -> ReLU -> z_max
    A_pos = F.relu(A)
    z = A_pos.amax(dim=(-1, -2)).squeeze(0).detach().cpu().numpy()  # (P,)

    # Predictions
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_id = int(torch.argmax(probs).item())
    pred_name = label_decoder.get(pred_id, f"class_{pred_id:03d}")
    pred_prob = float(probs[pred_id].item())
    top5_vals, top5_idx = torch.topk(probs, k=min(5, probs.numel()))
    top5 = [(int(i), float(v), label_decoder.get(int(i), f"class_{int(i):03d}")) for v, i in zip(top5_vals.tolist(), top5_idx.tolist())]

    # Build concept_id per channel (we stored them as "resnet18.layer3.1.conv2:00012")
    layer_name = None
    # try to infer from any concept
    if concepts:
        layer_name = next(iter(concepts.values())).get("layer", None)
    if not layer_name:
        layer_name = "layer3.1.conv2"

    P = A.shape[1]
    active: List[Tuple[str, int, float, float, float, float]] = []
    for ch in range(P):
        cid = f"resnet18.{layer_name}:{ch:05d}"
        thr = thresholds.get(cid, None)
        if thr is None:
            continue
        zc = float(z[ch])
        if zc <= float(thr):
            continue  # not ON
        # fetch concept stats for pred class
        cobj = concepts.get(cid, None)
        delta_pc = 0.0
        pmi_pc = 0.0
        if cobj is not None:
            top_classes = cobj.get("top_classes", [])
            # find the entry for our predicted class name
            for ent in top_classes:
                if str(ent.get("class", "")) == pred_name:
                    delta_pc = float(ent.get("delta_minus", 0.0))
                    pmi_pc = float(ent.get("PMI", 0.0))
                    break
        active.append((cid, ch, zc, float(thr), delta_pc, pmi_pc))

    # Rank active concepts: prioritize causal impact on predicted class (|Δ|), then z, then PMI
    active.sort(key=lambda t: (abs(t[4]), t[2], t[5]), reverse=True)

    return {
        "predicted_class": pred_id,
        "predicted_name": pred_name,
        "predicted_prob": pred_prob,
        "top5": top5,
        "z_max": z,
        "active_concepts": active,  # list of tuples as above
    }


# =========================
# Pretty print & LLM call
# =========================

def fmt_pct(p: float) -> str:
    return f"{int(round(p * 100))}%"

def build_why_text(report: Dict[str, Any],
                   concept_summaries: Dict[str, str],
                   top_k: int = DEFAULT_TOP_CONCEPTS) -> str:
    lines = []
    lines.append(f'Predicted class: "{report["predicted_name"]}" with probability {fmt_pct(report["predicted_prob"])}')
    lines.append("")
    lines.append("Top-5 classes:")
    for rank, (cid, pr, name) in enumerate(report["top5"], start=1):
        lines.append(f"{rank}. {name}: {pr:.3f} ({fmt_pct(pr)})")
    lines.append("")

    active = report["active_concepts"][:top_k]
    if not active:
        lines.append("No orthogonal concepts fired above threshold for this image.")
        return "\n".join(lines)

    lines.append(f"Top {len(active)} active orthogonal directions contributing to this decision:")
    lines.append("(ranked by causal |Δ⁻(pred)|, then activation z_max)")
    for i, (cid, ch, zc, thr, dpc, pmi) in enumerate(active, start=1):
        interp = concept_summaries.get(cid, "(no LLM summary)")
        role = "supports" if dpc < 0 else "suppresses" if dpc > 0 else "neutral"
        lines.append(
            f"{i:02d}. {cid}  [z={zc:.2f} > τ={thr:.2f}]  Δ⁻(pred)={dpc:+.3f} ({role}), PMI(pred)={pmi:+.2f}\n"
            f"    → LLM: {interp}"
        )

    return "\n".join(lines)

def call_llm_short(system_prompt: str, why_text: str,
                   model: str = DEFAULT_LLM_MODEL,
                   temperature: float = 0.2,
                   top_p: float = 0.9,
                   max_tokens: int = 256) -> Optional[str]:
    try:
        chunks = replicate.run(
            model,
            input={
                "system_prompt": system_prompt,
                "prompt": (
                    "Explain why the model chose the predicted class using clear, human-vision terms. "
                    "Do NOT mention internals or neurons. Be specific about the visual evidence and "
                    "how it supports the decision. Do not use bullet points. Use first person, start with \"I predicted ...\" \n\n"
                    "EVIDENCE:\n" + why_text
                ),
                "image_input": [],
                "temperature": temperature,
                "top_p": top_p,
                "max_completion_tokens": max_tokens,
                "verbosity": "low"
            },
        )
        return "".join(chunks).strip()
    except Exception as e:
        print("[warn] LLM call failed:", e)
        return None


# =========================
# Entry point
# =========================

def main():
    if not USER_IMAGE:
        raise RuntimeError("Please set USER_IMAGE at the top of why_did_you.py to the path of an image.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model, _ = load_ckpt(USER_CKPT, device)
    bn_after = get_bn_after_target(model)

    # Load explainer artifacts
    thresholds, concepts, label_decoder, meta = load_explainer_artifacts(USER_EXPLAIN_DIR)
    if not label_decoder:
        # fallback to generic names
        num_classes = model.fc.out_features
        label_decoder = {i: f"class_{i:03d}" for i in range(num_classes)}

    # Load LLM short answers (optional but nice)
    concept_summaries = load_llm_summaries(USER_EXPLAIN_DIR)

    # Run one image
    report = run_one_image(
        pil_or_path=USER_IMAGE,
        model=model,
        bn_after=bn_after,
        thresholds=thresholds,
        concepts=concepts,
        label_decoder=label_decoder,
        device=device,
    )

    # Pretty print “why”
    why_text = build_why_text(report, concept_summaries, top_k=USER_TOP_K)
    print(why_text)

    # Optional: one-sentence summary from LLM
    if not USER_NO_LLM:
        try:
            with open(USER_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                system_prompt = f.read()
        except Exception:
            system_prompt = "You summarize evidence into one sentence in plain language."
        short = call_llm_short(system_prompt, why_text, model=USER_LLM_MODEL)
        if short:
            print("\n— LLM explanation —")
            print(short)


if __name__ == "__main__":
    main()
