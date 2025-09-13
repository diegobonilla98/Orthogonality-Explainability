import os
import re
import io
import json
import math
import time
import base64
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: replicate client (kept same as your SAE script)
import dotenv
dotenv.load_dotenv()
import replicate


# ======================
# Config / Defaults
# ======================

DEFAULT_EXPLAIN_DIR = r"./runs/ortho_explainer"
DEFAULT_SYSTEM_PROMPT = r"./runs/ortho_explainer/system_prompt.txt"  # you uploaded this already
DEFAULT_OUTPUT = r"./runs/ortho_explainer/llm_output.jsonl"
DEFAULT_MAX_WORKERS = 8
DEFAULT_TEMP = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOK = 512
DEFAULT_MODEL = "openai/gpt-5-mini"  # same as your SAE script

SHORT_ANSWER_RE = re.compile(
    r'(?im)^\s*short\s*answer\s*:?\s*(.+?)\s*$'
)

# ======================
# Small utils
# ======================

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def mono_from_S(S_vals):
    """
    SAE-style "monosemanticity" proxy: 1 - H(p)/log K over the available S entries.
    """
    if not S_vals:
        return 0.0
    p = softmax(np.array(S_vals))
    H = -(p * (np.log(p + 1e-12))).sum()
    return float(max(0.0, min(1.0, 1.0 - H / (math.log(len(p) + 1e-12)))))


# ======================
# Image helpers
# ======================

def annotate_thumb(path_or_img: str, label: str, z: float, side: int = 224) -> Image.Image:
    img = Image.open(path_or_img).convert("L")
    img = img.resize((side, side), Image.BICUBIC)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = f"{label}\n(z={z:.2f})"
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # white box for contrast
    draw.rectangle([3, 3, 7 + tw, 7 + th], fill=255)
    draw.multiline_text((5, 5), text, fill=0, font=font, spacing=2)
    return img

def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "data:image/png;base64," + b64


# ======================
# LLM call
# ======================

def build_user_prompt(llm_obj_str: str) -> str:
    # Strong, explicit, minimal contract
    return (
        "You will analyze ONE orthogonal CNN concept.\n"
        "Reply in EXACTLY two sections:\n"
        "1) A single line starting with: SHORT ANSWER: <max 12 words>\n"
        "2) A blank line, then EVIDENCE SUMMARY: as bullet points.\n"
        "Do not include any other headings.\n\n"
        "NEURON JSON (single neuron / orthogonal concept):\n"
        "```json\n" + llm_obj_str + "\n```\n"
        "IMAGES: Top-5 thumbnails in descending z; each image is annotated (class, z)."
    )

def _extract_short_answer(text: str) -> Optional[str]:
    m = SHORT_ANSWER_RE.search(text or "")
    if not m:
        return None
    ans = m.group(1).strip()
    # strip surrounding quotes if present
    if len(ans) >= 2 and ((ans[0] == ans[-1] == '"') or (ans[0] == ans[-1] == '“') or (ans[0] == ans[-1] == '”')):
        ans = ans[1:-1].strip()
    # clip to ~12 words just in case
    words = ans.split()
    if len(words) > 12:
        ans = " ".join(words[:12])
    return ans or None

def replicate_run(
    system_prompt: str,
    concept_json_str: str,
    image_uris: List[str],
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMP,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOK,
    max_retries: int = 3,
) -> Optional[str]:
    base_prompt = build_user_prompt(concept_json_str)

    def _call(prompt_text: str) -> str:
        chunks = replicate.run(
            model,
            input={
                "system_prompt": system_prompt,
                "prompt": prompt_text,
                "image_input": image_uris,
                "temperature": temperature,
                "top_p": top_p,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "max_completion_tokens": max_tokens,
            },
        )
        return "".join(chunks).strip()

    # First attempt
    try:
        response = _call(base_prompt)
        ans = _extract_short_answer(response)
        if ans:
            return ans
        print("WARN: SHORT ANSWER not found.\n---\n", response, "\n---")
    except Exception as e:
        last_err = e
    else:
        last_err = None

    # One strict retry
    strict_suffix = (
        "\n\nYou DID NOT include the required 'SHORT ANSWER:' line.\n"
        "RETRY NOW. Follow EXACTLY:\n"
        "SHORT ANSWER: <max 12 words>\n\n"
        "EVIDENCE SUMMARY:\n- bullet 1\n- bullet 2\n- bullet 3\n"
    )
    try:
        response2 = _call(base_prompt + strict_suffix)
        ans2 = _extract_short_answer(response2)
        if ans2:
            return ans2
        print("WARN: SHORT ANSWER still not found.\n---\n", response2, "\n---")
    except Exception as e:
        last_err = e

    # If we reach here: synthesize a safe short answer later in the caller using stats.
    return None

def _synthesize_short_answer(llm_obj: dict) -> str:
    # Heuristic: pick the top class by (delta_minus weight + PMI).
    # Positive delta_minus means neuron suppresses that class; negative means it supports it.
    tcs = llm_obj.get("top_classes", [])
    if not tcs:
        return "no clear concept"
    # score: emphasize causal (|delta|), break ties by PMI
    scored = sorted(
        tcs,
        key=lambda d: (abs(float(d.get("delta_minus", 0.0))), float(d.get("PMI", 0.0))),
        reverse=True,
    )
    top = scored[0]
    cls = str(top.get("class", "unknown"))
    dm = float(top.get("delta_minus", 0.0))
    if dm > 0:
        # removing channel increases that class → neuron argues AGAINST it
        return f"anti-{cls} evidence"
    else:
        return f"{cls} evidence"

# ======================
# Concept → LLM JSON mapping
# ======================

def concept_to_llm_neuron_obj(concept: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map one line from concepts.jsonl into the JSON shape your system prompt expects.
    Fields we don't have ('w', 'coverage', 'precision', 'direction_gain') are left None.
    """
    # Stable numeric "neuron_id": use channel_index; keep also the string concept_id inside payload
    neuron_id = int(concept.get("channel_index", 0))
    sparsity = float(concept.get("sparsity", 0.0))
    threshold = float(concept.get("threshold", 0.0))
    mean_on = float(concept.get("mean_on_activation", 0.0))
    top_classes_in = concept.get("top_classes", []) or []

    top_classes_out = []
    S_vals = []
    for tc in top_classes_in:
        S_val = float(tc.get("S", 0.0))
        S_vals.append(S_val)
        top_classes_out.append({
            "class": str(tc.get("class", "")),
            "S": S_val,
            "PMI": float(tc.get("PMI", 0.0)),
            "delta_minus": float(tc.get("delta_minus", 0.0)),
            "w": None,               # unknown in ortho pipeline
            "count_on": int(tc.get("count_on", 0)),
            "coverage": None,        # not saved in ortho pipeline
            "precision": None,       # not saved in ortho pipeline
        })

    # Heuristic anti-classes (positive delta_minus ⇒ removal helps that class)
    anti_candidates = [tc for tc in top_classes_out if (tc.get("delta_minus", 0.0) > 0)]
    anti_candidates.sort(key=lambda d: d["delta_minus"], reverse=True)
    anti_classes = [{"class": d["class"], "delta_minus": float(d["delta_minus"])}
                    for d in anti_candidates[:3]]

    # SAE-style mono score from S
    mono_score = float(mono_from_S(S_vals))

    llm_obj = {
        "neuron_id": neuron_id,
        "concept_id": str(concept.get("concept_id", "")),  # keep for traceability
        "layer": str(concept.get("layer", "")),
        "sparsity": sparsity,
        "threshold": threshold,
        "mean_on_activation": mean_on,
        "top_classes": top_classes_out,
        "anti_classes": anti_classes,
        "direction_gain": None,   # N/A
        "mono_score": mono_score,
        "sprite_path": str(concept.get("sprite_path", "")),
        "top_examples": concept.get("top_examples", []),
        "auto_summary": str(concept.get("auto_summary", "")),
    }
    return llm_obj


def load_concepts(concepts_path: str) -> List[Dict[str, Any]]:
    with open(concepts_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def prepare_images_for_llm(top_examples: List[Dict[str, Any]], side: int = 224) -> List[str]:
    """
    Returns list of data URLs for the top-5 thumbs (PIL-annotated).
    """
    ims = []
    for te in top_examples[:5]:  # ensure at most 5
        thumb_path = te["thumb"]
        label = te.get("label", "")
        z = float(te.get("z", 0.0))
        img = annotate_thumb(thumb_path, label, z, side=side)
        ims.append(image_to_data_url(img))
    return ims


# ======================
# Main (CLI)
# ======================

def main():
    ap = argparse.ArgumentParser(description="Send orthogonal concept summaries to LLM (SAE-style interface).")
    ap.add_argument("--explain_dir", type=str, default=DEFAULT_EXPLAIN_DIR,
                    help="Directory containing concepts.jsonl, meta.json, thumbs/, sprites/")
    ap.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
                    help="Path to system_prompt.txt (SAE interpreter prompt).")
    ap.add_argument("--concept_id", type=str, default=None,
                    help="If provided, only process this exact concept_id (e.g., resnet18.layer3.1.conv2:00012).")
    ap.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                    help="Output JSONL with {'concept_id' or 'neuron_id', 'answer'} per line.")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL)
    ap.add_argument("--max_workers", type=int, default=DEFAULT_MAX_WORKERS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMP)
    ap.add_argument("--top_p", type=float, default=DEFAULT_TOP_P)
    ap.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOK)
    args = ap.parse_args()

    explain_dir = Path(args.explain_dir)
    concepts_path = explain_dir / "concepts.jsonl"

    if not concepts_path.is_file():
        raise FileNotFoundError(f"Concepts file not found: {concepts_path}")

    with open(args.system_prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    concepts = load_concepts(str(concepts_path))

    # If a single concept_id is requested, filter down
    if args.concept_id is not None:
        concepts = [c for c in concepts if c.get("concept_id", "") == args.concept_id]
        if not concepts:
            raise ValueError(f"Requested concept_id not found: {args.concept_id}")

    # Build tasks
    tasks: List[Tuple[str, Dict[str, Any], List[str], str]] = []
    for c in concepts:
        llm_obj = concept_to_llm_neuron_obj(c)
        # Prepare top-5 annotated thumbs as data URLs for the multimodal LLM
        image_uris = prepare_images_for_llm(llm_obj.get("top_examples", []), side=224)
        llm_obj_str = json.dumps({
            # Include only the fields the system prompt expects (and a couple of handy extras)
            "neuron_id": llm_obj["neuron_id"],
            "sparsity": llm_obj["sparsity"],
            "threshold": llm_obj["threshold"],
            "mean_on_activation": llm_obj["mean_on_activation"],
            "top_classes": llm_obj["top_classes"],
            "anti_classes": llm_obj["anti_classes"],
            "direction_gain": llm_obj["direction_gain"],
            "mono_score": llm_obj["mono_score"],
            "sprite_path": llm_obj["sprite_path"],
            "top_examples": llm_obj["top_examples"],
            "auto_summary": llm_obj["auto_summary"],
            # traceability
            "concept_id": llm_obj["concept_id"],
            "layer": llm_obj["layer"],
        }, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        concept_id = str(c.get("concept_id", f"channel:{llm_obj['neuron_id']}"))
        tasks.append((concept_id, llm_obj, image_uris, llm_obj_str))

    # Parallel LLM calls
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_f = open(args.output, "w", encoding="utf-8")

    max_workers = min(args.max_workers, len(tasks)) if len(tasks) > 0 else 0
    print(f"Submitting {len(tasks)} concept(s) to LLM with {max_workers} workers...")

    def _submit(task):
        concept_id, llm_obj, image_uris, llm_obj_str = task
        ans = replicate_run(
            system_prompt=system_prompt,
            concept_json_str=llm_obj_str,
            image_uris=image_uris,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        if ans is None:
            ans = _synthesize_short_answer(llm_obj)
        return (concept_id, llm_obj["neuron_id"], ans)

    if max_workers <= 1:
        for t in tqdm.tqdm(tasks, desc="Processing concepts"):
            concept_id, neuron_id, ans = _submit(t)
            if ans is not None:
                out_f.write(json.dumps({"concept_id": concept_id, "neuron_id": neuron_id, "answer": ans}, ensure_ascii=False) + "\n")
                out_f.flush()
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_submit, t) for t in tasks]
            for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing concepts"):
                concept_id, neuron_id, ans = fut.result()
                if ans is not None:
                    out_f.write(json.dumps({"concept_id": concept_id, "neuron_id": neuron_id, "answer": ans}, ensure_ascii=False) + "\n")
                    out_f.flush()

    out_f.close()
    print(f"Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()
