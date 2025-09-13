import os
from typing import Any, Dict, Tuple

import gradio as gr
from PIL import Image

import torch
import dotenv

# Reuse logic from why_did_you.py
from why_did_you import (
    DEFAULT_CKPT,
    DEFAULT_EXPLAIN_DIR,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOP_CONCEPTS,
    load_ckpt,
    get_bn_after_target,
    load_explainer_artifacts,
    load_llm_summaries,
    run_one_image,
    build_why_text,
    call_llm_short,
)


dotenv.load_dotenv()


# =========================
# Caching & resources
# =========================

_CACHE: Dict[str, Any] = {
    "device": None,
    "ckpt_path": None,
    "explain_dir": None,
    "model": None,
    "bn_after": None,
    "thresholds": None,
    "concepts": None,
    "label_decoder": None,
    "concept_summaries": None,
}


def get_resources(ckpt_path: str, explain_dir: str) -> Tuple[torch.device, Any, Any, Dict[str, float], Dict[str, Dict[str, Any]], Dict[int, str], Dict[str, str]]:
    reload_required = False

    device = _CACHE.get("device")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _CACHE["device"] = device

    if _CACHE.get("ckpt_path") != ckpt_path:
        reload_required = True
        _CACHE["ckpt_path"] = ckpt_path
    if _CACHE.get("explain_dir") != explain_dir:
        reload_required = True
        _CACHE["explain_dir"] = explain_dir

    if reload_required or _CACHE.get("model") is None:
        model, _ = load_ckpt(ckpt_path, device)
        _CACHE["model"] = model
        _CACHE["bn_after"] = get_bn_after_target(model)

        thresholds, concepts, label_decoder, _meta = load_explainer_artifacts(explain_dir)
        if not label_decoder:
            num_classes = _CACHE["model"].fc.out_features
            label_decoder = {i: f"class_{i:03d}" for i in range(num_classes)}
        _CACHE["thresholds"] = thresholds
        _CACHE["concepts"] = concepts
        _CACHE["label_decoder"] = label_decoder
        _CACHE["concept_summaries"] = load_llm_summaries(explain_dir)

    return (
        _CACHE["device"],
        _CACHE["model"],
        _CACHE["bn_after"],
        _CACHE["thresholds"],
        _CACHE["concepts"],
        _CACHE["label_decoder"],
        _CACHE["concept_summaries"],
    )


# =========================
# App logic
# =========================

def analyze_image(
    image: Image.Image,
    top_k: int = DEFAULT_TOP_CONCEPTS,
    ckpt_path: str = DEFAULT_CKPT,
    explain_dir: str = DEFAULT_EXPLAIN_DIR,
) -> Tuple[str, Dict[str, Any]]:
    if image is None:
        return "Please upload an image first.", {}

    device, model, bn_after, thresholds, concepts, label_decoder, concept_summaries = get_resources(
        ckpt_path, explain_dir
    )

    report = run_one_image(
        pil_or_path=image,
        model=model,
        bn_after=bn_after,
        thresholds=thresholds,
        concepts=concepts,
        label_decoder=label_decoder,
        device=device,
    )
    why_text = build_why_text(report, concept_summaries, top_k=top_k)
    return why_text, report


def generate_llm_explanation(
    why_text: str,
    system_prompt_file: str,
    model_id: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> str:
    if not why_text:
        return "No analysis available to summarize."
    try:
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    out = call_llm_short(
        system_prompt=system_prompt,
        why_text=why_text,
        model=model_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return out or "(No response from LLM)"


def format_concept_details(report: Dict[str, Any], top_k: int) -> str:
    if not report:
        return "No report available."
    lines = []
    active = report.get("active_concepts", [])[:top_k]
    if not active:
        return "No active concepts above threshold."
    lines.append("Active concept details (top by |Δ⁻(pred)|, then z_max):\n")
    for i, (cid, ch, zc, thr, dpc, pmi) in enumerate(active, start=1):
        role = "supports" if dpc < 0 else "suppresses" if dpc > 0 else "neutral"
        lines.append(
            f"{i:02d}. {cid} | channel={ch} | z={zc:.2f} > τ={thr:.2f} | Δ⁻={dpc:+.3f} ({role}) | PMI={pmi:+.2f}"
        )
    return "\n".join(lines)


# =========================
# Gradio UI
# =========================

def create_interface():
    with gr.Blocks(title="Orthogonality Explainer") as demo:
        gr.Markdown("# 🧩 Orthogonality Explainer\nUpload an image to see the model's reasoning via orthogonal concepts.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Image", type="pil", height=320)
                top_k = gr.Slider(minimum=1, maximum=20, step=1, value=DEFAULT_TOP_CONCEPTS, label="Top-K Concepts")
                enable_llm = gr.Checkbox(value=True, label="Enable LLM explanation")
                system_prompt_file = gr.Textbox(value=str(DEFAULT_EXPLAIN_DIR + "\\system_prompt.txt"), label="System prompt file")
                model_id = gr.Textbox(value="openai/gpt-5", label="LLM model id")
                analyze_btn = gr.Button("Analyze")

            with gr.Column(scale=2):
                with gr.Tab("Why"):
                    why_md = gr.Markdown(value="Upload an image and click Analyze.")
                with gr.Tab("LLM Explanation"):
                    llm_md = gr.Markdown(value="Enable LLM to generate an explanation.")
                with gr.Tab("Concept Details"):
                    details_md = gr.Markdown(value="Active concept details will appear here.")

        def on_analyze(img, k, do_llm, sp_file, mdl):
            why_text, report = analyze_image(img, top_k=int(k))
            details = format_concept_details(report, top_k=int(k))
            if do_llm:
                llm = generate_llm_explanation(why_text, sp_file, mdl)
            else:
                llm = "LLM explanation disabled."
            return why_text, llm, details

        analyze_btn.click(
            fn=on_analyze,
            inputs=[image_input, top_k, enable_llm, system_prompt_file, model_id],
            outputs=[why_md, llm_md, details_md],
        )

        image_input.change(
            fn=on_analyze,
            inputs=[image_input, top_k, enable_llm, system_prompt_file, model_id],
            outputs=[why_md, llm_md, details_md],
        )

    return demo


if __name__ == "__main__":
    # Basic file checks
    missing = []
    if not os.path.exists(DEFAULT_CKPT):
        missing.append(DEFAULT_CKPT)
    for fname in ["thresholds.json", "concepts.jsonl"]:
        if not os.path.exists(os.path.join(DEFAULT_EXPLAIN_DIR, fname)):
            missing.append(os.path.join(DEFAULT_EXPLAIN_DIR, fname))
    if missing:
        print(f"⚠️ Missing files: {missing}")

    app = create_interface()
    app.launch()


