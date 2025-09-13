## Orthogonality Explainability

### Overview
This project turns a standard CNN into a self-explaining model by enforcing strict orthogonality on one chosen convolutional layer and then interpreting each channel of that layer as an independent concept. Orthogonality is maintained at every gradient update using Riemannian optimization on the Stiefel manifold. As a result, mid-level channels form a clean, non-redundant basis where each filter captures a distinct axis of variation, reducing feature interference and making representations easier to probe.

- **Goal (interpretability, not accuracy)**: Freeze most of a pretrained network, pick a mid/late block where concepts are more disentangled, and finetune only that block under an orthogonality constraint. Afterwards, analyze per-channel activations and causality, and generate human-language explanations with an LLM.
- **End-to-end steps**:
  - **Train base CNN** (standard supervised training)
  - **Finetune one orthogonal layer** via Stiefel-constrained Riemannian Adam
  - **Extract/score concepts** from that layer on a sample of data
  - **Summarize concepts with an LLM** (short natural-language labels)
  - **Ask “why?” on an image**: attribute the decision to the top active orthogonal concepts

### Key idea (what this repo explores)
This orthogonalization idea takes a pretrained network and, instead of letting its mid-level features overlap arbitrarily, it constrains one chosen convolutional layer so that its filters are always orthogonal directions in feature space, maintained at every gradient update using Riemannian optimization on the Stiefel manifold. By doing this, each channel of that layer encodes an independent axis of variation rather than a correlated mixture, which reduces interference between features and makes it easier to probe what each direction contributes to the representation. In practice, this means you freeze most of the network, pick a mid/late convolutional block where concepts are already somewhat disentangled, and finetune only that block under strict orthogonality. The goal isn’t higher accuracy but greater interpretability: the outputs of that layer can be analyzed as a clean, non-redundant basis where each filter captures a distinct concept or statistical factor, making the learned representations more transparent to inspection and probing.


## Project layout
- `train_standard_cnn.py` — Standard training loop for ResNet-18 on TU-Berlin (single or multi-GPU). Saves best checkpoint to `runs/tu-berlin/checkpoint_best.pt`.
- `local_train_ortho.py` — Finetunes a single conv layer with Stiefel-constrained Riemannian Adam. Saves best to `runs/ortho_finetune/checkpoint_best.pt`.
- `ortho_explain_report.py` — Runs the orthogonal concept analysis pipeline: thresholds, PMI, causal deltas (BN-aware knockouts), sprites, thumbs, diagnostics. Writes to `runs/ortho_explainer/`.
- `ortho_report_to_llm.py` — Sends each concept (from `concepts.jsonl`) to an LLM via Replicate to obtain short, human-readable labels; writes `llm_output.jsonl`.
- `why_did_you.py` — Command-line “why” explainer for a single image using the artifacts above; optionally adds a one-sentence LLM summary.
- `app.py` — Gradio UI to upload an image and see the “why” analysis and LLM explanation.
- `download_dataset.py` — Minimal helper to snapshot the TU‑Berlin dataset locally (optional; training scripts can also trigger download).
- `runs/` — Outputs: base training, ortho checkpoint, and explainer artifacts live here.


## Environment setup
### Python
- Python 3.10+ recommended
- Windows and CUDA are supported by the scripts (Windows-safe dataloader defaults are used)

### Install dependencies
If you have a GPU, please follow PyTorch’s install matrix for your CUDA version. As a starting point:

```bash
python -m venv .venv
.\.venv\Scripts\activate  # on Windows PowerShell
pip install --upgrade pip

# Core ML stack (install the correct build for your system/CUDA)
pip install torch torchvision

# Project deps
pip install pillow numpy pandas pyarrow pyyaml tqdm gradio huggingface_hub replicate python-dotenv matplotlib
```

### Replicate API token (for LLM calls)
Set an environment variable before running the LLM steps or the Gradio app with LLM enabled:

```bash
$env:REPLICATE_API_TOKEN = "<your-token>"  # PowerShell on Windows
```

You can also create a `.env` file in the repo root with:
```
REPLICATE_API_TOKEN=your_token_here
```


## Dataset: TU‑Berlin sketches
Scripts expect a local snapshot of the TU‑Berlin dataset as Parquet shards downloaded from the Hugging Face dataset `kmewhort/tu-berlin-png`.

- Default path in configs: `D:\TuBerlin`
- On first run, training scripts will call `huggingface_hub.snapshot_download` to fetch data under that folder.
- Structure expected: `D:\TuBerlin\data\*.parquet` plus a README that includes class name mappings.

You can also run the helper once:
```bash
python download_dataset.py
```

If your dataset lives elsewhere, update the `data_root` fields in the config dataclasses inside the scripts.


## Step 1 — Train the base CNN
Script: `train_standard_cnn.py`

- Config class: `TrainConfig`
  - Important fields:
    - `data_root`: dataset path (default `D:\TuBerlin`)
    - `output_dir`: where checkpoints go (default `runs/tu-berlin`)
    - `batch_size`, `epochs`, `lr`, `grad_accum_steps`, etc.
  - Multi‑GPU: automatically enabled if multiple CUDA devices are present (DDP with `mp.spawn`).

Run it:
```bash
python train_standard_cnn.py
```
Outputs:
- Best checkpoint: `runs/tu-berlin/checkpoint_best.pt`
- Optional probe activations: `runs/tu-berlin/probe/` (periodic)


## Step 2 — Finetune one orthogonal layer
Script: `local_train_ortho.py`

- Loads the standard model weights and finetunes only the last 3×3 conv in `layer3`.
- Enforces Stiefel orthogonality on the chosen conv’s filters using a custom `RiemannianAdamStiefel` optimizer:
  - Project gradients to the tangent space
  - Adam moments on-manifold
  - QR retraction back to the Stiefel manifold after each update
- BatchNorm policy: all BN layers frozen except the BN immediately after the target conv, where only running stats update (affine stays frozen). This matches how activations are read out.

Update the config:
- Set `OrthoConfig.ckpt_path` to point at your base checkpoint, e.g. `runs/tu-berlin/checkpoint_best.pt`.

Run it:
```bash
python local_train_ortho.py
```
Outputs:
- Best checkpoint: `runs/ortho_finetune/checkpoint_best.pt`
- TensorBoard logs: `runs/ortho_finetune/tensorboard/`


## Step 3 — Build concept report from the orthogonal layer
Script: `ortho_explain_report.py`

What it does on a stratified sample per class:
- Caches `bn2` activations after the target conv and computes per-channel statistics (`z_max`, `z_avg`).
- Thresholds channels by a high quantile (default q=0.99) to define ON events per image.
- Computes class-conditional PMI and a causal score `Δ⁻` by BN‑aware channel knockout (replace the channel at BN with its baseline, recompute logits, measure change per class).
- Ranks class associations by a combined score S = 2·z(Δ⁻) + 1·z(PMI).
- Saves assets per channel: sprite, top example thumbnails, and a JSONL record with scores and metadata.
- Diagnostics: co-activation vs independent baseline, correlation stats, BN ablation fidelity check.

Update the config:
- `ExplainConfig.ckpt_path`: `runs/ortho_finetune/checkpoint_best.pt`
- `ExplainConfig.data_root`: dataset path
- `ExplainConfig.out_dir`: e.g. `runs/ortho_explainer`

Run it:
```bash
python ortho_explain_report.py
```
Artifacts (under `runs/ortho_explainer/` by default):
- `concepts.jsonl` — one line per channel with top classes, thresholds, examples, auto summary
- `thresholds.json` — per-channel ON thresholds and quantile q
- `meta.json` — dataset and model metadata
- `activation_index.json` — for sampled images, which concepts fired
- `sprites/` and `thumbs/` — visual evidence per concept
- `coactivation.json`, `correlations.json` — diagnostics
- `ablation_fidelity.json` — verifies the BN-aware knockout behavior


## Step 4 — Summarize concepts with an LLM
Script: `ortho_report_to_llm.py`

This takes each concept (channel) from `concepts.jsonl`, renders annotated top-5 thumbnails, and prompts an LLM (via Replicate) to produce a short human-readable label. Results are written to `llm_output.jsonl` and used by the “why” tools.

Prepare a system prompt file (required): `runs/ortho_explainer/system_prompt.txt`

Example content you can start with:
```text
You label a single orthogonal CNN concept using the JSON and 5 images.
Reply in EXACTLY two sections:
1) SHORT ANSWER: <max 12 words>
2) (blank line) then EVIDENCE SUMMARY: bullet points about what the neuron fires on.
Be crisp and visual; do not mention neurons or internals.
```

Run it (defaults shown):
```bash
python ortho_report_to_llm.py \
  --explain_dir runs/ortho_explainer \
  --system_prompt runs/ortho_explainer/system_prompt.txt \
  --output runs/ortho_explainer/llm_output.jsonl \
  --model openai/gpt-5-mini \
  --max_workers 8
```
Make sure `REPLICATE_API_TOKEN` is set.


## Step 5 — Ask “why did you choose that class?”
Script: `why_did_you.py`

- Loads the orthogonal checkpoint and explainer artifacts (thresholds, concepts, LLM short answers).
- Runs one image through the model, finds the top active concepts supporting/suppressing the predicted class (ranked by |Δ⁻| then activation).
- Optionally calls an LLM to compress the reasoning into one sentence.

Edit the top of `why_did_you.py` to set:
- `USER_IMAGE` — path to the image to explain (any grayscale image is OK; it will be normalized like training)
- Optional: `USER_CKPT`, `USER_EXPLAIN_DIR`, `USER_TOP_K`, `USER_NO_LLM`, `USER_SYSTEM_PROMPT_FILE`, `USER_LLM_MODEL`

Run it:
```bash
python why_did_you.py
```
Example output includes the predicted class/probability, top‑5 classes, and the ranked list of active orthogonal directions with their z, threshold, Δ⁻, PMI, and the LLM’s short label.


## Web UI (Gradio)
Script: `app.py`

Launch the UI:
```bash
python app.py
```
- Upload an image; the app runs the same “why” analysis and (optionally) the one-sentence LLM summary.
- Edit the sidebar to change Top‑K, paths to `checkpoint_best.pt` and `runs/ortho_explainer`, system prompt, and model id.
- The app warns if core files are missing (checkpoint, concepts/thresholds).


## How orthogonality is enforced (implementation sketch)
- Target: last 3×3 conv in `layer3` of ResNet‑18 (`BasicBlock.conv2`).
- Flatten conv weights to a tall matrix `W ∈ R^{n×p}` with columns as filters; maintain `WᵀW = I`.
- Optimizer `RiemannianAdamStiefel`:
  - Project Euclidean gradient to tangent: `G_R = G − W·sym(WᵀG)`
  - Adam moments on `G_R`
  - Retract with QR: `W ← qf(W − η·step)` with sign-fix on `R` diagonal
  - Transport first moment by tangent projection at the new point
- BN policy guarantees that “knockout at BN” faithfully removes the target channel’s contribution while keeping running-stat baselines.


## Metrics and diagnostics
- **Orthogonality** (reported during finetune): `∥WᵀW − I∥_F`, per‑column Fro, max off‑diagonal.
- **PMI vs class**: association between channel ON events and class labels.
- **Causal Δ⁻**: logit drop when removing channel at BN; negative supports class, positive suppresses it.
- **Co-activation**: `Pr(I_i=1 & I_j=1)` vs independent baseline `p_i p_j`.
- **Correlations**: unconditional and class‑residualized Pearson on `z_avg`.
- **Ablation fidelity**: checks that BN‑aware knockout yields `ReLU(beta)` as the residual activation baseline.


## Artifacts produced
- `runs/tu-berlin/checkpoint_best.pt` — base model checkpoint
- `runs/ortho_finetune/checkpoint_best.pt` — orthogonally constrained model (same architecture; only one conv updated)
- `runs/ortho_explainer/` — concept library and diagnostics used by the explainers and the app


## Customization tips
- To pick a different conv: adjust `pick_target_conv(...)` in the scripts.
- To change ON thresholding: modify `quantile_q` in `ExplainConfig`.
- To change the LLM or prompt: pass different `--model` and `--system_prompt` to `ortho_report_to_llm.py`.
- To run on CPU only: everything works but will be slower; set small batch sizes.


## Troubleshooting
- **Missing files in the app**: ensure `runs/ortho_finetune/checkpoint_best.pt`, `runs/ortho_explainer/concepts.jsonl`, and `thresholds.json` exist.
- **Cannot load base checkpoint in ortho finetune**: set `OrthoConfig.ckpt_path` to your base CNN checkpoint (e.g., `runs/tu-berlin/checkpoint_best.pt`).
- **LLM steps fail**: check `REPLICATE_API_TOKEN`; reduce `--max_workers` on low‑memory systems.
- **CUDA out of memory**: lower `batch_size` and/or increase `grad_accum_steps`.
- **Windows dataloader issues**: the scripts default to safe `num_workers=0` in places; keep it small on Windows.


## Credits and citations
- Dataset: TU‑Berlin sketch dataset (`kmewhort/tu-berlin-png` snapshot). Please cite the original dataset if you publish results.
- Backbone: torchvision ResNet‑18 (ImageNet weights).
- Riemannian optimization on Stiefel manifold: QR retraction and tangent‑space projection as implemented in this repo.


## License
No license file is included in this repository. If you plan to release or reuse parts of this work, please add an appropriate license.
