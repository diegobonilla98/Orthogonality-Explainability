import os
import io
import math
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


# =========================
# Config
# =========================

@dataclass
class ExplainConfig:
    # Paths
    data_root: str = r"D:\TuBerlin"                  # path containing "data/*.parquet" and README.md
    ckpt_path: str = r"runs\ortho_finetune\checkpoint_best.pt"
    out_dir: str = r"runs\ortho_explainer"

    # Data / loader
    image_size: int = 224
    batch_size: int = 128
    workers: int = 0  # 0 is safe on Windows

    # Subsample for analysis
    n_per_class: int = 60      # stratified sample per class
    seed: int = 1234

    # Feature activation threshold
    quantile_q: float = 0.99   # detect “ON” by z_max > q-quantile (per channel)

    # Causal knockout sampling
    top_classes_for_delta: int = 5   # by PMI
    control_classes_rand: int = 8
    m_images_per_class: int = 8

    # Assets
    k_top_examples: int = 5
    thumb_side: int = 128

    # Diagnostics
    report_orthogonality: bool = True
    compute_coactivation: bool = True
    compute_correlations: bool = True

# =========================
# Dataset & transforms
# =========================

class TuBerlinParquetDataset(Dataset):
    """
    Loads parquet shards: columns ["image", "label"], where image stores {"bytes": ...}.
    """
    def __init__(self, parquet_paths: List[str], transform=None):
        self.parquet_paths = parquet_paths
        self.transform = transform
        frames = []
        for p in parquet_paths:
            table = pq.read_table(p, columns=["image", "label"])
            frames.append(table.to_pandas())
        import pandas as pd
        self.df = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_dict = row["image"]
        label = int(row["label"])
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def build_transforms(image_size: int):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tfm

def find_parquet_files(data_root: str) -> List[str]:
    parquet_dir = os.path.join(data_root, "data")
    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")
    return sorted(files)

def load_label_decoder(data_root: str) -> Dict[int, str]:
    """
    Parse README.md from the dataset snapshot to decode class labels.
    """
    readme_path = os.path.join(data_root, "README.md")
    if not os.path.isfile(readme_path):
        return {}
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find("class_label:")
    if start == -1:
        return {}
    start = content.find("names:", start)
    if start == -1:
        return {}
    end = content.find("splits:", start)
    yaml_str = content[start:end]
    yaml_str = "names:\n" + "\n".join(line for line in yaml_str.splitlines()[1:])
    import yaml
    names = yaml.safe_load(yaml_str)["names"]
    return {int(k): v for k, v in names.items()}

# =========================
# Model helpers
# =========================

def build_model(num_classes: int, pretrained_backbone: bool = True) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained_backbone else None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def pick_target_conv(model: nn.Module) -> nn.Conv2d:
    """
    Match the layer used during orthogonal finetuning: last 3x3 conv in layer3 (BasicBlock.conv2).
    """
    m = model
    return m.layer3[-1].conv2

def get_bn_after_target(model: nn.Module) -> nn.BatchNorm2d:
    block = model.layer3[-1]
    return block.bn2

def module_path(model: nn.Module, target: nn.Module) -> str:
    for name, mod in model.named_modules():
        if mod is target:
            return name
    return "unknown"

# =========================
# Ortho diagnostics
# =========================

@torch.no_grad()
def stiefel_orth_error_from_conv(conv: nn.Conv2d) -> Dict[str, float]:
    """
    Compute ||W^T W - I||_F and max off-diagonal for flattened conv weights (columns=filters).
    """
    out, in_ch, kh, kw = conv.weight.shape
    W = conv.weight.data.view(out, -1).t().contiguous()  # (n, p)
    p = W.shape[1]
    M = W.T @ W
    I = torch.eye(p, device=W.device, dtype=W.dtype)
    E = M - I
    fro = torch.linalg.norm(E).item()
    off = M - torch.diag(torch.diag(M))
    max_off = off.abs().max().item() if p > 1 else 0.0
    return {"fro": fro, "max_offdiag": max_off, "p": float(p), "n": float(W.shape[0])}

# =========================
# Hook utilities
# =========================

class CachedActivation:
    """
    Cache a single module's forward output.
    """
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

class BNAwareChannelKnockout:
    """
    Knock out selected channels at BN (bn2) by replacing with BN baseline:
      - If affine: set to beta (bn.bias).
      - Else: set to 0.
    This matches "remove conv2 contribution under eval BN stats" and avoids leakage
    from running-mean offsets. Works pre-residual-add.
    """
    def __init__(self, bn: nn.BatchNorm2d, channels: List[int]):
        self.channels = list(set(int(c) for c in channels))
        self.bn = bn
        self.h = bn.register_forward_hook(self._hook)

    def _hook(self, m, inp, out):
        out = out.clone()
        if getattr(self.bn, "affine", False) and (self.bn.bias is not None):
            beta = self.bn.bias[self.channels].view(1, -1, 1, 1)
            out[:, self.channels, :, :] = beta
        else:
            out[:, self.channels, :, :] = 0
        return out

    def remove(self):
        self.h.remove()

# =========================
# Utilities
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_square_thumbnail(pil_img: Image.Image, side: int = 128) -> Image.Image:
    w, h = pil_img.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), color=255)
    canvas.paste(pil_img, ((s - w)//2, (s - h)//2))
    canvas = canvas.resize((side, side), resample=Image.BICUBIC)
    return canvas

def zscore(v: np.ndarray, axis=None, eps=1e-8):
    m = v.mean(axis=axis, keepdims=True)
    s = v.std(axis=axis, keepdims=True)
    return (v - m) / (s + eps)

# =========================
# Main pipeline
# =========================

def main():
    cfg = ExplainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.out_dir, exist_ok=True)
    sprites_dir = Path(cfg.out_dir) / "sprites"; sprites_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir  = Path(cfg.out_dir) / "thumbs";  thumbs_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path  = Path(cfg.out_dir) / "concepts.jsonl"
    thresholds_path = Path(cfg.out_dir) / "thresholds.json"
    actindex_path = Path(cfg.out_dir) / "activation_index.json"
    meta_path = Path(cfg.out_dir) / "meta.json"
    coact_path = Path(cfg.out_dir) / "coactivation.json"
    corr_path  = Path(cfg.out_dir) / "correlations.json"
    abl_fidelity_path = Path(cfg.out_dir) / "ablation_fidelity.json"

    # ---------- Load checkpoint ----------
    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    num_classes = int(ckpt.get("num_classes", -1))
    if num_classes <= 0:
        raise RuntimeError("num_classes not found in checkpoint; ensure ortho finetune saved it.")

    model = build_model(num_classes=num_classes, pretrained_backbone=False).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"[load] Missing={len(missing)} Unexpected={len(unexpected)}")

    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    # ---------- Identify target layer ----------
    target_conv = pick_target_conv(model)
    bn_after    = get_bn_after_target(model)
    layer_path  = module_path(model, target_conv)
    print(f"[target] Using layer: {layer_path}  (channels={target_conv.out_channels})")

    # ---------- Optional orthogonality report ----------
    ortho = stiefel_orth_error_from_conv(target_conv) if cfg.report_orthogonality else None
    if ortho:
        print(f"[ortho] Fro ||W^T W - I||={ortho['fro']:.3e} | max offdiag={ortho['max_offdiag']:.3e} | n={int(ortho['n'])}, p={int(ortho['p'])}")

    # ---------- Data ----------
    parquet_paths = find_parquet_files(cfg.data_root)
    tfm = build_transforms(cfg.image_size)
    full_ds = TuBerlinParquetDataset(parquet_paths, transform=tfm)

    # Build label names if available
    label_decoder = load_label_decoder(cfg.data_root)
    class_names = [label_decoder.get(i, f"class_{i:03d}") for i in range(num_classes)]

    # Stratified subsample indices (global indices)
    import pandas as pd
    labels_series = full_ds.df["label"].astype(int)
    by_class: Dict[int, List[int]] = {}
    for gi, lab in enumerate(labels_series.tolist()):
        by_class.setdefault(int(lab), []).append(int(gi))

    rng = np.random.RandomState(cfg.seed)
    picked_global_indices: List[int] = []
    for c, idxs in by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        picked_global_indices.extend(idxs[:cfg.n_per_class])
    picked_global_indices = sorted(picked_global_indices)
    N = len(picked_global_indices)
    print(f"[sample] N={N} images ({cfg.n_per_class} per class × {len(by_class)} classes)")

    # Loader over subsample
    class SubsetDataset(Dataset):
        def __init__(self, base: TuBerlinParquetDataset, indices: List[int]):
            self.base = base
            self.indices = list(indices)
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            gi = self.indices[i]
            x, y = self.base[gi]
            return x, int(y), int(gi)
    sub_ds = SubsetDataset(full_ds, picked_global_indices)
    loader = DataLoader(sub_ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.workers, pin_memory=True, drop_last=False)

    # ---------- Cache activations at bn2 ----------
    P = target_conv.out_channels
    z_max = torch.zeros((N, P), dtype=torch.float32)
    z_avg = torch.zeros((N, P), dtype=torch.float32)
    labels_np = np.zeros(N, dtype=np.int64)
    global_idx_np = np.zeros(N, dtype=np.int64)

    cache_bn = CachedActivation(bn_after)
    with torch.no_grad():
        offset = 0
        for x, y, gi in tqdm(loader, desc="Caching concept activations (bn2→ReLU)", ncols=100):
            b = x.size(0)
            x = x.to(device, non_blocking=True)
            _ = model(x)  # triggers hook
            A = cache_bn.pop()  # (B, P, H, W) pre-residual, post-BN
            if A is None:
                raise RuntimeError("bn2 hook returned None.")
            A_pos = F.relu(A)  # define ON by bn2→ReLU
            zmax = A_pos.amax(dim=(-1, -2))   # (B, P)
            zmean = A_pos.mean(dim=(-1, -2))  # (B, P)
            z_max[offset:offset+b] = zmax.cpu()
            z_avg[offset:offset+b] = zmean.cpu()
            labels_np[offset:offset+b] = y.numpy()
            global_idx_np[offset:offset+b] = gi.numpy()
            offset += b
    cache_bn.remove()

    # ---------- Thresholds & PMI ----------
    q = cfg.quantile_q
    tau = torch.quantile(z_max, q=q, dim=0).numpy()      # (P,)
    Z = z_max.numpy()                                    # (N, P)
    I_on = (Z > tau.reshape(1, -1))                      # boolean (N, P)
    n_i = I_on.sum(axis=0)                               # (P,)
    p_hat_i = n_i / float(N)                             # sparsity

    # mean-on (for logging)
    mu_on = np.zeros(P, dtype=np.float32)
    for i in range(P):
        idx = np.where(I_on[:, i])[0]
        mu_on[i] = float(Z[idx, i].mean()) if idx.size > 0 else 0.0

    classes = np.sort(np.unique(labels_np))
    K = classes.size
    lam = 1.0
    # class priors
    n_c = np.array([np.sum(labels_np == c) for c in classes], dtype=np.int64)
    P_c = (n_c + lam) / (N + lam * K)

    # n_{i,c}
    n_i_c = np.zeros((P, K), dtype=np.int64)
    for k_idx, c in enumerate(classes):
        mask_c = (labels_np == c)
        I_c = I_on[mask_c, :]
        n_i_c[:, k_idx] = I_c.sum(axis=0)
    P_I1 = (n_i + lam) / (N + lam)  # not used in ranking
    P_c_given_I1 = (n_i_c + lam) / (n_i.reshape(-1, 1) + lam * K)  # (P, K)
    PMI = np.log(np.maximum(P_c_given_I1, 1e-12)) - np.log(P_c.reshape(1, -1))  # (P, K)

    # ---------- Helper: causal Δ^- by knockout at bn2 ----------
    def delta_minus_for_feature(
        feature_i: int,
        eval_k_indices: List[int],                 # class index positions (0..K-1) to evaluate
        m_images_per_class: int,
        seed_local: int = 1234
    ) -> Dict[int, float]:
        """
        Return dict {abs_class_label: mean delta_logit_c} for classes in eval_k_indices.
        Δy_c = y_c(x) - y_c(x | bn2[channel i] := β). Average over images in that class with feature_i ON.
        """
        rng_local = np.random.RandomState(seed_local + feature_i)
        results: Dict[int, float] = {}

        for kk in eval_k_indices:
            abs_c = int(classes[kk])
            idxs = np.where((labels_np == abs_c) & I_on[:, feature_i])[0]
            if idxs.size == 0:
                continue
            rng_local.shuffle(idxs)
            idxs = idxs[:m_images_per_class]

            cur_ds = torch.utils.data.Subset(sub_ds, indices=idxs.tolist())
            cur_loader = DataLoader(cur_ds, batch_size=min(len(cur_ds), cfg.batch_size),
                                    shuffle=False, num_workers=0, pin_memory=True)
            deltas: List[float] = []

            with torch.no_grad():
                for x, y, gi in cur_loader:
                    x = x.to(device, non_blocking=True)
                    # baseline logits
                    logits_base = model(x)
                    # knockout at bn2 with baseline-preserving replacement
                    ko = BNAwareChannelKnockout(bn_after, [feature_i])
                    logits_ko = model(x)
                    ko.remove()
                    deltas.extend((logits_base[:, abs_c] - logits_ko[:, abs_c]).detach().cpu().tolist())

            if len(deltas) > 0:
                results[abs_c] = float(np.mean(deltas))

        return results

    # ---------- Assets helpers ----------
    def save_sprite_and_thumbs(fid: int, top_examples: List[Tuple[int, float]]):
        thumbs_info = []
        tiles = []
        for k, (j_idx, zval) in enumerate(top_examples):
            gi = int(global_idx_np[j_idx])
            row = full_ds.df.iloc[gi]
            img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            pil = Image.open(io.BytesIO(img_bytes)).convert("L")
            thumb = to_square_thumbnail(pil, side=cfg.thumb_side)
            tpath = Path(thumbs_dir) / f"feature_{fid:05d}_img{k}.webp"
            thumb.save(tpath, "WEBP", quality=90)
            tiles.append(thumb.convert("L"))
            thumbs_info.append({
                "thumb": str(tpath.as_posix()),
                "label": class_names[int(row["label"])],
                "z": float(zval),
                "global_index": int(gi),
            })
        # sprite (horizontal)
        sprite = Image.new("L", (cfg.thumb_side * len(tiles), cfg.thumb_side), color=255)
        for i, tile in enumerate(tiles):
            sprite.paste(tile, (i * cfg.thumb_side, 0))
        spath = Path(sprites_dir) / f"feature_{fid:05d}.webp"
        sprite.save(spath, "WEBP", quality=90)
        return str(spath.as_posix()), thumbs_info

    # Keep an activation index: which concept IDs are ON for each sampled image (handy later)
    activation_index: Dict[int, List[str]] = {int(gi): [] for gi in global_idx_np.tolist()}

    # ---------- Concept loop ----------
    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    label_to_k = {int(c): int(i) for i, c in enumerate(classes)}
    k_to_label = {int(i): int(c) for i, c in enumerate(classes)}

    for i in tqdm(range(P), desc="Interpreting orthogonal concepts (bn2-aware)", ncols=100):
        sparsity = float(p_hat_i[i])
        threshold = float(tau[i])
        mean_on   = float(mu_on[i])

        pmi_i = PMI[i, :]
        top_by_pmi = np.argsort(-pmi_i)[:cfg.top_classes_for_delta]
        # add random controls
        all_k = list(range(K))
        rng.shuffle(all_k)
        controls = []
        for kk in all_k:
            if kk not in top_by_pmi:
                controls.append(kk)
            if len(controls) >= cfg.control_classes_rand:
                break
        eval_k = list(np.unique(np.concatenate([top_by_pmi, np.array(controls)], axis=0)).tolist())

        # Δ^- via BN-aware knockout
        delta_map = delta_minus_for_feature(
            feature_i=i,
            eval_k_indices=eval_k,
            m_images_per_class=cfg.m_images_per_class,
            seed_local=cfg.seed
        )
        delta_vec = np.zeros(K, dtype=np.float32)
        have_delta = np.zeros(K, dtype=bool)
        for c_abs, val in delta_map.items():
            kk = label_to_k[c_abs]
            delta_vec[kk] = float(val)
            have_delta[kk] = True

        # normalized scores
        zPMI = zscore(pmi_i.copy())
        if have_delta.any():
            zDelta = np.zeros_like(delta_vec)
            sub = delta_vec[have_delta]
            zDelta[have_delta] = zscore(sub)
        else:
            zDelta = np.zeros_like(delta_vec)

        # Combined ranking (heavier weight to Δ^- which is causal)
        S = 2.0 * zDelta + 1.0 * zPMI

        # Top examples by z_i(x)
        z_col = Z[:, i]
        top_j = np.argsort(-z_col)[:cfg.k_top_examples]
        top_examples = [(int(j_idx), float(z_col[int(j_idx)])) for j_idx in top_j]
        sprite_path, thumbs_info = save_sprite_and_thumbs(i, top_examples)

        # Per-class counts n_{i,c}
        i_counts = n_i_c[i, :]

        # Build top classes table
        topK = 5
        top_classes_idx = np.argsort(-S)[:topK]
        top_classes = []
        for kk in top_classes_idx:
            top_classes.append({
                "class": class_names[int(k_to_label[kk])],
                "S": float(S[kk]),
                "PMI": float(pmi_i[kk]),
                "delta_minus": float(delta_vec[kk]) if have_delta[kk] else 0.0,
                "count_on": int(i_counts[kk]),
            })

        concept_id = f"resnet18.{layer_path}:{i:05d}"

        # Auto summary
        if len(top_classes) > 0:
            best = top_classes[0]
            dm = best.get("delta_minus", 0.0)
            sign_word = "increases" if dm > 0 else "reduces"
            summary = (f"{concept_id} fires on {sparsity*100:.1f}% images; "
                       f"knockout {sign_word} '{best['class']}' logit by {abs(dm):.2f}. "
                       f"Top classes: " + ", ".join([tc['class'] for tc in top_classes]))
        else:
            summary = f"{concept_id} fires on {sparsity*100:.1f}% images; no strong class association."

        obj = {
            "concept_id": concept_id,
            "layer": layer_path,
            "channel_index": int(i),
            "threshold_q": float(q),
            "threshold": float(threshold),
            "sparsity": float(sparsity),
            "mean_on_activation": float(mean_on),
            "top_classes": top_classes,
            "sprite_path": sprite_path,
            "top_examples": thumbs_info,
            "auto_summary": summary,
        }
        jsonl_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # Update activation index for the subsample (who activated this concept)
        on_images = np.where(I_on[:, i])[0]
        for j in on_images:
            gi = int(global_idx_np[j])
            activation_index[gi].append(concept_id)

    jsonl_f.close()

    # ---------- Diagnostics: co-activation & correlations ----------
    if cfg.compute_coactivation:
        # C_ij vs independent baseline p_i p_j (symmetric summaries)
        p_i = p_hat_i  # (P,)
        C = (I_on.T @ I_on) / float(N)  # Pr(I_i=1 & I_j=1)
        Indep = np.outer(p_i, p_i)
        excess = C - Indep
        coact = {
            "mean_abs_excess": float(np.mean(np.abs(excess))),
            "max_excess": float(np.max(excess)),
            "top_pairs": [
                {"i": int(i), "j": int(j), "excess": float(excess[i, j])}
                for i, j in zip(*np.unravel_index(np.argsort(-excess.ravel())[:50], excess.shape))
                if i < j
            ],
        }
        with open(coact_path, "w", encoding="utf-8") as f:
            json.dump(coact, f, ensure_ascii=False, indent=2)

    if cfg.compute_correlations:
        # Unconditional Pearson on z_avg and class-conditional (residualized) Pearson
        Za = z_avg.numpy()  # (N, P)
        # unconditional
        corr_uncond = np.corrcoef(Za, rowvar=False)
        # residualize by class mean
        Za_res = Za.copy()
        for c in classes:
            mask = (labels_np == c)
            if mask.any():
                Za_res[mask] -= Za[mask].mean(axis=0, keepdims=True)
        corr_cond = np.corrcoef(Za_res, rowvar=False)
        corrs = {
            "summary": {
                "uncond_mean_abs": float(np.nanmean(np.abs(corr_uncond[np.isfinite(corr_uncond)]))),
                "cond_mean_abs": float(np.nanmean(np.abs(corr_cond[np.isfinite(corr_cond)]))),
            }
        }
        with open(corr_path, "w", encoding="utf-8") as f:
            json.dump(corrs, f, ensure_ascii=False, indent=2)

    # ---------- Ablation fidelity (sanity) ----------
    # Verify that bn2-aware knockout drives A_pos[channel] ≈ ReLU(beta)
    # on a small probe batch for a few random channels.
    abl_report = {}
    probe_loader = DataLoader(sub_ds, batch_size=min(64, cfg.batch_size),
                              shuffle=True, num_workers=0, pin_memory=True)
    x_probe, y_probe, gi_probe = next(iter(probe_loader))
    x_probe = x_probe.to(device, non_blocking=True)
    with torch.no_grad():
        # baseline bn2 activations
        cache0 = CachedActivation(bn_after)
        _ = model(x_probe)
        A0 = cache0.pop()  # (B,P,H,W)
        cache0.remove()
        # test a handful of channels
        test_channels = list(np.linspace(0, P-1, num=min(P, 8), dtype=int))
        for ch in test_channels:
            ko = BNAwareChannelKnockout(bn_after, [ch])
            cache1 = CachedActivation(bn_after)
            _ = model(x_probe)
            A1 = cache1.pop()
            cache1.remove()
            ko.remove()
            # compare ReLU(A1[:,ch]) to ReLU(beta)
            if getattr(bn_after, "affine", False) and (bn_after.bias is not None):
                beta = float(bn_after.bias[ch].detach().cpu().item())
            else:
                beta = 0.0
            target = max(beta, 0.0)
            A1_pos = F.relu(A1[:, ch:ch+1, :, :]).mean(dim=(0,2,3)).squeeze(0).detach().cpu().numpy()  # ( )
            mae = float(np.mean(np.abs(A1_pos - target)))
            abl_report[int(ch)] = {"relu_beta": float(target), "mean_abs_error": mae}
    with open(abl_fidelity_path, "w", encoding="utf-8") as f:
        json.dump(abl_report, f, ensure_ascii=False, indent=2)

    # ---------- Save thresholds & activation index & meta ----------
    thresholds = {
        "layer": layer_path,
        "concept_thresholds": {f"resnet18.{layer_path}:{i:05d}": float(tau[i]) for i in range(P)},
        "quantile_q": float(q),
    }
    with open(thresholds_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)

    with open(actindex_path, "w", encoding="utf-8") as f:
        json.dump(activation_index, f, ensure_ascii=False)

    meta = {
        "ckpt_path": os.path.abspath(cfg.ckpt_path),
        "data_root": os.path.abspath(cfg.data_root),
        "num_classes": int(num_classes),
        "class_names": class_names,
        "target_layer": layer_path,
        "orthogonality": ortho if ortho else {},
        "subsample": {"n_per_class": cfg.n_per_class, "N_total": N},
        "config": vars(cfg),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[done] Concepts JSONL : {jsonl_path}")
    print(f"[done] Thresholds     : {thresholds_path}")
    print(f"[done] Act index      : {actindex_path}")
    print(f"[done] Sprites dir    : {sprites_dir}")
    print(f"[done] Thumbs  dir    : {thumbs_dir}")
    print(f"[done] Coactivation   : {coact_path}")
    print(f"[done] Correlations   : {corr_path}")
    print(f"[done] Ablation check : {abl_fidelity_path}")

if __name__ == "__main__":
    main()
