import os
import io
import math
import random
import json
from dataclasses import dataclass
from typing import Iterator, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image, ImageFilter
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import snapshot_download


@torch.no_grad()
def stiefel_qr_retraction(Y: torch.Tensor) -> torch.Tensor:
    # Y: (n, p), n >= p
    Q, R = torch.linalg.qr(Y, mode='reduced')
    # Cheap, stable column sign-fix (no diag mat)
    d = torch.sign(torch.diag(R))
    d[d == 0] = 1
    Q = Q * d  # broadcast over columns
    return Q

def stiefel_project_tangent(W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    # W, G: (n, p), with W^T W = I
    WTG = W.T @ G
    sym = 0.5 * (WTG + WTG.T)
    return G - W @ sym

def stiefel_orth_error(W: torch.Tensor):
    # Returns (fro_norm, per_col, max_offdiag)
    p = W.shape[1]
    M = W.T @ W
    identity = torch.eye(p, device=W.device, dtype=W.dtype)
    E = M - identity
    fro = torch.linalg.norm(E).item()
    per_col = fro / p
    off = M - torch.diag(torch.diag(M))
    max_off = off.abs().max().item() if p > 1 else 0.0
    return fro, per_col, max_off


class RiemannianAdamStiefel(torch.optim.Optimizer):
    """
    Riemannian Adam for a single Conv2d weight, enforcing column-orthonormality
    on W_flat ∈ R^{(in*kh*kw) × out}. Transport only the first moment m.
    """
    def __init__(
        self,
        params: Iterator[nn.Parameter],
        conv_shape: Tuple[int, int, int, int],  # (out, in, kh, kw)
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.out, self.in_ch, self.kh, self.kw = conv_shape

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr    = group['lr']
            beta1, beta2 = group['betas']
            eps   = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                out, in_ch, kh, kw = p.shape
                # Flatten to (n, p): columns are filters
                W = p.data.view(out, -1).t().contiguous()
                G = p.grad.view(out, -1).t().contiguous()

                # Ensure on-manifold at base point
                W = stiefel_qr_retraction(W)

                # State
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(W)  # first moment
                    state['v'] = torch.zeros_like(W)  # second moment (no transport)
                state['step'] += 1
                t = state['step']
                m = state['m']
                v = state['v']

                # Riemannian (tangent) grad
                G_R = stiefel_project_tangent(W, G)

                # Adam moments
                m.mul_(beta1).add_(G_R, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(G_R, G_R, value=1 - beta2)
                m_hat = m / (1 - beta1**t)
                v_hat = v / (1 - beta2**t)

                # Tangent step
                step_dir = m_hat / (v_hat.sqrt() + eps)
                Y = W - lr * step_dir

                # Retract
                W_new = stiefel_qr_retraction(Y)

                # Transport first moment to new tangent (cheap projection)
                m.copy_(stiefel_project_tangent(W_new, m))
                # Do NOT transport v

                # Write back
                p.data.copy_(W_new.t().contiguous().view(out, in_ch, kh, kw))
        return loss


def _register_layer4_hook(model: nn.Module):
    cache = {'A': None}
    def hook(module, input, output):
        cache['A'] = output.detach()
    handle = model.layer4.register_forward_hook(hook)
    return cache, handle

def pick_target_conv(model: nn.Module) -> nn.Conv2d:
    # Use the last 3×3 conv in layer3 of the underlying backbone
    m = model.backbone if hasattr(model, 'backbone') else model
    return m.layer3[-1].conv2

def freeze_all_except_conv(model: nn.Module, target: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in target.parameters():
        p.requires_grad_(True)
    # keep target conv in fp32 (QR is numerically happier)
    target.weight.data = target.weight.data.float()

def set_bn_policy(model: nn.Module, bn_after: nn.BatchNorm2d):
    # Freeze all BN (eval + freeze params)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
    # Except: the BN right after target conv updates running stats (affine still frozen)
    bn_after.train()
    for p in bn_after.parameters():
        p.requires_grad_(False)

@torch.no_grad()
def preproject_conv_to_stiefel(conv: nn.Conv2d):
    assert conv.groups == 1, "Stiefel logic assumes groups=1."
    out, in_ch, kh, kw = conv.weight.shape
    n = in_ch * kh * kw
    p = out
    assert n >= p, f"Need tall matrix (n>=p). Got n={n}, p={p}."
    W = conv.weight.data.view(out, -1).t().contiguous()  # (n, p)
    W = stiefel_qr_retraction(W)
    conv.weight.data.copy_(W.t().contiguous().view(out, in_ch, kh, kw))


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class RandomStrokeJitter:
    def __init__(self, p: float = 0.25, sizes: Tuple[int, int] = (3, 5)):
        self.p = p
        self.sizes = sizes

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        k = random.choice(self.sizes)
        if random.random() < 0.5:
            return img.filter(ImageFilter.MinFilter(size=k))
        else:
            return img.filter(ImageFilter.MaxFilter(size=k))

def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            shear=5,
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomStrokeJitter(p=0.25, sizes=(3, 5)),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, val_tfms

class TuBerlinParquetDataset(Dataset):
    def __init__(self, parquet_paths: List[str], transform=None):
        self.parquet_paths = parquet_paths
        self.transform = transform
        frames: List[pd.DataFrame] = []
        for p in parquet_paths:
            table = pq.read_table(p)
            frames.append(table.to_pandas())
        self.df = pd.concat(frames, ignore_index=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_dict = row["image"]
        label = int(row["label"])  # ensure int
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale PIL
        if self.transform is not None:
            img = self.transform(img)
        return img, label

class TransformSubset(Dataset):
    def __init__(self, base_df: pd.DataFrame, indices: List[int], transform):
        self.base_df = base_df
        self.indices = list(indices)
        self.transform = transform
        
    def __len__(self):
        return len(self.indices)
        
    def __getitem__(self, i):
        idx = self.indices[i]
        row = self.base_df.iloc[idx]
        img_dict = row["image"]
        label = int(row["label"])  # ensure int
        img_bytes = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img = self.transform(img)
        return img, label

def ensure_dataset_downloaded(data_root: str) -> None:
    if not os.path.exists(data_root) or not os.path.exists(os.path.join(data_root, "data")):
        print(f"Dataset not found at {data_root}. Downloading...")
        snapshot_download(
            repo_id="kmewhort/tu-berlin-png",
            repo_type="dataset",
            local_dir=data_root
        )
        print(f"Dataset downloaded to {data_root}")
    else:
        print(f"Dataset found at {data_root}")

def load_label_decoder(data_root: str) -> Dict[int, str]:
    readme_path = os.path.join(data_root, "README.md")
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find("class_label:")
    if start == -1:
        raise ValueError("class_label section not found in README.md")
    start = content.find("names:", start)
    if start == -1:
        raise ValueError("names section not found in README.md")
    end = content.find("splits:", start)
    yaml_str = content[start:end]
    yaml_str = "names:\n" + "\n".join(line for line in yaml_str.splitlines()[1:])
    import yaml
    names = yaml.safe_load(yaml_str)["names"]
    return {int(k): v for k, v in names.items()}

def find_parquet_files(data_root: str) -> List[str]:
    parquet_dir = os.path.join(data_root, "data")
    if not os.path.isdir(parquet_dir):
        raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")
    files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {parquet_dir}")
    return sorted(files)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    pbar = tqdm(loader, desc="Validation", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        pbar.set_postfix({'acc': f'{(correct / max(1,total)):.3f}'})
    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

@torch.no_grad()
def compute_A_stats(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, List[float]]:
    model.eval()
    sums = None
    sumsqs = None
    count = 0
    cache, handle = _register_layer4_hook(model)
    try:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            cache['A'] = None
            _ = model(images)
            A = cache['A']
            if A is None:
                continue
            B, C, H, W = A.shape
            a = A.permute(1, 0, 2, 3).contiguous().view(C, -1)
            if sums is None:
                sums = a.sum(dim=1)
                sumsqs = (a * a).sum(dim=1)
            else:
                sums += a.sum(dim=1)
                sumsqs += (a * a).sum(dim=1)
            count += a.shape[1]
    finally:
        handle.remove()
    if count == 0:
        return {"mean": [], "std": []}
    mean = (sums / count).tolist()
    var = (sumsqs / count - (sums / count) ** 2).clamp_min(0)
    std = var.sqrt().tolist()
    return {"mean": mean, "std": std}

@dataclass
class OrthoConfig:
    data_root: str = r"D:\TuBerlin"
    output_dir: str = r"runs\ortho_finetune"
    image_size: int = 224
    batch_size: int = 4
    workers: int = 0  # Windows safe default
    epochs: int = 20
    lr: float = 1e-4
    warmup_epochs: int = 2
    val_every: int = 1
    log_probe_every: int = 5
    seed: int = 42
    # Single expected checkpoint path; if missing or incompatible, raise error
    ckpt_path: str = "./checkpoint_best.pt"
    grad_accum_steps: int = 16
    # ReduceLROnPlateau
    plateau_patience: int = 3
    plateau_factor: float = 0.5
    plateau_min_lr: float = 1e-6
    # Early stopping
    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_loaders(cfg: OrthoConfig) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    train_tfms, val_tfms = build_transforms(cfg.image_size)
    parquet_paths = find_parquet_files(cfg.data_root)
    full_ds = TuBerlinParquetDataset(parquet_paths, transform=None)

    # Simple random split (follow standard script behavior)
    num_total = len(full_ds)
    num_val = max(1, int(0.1 * num_total))
    num_train = num_total - num_val
    generator = torch.Generator().manual_seed(cfg.seed)
    train_indices, val_indices = torch.utils.data.random_split(range(num_total), lengths=[num_train, num_val], generator=generator)

    train_ds = TransformSubset(full_ds.df, list(train_indices.indices), train_tfms)
    val_ds = TransformSubset(full_ds.df, list(val_indices.indices), val_tfms)

    # Probe set: fixed subset from validation set
    probe_size = min(200, len(val_ds))
    probe_indices = list(range(probe_size))
    probe_ds = torch.utils.data.Subset(val_ds, probe_indices)

    # Infer num_classes from labels present
    num_classes = int(max(full_ds.df["label"]) + 1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.workers, pin_memory=True)
    probe_loader = DataLoader(probe_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.workers, pin_memory=True)

    return train_loader, val_loader, probe_loader, num_classes


@torch.no_grad()
def log_probe_activations(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: str, step_tag: str, max_items: int = 50):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0
    batch_idx = 0
    cache, handle = _register_layer4_hook(model)
    try:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            cache['A'] = None
            logits = model(images)
            A = cache['A']
            if A is None:
                continue
            for i in range(images.size(0)):
                if saved >= max_items:
                    return
                item = {
                    "target": int(targets[i].item()),
                    "logits": logits[i].detach().cpu().tolist(),
                }
                torch.save(A[i].detach().cpu(), os.path.join(out_dir, f"{step_tag}_A_{batch_idx}_{i}.pt"))
                with open(os.path.join(out_dir, f"{step_tag}_meta_{batch_idx}_{i}.json"), "w") as f:
                    json.dump(item, f)
                saved += 1
            batch_idx += 1
    finally:
        handle.remove()

def load_pretrained_standard_model(model: nn.Module, ckpt_path: str) -> Tuple[bool, int]:
    """Load weights from standard CNN training. Returns (loaded, num_classes_from_ckpt_or_-1).

    Only attempts a single checkpoint path. Raises a RuntimeError on failure.
    """
    device = next(model.parameters()).device

    # Resolve path relative to CWD and script dir; prefer CWD
    candidates: List[str] = []
    candidates.append(os.path.abspath(ckpt_path))
    here = os.path.dirname(__file__)
    candidates.append(os.path.abspath(os.path.join(here, ckpt_path)))
    # de-dup
    dedup: List[str] = []
    seen = set()
    for q in candidates:
        if q not in seen:
            seen.add(q)
            dedup.append(q)

    # Find first existing
    found_path = None
    for q in dedup:
        if os.path.isfile(q):
            found_path = q
            break
    if found_path is None:
        raise FileNotFoundError(f"Standard CNN checkpoint not found at {ckpt_path}. Expected path relative to CWD or script dir.")

    ckpt = torch.load(found_path, map_location=device)

    # Extract state dict from supported formats
    state_dict = None
    num_classes = -1
    if isinstance(ckpt, dict):
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
            num_classes = int(ckpt.get('num_classes', -1))
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            state_dict = ckpt['model']
        else:
            # maybe raw state dict
            candidate_keys = list(ckpt.keys())
            if any(k.startswith('backbone.') or k.startswith('fc.') for k in candidate_keys):
                state_dict = ckpt
    if state_dict is None:
        raise RuntimeError(f"Unrecognized checkpoint format at {found_path}. Available top-level keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")

    # Attempt key remapping if needed (e.g., CamFriendlyResNet to SketchResNet18)
    model_keys = set(model.state_dict().keys())
    provided_keys = set(state_dict.keys())
    overlap = len([k for k in provided_keys if k in model_keys])

    if overlap < 10:
        # Try simple remaps
        remapped = {}
        for k, v in state_dict.items():
            k2 = k
            # Common prefixes that differ across scripts
            if k2.startswith('stem.'):  # CamFriendlyResNet stem -> backbone.*
                k2 = 'backbone.' + k2.replace('stem.', '')
            if k2.startswith('fc.'):   # fc names often compatible
                k2 = k2
            remapped[k2] = v
        provided_keys = set(remapped.keys())
        overlap = len([k for k in provided_keys if k in model_keys])
        if overlap < 10:
            raise RuntimeError(f"Checkpoint at {found_path} appears incompatible with model (only {overlap} keys overlap).")
        state_dict = remapped

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights from {found_path}\nMissing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    return True, num_classes

def main():
    cfg = OrthoConfig()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "tensorboard"))

    # Data (same as standard CNN)
    ensure_dataset_downloaded(cfg.data_root)
    train_loader, val_loader, probe_loader, num_classes = create_loaders(cfg)

    # Model: standard torchvision resnet18, replace final fc for dataset classes
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Load pretrained weights from standard CNN run
    try:
        _loaded, ckpt_num_classes = load_pretrained_standard_model(model, cfg.ckpt_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load standard CNN checkpoint: {e}")

    # Target conv & guardrails
    target_conv = pick_target_conv(model)
    assert isinstance(target_conv, nn.Conv2d)
    assert target_conv.kernel_size == (3,3), "Expected 3×3 conv."
    assert target_conv.groups == 1, "Stiefel logic assumes groups=1."
    n = target_conv.in_channels * target_conv.kernel_size[0] * target_conv.kernel_size[1]
    p = target_conv.out_channels
    assert n >= p, f"Need tall matrix (n>=p). Got n={n}, p={p}."

    # Freeze everything but target conv (and keep it fp32)
    freeze_all_except_conv(model, target_conv)

    # BN after target conv (BasicBlock: bn2) — allow running stats only
    m_backbone = model.backbone if hasattr(model, 'backbone') else model
    block = m_backbone.layer3[-1]
    bn_after = block.bn2

    # Pre-project once to Stiefel
    preproject_conv_to_stiefel(target_conv)

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer (only over target conv weight)
    out, in_ch, kh, kw = target_conv.weight.shape
    riem_opt = RiemannianAdamStiefel(params=[target_conv.weight],
                                     conv_shape=(out, in_ch, kh, kw),
                                     lr=cfg.lr)

    # Scheduler: warmup over optimizer update steps (accum-aware), then ReduceLROnPlateau
    steps_per_epoch = len(train_loader)
    updates_per_epoch = math.ceil(steps_per_epoch / max(1, cfg.grad_accum_steps))
    warmup_updates = cfg.warmup_epochs * updates_per_epoch
    global_update = 0
    global_step = 0  # micro-steps for logging
    best_val_acc = 0.0
    epochs_no_improve = 0

    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        riem_opt,
        mode='max',
        factor=cfg.plateau_factor,
        patience=cfg.plateau_patience,
        threshold=cfg.early_stop_min_delta,
        threshold_mode='abs',
        min_lr=cfg.plateau_min_lr,
        verbose=True,
    )

    # TQDM epoch loop
    epoch_pbar = tqdm(range(cfg.epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:
        model.train()
        set_bn_policy(model, bn_after)

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}", leave=False, unit="batch")
        for batch_idx, (x, y) in enumerate(train_pbar):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # Accumulation bookkeeping
            accumulate_steps = max(1, cfg.grad_accum_steps)
            is_update_step = (((batch_idx + 1) % accumulate_steps) == 0) or ((batch_idx + 1) == steps_per_epoch)

            # Forward / loss scaled for accumulation
            with torch.cuda.amp.autocast(enabled=True):
                logits = model(x)
                loss = criterion(logits, y) / accumulate_steps
            loss.backward()

            # Only step optimizer on accumulation boundary
            if is_update_step:
                # LR schedule: linear warmup by updates, then keep last LR for plateau scheduler to adjust
                if global_update < warmup_updates:
                    lr = cfg.lr * (global_update + 1) / max(1, warmup_updates)
                    for g in riem_opt.param_groups:
                        g['lr'] = lr
                else:
                    lr = riem_opt.param_groups[0]['lr']

                riem_opt.step()
                riem_opt.zero_grad(set_to_none=True)
                global_update += 1
            else:
                # keep last lr for display
                lr = riem_opt.param_groups[0]['lr']

            batch_loss = loss.item() * accumulate_steps
            preds = logits.argmax(1)
            batch_correct = (preds == y).sum().item()
            batch_size = x.size(0)

            epoch_loss += batch_loss * batch_size
            epoch_correct += batch_correct
            epoch_total += batch_size

            # TQDM batch metrics
            train_pbar.set_postfix({'loss': f'{batch_loss:.3f}', 'acc': f'{(batch_correct/batch_size):.3f}', 'lr': f'{lr:.2e}'})

            # TensorBoard batch logs (sparse)
            if batch_idx % 10 == 0:
                writer.add_scalar('Train/BatchLoss', batch_loss, global_step)
                writer.add_scalar('Train/BatchAcc', batch_correct / max(1, batch_size), global_step)
                writer.add_scalar('Train/LearningRate', lr, global_step)

            global_step += 1

        # Epoch aggregates
        train_loss = epoch_loss / max(1, epoch_total)
        train_acc = epoch_correct / max(1, epoch_total)
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('Train/EpochAcc', train_acc, epoch)

        # Validation
        if ((epoch + 1) % cfg.val_every == 0) or (epoch == cfg.epochs - 1):
            val_loss, val_acc = evaluate(model, val_loader, device)
            writer.add_scalar('Val/Loss', val_loss, epoch)
            writer.add_scalar('Val/Acc', val_acc, epoch)

            # Orthogonality metrics
            with torch.no_grad():
                W = target_conv.weight.data.view(out, -1).t().contiguous()
                fro, per_col, max_off = stiefel_orth_error(W)
            writer.add_scalar('Orth/Fro', fro, epoch)
            writer.add_scalar('Orth/PerCol', per_col, epoch)
            writer.add_scalar('Orth/MaxOffDiag', max_off, epoch)

            # Save best and log A stats
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                epochs_no_improve = 0

                A_stats = compute_A_stats(model, val_loader, device)
                if A_stats.get("mean") and A_stats.get("std"):
                    writer.add_histogram('Features/ChannelMeans', torch.tensor(A_stats["mean"]), epoch)
                    writer.add_histogram('Features/ChannelStds', torch.tensor(A_stats["std"]), epoch)

                ckpt = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'cfg': vars(cfg),
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'orth_fro': fro,
                    'orth_per_col': per_col,
                    'orth_max_offdiag': max_off,
                    'num_classes': num_classes,
                    'global_step': global_step,
                }
                torch.save(ckpt, os.path.join(cfg.output_dir, 'checkpoint_best.pt'))
            else:
                epochs_no_improve += 1

            # Step plateau scheduler with validation accuracy
            plateau_sched.step(val_acc)

            # Early stopping check
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best val acc: {best_val_acc:.4f}")
                epoch_pbar.set_postfix({'train_acc': f'{train_acc:.3f}', 'val_acc': f'{best_val_acc:.3f}', 'best': '★'})
                break

            epoch_pbar.set_postfix({'train_acc': f'{train_acc:.3f}', 'val_acc': f'{best_val_acc:.3f}', 'best': '★' if is_best else ''})

        else:
            epoch_pbar.set_postfix({'train_acc': f'{train_acc:.3f}'})

    writer.close()
    print(f"Finished training. Best val acc: {best_val_acc:.4f}")
    print(f"Best checkpoint saved to: {os.path.join(cfg.output_dir, 'checkpoint_best.pt')}")

if __name__ == "__main__":
    main()
