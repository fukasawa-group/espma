"""
PIACO2 training script.

Point clouds are forwarded with xyz + all per-point feature channels
(unless --nullify_points is set).
ESM-2 mean+max pooled embeddings (2560-d) are loaded from HDF5 files
when --esm_pooling is set.
"""

import argparse
import datetime
import logging
import os
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch.utils.data import Dataset
from torchsampler import ImbalancedDatasetSampler
from tqdm import tqdm

import utils.provider as provider
from utils.provider import EarlyStopping

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "model"))
sys.path.append(os.path.join(BASE_DIR, "utils"))

from model.piaco2_architecture import Piaco2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PIACO2 training")

    # Training
    p.add_argument("--batch_size",     type=int,   default=24,    help="mini-batch size")
    p.add_argument("--epoch",          type=int,   default=50,    help="total training epochs")
    p.add_argument("--learning_rate",  type=float, default=1e-3,  help="initial learning rate")
    p.add_argument("--decay_rate",     type=float, default=1e-4,  help="weight decay")
    p.add_argument("--optimizer",      type=str,   default="Adam",help="Adam | SGD")
    p.add_argument("--seed",           type=int,   default=42,    help="global random seed")
    p.add_argument("--gpu",            type=str,   default="0",   help="GPU index (or 'cpu')")

    # Data
    p.add_argument("--num_point",      type=int,   default=1000,  help="interface points per sample")
    p.add_argument("--dataset_dir",   type=str,   default="dataset",
                   help="directory containing train.txt / valid.txt / test.txt")
    p.add_argument("--data_dir",       type=str,   default="data/interface_points",
                   help="root directory for .npy point cloud files")
    p.add_argument("--esm_pooling",    type=bool,  default=True,  help="use ESM-2 pooled embeddings")
    p.add_argument("--esm_crossattn",  type=bool,  default=False, help="use per-residue ESM-2 cross-attention")
    p.add_argument("--nullify_points", action="store_true",
                   help="zero out all point features except xyz (ablation: geometry only)")

    # Paths
    p.add_argument("--log_dir",        type=str,   default="piaco2",   help="experiment log sub-folder (logs + TensorBoard written here)")
    p.add_argument("--checkpoint",     type=str,   default="piaco2/run1",
                   help="checkpoint sub-folder under ./checkpoint/")

    args, _ = p.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------------

def centroid_scale_params(xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return (centroid, max-radius) for a point cloud in Angstroms."""
    c = xyz.mean(axis=0)
    r = float(np.sqrt(((xyz - c) ** 2).sum(axis=1)).max())
    return c.astype(np.float32), r + 1e-12


def apply_norm(xyz: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return (xyz - center) / scale


# ---------------------------------------------------------------------------
# HDF5 helpers (ESM-2 embeddings)
# ---------------------------------------------------------------------------

def _read_chain_group(grp: h5py.Group) -> Dict[str, np.ndarray]:
    """Load one chain's ESM embeddings and sidechain centroids from HDF5 group."""
    ds = grp["chain_ids"]
    if hasattr(ds, "asstr"):
        chain_ids = ds.asstr()[:].reshape(-1)
    else:
        raw = np.asarray(ds[:]).reshape(-1)
        chain_ids = np.array(
            [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw],
            dtype=np.str_,
        )

    emb = np.asarray(grp["esm_representations"][:], dtype=np.float32)   # [L, 1280]
    xyz = np.asarray(grp["sidechain_centroids"][:],  dtype=np.float32)  # [L, 3]

    if emb.ndim != 2 or emb.shape[1] != 1280:
        raise ValueError(f"Expected ESM shape [L,1280], got {emb.shape}")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected centroid shape [L,3], got {xyz.shape}")

    return {"chain_ids": chain_ids, "embedding": emb, "centroids": xyz}


def load_complexes_hdf5(
    h5_path: str,
    names: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Load receptor+ligand embeddings for a set of complexes from one HDF5 file."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)
    result = {}
    with h5py.File(h5_path, "r") as f:
        keys = names if names is not None else list(f.keys())
        for name in keys:
            if name not in f:
                raise KeyError(f"Complex '{name}' not found in {h5_path}")
            g = f[name]
            if "receptor" not in g or "ligand" not in g:
                raise KeyError(f"'{name}' missing receptor/ligand in {h5_path}")
            result[name] = {
                "receptor": _read_chain_group(g["receptor"]),
                "ligand":   _read_chain_group(g["ligand"]),
            }
    return result


def merge_pos_neg_hdf5(pos_h5: Optional[str], neg_h5: Optional[str]) -> Dict[str, Dict]:
    """Merge positive (label=1) and negative (label=0) HDF5 into one dict."""
    merged: Dict[str, Dict] = {}
    for h5_path, label in ((pos_h5, 1), (neg_h5, 0)):
        if not h5_path:
            continue
        for name, chains in load_complexes_hdf5(h5_path).items():
            if name in merged:
                raise ValueError(f"Duplicate complex name across pos/neg: {name}")
            merged[name] = {"label": label, **chains}
    return merged


def esm_pooling(complex_entry: Dict) -> Optional[np.ndarray]:
    """Mean+max pooling over ESM-2 tokens for receptor and ligand; returns 2560-d vector."""
    rec = complex_entry.get("receptor")
    lig = complex_entry.get("ligand")
    if rec is None or lig is None:
        return None

    emb_r = rec["embedding"]   # [L_r, 1280]
    emb_l = lig["embedding"]   # [L_l, 1280]
    if emb_r.shape[0] == 0 or emb_l.shape[0] == 0:
        return None

    pooled_r = emb_r.mean(0)   # [1280]
    pooled_l = emb_l.mean(0)   # [1280]
    return np.concatenate([pooled_r, pooled_l])  # [2560]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class InterfaceDataset(Dataset):
    """Protein–protein interface dataset.

    Each sample is one protein complex loaded from a .npy point cloud file.
    Optionally pairs with ESM-2 embeddings stored in an HDF5 file.

    Split files (train.txt / valid.txt / test.txt) contain tab-separated lines:
        <relative_path_to_npy>  <label (0 or 1)>
    """

    def __init__(
        self,
        data_root:    str,
        dataset_dir:  str,
        split:            str  = "train",
        use_aa:          bool = True,
        use_esm_pooling:  bool = False,
        use_esm_crossattn: bool = False,
    ):
        assert split in ("train", "valid", "test")
        self.data_root        = data_root
        self.use_aa          = use_aa
        self.use_esm_pooling  = use_esm_pooling
        self.use_esm_crossattn = use_esm_crossattn

        # Read file list
        lines = open(os.path.join(dataset_dir, f"{split}.txt")).readlines()

        entries = [ln.rstrip().split("\t") for ln in lines]
        self._paths  = [os.path.join(data_root, e[0]) for e in entries]
        self._labels = [int(e[1]) for e in entries]
        print(f"[{split}] {len(self._paths)} samples")

        # Load ESM
        h5_index: Dict[str, Dict] = {}
        if use_esm_crossattn or use_esm_pooling:
            esm_root = os.path.join(data_root, "..", "esm_interface", split)
            pos_h5   = os.path.join(esm_root, "bio.h5")
            neg_h5   = os.path.join(esm_root, "xtal.h5")
            h5_index = merge_pos_neg_hdf5(
                pos_h5 if os.path.exists(pos_h5) else None,
                neg_h5 if os.path.exists(neg_h5) else None,
            )

        # Pre-load point clouds and embeddings
        self._points:     List[Optional[np.ndarray]] = []
        self._pl_feats:  List[Optional[np.ndarray]] = []
        self._esm_tokens: List[Optional[Dict]]       = []

        for path in tqdm(self._paths, desc=f"Loading {split}"):
            # ── point cloud ────────────────────────────────────────────────
            pt = None
            center, scale = None, None
            try:
                pt = np.load(path + ".npy").astype(np.float32)
                center, scale = centroid_scale_params(pt[:, 0:3])
                pt[:, 0:3]    = apply_norm(pt[:, 0:3], center, scale)
                if not self.use_aa:
                    # Drop residue-type features (dims 14:34); keep xyz+atom+R/L only
                    pt = np.concatenate((pt[:, 0:7], pt[:, -2:]), axis=1)
            except Exception as e:
                print(f"  Warning: could not load {path}: {e}")

            self._points.append(pt)

            # ── ESM ──────────────────────────────────────────────────
            pl_feat  = None
            esm_entry = None

            if h5_index:
                cname = os.path.basename(path)
                if cname.endswith("_dmasif"):
                    cname = cname[:-7]

                entry = h5_index.get(cname)
                if entry is not None:
                    if use_esm_pooling:
                        pl_feat = esm_pooling(entry)

                    if center is not None and entry.get("receptor") is not None:
                        rec, lig = entry["receptor"], entry["ligand"]
                        esm_entry = {
                            "name":        cname,
                            "chain_ids_r": rec["chain_ids"],
                            "chain_ids_l": lig["chain_ids"],
                            "esm_r":       rec["embedding"],
                            "esm_l":       lig["embedding"],
                            "xyz_r":       apply_norm(rec["centroids"], center, scale),
                            "xyz_l":       apply_norm(lig["centroids"], center, scale),
                        }

            self._pl_feats.append(pl_feat)
            self._esm_tokens.append(esm_entry)

    # ── Dataset protocol ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._paths)

    def get_labels(self) -> List[int]:
        return self._labels

    def __getitem__(self, idx: int):
        return (
            self._points[idx],
            self._labels[idx],
            self._pl_feats[idx]  if self.use_esm_pooling  else None,
            self._esm_tokens[idx] if self.use_esm_crossattn else None,
        )


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple],
    max_pts: int             = 1000,
    lcap:    Optional[int]   = None,
    deterministic: bool      = False,
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Any]:
    """Collate a list of (point_array, label, plm, esm) tuples into batch tensors.

    Point clouds are split into receptor (first half) and ligand (second half),
    each capped / zero-padded to max_pts // 2 points, then re-concatenated.
    This ensures a consistent [B, max_pts, F] layout with R before L.

    Returns:
        points:    [B, max_pts, F]  float32
        atom_mask: [B, max_pts]     bool   (True = real point)
        labels:    [B]              long
        plms:      [B, 2560] float32 or None
        esms:      padded dict (lcap is set) or list[dict|None]
    """
    B    = len(batch)
    half = max_pts // 2

    # infer feature width from first valid sample
    F = next(item[0] for item in batch if item[0] is not None).shape[1]

    points    = torch.zeros(B, max_pts, F, dtype=torch.float32)
    atom_mask = torch.zeros(B, max_pts,    dtype=torch.bool)
    labels    = torch.zeros(B,             dtype=torch.long)
    plm_list: List[Optional[Tensor]]      = []
    esm_list: List[Optional[Dict[str, Any]]] = []

    for i, (pt_np, label, plm_np, esm_np) in enumerate(batch):
        labels[i] = int(label)

        if pt_np is not None:
            # split by R/L flag columns (last two)
            r_pts = pt_np[pt_np[:, -2] > 0.5]   # receptor rows
            l_pts = pt_np[pt_np[:, -1] > 0.5]   # ligand rows

            def _cap_pad(arr: np.ndarray) -> Tuple[np.ndarray, int]:
                """Sample to half, or zero-pad; returns (array[half,F], n_real)."""
                n = arr.shape[0]
                if n > half:
                    if deterministic:
                        return arr[:half], half
                    idx = np.random.choice(n, half, replace=False)
                    return arr[idx], half
                pad = np.zeros((half, F), dtype=np.float32)
                if n > 0:
                    pad[:n] = arr
                return pad, n

            r_final, r_real = _cap_pad(r_pts)
            l_final, l_real = _cap_pad(l_pts)

            points[i] = torch.from_numpy(np.vstack([r_final, l_final]))
            atom_mask[i, :r_real]            = True
            atom_mask[i, half:half + l_real] = True

        # PLM
        if plm_np is not None:
            t = torch.from_numpy(np.asarray(plm_np, dtype=np.float32))
            if t.ndim == 2 and t.shape[1] == 1:
                t = t.squeeze(1)
            plm_list.append(t)
        else:
            plm_list.append(None)

        # ESM (convert ndarray → Tensor)
        if esm_np is None:
            esm_list.append(None)
        else:
            e: Dict[str, Any] = {}
            for k in ("xyz_r", "xyz_l", "esm_r", "esm_l"):
                e[k] = torch.from_numpy(np.asarray(esm_np[k], dtype=np.float32))
            for k in ("chain_ids_r", "chain_ids_l"):
                if k in esm_np:
                    raw = np.asarray(esm_np[k]).reshape(-1)
                    e[k] = [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in raw]
            e["name"] = esm_np.get("name", f"sample_{i}")
            esm_list.append(e)

    # stack PLM if all present and same shape
    plms: Optional[Tensor] = None
    valid_plms = [t for t in plm_list if t is not None]
    if len(valid_plms) == B and len({tuple(t.shape) for t in valid_plms}) == 1:
        plms = torch.stack(valid_plms).float()

    # ESM: pad to fixed lcap if requested
    if lcap is None:
        esms = esm_list
    else:
        def _pad_side(xyz_key: str, esm_key: str):
            xyz_t = torch.zeros(B, lcap,  3,    dtype=torch.float32)
            emb_t = torch.zeros(B, lcap, 1280,  dtype=torch.float32)
            msk   = torch.zeros(B, lcap,        dtype=torch.bool)
            for bi, e in enumerate(esm_list):
                if e is None:
                    continue
                L = min(e[xyz_key].shape[0], lcap)
                xyz_t[bi, :L]  = e[xyz_key][:L]
                emb_t[bi, :L]  = e[esm_key][:L]
                msk[bi,   :L]  = True
            return xyz_t, emb_t, msk

        xyz_r, esm_r, mask_r = _pad_side("xyz_r", "esm_r")
        xyz_l, esm_l, mask_l = _pad_side("xyz_l", "esm_l")
        esms = {
            "xyz_r": xyz_r, "esm_r": esm_r, "mask_r": mask_r,
            "xyz_l": xyz_l, "esm_l": esm_l, "mask_l": mask_l,
        }

    return points, atom_mask, labels, plms, esms


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_esms_to(esms: Any, device: torch.device) -> Any:
    """Move ESM tensors from collate_fn to a device."""
    if esms is None:
        return None
    if isinstance(esms, list):
        for entry in esms:
            if entry is None:
                continue
            for k in ("xyz_r", "xyz_l", "esm_r", "esm_l", "mask_r", "mask_l"):
                if k in entry and torch.is_tensor(entry[k]):
                    entry[k] = entry[k].to(device)
        return esms
    for k in ("xyz_r", "xyz_l", "esm_r", "esm_l", "mask_r", "mask_l"):
        if k in esms:
            esms[k] = esms[k].to(device)
    return esms


def select_point_channels(points: Tensor) -> Tensor:
    """Keep only xyz (dims 0:3) and R/L flags (dims -2:) → [B, 5, N].

    This discards chemical feature channels, leaving only the geometric
    and chain-identity signal used by the default Piaco2(in_channels=2) model.
    """
    return torch.cat([points[:, :3, :], points[:, -2:, :]], dim=1)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # ── directories ───────────────────────────────────────────────────────
    ckpt_dir = Path(BASE_DIR) / "checkpoint" / args.checkpoint
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(BASE_DIR) / "log" / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    # ── logging ───────────────────────────────────────────────────────────
    logger = logging.getLogger("piaco2_train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    fh  = logging.FileHandler(log_dir / "train.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    def log(msg: str) -> None:
        logger.info(msg)
        print(msg)

    log(f"Args: {vars(args)}")

    # ── data ──────────────────────────────────────────────────────────────
    dataset_dir  = os.path.join(BASE_DIR, "dataset", args.dataset_dir)
    data_dir = os.path.join(BASE_DIR, "data",     args.data_dir)

    train_ds = InterfaceDataset(data_dir, dataset_dir, split="train",
                                use_esm_pooling=args.esm_pooling, use_esm_crossattn=args.esm_crossattn)
    valid_ds = InterfaceDataset(data_dir, dataset_dir, split="valid",
                                use_esm_pooling=args.esm_pooling, use_esm_crossattn=args.esm_crossattn)

    cf_train = partial(collate_fn, max_pts=args.num_point, lcap=100, deterministic=False)
    cf_val   = partial(collate_fn, max_pts=args.num_point, lcap=100, deterministic=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=ImbalancedDatasetSampler(train_ds),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        collate_fn=cf_train,
        worker_init_fn=lambda wid: (
            np.random.seed(torch.initial_seed() % 2**32),
            random.seed(torch.initial_seed() % 2**32),
        ),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        collate_fn=cf_val,
        worker_init_fn=lambda wid: (
            np.random.seed(torch.initial_seed() % 2**32),
            random.seed(torch.initial_seed() % 2**32),
        ),
    )

    # ── model ─────────────────────────────────────────────────────────────
    # Infer in_channels from actual feature count (F - 3 xyz dims)
    _sample_pt = next(pt for pt in train_ds._points if pt is not None)
    in_channels = _sample_pt.shape[1] - 3
    log(f"in_channels inferred from data: {in_channels} (F={_sample_pt.shape[1]})")

    model = Piaco2(
        in_channels=in_channels,
        use_esm=args.esm_crossattn,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable parameters: {n_params / 1e6:.3f} M")

    criterion  = nn.BCEWithLogitsLoss()
    early_stop = EarlyStopping(patience=10, verbose=True)

    # ── optimiser ──────────────────────────────────────────────────────────
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # ── resume ────────────────────────────────────────────────────────────
    last_ckpt = ckpt_dir / "last_model.pth"
    start_epoch = 0
    if last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device)
        try:
            model.load_state_dict(ck["model_state_dict"])
        except RuntimeError as e:
            log(f"Warning: strict checkpoint load failed ({e}). Retrying with strict=False.")
            incompatible = model.load_state_dict(ck["model_state_dict"], strict=False)
            log(
                "Loaded with strict=False. "
                f"missing_keys={len(incompatible.missing_keys)} "
                f"unexpected_keys={len(incompatible.unexpected_keys)}"
            )
        start_epoch = ck["epoch"]
        log(f"Resumed from epoch {start_epoch}")
    else:
        log("Starting from scratch")

    best_class_acc    = 0.0
    best_inst_acc     = 0.0
    global_step       = 0

    # ====================================================================
    for epoch in range(start_epoch, args.epoch):
        log(f"── Epoch {epoch + 1}/{args.epoch} ──")

        # ── train ─────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_correct    = 0
        n_total      = 0
        train_targets: List[float] = []
        train_probs:   List[float] = []

        for pts, mask, target, plms, esms in tqdm(train_loader, desc="train", leave=False):
            esms   = move_esms_to(esms, device)
            target = target.float().to(device)
            plms   = plms.to(device) if plms is not None else None

            # augmentation (operates on [B, N, 3] prefix)
            pts = provider.random_point_dropout(pts)
            pts[:, :, 0:3] = provider.shift_point_cloud(pts[:, :, 0:3])
            pts[:, :, 0:3] = provider.rotate_point_cloud_so3(pts[:, :, 0:3])

            # [B, N, F] → [B, F, N]
            pts = pts.transpose(2, 1).to(device)
            if args.nullify_points:
                # Keep xyz (0:3) and R/L flags (-2:); nullify all other features.
                pts[:, 3:-2, :] = 0

            optimizer.zero_grad()
            logits = model(pts, plm=plms, esms=esms)
            loss   = criterion(logits, target)
            loss.backward()
            optimizer.step()

            probs   = torch.sigmoid(logits).detach()
            preds   = (logits.detach() > 0).float()
            n_correct += preds.eq(target).sum().item()
            n_total   += len(target)
            running_loss += loss.item()
            train_targets.extend(target.cpu().tolist())
            train_probs.extend(probs.cpu().tolist())
            global_step += 1

        scheduler.step()

        train_acc  = n_correct / max(n_total, 1)
        train_auc  = roc_auc_score(train_targets, train_probs)
        train_loss = running_loss / max(n_total // args.batch_size, 1)
        log(f"  Train  acc={train_acc:.4f}  auc={train_auc:.4f}  loss={train_loss:.5f}")

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        valid_loss   = 0.0
        class_acc    = np.zeros((2, 3))
        n_correct    = 0
        n_total      = 0
        valid_targets: List[float] = []
        valid_probs:   List[float] = []

        with torch.no_grad():
            for pts, mask, target, plms, esms in tqdm(valid_loader, desc="valid", leave=False):
                esms   = move_esms_to(esms, device)
                target = target.float().to(device)
                plms   = plms.to(device) if plms is not None else None

                pts = pts.transpose(2, 1).to(device)
                if args.nullify_points:
                    # Keep xyz (0:3) and R/L flags (-2:); nullify all other features.
                    pts[:, 3:-2, :] = 0
                logits = model(pts, plm=plms, esms=esms)
                loss   = criterion(logits, target)

                probs  = torch.sigmoid(logits)
                preds  = (logits > 0).float()
                n_correct  += preds.eq(target).sum().item()
                n_total    += len(target)
                valid_loss += loss.item()
                valid_targets.extend(target.cpu().tolist())
                valid_probs.extend(probs.cpu().tolist())

                # per-class accuracy
                for cat in np.unique(target.cpu().numpy()).astype(int):
                    mask_c = (target == cat)
                    class_acc[cat, 0] += preds[mask_c].eq(target[mask_c]).sum().item()
                    class_acc[cat, 1] += mask_c.sum().item()

        valid_auc       = roc_auc_score(valid_targets, valid_probs)
        valid_inst_acc  = n_correct / max(n_total, 1)
        valid_loss     /= max(n_total // args.batch_size, 1)
        class_acc[:, 2] = class_acc[:, 0] / np.maximum(class_acc[:, 1], 1)
        mean_class_acc  = float(class_acc[:, 2].mean())

        log(f"  Valid  inst_acc={valid_inst_acc:.4f}  class_acc={mean_class_acc:.4f}"
            f"  auc={valid_auc:.4f}  loss={valid_loss:.5f}")

        # ── checkpoint ────────────────────────────────────────────────────
        state = {
            "epoch":              epoch + 1,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "instance_acc":       valid_inst_acc,
            "class_acc":          mean_class_acc
        }

        torch.save(state, ckpt_dir / f"epoch_{epoch:d}.pth")
        torch.save(state, ckpt_dir / "last_model.pth")

        if mean_class_acc >= best_class_acc:
            best_class_acc = mean_class_acc
            best_inst_acc  = valid_inst_acc
            torch.save(state, ckpt_dir / "best_model.pth")
            log(f"  ✓ New best  class_acc={best_class_acc:.4f}")

        log(f"  Best so far  class_acc={best_class_acc:.4f}  inst_acc={best_inst_acc:.4f}")

        if epoch > 5:
            early_stop(1.0 - mean_class_acc, model)
        if early_stop.early_stop:
            log("Early stopping triggered.")
            break

    # ====================================================================
    log("Training complete.")
    logger.removeHandler(fh)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(parse_args())
