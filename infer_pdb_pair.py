#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""infer_pdb_pair.py

PDB (chain A/B) -> PIACO2 probability.

This script is intended as an inference entrypoint that mirrors the
`train_piaco2.py` data conventions:
  - Build an interface point cloud (xyz + features) from a 2-chain PDB
  - Optionally compute ESM-2 embeddings from the PDB chains:
      * pooled ESM vector (2560-d) via mean pooling per chain
      * per-residue tokens + centroids for cross-attention (when --esm_crossattn)
  - Load a PIACO2 checkpoint and output sigmoid(logit)

Notes:
  - dMaSIF surface features are supported if you already have per-chain
    .npy files (see --dmasif_npy_dir). Computing dMaSIF from scratch is
    environment-dependent and is not bundled here.
  - ESM-2 requires `fair-esm` (pip install fair-esm).

Example:
  python infer_pdb_pair.py \
    --pdb example.pdb --receptor A --ligand B \
    --checkpoint checkpoint/piaco2/run1/best_model.pth \
    --esm_pooling
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from Bio.PDB import MMCIFParser, PDBParser

from model.piaco2_architecture import Piaco2
from run_preprocess_piaco2 import process_pdb, parse_atoms, filter_to_interface

# ---------------------------------------------------------------------------
# FreeSASA / interface_analyzer (distance + SASA, matching training pipeline)
# ---------------------------------------------------------------------------
try:
    import freesasa as _freesasa
    _freesasa.setVerbosity(_freesasa.nowarnings)
    _FREESASA_AVAILABLE = True
except ImportError:
    _FREESASA_AVAILABLE = False

try:
    if not _FREESASA_AVAILABLE:
        raise ImportError("freesasa is not installed")
    from interface_analyzer import get_interface_residues_with_sasa as _get_iface_sasa
    _INTERFACE_ANALYZER_AVAILABLE = True
except ImportError:
    _INTERFACE_ANALYZER_AVAILABLE = False
    print("[warn] freesasa or interface_analyzer.py not found.")


AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",  # selenomethionine
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Infer PIACO2 probability from a 2-chain PDB")
    p.add_argument("--pdb", type=str, required=True, help="Path to PDB/mmCIF with both chains")
    p.add_argument("--receptor", type=str, required=True, help="Receptor chain id (e.g. A)")
    p.add_argument("--ligand", type=str, required=True, help="Ligand chain id (e.g. B)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pth (best_model.pth etc)")

    p.add_argument("--device", type=str, default="cuda:0", help="cuda:0 | cpu")
    p.add_argument("--npoint", type=int, default=1000, help="Number of interface points (should match training)")
    p.add_argument("--nullify_points", action="store_true", help="Zero out all point features except xyz and R/L flags")

    # ESM-2
    p.add_argument("--esm_pooling", action="store_true", help="Compute mean-pooled ESM-2 features per chain (2560-d) and pass to model")
    p.add_argument("--esm_crossattn", action="store_true", help="Enable per-residue ESM cross-attention (also computes tokens)")
    p.add_argument(
        "--esm2_model",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="fair-esm pretrained model function name (must output 1280-d representations)",
    )
    p.add_argument("--esm_lcap", type=int, default=100, help="Max residues per side for cross-attn padding")
    p.add_argument("--distance_cutoff", type=float, default=8.0,
                   help="Residue-residue distance cutoff (Å) for interface definition (default: 8.0)")
    p.add_argument("--sasa_cutoff", type=float, default=1.0,
                   help="SASA burial cutoff (Å²) for interface definition (default: 1.0; requires freesasa)")

    # dMaSIF
    p.add_argument(
        "--dmasif_npy_dir",
        type=str,
        default=None,
        help=(
            "(Optional) Folder with dMaSIF per-chain .npy files named {pdbid}_chain_{A}.npy. "
            "When provided, pooled dMaSIF features are appended and the checkpoint must expect them."
        ),
    )

    # Debug
    p.add_argument(
        "--interface_json",
        type=str,
        default=None,
        help=(
            "(Debug) Path to a pre-computed interface JSON produced by "
            "esm_20aa_feature_extractor_batch.py. "
            "When supplied, interface residue IDs are read directly from this file instead of being "
            "re-computed via distance/SASA. "
            'Format: {"receptor": [{"chain_id": "A", "residue_id": 123}, ...], "ligand": [...]}. '
            "Use this to compare infer_pdb_pair.py and evaluate_piaco2.py with identical interface residues."
        ),
    )

    # Output
    p.add_argument("--out_json", type=str, default=None, help="Optional path to write a JSON result")
    return p


def load_structure(path: str):
    if path.lower().endswith((".cif", ".mmcif")):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure("S", path)


def residue_to_oneletter(resname: str) -> str:
    return AA3_TO_1.get(resname.strip().upper(), "X")


def sidechain_centroid(residue) -> Optional[np.ndarray]:
    """Compute sidechain centroid for one residue. Fallback to CA when needed.

    Matches esm_20aa_feature_extractor_batch.py::get_sidechain_centroid() exactly:
    - Excludes backbone atoms {N, CA, C, O} only (hydrogens are NOT filtered so
      that PDB files with explicit H atoms yield the same centroid as the HDF5
      training pipeline).
    - Falls back to CA, then to mean of all atoms.
    """
    _BACKBONE = {"N", "CA", "C", "O"}
    sidechain_atoms = [
        atom for atom in residue.get_atoms()
        if atom.get_id() not in _BACKBONE
    ]
    if sidechain_atoms:
        coords = np.array([a.get_coord() for a in sidechain_atoms], dtype=np.float32)
        return coords.mean(axis=0)
    if residue.has_id("CA"):
        return residue["CA"].get_coord().astype(np.float32)
    all_coords = [a.get_coord() for a in residue.get_atoms()]
    if all_coords:
        return np.array(all_coords, dtype=np.float32).mean(axis=0)
    return None


def extract_chain_sequence_and_centroids(
    structure,
    chain_id: str,
) -> Tuple[str, np.ndarray, List[int]]:
    """Return (sequence, centroids[L,3], res_ids[L]) for one chain."""
    model = next(structure.get_models())
    if chain_id not in model:
        present = [c.id for c in model.get_chains()]
        raise ValueError(f"Chain '{chain_id}' not found. present={present}")

    chain = model[chain_id]
    seq: List[str] = []
    xyz: List[np.ndarray] = []
    res_ids: List[int] = []

    for residue in chain.get_residues():
        hetflag, resseq, icode = residue.id
        if str(hetflag).strip():
            continue
        resname = residue.get_resname().strip().upper()
        aa = residue_to_oneletter(resname)
        if aa == "X":
            continue
        c = sidechain_centroid(residue)
        if c is None:
            continue
        seq.append(aa)
        xyz.append(c)
        res_ids.append(resseq)

    if not seq:
        raise ValueError(f"No residues extracted for chain {chain_id}.")

    return "".join(seq), np.asarray(xyz, dtype=np.float32), res_ids


def centroid_scale_params(xyz: np.ndarray) -> Tuple[np.ndarray, float]:
    c = xyz.mean(axis=0)
    r = float(np.sqrt(((xyz - c) ** 2).sum(axis=1)).max())
    return c.astype(np.float32), r + 1e-12


def apply_norm(xyz: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return (xyz - center) / scale


def infer_in_channels_from_state_dict(sd: Dict[str, torch.Tensor]) -> int:
    # encoder.feat_embed.net.0.weight: [embed_dim, in_channels, 1]
    for k, v in sd.items():
        if k.endswith("encoder.feat_embed.net.0.weight") and torch.is_tensor(v):
            if v.ndim != 3:
                break
            return int(v.shape[1])
    raise KeyError("Could not infer in_channels from state_dict (missing encoder.feat_embed.net.0.weight)")


def adapt_points_to_in_channels(points_nf: np.ndarray, in_channels: int) -> np.ndarray:
    """Return points array shaped [N, 3+in_channels] matching checkpoint expectation."""
    if points_nf.ndim != 2 or points_nf.shape[1] < 5:
        raise ValueError(f"Expected points as [N,F>=5], got {points_nf.shape}")

    F = points_nf.shape[1]
    want_F = 3 + in_channels
    if F == want_F:
        return points_nf

    # Common special-case: checkpoint expects only R/L flags (in_channels=2)
    if in_channels == 2 and F >= 5:
        xyz = points_nf[:, :3]
        flags = points_nf[:, -2:]
        return np.concatenate([xyz, flags], axis=1).astype(np.float32)

    raise ValueError(
        "Point feature width does not match checkpoint. "
        f"checkpoint expects F={want_F} (in_channels={in_channels}), but got F={F}. "
        "If the checkpoint expects dMaSIF features, pass --dmasif_npy_dir (or use a checkpoint trained without dMaSIF)."
    )


def _detect_interface(
    pdb_path: str,
    receptor_chains: List[str],
    ligand_chains: List[str],
    distance_cutoff: float = 8.0,
    sasa_cutoff: float = 1.0,
) -> Tuple[set, set]:
    """Return (r_iface_res_ids, l_iface_res_ids) using distance+SASA when available.

    Delegates to interface_analyzer.get_interface_residues_with_sasa (same method
    used by esm_20aa_feature_extractor_batch.py to build the training HDF5).
    Falls back to distance-only if interface_analyzer is unavailable.
    """
    if _INTERFACE_ANALYZER_AVAILABLE:
        iface = _get_iface_sasa(
            pdb_path,
            chain_ids_1=receptor_chains,
            chain_ids_2=ligand_chains,
            distance_cutoff=distance_cutoff,
            sasa_cutoff=sasa_cutoff,
        )
        r_ids: set = {r["residue_id"] for r in iface["receptor"]}
        l_ids: set = {r["residue_id"] for r in iface["ligand"]}
        print(f"[ESM interface] distance+SASA: {len(r_ids)} receptor, {len(l_ids)} ligand residues")
    else:
        r_atoms, l_atoms = parse_atoms(pdb_path, receptor_chains, ligand_chains)
        r_iface, l_iface = filter_to_interface(r_atoms, l_atoms, cutoff=distance_cutoff)
        r_ids = {a[5] for a in r_iface}
        l_ids = {a[5] for a in l_iface}
        print(f"[ESM interface] distance-only fallback: {len(r_ids)} receptor, {len(l_ids)} ligand residues")
    return r_ids, l_ids


def _load_interface_json(
    json_path: str,
    receptor_chain: str,
    ligand_chain: str,
) -> Tuple[set, set]:
    """Load interface residue IDs from a pre-computed JSON file.

    The JSON is produced by esm_20aa_feature_extractor_batch.py and has the form::

        {
            "receptor": [{"chain_id": "A", "residue_id": 123, ...}, ...],
            "ligand":   [{"chain_id": "B", "residue_id": 456, ...}, ...]
        }

    Returns (r_ids, l_ids) where each element is a set of integer residue IDs,
    matching the format returned by _detect_interface().
    """
    import json as _json
    with open(json_path, "r") as fh:
        data = _json.load(fh)

    r_ids: set = set()
    l_ids: set = set()
    for entry in data.get("receptor", []):
        r_ids.add(int(entry["residue_id"]))
    for entry in data.get("ligand", []):
        l_ids.add(int(entry["residue_id"]))

    print(
        f"[ESM interface] loaded from JSON {json_path}: "
        f"{len(r_ids)} receptor, {len(l_ids)} ligand residues"
    )
    return r_ids, l_ids


@dataclass
class ESM2Bundle:
    plm: Optional[torch.Tensor]
    esms: Optional[Dict[str, torch.Tensor]]


def _load_esm2_model(model_fn_name: str, device: torch.device):
    try:
        import esm  # fair-esm
    except ImportError as e:
        raise SystemExit(
            "Cannot import 'esm'. Install fair-esm: pip install fair-esm\n"
            f"ImportError: {e}"
        )

    if not hasattr(esm.pretrained, model_fn_name):
        names = [n for n in dir(esm.pretrained) if n.startswith("esm2_")]
        raise ValueError(f"Unknown esm.pretrained.{model_fn_name}. Available: {names}")

    model_fn = getattr(esm.pretrained, model_fn_name)
    model, alphabet = model_fn()
    model = model.to(device).eval()
    return model, alphabet


def _esm2_representations(
    model,
    alphabet,
    seq: str,
    device: torch.device,
    layer: Optional[int] = None,
    mask_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return per-residue embedding array [L, d].
    
    If mask_indices (boolean array of length L) is provided, computes representations
    for those positions using Masked Language Modeling (replacing the token with <mask>),
    which exactly matches how training data was constructed.
    Positions that are False in mask_indices will remain 0.0 vectors.
    """
    if layer is None:
        layer = max(model.num_layers, 1)

    batch_converter = alphabet.get_batch_converter()
    _, _, toks = batch_converter([("x", seq)])
    toks = toks.to(device)

    # Standard (Unmasked) representation if no mask provided
    if mask_indices is None or not mask_indices.any():
        with torch.no_grad():
            out = model(toks, repr_layers=[layer], return_contacts=False)
        reps = out["representations"][layer]  # [1, L+2, d]
        return reps[0, 1 : len(seq) + 1].float().cpu().numpy().astype(np.float32)

    # Masked representation for interface residues (matching training data pipeline)
    mask_idx = batch_converter.alphabet.mask_idx
    positions = np.where(mask_indices)[0] + 1  # 1-based for ESM sequence ([CLS] is 0)
    pos_t = torch.tensor(positions, device=device)
    
    L = len(seq)
    results = np.zeros((L, 1280), dtype=np.float32)
    max_batch = 16  # avoid OOM

    for start in range(0, len(positions), max_batch):
        chunk = pos_t[start : start + max_batch]
        B = chunk.shape[0]
        toks_rep = toks.repeat(B, 1)
        toks_rep[torch.arange(B, device=device), chunk] = mask_idx

        use_fp16 = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            with torch.no_grad():
                out = model(toks_rep, repr_layers=[layer], return_contacts=False)
                
        reprs = out["representations"][layer]  # [B, L+2, 1280]
        chunk_reprs = reprs[torch.arange(B, device=device), chunk].detach().float().cpu().numpy()
        
        chunk_0based = chunk.cpu().numpy() - 1
        results[chunk_0based] = chunk_reprs

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def compute_esm2_bundle(
    pdb_path: str,
    receptor_chain: str,
    ligand_chain: str,
    center: np.ndarray,
    scale: float,
    device: torch.device,
    model_fn_name: str,
    use_esm_pooling: bool,
    use_esm_tokens: bool,
    lcap: int,
    distance_cutoff: float = 8.0,
    sasa_cutoff: float = 1.0,
    interface_json: Optional[str] = None,
) -> ESM2Bundle:
    if not use_esm_pooling and not use_esm_tokens:
        return ESM2Bundle(plm=None, esms=None)

    structure = load_structure(pdb_path)
    seq_r, xyz_r, res_ids_r = extract_chain_sequence_and_centroids(structure, receptor_chain)
    seq_l, xyz_l, res_ids_l = extract_chain_sequence_and_centroids(structure, ligand_chain)

    # Interface residues: use pre-computed JSON (debug) or live detection
    if interface_json is not None:
        r_iface_res_ids, l_iface_res_ids = _load_interface_json(
            interface_json, receptor_chain, ligand_chain
        )
    else:
        r_iface_res_ids, l_iface_res_ids = _detect_interface(
            pdb_path, [receptor_chain], [ligand_chain],
            distance_cutoff=distance_cutoff,
            sasa_cutoff=sasa_cutoff,
        )

    r_iface_mask = np.array([rid in r_iface_res_ids for rid in res_ids_r], dtype=bool)
    l_iface_mask = np.array([rid in l_iface_res_ids for rid in res_ids_l], dtype=bool)

    model, alphabet = _load_esm2_model(model_fn_name, device)
    layer = int(model.num_layers)

    emb_r_full = _esm2_representations(model, alphabet, seq_r, device=device, layer=layer, mask_indices=r_iface_mask)
    emb_l_full = _esm2_representations(model, alphabet, seq_l, device=device, layer=layer, mask_indices=l_iface_mask)

    if emb_r_full.shape[1] != 1280 or emb_l_full.shape[1] != 1280:
        raise ValueError(
            f"ESM-2 embedding dim must be 1280 for this codebase; got rec={emb_r_full.shape}, lig={emb_l_full.shape}. "
            "Use an esm2_* model that outputs 1280-d (e.g., esm2_t33_650M_UR50D)."
        )

    # Filter to interface residues (matches HDF5 training data scope)
    emb_r = emb_r_full[r_iface_mask] if r_iface_mask.any() else emb_r_full
    emb_l = emb_l_full[l_iface_mask] if l_iface_mask.any() else emb_l_full
    xyz_r = xyz_r[r_iface_mask] if r_iface_mask.any() else xyz_r
    xyz_l = xyz_l[l_iface_mask] if l_iface_mask.any() else xyz_l

    plm_t: Optional[torch.Tensor] = None
    if use_esm_pooling:
        pooled = np.concatenate([emb_r.mean(0), emb_l.mean(0)], axis=0).astype(np.float32)  # (2560,)
        plm_t = torch.from_numpy(pooled).unsqueeze(0).to(device)

    esms: Optional[Dict[str, torch.Tensor]] = None
    if use_esm_tokens:
        # normalize residue centroids with the *same* center/scale as point cloud
        xyz_r_n = apply_norm(xyz_r, center, scale)
        xyz_l_n = apply_norm(xyz_l, center, scale)

        def pad_side(xyz: np.ndarray, emb: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            L = min(xyz.shape[0], emb.shape[0])
            L2 = min(L, lcap)
            xyz_t = torch.zeros((1, lcap, 3), dtype=torch.float32, device=device)
            emb_t = torch.zeros((1, lcap, 1280), dtype=torch.float32, device=device)
            msk_t = torch.zeros((1, lcap), dtype=torch.bool, device=device)
            if L2 > 0:
                xyz_t[0, :L2] = torch.from_numpy(xyz[:L2]).to(device)
                emb_t[0, :L2] = torch.from_numpy(emb[:L2]).to(device)
                msk_t[0, :L2] = True
            return xyz_t, emb_t, msk_t

        xr, er, mr = pad_side(xyz_r_n, emb_r)
        xl, el, ml = pad_side(xyz_l_n, emb_l)
        esms = {
            "xyz_r": xr,
            "esm_r": er,
            "mask_r": mr,
            "xyz_l": xl,
            "esm_l": el,
            "mask_l": ml,
        }

    return ESM2Bundle(plm=plm_t, esms=esms)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # assume it's already a state_dict
        return ckpt  # type: ignore[return-value]
    raise ValueError(f"Unrecognized checkpoint format at {path}")


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    state_dict = load_checkpoint(args.checkpoint, device)
    in_channels = infer_in_channels_from_state_dict(state_dict)

    # preprocess points
    cloud = process_pdb(
        args.pdb,
        receptor_chains=[args.receptor],
        ligand_chains=[args.ligand],
        npoint=args.npoint,
        npy_folder=args.dmasif_npy_dir,
        pdb_id=os.path.splitext(os.path.basename(args.pdb))[0],
    ).astype(np.float32)

    # Validate that at least some real points exist
    real_mask = (cloud[:, -2:].sum(axis=1) > 0.5)
    if not np.any(real_mask):
        raise ValueError("Preprocess produced only padding rows (no real points).")

    # Normalise ALL rows (including padding zeros) to match train_piaco2.py behaviour,
    # which calls centroid_scale_params/apply_norm on the full (npoint, F) array.
    center, scale = centroid_scale_params(cloud[:, 0:3])
    cloud[:, 0:3] = apply_norm(cloud[:, 0:3], center, scale)

    # Adapt feature width to match checkpoint expectation
    cloud = adapt_points_to_in_channels(cloud, in_channels=in_channels)  # [N, 3+in_channels]

    # Build input tensor [B, C, N]
    pts = torch.from_numpy(cloud).unsqueeze(0)  # [1, N, C]
    pts = pts.transpose(2, 1).contiguous().to(device)  # [1, C, N]

    if args.nullify_points:
        # Keep xyz and R/L flags; nullify everything else.
        pts[:, 3:-2, :] = 0

    # ESM-2
    esm_bundle = compute_esm2_bundle(
        pdb_path=args.pdb,
        receptor_chain=args.receptor,
        ligand_chain=args.ligand,
        center=center,
        scale=scale,
        device=device,
        model_fn_name=args.esm2_model,
        use_esm_pooling=bool(args.esm_pooling),
        use_esm_tokens=bool(args.esm_crossattn),
        lcap=int(args.esm_lcap),
        distance_cutoff=float(args.distance_cutoff),
        sasa_cutoff=float(args.sasa_cutoff),
        interface_json=args.interface_json,
    )

    # Model
    model = Piaco2(in_channels=in_channels, use_esm=bool(args.esm_crossattn)).to(device)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        # For inference, strict=False is helpful when experimenting with ablations.
        # Still, surface the mismatch so users know what happened.
        print(
            f"[checkpoint] loaded with strict=False: "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )

    model.eval()
    with torch.no_grad():
        logits = model(pts, plm=esm_bundle.plm, esms=esm_bundle.esms)
        prob = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[0].item()
        logit = logits.detach().cpu().numpy().reshape(-1)[0].item()

    result = {
        "pdb": os.path.abspath(args.pdb),
        "receptor": args.receptor,
        "ligand": args.ligand,
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": str(device),
        "in_channels": int(in_channels),
        "npoint": int(args.npoint),
        "used_esm_pooling": bool(esm_bundle.plm is not None),
        "used_esm_crossattn": bool(args.esm_crossattn),
        "logit": float(logit),
        "prob": float(prob),
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
