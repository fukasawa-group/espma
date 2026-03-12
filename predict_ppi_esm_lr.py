#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict_ppi_esm_lr.py
=======================
Colab-friendly PPI prediction pipeline using ESM-2 features and a pre-trained Logistic Regression.

Pipeline (mirrors the training data generation pipeline):
  1. Interface detection using freesasa (SASA-buried cutoff) with automatic
     fallback to distance-only if freesasa is unavailable.
  2. Build per-chain amino acid sequences directly from the PDB file
     (no external FASTA required).
  3. For each interface residue, extract ESM-2 representation via
     masked language-modelling. Use --no_masking for a
     faster but slightly different forward pass.
  4. Mean-pool receptor representations (1280-d) and ligand
     representations (1280-d), concatenate to 2560-d feature vector.
  5. Load a pre-trained LogisticRegression from a joblib file and
     return the probability for class 1.

Required installs:
    pip install freesasa biopython fair-esm torch scikit-learn joblib numpy

Usage:
    python predict_ppi_esm_lr.py \\
        --pdb complex.pdb --receptor A --ligand B \\
        --lr_model lr_model.joblib \\
        [--distance_cutoff 8.0] [--sasa_cutoff 1.0] \\
        [--no_masking] [--max_masks_per_step 256] \\
        [--device cuda:0] [--save_json result.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# FreeSASA availability (for display in result JSON)
# ---------------------------------------------------------------------------
try:
    import freesasa as _freesasa_check
    _freesasa_check.setVerbosity(_freesasa_check.nowarnings)  # suppress "warning: atom..." messages
    FREESASA_AVAILABLE = True
except ImportError:
    FREESASA_AVAILABLE = False
    print("[warn] freesasa not found.  Install with: pip install freesasa")
    print("[warn] interface_analyzer will fall back to distance-only definition.")

from Bio.PDB import PDBParser

# ---------------------------------------------------------------------------
# interface_analyzer  (must be in the same directory or on PYTHONPATH)
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from interface_analyzer import get_interface_residues_with_sasa as _get_iface_sasa
    INTERFACE_ANALYZER_AVAILABLE = True
except ImportError:
    INTERFACE_ANALYZER_AVAILABLE = False
    print("[warn] interface_analyzer.py not found.  Using built-in distance-only fallback.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

ESM_AA_ORDER = ["A","R","N","D","C","Q","E","G","H","I",
                "L","K","M","F","P","S","T","W","Y","V"]


# ===========================================================================
# STEP 1  –  Interface residue detection
# ===========================================================================

def _distance_interface_fallback(pdb_path: str,
                                 chain_ids_1: List[str],
                                 chain_ids_2: List[str],
                                 cutoff: float) -> Dict[str, list]:
    """
    Pure-Python fallback (no interface_analyzer.py).
    Reads raw ATOM lines, matching the same logic as interface_analyzer.get_interface_residues().
    """
    receptor_residues, rlist = [], []
    ligand_residues,  llist = [], []

    tmp_r, pre_r_rid, pre_r_cid, pre_r_rname = [], None, None, None
    tmp_l, pre_l_rid, pre_l_cid, pre_l_rname = [], None, None, None

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            try:
                cid   = line[21]
                rid   = int(line[22:26].strip())
                rname = line[17:20].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except (ValueError, IndexError):
                continue

            if cid in chain_ids_1:
                if pre_r_rid is not None and rid != pre_r_rid:
                    if tmp_r:
                        rlist.append(tmp_r)
                        receptor_residues.append(
                            {"chain_id": pre_r_cid, "residue_id": pre_r_rid,
                             "residue_name": pre_r_rname})
                    tmp_r = []
                tmp_r.append([x, y, z])
                pre_r_rid, pre_r_cid, pre_r_rname = rid, cid, rname
            elif cid in chain_ids_2:
                if pre_l_rid is not None and rid != pre_l_rid:
                    if tmp_l:
                        llist.append(tmp_l)
                        ligand_residues.append(
                            {"chain_id": pre_l_cid, "residue_id": pre_l_rid,
                             "residue_name": pre_l_rname})
                    tmp_l = []
                tmp_l.append([x, y, z])
                pre_l_rid, pre_l_cid, pre_l_rname = rid, cid, rname

    if tmp_r:
        rlist.append(tmp_r)
        receptor_residues.append({"chain_id": pre_r_cid, "residue_id": pre_r_rid,
                                   "residue_name": pre_r_rname})
    if tmp_l:
        llist.append(tmp_l)
        ligand_residues.append({"chain_id": pre_l_cid, "residue_id": pre_l_rid,
                                 "residue_name": pre_l_rname})

    cutoff_sq = cutoff ** 2
    r_idx, l_idx = set(), set()
    for i, ra in enumerate(rlist):
        for j, la in enumerate(llist):
            min_dist_sq = min(
                sum((r[k] - l[k]) ** 2 for k in range(3))
                for r in ra for l in la
            )
            if min_dist_sq <= cutoff_sq:
                r_idx.add(i)
                l_idx.add(j)

    return {
        "receptor": [receptor_residues[i] for i in r_idx],
        "ligand":   [ligand_residues[j]   for j in l_idx],
    }


def detect_interface(pdb_path: str,
                     receptor_chains: List[str],
                     ligand_chains: List[str],
                     distance_cutoff: float = 8.0,
                     sasa_cutoff: float = 1.0) -> Dict[str, list]:
    """
    Delegates to ``interface_analyzer.get_interface_residues_with_sasa`` when
    available (guarantees identical results to the training pipeline).
    Falls back to a built-in distance-only implementation otherwise.
    """
    if INTERFACE_ANALYZER_AVAILABLE:
        print(f"[interface] using interface_analyzer.py  "
              f"(distance={distance_cutoff} Å, SASA={sasa_cutoff} Å²)")
        iface = _get_iface_sasa(
            pdb_path,
            chain_ids_1=receptor_chains,
            chain_ids_2=ligand_chains,
            distance_cutoff=distance_cutoff,
            sasa_cutoff=sasa_cutoff,
        )
    else:
        print(f"[interface] interface_analyzer.py unavailable; "
              f"falling back to built-in distance-only (cutoff={distance_cutoff} Å)")
        iface = _distance_interface_fallback(
            pdb_path, receptor_chains, ligand_chains, distance_cutoff)

    print(f"[interface] result: "
          f"{len(iface['receptor'])} receptor, {len(iface['ligand'])} ligand residues")
    return iface


# ===========================================================================
# STEP 2  –  Sequence extraction from PDB
# ===========================================================================

def _has_backbone(residue) -> bool:
    return {"N", "CA", "C"}.issubset({a.get_id() for a in residue})


def _sidechain_centroid(residue) -> np.ndarray:
    bb = {"N", "CA", "C", "O"}
    sc = [a.get_coord() for a in residue if a.get_id() not in bb]
    if sc:
        return np.mean(sc, axis=0).astype(np.float32)
    if residue.has_id("CA"):
        return residue["CA"].get_coord().astype(np.float32)
    return np.full(3, np.nan, dtype=np.float32)


def extract_chain_info(pdb_path: str, chain_id: str
                       ) -> Tuple[str, Dict[int, int], Dict[int, np.ndarray]]:
    """
    Returns:
        sequence       : full amino-acid sequence (1-letter) for the chain
        resid_to_seqidx: {pdb_residue_id → 0-based sequence index}
        centroid_map   : {pdb_residue_id → sidechain centroid xyz}
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)
    model = next(structure.get_models())

    if chain_id not in model:
        avail = [c.id for c in model.get_chains()]
        raise ValueError(f"Chain '{chain_id}' not in PDB.  Available: {avail}")

    seq: List[str] = []
    resid_map: Dict[int, int] = {}
    centroid_map: Dict[int, np.ndarray] = {}

    seq_idx = 0
    for res in model[chain_id].get_residues():
        resname = res.get_resname().strip().upper()
        is_std = resname in THREE_TO_ONE
        if not (is_std or _has_backbone(res)):
            continue
        aa = THREE_TO_ONE.get(resname, "X")
        pdb_rid = int(res.id[1])
        seq.append(aa)
        resid_map[pdb_rid]    = seq_idx
        centroid_map[pdb_rid] = _sidechain_centroid(res)
        seq_idx += 1

    return "".join(seq), resid_map, centroid_map


# ===========================================================================
# STEP 3  –  ESM-2 representations
# ===========================================================================

def load_esm2(model_name: str = "esm2_t33_650M_UR50D",
              device: torch.device = torch.device("cpu")):
    try:
        import esm as esm_lib
    except ImportError:
        sys.exit("fair-esm not found.  Install with: pip install fair-esm")

    fn = getattr(esm_lib.pretrained, model_name, None)
    if fn is None:
        avail = [n for n in dir(esm_lib.pretrained) if n.startswith("esm")]
        sys.exit(f"Unknown model '{model_name}'.  Available: {avail}")

    model, alphabet = fn()
    model = model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def _esm_masked_representations(model,
                                  batch_converter,
                                  sequence: str,
                                  positions: List[int],   # 1-based token positions (ESM convention)
                                  max_batch: int,
                                  device: torch.device) -> List[np.ndarray]:
    """
    Masked-prediction representation for each position in `positions`.
    Mirrors calculate_chain_masking_batch() in esm_20aa_feature_extractor_batch.py.

    Returns list of (1280,) float32 arrays, one per position.
    """
    repr_layer = model.num_layers
    _, _, toks = batch_converter([("seq", sequence)])
    toks = toks.to(device)
    mask_idx = batch_converter.alphabet.mask_idx

    pos_t = torch.tensor(positions, device=device)
    results: List[torch.Tensor] = []

    for start in range(0, len(positions), max_batch):
        chunk = pos_t[start : start + max_batch]
        B = chunk.shape[0]
        toks_rep = toks.repeat(B, 1)
        toks_rep[torch.arange(B, device=device), chunk] = mask_idx

        use_fp16 = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            with torch.no_grad():
                out = model(toks_rep, repr_layers=[repr_layer], return_contacts=False)
        reprs = out["representations"][repr_layer]   # [B, L+2, 1280]
        results.extend(
            reprs[torch.arange(B, device=device), chunk].detach().float().cpu()
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return [r.numpy().astype(np.float32) for r in results]


def _esm_standard_representations(model,
                                    batch_converter,
                                    sequence: str,
                                    positions: List[int],
                                    device: torch.device) -> List[np.ndarray]:
    """
    Standard (non-masked) forward pass.  Faster but slightly different
    from training data.  Use --no_masking to select this path.
    """
    repr_layer = model.num_layers
    _, _, toks = batch_converter([("seq", sequence)])
    toks = toks.to(device)

    use_fp16 = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
        with torch.no_grad():
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
    reprs = out["representations"][repr_layer][0]    # [L+2, 1280]
    return [reprs[p].float().cpu().numpy().astype(np.float32) for p in positions]


def extract_interface_esm_reps(model,
                                batch_converter,
                                pdb_path: str,
                                interface: Dict[str, list],
                                receptor_chains: List[str],
                                ligand_chains: List[str],
                                max_masks_per_step: int,
                                use_masking: bool,
                                device: torch.device
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        rec_reps  : (N_r, 1280) float32  – representations at receptor interface residues
        lig_reps  : (N_l, 1280) float32  – representations at ligand interface residues
    """
    # Build {chain_id → {pdb_resid → seq_idx}}
    all_chains = list(dict.fromkeys(receptor_chains + ligand_chains))
    chain_seq:     Dict[str, str]           = {}
    chain_resid_map: Dict[str, Dict[int, int]] = {}

    for cid in all_chains:
        seq, rid_map, _ = extract_chain_info(pdb_path, cid)
        chain_seq[cid]      = seq
        chain_resid_map[cid] = rid_map

    def _reps_for_side(residue_list: list) -> np.ndarray:
        """Collect representations for one side (receptor or ligand)."""
        # Group by chain to run one model call per chain
        per_chain: Dict[str, List[int]] = {}   # chain_id → list of seq positions (1-based for ESM)
        order:     Dict[str, List[int]] = {}   # chain_id → original list indices (to preserve output order)

        for k, r in enumerate(residue_list):
            cid  = r["chain_id"]
            prid = r["residue_id"]
            seqmap = chain_resid_map.get(cid, {})
            if prid not in seqmap:
                continue
            seq_idx = seqmap[prid]     # 0-based index in sequence
            esm_pos = seq_idx + 1      # ESM tokens: [CLS] at 0, residues start at 1
            per_chain.setdefault(cid, []).append(esm_pos)
            order.setdefault(cid, []).append(k)

        side_reps: Dict[int, np.ndarray] = {}   # original index → 1280-d

        for cid, positions in per_chain.items():
            seq = chain_seq[cid]
            if use_masking:
                reps = _esm_masked_representations(
                    model, batch_converter, seq, positions,
                    max_masks_per_step, device)
            else:
                reps = _esm_standard_representations(
                    model, batch_converter, seq, positions, device)

            for orig_k, rep in zip(order[cid], reps):
                side_reps[orig_k] = rep

        # Preserve residue order
        out = [side_reps[k] for k in sorted(side_reps)]
        if not out:
            return np.zeros((0, 1280), dtype=np.float32)
        return np.stack(out, axis=0).astype(np.float32)

    print(f"[esm] extracting receptor representations "
          f"({len(interface['receptor'])} residues)...")
    rec_reps = _reps_for_side(interface["receptor"])

    print(f"[esm] extracting ligand representations "
          f"({len(interface['ligand'])} residues)...")
    lig_reps = _reps_for_side(interface["ligand"])

    return rec_reps, lig_reps


# ===========================================================================
# STEP 4  –  Feature pooling
# ===========================================================================

def pool_features(rec_reps: np.ndarray, lig_reps: np.ndarray) -> np.ndarray:
    """
    Mean-pool each side → concat → 2560-d.
    Matches the logic in extract_esm_features.py / train_piaco2.py.
    """
    if rec_reps.shape[0] == 0:
        raise ValueError("No receptor representations.  Check interface detection.")
    if lig_reps.shape[0] == 0:
        raise ValueError("No ligand representations.  Check interface detection.")
    return np.concatenate([rec_reps.mean(0), lig_reps.mean(0)]).astype(np.float32)


# ===========================================================================
# STEP 5  –  Scikit-learn LR prediction
# ===========================================================================

def predict_lr(feature_vec: np.ndarray, model_path: str) -> Tuple[float, float]:
    """
    Load a joblib scikit-learn estimator and predict.

    Returns:
        prob_class1 : probability for class 1
        logit       : log-odds (logit) – useful for debugging
    """
    try:
        import joblib
    except ImportError:
        sys.exit("joblib not found.  Install with: pip install scikit-learn joblib")

    lr = joblib.load(model_path)

    # Compatibility patch: sklearn ≥1.7 removed multi_class; older versions need it.
    if not hasattr(lr, "multi_class"):
        lr.multi_class = "auto"

    X = feature_vec.reshape(1, -1)

    if hasattr(lr, "predict_proba"):
        proba = lr.predict_proba(X)[0]
        # proba might be [p_class0, p_class1] or [p_class1] depending on model
        prob1 = float(proba[1]) if len(proba) == 2 else float(proba[0])
    elif hasattr(lr, "decision_function"):
        score = float(lr.decision_function(X)[0])
        import math
        prob1 = 1.0 / (1.0 + math.exp(-score))
    else:
        raise AttributeError("Model has no predict_proba or decision_function.")

    logit = float(np.log(prob1 / max(1.0 - prob1, 1e-12)))
    return prob1, logit


# ===========================================================================
# CLI
# ===========================================================================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PDB → interface → ESM-2 features → LR probability",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input
    p.add_argument("--pdb",      required=True, help="Input PDB file")
    p.add_argument("--receptor", required=True, help="Receptor chain ID(s), e.g. A or AB")
    p.add_argument("--ligand",   required=True, help="Ligand chain ID(s), e.g. B")

    # Interface
    p.add_argument("--distance_cutoff", type=float, default=8.0,
                   help="Minimum inter-residue atom distance cutoff (Å)")
    p.add_argument("--sasa_cutoff",     type=float, default=1.0,
                   help="Buried SASA cutoff (Å²); ignored when freesasa unavailable")
    p.add_argument("--save_interface_json", type=str, default=None,
                   help="Optional path to save the interface JSON")

    # ESM-2
    p.add_argument("--esm2_model",        type=str, default="esm2_t33_650M_UR50D")
    p.add_argument("--no_masking",        action="store_true",
                   help="Use standard (non-masked) ESM forward pass "
                        "(faster, slight accuracy difference from training data)")
    p.add_argument("--max_masks_per_step", type=int, default=256,
                   help="Max residues per masked forward pass (reduce if OOM)")

    # Model
    p.add_argument("--lr_model", required=True,
                   help="Path to joblib-saved scikit-learn LR model")

    # Runtime
    p.add_argument("--device",    type=str, default="cuda:0",
                   help="Torch device.  Falls back to CPU if cuda unavailable.")
    p.add_argument("--save_json", type=str, default=None,
                   help="Optional path to save full result as JSON")
    return p


def main() -> None:
    args = build_cli().parse_args()

    # Device
    if args.device != "cpu" and not torch.cuda.is_available():
        print(f"[warn] CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    receptor_chains = list(args.receptor)   # "AB" → ['A','B']
    ligand_chains   = list(args.ligand)

    # ------------------------------------------------------------------
    # 1. Interface
    # ------------------------------------------------------------------
    interface = detect_interface(
        args.pdb, receptor_chains, ligand_chains,
        distance_cutoff=args.distance_cutoff,
        sasa_cutoff=args.sasa_cutoff,
    )

    if not interface["receptor"] or not interface["ligand"]:
        sys.exit("[error] Interface detection returned no residues on one or both sides. "
                 "Check chain IDs or lower --distance_cutoff / --sasa_cutoff.")

    if args.save_interface_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_interface_json)), exist_ok=True)
        with open(args.save_interface_json, "w", encoding="utf-8") as f:
            json.dump(interface, f, indent=2, ensure_ascii=False)
        print(f"[interface] saved → {args.save_interface_json}")

    # ------------------------------------------------------------------
    # 2 & 3. ESM-2 feature extraction
    # ------------------------------------------------------------------
    print(f"[esm] loading {args.esm2_model} ...")
    esm_model, batch_converter = load_esm2(args.esm2_model, device)

    use_masking = not args.no_masking
    print(f"[esm] representation mode: {'masked (matches training)' if use_masking else 'standard (fast)'}")

    rec_reps, lig_reps = extract_interface_esm_reps(
        esm_model, batch_converter,
        args.pdb, interface,
        receptor_chains, ligand_chains,
        args.max_masks_per_step,
        use_masking, device,
    )
    print(f"[esm] receptor reps: {rec_reps.shape}  ligand reps: {lig_reps.shape}")

    # ------------------------------------------------------------------
    # 4. Pool → 2560-d
    # ------------------------------------------------------------------
    feat = pool_features(rec_reps, lig_reps)
    print(f"[pool] feature vector shape: {feat.shape}")

    # ------------------------------------------------------------------
    # 5. LR prediction
    # ------------------------------------------------------------------
    prob, logit = predict_lr(feat, args.lr_model)

    result = {
        "pdb":              os.path.abspath(args.pdb),
        "receptor":         args.receptor,
        "ligand":           args.ligand,
        "lr_model":         os.path.abspath(args.lr_model),
        "device":           str(device),
        "esm2_model":       args.esm2_model,
        "masking":          use_masking,
        "interface": {
            "n_receptor": len(interface["receptor"]),
            "n_ligand":   len(interface["ligand"]),
            "freesasa":   FREESASA_AVAILABLE,
        },
        "feature_dim":  int(feat.shape[0]),
        "prob":         float(prob),
        "logit":        float(logit),
    }

    # Print summary
    print("\n" + "=" * 50)
    print(f"  Probability (class 1) : {prob:.4f}")
    print(f"  Logit                 : {logit:+.4f}")
    print(f"  Interface residues    : {len(interface['receptor'])} receptor / "
          f"{len(interface['ligand'])} ligand")
    print(f"  FreeSASA used         : {FREESASA_AVAILABLE}")
    print("=" * 50)

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[result] saved → {args.save_json}")


if __name__ == "__main__":
    main()
