"""
run_preprocess_piaco2.py
========================
Preprocess PDB files into fixed-size atom-level point clouds for PIACO2.

Output feature layout per atom point (36 dimensions):
  dims  0: 3  - xyz coordinates
  dims  3:14  - atom type one-hot (11 classes: C N O S F P Cl Br B H Others)
  dims 14:34  - residue type one-hot (20 standard amino acids)
  dims 34:36  - receptor/ligand flag: receptor=[1,0], ligand=[0,1]

Output file:
  <output_dir>/<stem>.npy  shape (npoint, 36), dtype float32
  Receptor atoms occupy rows [:npoint//2], ligand rows [npoint//2:].
  Rows beyond actual atom count are zero-padded.

Usage (single file):
  python run_preprocess_piaco2.py protein.pdb --receptor A --ligand B --npoint 1000

Usage (batch):
  python run_preprocess_piaco2.py dataset_dir/ --batch --receptor A --ligand B
  # Each sub-directory is expected to contain one PDB file.

Requirements:
  pip install biopython numpy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from Bio.PDB import PDBParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ATOM_TYPES: List[str] = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "B", "H", "Others"]
RES_TYPES: List[str] = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
# Selenomethionine treated as methionine
_MSE_TO_MET = {"MSE": "MET"}

ATOM_INDEX = {a: i for i, a in enumerate(ATOM_TYPES)}
RES_INDEX  = {r: i for i, r in enumerate(RES_TYPES)}

INTERFACE_CUTOFF_ANGSTROM = 8.0   # residue-level interface filter
FEATURE_DIM = 36                  # xyz(3) + atom_oh(11) + res_oh(20) + rl(2)


# ---------------------------------------------------------------------------
# One-hot helpers
# ---------------------------------------------------------------------------

def _atom_onehot(atom_name: str) -> np.ndarray:
    """Return length-11 one-hot vector for atom element."""
    vec = np.zeros(len(ATOM_TYPES), dtype=np.float32)
    vec[ATOM_INDEX.get(atom_name, ATOM_INDEX["Others"])] = 1.0
    return vec


def _res_onehot(res_name: str) -> np.ndarray:
    """Return length-20 one-hot vector. Unknown residues → zero vector."""
    vec = np.zeros(len(RES_TYPES), dtype=np.float32)
    idx = RES_INDEX.get(res_name)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _rl_flag(is_receptor: bool) -> np.ndarray:
    """Return [1,0] for receptor, [0,1] for ligand."""
    return np.array([1.0, 0.0], dtype=np.float32) if is_receptor else np.array([0.0, 1.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def _element_from_name(atom_name: str) -> str:
    """
    Infer atom element from PDB atom name string (e.g. ' CA ', ' OG1').
    Returns one of the ATOM_TYPES strings.
    """
    name = atom_name.strip()
    # BioPython sometimes provides element directly; if not, use first non-digit char
    for elem in ("Cl", "Br"):
        if name.upper().startswith(elem.upper()):
            return elem
    first = name.lstrip("0123456789")[:1].upper()
    mapping = {"C": "C", "N": "N", "O": "O", "S": "S",
               "F": "F", "P": "P", "B": "B", "H": "H"}
    return mapping.get(first, "Others")


def parse_atoms(
    pdb_path: str,
    receptor_chains: List[str],
    ligand_chains: List[str],
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Parse a PDB file and return (receptor_atoms, ligand_atoms).

    Each atom is a tuple:
        (x, y, z, element_str, res_name_str, res_id_int, chain_id_str)

    Only standard amino-acid residues are kept (unknown residues silently dropped).
    MSE is mapped to MET.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)

    receptor_atoms: List[Tuple] = []
    ligand_atoms:   List[Tuple] = []

    for model in structure.get_models():
        for chain in model.get_chains():
            cid = chain.id
            if cid not in receptor_chains and cid not in ligand_chains:
                continue
            is_receptor = cid in receptor_chains

            for residue in chain.get_residues():
                res_name = residue.get_resname().strip()
                res_name = _MSE_TO_MET.get(res_name, res_name)
                if res_name not in RES_INDEX:
                    continue  # skip non-standard residues
                res_id = residue.get_id()[1]

                for atom in residue.get_atoms():
                    try:
                        coord = atom.get_vector().get_array().astype(np.float32)
                    except Exception:
                        continue
                    element = _element_from_name(atom.get_name())
                    entry = (float(coord[0]), float(coord[1]), float(coord[2]),
                             element, res_name, res_id, cid)
                    if is_receptor:
                        receptor_atoms.append(entry)
                    else:
                        ligand_atoms.append(entry)
        break  # use first model only

    return receptor_atoms, ligand_atoms


# ---------------------------------------------------------------------------
# Interface filtering
# ---------------------------------------------------------------------------

def _residue_min_dist(
    r_atoms: List[Tuple],
    l_atoms: List[Tuple],
    cutoff: float = INTERFACE_CUTOFF_ANGSTROM,
) -> Tuple[set, set]:
    """
    Return sets of (chain_id, res_id) pairs for residues within `cutoff` Å.
    Operates at the atom-level for distance, grouped by residue.
    """
    r_coords = np.array([[a[0], a[1], a[2]] for a in r_atoms], dtype=np.float32)
    l_coords = np.array([[a[0], a[1], a[2]] for a in l_atoms], dtype=np.float32)

    # Pairwise distances via broadcasting (memory-friendly for typical interface sizes)
    diff = r_coords[:, None, :] - l_coords[None, :, :]  # (Nr, Nl, 3)
    dist = np.sqrt((diff ** 2).sum(-1))                  # (Nr, Nl)

    close_r, close_l = np.where(dist <= cutoff)
    receptor_res = {(r_atoms[i][6], r_atoms[i][5]) for i in close_r}
    ligand_res   = {(l_atoms[i][6], l_atoms[i][5]) for i in close_l}
    return receptor_res, ligand_res


def filter_to_interface(
    receptor_atoms: List[Tuple],
    ligand_atoms:   List[Tuple],
    cutoff: float = INTERFACE_CUTOFF_ANGSTROM,
) -> Tuple[List[Tuple], List[Tuple]]:
    """Keep only atoms belonging to interface residues (within cutoff Å)."""
    if not receptor_atoms or not ligand_atoms:
        return receptor_atoms, ligand_atoms

    r_res, l_res = _residue_min_dist(receptor_atoms, ligand_atoms, cutoff)
    r_filtered = [a for a in receptor_atoms if (a[6], a[5]) in r_res]
    l_filtered = [a for a in ligand_atoms   if (a[6], a[5]) in l_res]

    print(f"  Interface residues: {len(r_res)} receptor, {len(l_res)} ligand")
    print(f"  Interface atoms:    {len(r_filtered)} receptor, {len(l_filtered)} ligand")
    return r_filtered, l_filtered


# ---------------------------------------------------------------------------
# Atom selection: nearest N/2 pairs by inter-chain distance
# ---------------------------------------------------------------------------

def select_interface_atoms(
    receptor_atoms: List[Tuple],
    ligand_atoms:   List[Tuple],
    n_each: int,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Select up to n_each atoms from each chain without repetition.

    # Greedy bipartite matching: sort all receptor–ligand atom pairs by distance,
    # then pick the closest non-overlapping 1-to-1 pairs (each atom used at most once).
    """
    if not receptor_atoms or not ligand_atoms:
        return receptor_atoms[:n_each], ligand_atoms[:n_each]

    r_coords = np.array([[a[0], a[1], a[2]] for a in receptor_atoms], dtype=np.float32)
    l_coords = np.array([[a[0], a[1], a[2]] for a in ligand_atoms],   dtype=np.float32)

    diff = r_coords[:, None, :] - l_coords[None, :, :]
    dist_mat = np.sqrt((diff ** 2).sum(-1))              # (Nr, Nl)

    # Flatten and sort
    flat_idx = np.argsort(dist_mat, axis=None)
    Nr, Nl = dist_mat.shape

    selected_r: List[int] = []
    selected_l: List[int] = []
    seen_r: set = set()
    seen_l: set = set()

    for idx in flat_idx:
        ri, li = divmod(int(idx), Nl)
        if ri not in seen_r and li not in seen_l:
            selected_r.append(ri)
            selected_l.append(li)
            seen_r.add(ri)
            seen_l.add(li)
        if len(selected_r) >= n_each:
            break

    r_out = [receptor_atoms[i] for i in selected_r]
    l_out = [ligand_atoms[i]   for i in selected_l]
    print(f"  Selected atoms: {len(r_out)} receptor, {len(l_out)} ligand")
    return r_out, l_out


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def encode_atom(atom: Tuple, is_receptor: bool) -> np.ndarray:
    """
    Encode a single atom tuple into a 36-dim feature vector.

    Layout:
      [0:3]   xyz
      [3:14]  atom-type one-hot (11 dims)
      [14:34] residue-type one-hot (20 dims)
      [34:36] R/L flag: receptor=[1,0], ligand=[0,1]
    """
    x, y, z, element, res_name, _res_id, _chain = atom
    xyz     = np.array([x, y, z], dtype=np.float32)
    atom_oh = _atom_onehot(element)
    res_oh  = _res_onehot(res_name)
    rl      = _rl_flag(is_receptor)
    return np.concatenate([xyz, atom_oh, res_oh, rl])


def build_point_cloud(
    receptor_atoms: List[Tuple],
    ligand_atoms:   List[Tuple],
    npoint: int,
) -> np.ndarray:
    """
    Encode atoms and zero-pad to (npoint, 36).

    Rows   [:npoint//2]   → receptor
    Rows   [npoint//2:]   → ligand
    """
    half = npoint // 2

    r_feats = [encode_atom(a, is_receptor=True)  for a in receptor_atoms[:half]]
    l_feats = [encode_atom(a, is_receptor=False) for a in ligand_atoms[:half]]

    def pad(feats: list, target: int) -> np.ndarray:
        arr = np.zeros((target, FEATURE_DIM), dtype=np.float32)
        if feats:
            arr[:len(feats)] = np.stack(feats)
        return arr

    r_block = pad(r_feats, half)
    l_block = pad(l_feats, half)
    return np.concatenate([r_block, l_block], axis=0)  # (npoint, 36)


# ---------------------------------------------------------------------------
# Optional: pool dMaSIF surface features onto atom points
# ---------------------------------------------------------------------------

def load_dmasif_npy(npy_folder: str, pdb_id: str, chain_id: str) -> np.ndarray:
    """
    Load a dMaSIF surface .npy file.
    Expected filename: {npy_folder}/{pdb_id}_chain_{chain_id}.npy
    Each row is [x, y, z, feat_0, ..., feat_15] (shape: M, 19).
    """
    path = os.path.join(npy_folder, f"{pdb_id}_chain_{chain_id}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"dMaSIF file not found: {path}")
    arr = np.load(path).astype(np.float32)
    print(f"  Loaded dMaSIF: {path}  shape={arr.shape}")
    return arr


def append_dmasif_features(
    cloud: np.ndarray,
    r_dmasif: np.ndarray,
    l_dmasif: np.ndarray,
    npoint: int,
    threshold: float = 4.5,
) -> np.ndarray:
    """
    Pool dMaSIF surface features onto the atom-level point cloud and append.

    For each atom point, the dMaSIF surface points within `threshold` Å are
    averaged.  If none are found, a zero vector is used.

    Parameters:
        cloud     : (npoint, 36) atom point cloud from build_point_cloud()
        r_dmasif  : (Mr, 3+F) receptor dMaSIF surface points
        l_dmasif  : (Ml, 3+F) ligand   dMaSIF surface points
        npoint    : total number of points (half receptor, half ligand)
        threshold : neighbourhood radius in Å

    Returns:
        np.ndarray of shape (npoint, 36 + F), dtype float32
        where F = r_dmasif.shape[1] - 3  (typically 16)
    """
    half = npoint // 2
    F = r_dmasif.shape[1] - 3  # feature dims (excluding xyz)

    def pool_one_side(atom_block: np.ndarray, surface: np.ndarray) -> np.ndarray:
        """atom_block: (half, 36), surface: (M, 3+F) → pooled: (half, F)"""
        atom_coords   = atom_block[:, :3]            # (half, 3)
        surface_coords = surface[:, :3]              # (M, 3)
        surface_feats  = surface[:, 3:]              # (M, F)
        pooled = np.zeros((len(atom_block), F), dtype=np.float32)
        for i, coord in enumerate(atom_coords):
            if np.all(coord == 0.0):
                continue  # padding row – leave as zeros
            dists = np.linalg.norm(surface_coords - coord, axis=1)
            mask  = dists <= threshold
            if mask.any():
                pooled[i] = surface_feats[mask].mean(axis=0)
        return pooled

    r_block  = cloud[:half]
    l_block  = cloud[half:]
    r_pooled = pool_one_side(r_block, r_dmasif)
    l_pooled = pool_one_side(l_block, l_dmasif)

    combined = np.concatenate([
        np.concatenate([r_block, r_pooled], axis=1),
        np.concatenate([l_block, l_pooled], axis=1),
    ], axis=0)  # (npoint, 36+F)
    print(f"  After dMaSIF append: {combined.shape}")
    return combined


# ---------------------------------------------------------------------------
# High-level single-file processing
# ---------------------------------------------------------------------------

def process_pdb(
    pdb_path: str,
    receptor_chains: List[str],
    ligand_chains:   List[str],
    npoint: int = 1000,
    npy_folder: str | None = None,
    pdb_id: str | None = None,
) -> np.ndarray:
    """
    Full preprocessing pipeline for one PDB file.

    Returns:
        np.ndarray of shape (npoint, 36) when npy_folder is None
        np.ndarray of shape (npoint, 52) when npy_folder is provided (36 + 16 dMaSIF dims)
    """
    print(f"\n[preprocess] {pdb_path}")
    print(f"  Receptor chains: {receptor_chains}  Ligand chains: {ligand_chains}")

    # 1. Parse
    r_atoms, l_atoms = parse_atoms(pdb_path, receptor_chains, ligand_chains)
    print(f"  Raw atoms: {len(r_atoms)} receptor, {len(l_atoms)} ligand")

    if not r_atoms:
        raise ValueError(f"No receptor atoms found in chains {receptor_chains}")
    if not l_atoms:
        raise ValueError(f"No ligand atoms found in chains {ligand_chains}")

    # 2. Interface filter (residue-level)
    r_atoms, l_atoms = filter_to_interface(r_atoms, l_atoms)

    # 3. Select N/2 nearest pairs
    r_atoms, l_atoms = select_interface_atoms(r_atoms, l_atoms, n_each=npoint // 2)

    # 4. Encode + pad
    cloud = build_point_cloud(r_atoms, l_atoms, npoint)

    # 5. Optional: append pooled dMaSIF surface features
    if npy_folder is not None:
        # Infer pdb_id from filename if not supplied
        _pid = pdb_id or Path(pdb_path).stem
        r_dmasif = load_dmasif_npy(npy_folder, _pid, receptor_chains[0])
        l_dmasif = load_dmasif_npy(npy_folder, _pid, ligand_chains[0])
        cloud = append_dmasif_features(cloud, r_dmasif, l_dmasif, npoint)

    print(f"  Output shape: {cloud.shape}")
    return cloud


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _find_chain_pairs(pdb_path: str) -> Tuple[str, str]:
    """Return (first_chain_id, second_chain_id) from a PDB file."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("X", pdb_path)
    model = next(struct.get_models())
    chains = [c.id for c in model.get_chains()]
    if len(chains) < 2:
        raise ValueError(f"PDB file must contain at least 2 chains, found: {chains}")
    return chains[0], chains[1]


def _resolve_chains(
    pdb_path: str,
    receptor_arg: str | None,
    ligand_arg: str | None,
) -> Tuple[List[str], List[str]]:
    """
    Determine receptor/ligand chains.  Falls back to first/second chain when
    --receptor / --ligand are not supplied.
    """
    if receptor_arg and ligand_arg:
        return list(receptor_arg), list(ligand_arg)
    c1, c2 = _find_chain_pairs(pdb_path)
    print(f"  Auto-detected chains: receptor={c1}, ligand={c2}")
    return [c1], [c2]


def _save(cloud: np.ndarray, out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, cloud)
    print(f"  Saved → {out_path}  {cloud.shape}")


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def batch_process(
    dataset_dir: str,
    output_dir: str,
    receptor_chains: List[str],
    ligand_chains:   List[str],
    npoint: int,
    npy_folder: str | None = None,
) -> None:
    """
    Process all PDB files found recursively under dataset_dir.

    Expected layout (one PDB per sub-directory):
        dataset_dir/
            complex_001/
                complex_001.pdb
            complex_002/
                complex_002.pdb
            ...
    """
    dataset_path = Path(dataset_dir)
    pdb_files = sorted(dataset_path.rglob("*.pdb"))
    print(f"Found {len(pdb_files)} PDB files under {dataset_dir}")

    errors = []
    for pdb_file in pdb_files:
        rel = pdb_file.relative_to(dataset_path)
        out_path = Path(output_dir) / rel.with_suffix(".npy")

        # Resolve chains per-file when not explicitly given
        try:
            r_chains, l_chains = _resolve_chains(
                str(pdb_file),
                receptor_chains[0] if receptor_chains else None,
                ligand_chains[0]   if ligand_chains   else None,
            )
            cloud = process_pdb(str(pdb_file), r_chains, l_chains, npoint,
                               npy_folder=npy_folder)
            _save(cloud, str(out_path))
        except Exception as exc:
            print(f"  [SKIP] {pdb_file}: {exc}")
            errors.append((str(pdb_file), str(exc)))

    print(f"\nDone. {len(pdb_files) - len(errors)}/{len(pdb_files)} succeeded.")
    if errors:
        err_file = Path(output_dir) / "preprocess_errors.txt"
        with open(err_file, "w") as f:
            for path, msg in errors:
                f.write(f"{path}\t{msg}\n")
        print(f"Errors written to {err_file}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess PDB files into 36-dim atom point clouds for PIACO2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", help="Path to a PDB file (single mode) or dataset directory (--batch mode).")
    p.add_argument("--output_dir", default="preprocessed", help="Output directory (default: preprocessed)")
    p.add_argument("--npoint",     type=int, default=1000,  help="Total number of points (default: 1000)")
    p.add_argument("--receptor",   default=None,            help="Receptor chain ID(s), e.g. A or AB")
    p.add_argument("--ligand",     default=None,            help="Ligand chain ID(s), e.g. B or CD")
    p.add_argument("--batch",      action="store_true",     help="Process all PDB files under input directory")
    p.add_argument(
        "--npy_folder", default=None,
        help="(Optional) Path to folder with dMaSIF .npy files. "
             "When provided, 16-dim pooled dMaSIF features are appended → output is (npoint, 52) "
             "instead of (npoint, 36). Required to reproduce the original trained model behaviour.",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: --batch requires a directory, got: {args.input}", file=sys.stderr)
            sys.exit(1)
        r_chains = list(args.receptor) if args.receptor else []
        l_chains = list(args.ligand)   if args.ligand   else []
        batch_process(args.input, args.output_dir, r_chains, l_chains, args.npoint,
                      npy_folder=args.npy_folder)
    else:
        if not os.path.isfile(args.input):
            print(f"Error: PDB file not found: {args.input}", file=sys.stderr)
            sys.exit(1)
        r_chains, l_chains = _resolve_chains(args.input, args.receptor, args.ligand)
        cloud = process_pdb(args.input, r_chains, l_chains, args.npoint,
                            npy_folder=args.npy_folder)
        stem     = Path(args.input).stem
        out_path = os.path.join(args.output_dir, stem + ".npy")
        _save(cloud, out_path)


if __name__ == "__main__":
    main()
