"""
ESM feature extraction for interface residues (Batch Processing Version)
指定されたディレクトリ構造に基づき、複数のサンプルをバッチ処理する
"""

import torch
import esm
import json
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser
from pathlib import Path
import h5py
import argparse
from tqdm import tqdm

# --- ヘルパー関数 (変更なし) ---
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',  
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',  
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',  
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',  
    'MSE': 'M'
}

def has_backbone_atoms(residue):
    backbone_atoms = {"N", "CA", "C"}
    residue_atoms = {atom.get_id() for atom in residue}
    return backbone_atoms.issubset(residue_atoms)

def get_sidechain_centroid(residue):
    sidechain_atoms = []
    backbone_ids = {"N", "CA", "C", "O"}
    for atom in residue:
        if atom.get_id() not in backbone_ids:
            sidechain_atoms.append(atom)
    if not sidechain_atoms:
        if "CA" in residue:
            return residue["CA"].get_coord()
        coords = [atom.get_coord() for atom in residue]
        return np.mean(coords, axis=0) if coords else np.array([np.nan, np.nan, np.nan])
    return np.mean([atom.get_coord() for atom in sidechain_atoms], axis=0)

def load_esm_model(model_name="esm2_t33_650M_UR50D"):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Using GPU for ESM calculations.")
    else:
        print("Using CPU for ESM calculations.")
    return model, batch_converter

def read_sequences_from_fasta(fasta_path):
    sequences = {}
    with open(fasta_path, 'r') as f:
        for record in SeqIO.parse(f, 'fasta'):
            try:
                chain_id = record.id.split('_')[-1]
                if len(chain_id) == 1:
                    sequences[chain_id] = str(record.seq)
            except Exception:
                pass
    return sequences

def get_residue_mapping_from_pdb(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    residue_mapping, centroid_mapping = {}, {}
    if chain_id in structure[0]:
        chain = structure[0][chain_id]
        seq_index = 0
        for residue in chain:
            is_counted = residue.get_resname() in three_to_one or has_backbone_atoms(residue)
            if is_counted:
                pdb_residue_id = residue.get_id()[1]
                residue_mapping[pdb_residue_id] = seq_index
                centroid_mapping[pdb_residue_id] = get_sidechain_centroid(residue)
                seq_index += 1
    return residue_mapping, centroid_mapping

def calculate_chain_masking_batch(model, batch_converter, sequence, positions, max_masks_per_step, device):
    _, _, toks = batch_converter([("seq", sequence)])
    toks = toks.to(device)
    pos_tok = torch.tensor(positions, device=device)
    log_probs_original, full_probs_distributions, representations_list = [], [], []
   
    # --- BUG FIX ---
    # 3文字コードではなく、標準的な1文字コードを使用する
    aa_tokens = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    aa_indices = torch.tensor([batch_converter.alphabet.get_idx(aa) for aa in aa_tokens], device=device)
   
    for start in range(0, len(positions), max_masks_per_step):
        chunk = pos_tok[start:start + max_masks_per_step]
        B = chunk.shape[0]
        toks_rep = toks.repeat(B, 1)
        wt_ids = toks_rep[torch.arange(B, device=device), chunk]
        toks_rep[torch.arange(B, device=device), chunk] = batch_converter.alphabet.get_idx("<mask>")
       
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            with torch.no_grad():
                out = model(toks_rep, repr_layers=[33], return_contacts=False)
                logits_pos = out["logits"][torch.arange(B, device=device), chunk]
                probs = torch.softmax(logits_pos, dim=-1)
                reprs_pos = out["representations"][33][torch.arange(B, device=device), chunk]
       
        # --- MEMORY LEAK FIX: .detach() を追加して計算グラフから切り離す ---
        log_probs_original.extend(torch.log(probs[torch.arange(B, device=device), wt_ids]).detach().cpu())
        full_probs_distributions.extend(probs[:, aa_indices].detach().cpu())
        representations_list.extend(reprs_pos.detach().cpu())
       
    return log_probs_original, full_probs_distributions, representations_list

# --- メインの計算関数 (モデルを引数で受け取るように変更) ---
def process_single_sample(model, batch_converter, pdb_path, fasta_path, interface_path, max_masks_per_step):
    device = next(model.parameters()).device
    with open(interface_path, 'r') as f:
        interface_residues = json.load(f)
   
    chain_sequences = read_sequences_from_fasta(fasta_path)
    if not chain_sequences:
        raise ValueError(f"No sequences in {fasta_path}")

    chain_residue_mapping, chain_centroid_mapping = {}, {}
   
    # --- NEW LOGIC: PDBをパースして順序付きチェインリストを取得 ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ordered_chain_ids = [chain.id for chain in structure[0]]

    for chain_id in ordered_chain_ids:
        if chain_id not in chain_sequences:
            continue # FASTAに存在しないチェインはスキップ
        residue_map, centroid_map = get_residue_mapping_from_pdb(pdb_path, chain_id)
        if len(chain_sequences[chain_id]) != len(residue_map):
            raise ValueError(f"Seq/Map length mismatch in {pdb_path} chain {chain_id}")
        chain_residue_mapping[chain_id] = residue_map
        chain_centroid_mapping[chain_id] = centroid_map

    chain_interface_positions, chain_interface_residues = {}, {}
   
    # --- DEFINITION CHANGE: run_preprocess.pyのロジックに合わせる ---
    # PDBの最初のチェインをreceptorと定義する
    if not ordered_chain_ids:
        raise ValueError(f"No chains found in PDB file: {pdb_path}")
    receptor_chain_id = ordered_chain_ids[0]
    receptor_chains = {receptor_chain_id}
    print(f"  Info for {Path(pdb_path).name}: Receptor chain set to '{receptor_chain_id}' based on PDB order.")
   
    for res_type in ['receptor', 'ligand']:
        for residue in interface_residues.get(res_type, []):
            chain_id, res_id = residue['chain_id'], residue['residue_id']
            if chain_id not in chain_interface_positions:
                chain_interface_positions[chain_id] = []
                chain_interface_residues[chain_id] = []
            if res_id in chain_residue_mapping.get(chain_id, {}):
                seq_idx = chain_residue_mapping[chain_id][res_id]
                chain_interface_positions[chain_id].append(seq_idx + 1)
                centroid = chain_centroid_mapping[chain_id].get(res_id)
                residue['sidechain_centroid'] = centroid.tolist() if centroid is not None else [np.nan]*3
                chain_interface_residues[chain_id].append(residue)

    masking_results = {'receptor': [], 'ligand': []}
    aa_tokens_1_char = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
   
    for chain_id in chain_interface_positions:
        sequence, positions, residues = chain_sequences[chain_id], chain_interface_positions[chain_id], chain_interface_residues[chain_id]
        log_probs, full_probs, reprs = calculate_chain_masking_batch(model, batch_converter, sequence, positions, max_masks_per_step, device)
       
        for i, (residue, log_prob, prob_dist, repr_vec) in enumerate(zip(residues, log_probs, full_probs, reprs)):
            res_with_probs = residue.copy()
            seq_idx = positions[i] - 1
            res_with_probs.update({
                'sequence_index': seq_idx, 'original_aa': sequence[seq_idx],
                'confidence_score': torch.exp(log_prob).item(),
                'mask_probabilities': {aa: p.item() for aa, p in zip(aa_tokens_1_char, prob_dist)},
                'esm_representation': repr_vec.tolist()
            })
           
            # --- この判定ロジックは変更不要 (receptor_chainsの定義が変わったため) ---
            res_type = 'receptor' if chain_id in receptor_chains else 'ligand'
            masking_results[res_type].append(res_with_probs)
           
    return masking_results

# --- HDF5保存関数 (HDF5グループオブジェクトを受け取るように変更) ---
def save_results_to_hdf5_group(results, h5_group):
    # --- BUG FIX ---
    # 上の calculate_chain_masking_batch と同じ順序のアミノ酸リストを使用する
    aa_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    h5_group.attrs['aa_order'] = json.dumps(aa_order)
    for res_type in ['receptor', 'ligand']:
        if not results[res_type]: continue
        res_group = h5_group.create_group(res_type)
        data_arrays = {
            'chain_ids': np.array([r['chain_id'] for r in results[res_type]], dtype='S1'),
            'residue_ids': np.array([r['residue_id'] for r in results[res_type]], dtype=np.int32),
            #'sequence_indices': np.array([r['sequence_index'] for r in results[res_type]], dtype=np.int32),
            'confidence_scores': np.array([r['confidence_score'] for r in results[res_type]], dtype=np.float32),
            'sidechain_centroids': np.array([r['sidechain_centroid'] for r in results[res_type]], dtype=np.float32),
            'mask_probabilities': np.array([[r['mask_probabilities'][aa] for aa in aa_order] for r in results[res_type]], dtype=np.float32),
            'esm_representations': np.array([r['esm_representation'] for r in results[res_type]], dtype=np.float32)
        }
        for name, data in data_arrays.items():
            res_group.create_dataset(name, data=data, compression='gzip')

# --- メインのバッチ処理 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process ESM feature extraction for protein interfaces.")
    parser.add_argument("--base-dir", required=True, help="Base directory containing 'pdb', 'fasta', and 'interface' subdirectories.")
    parser.add_argument("--output-hdf5", required=True, help="Path for the single output HDF5 file.")
    parser.add_argument("--max-masks-per-step", type=int, default=256, help="Max residues to mask in a single batch.")
    args = parser.parse_args()

    # 1. モデルを最初に1回だけ読み込む
    print("Loading ESM model (this may take a moment)...")
    model, batch_converter = load_esm_model()

    # 2. 処理対象のファイルを探索
    base_dir = Path(args.base_dir)
    pdb_files = sorted(list((base_dir / "pdb").glob("*.pdb")))
    if not pdb_files:
        print(f"Error: No PDB files found in '{base_dir / 'pdb'}'.")
        exit(1)
   
    print(f"Found {len(pdb_files)} PDB files to process.")

    # 3. 単一のHDF5ファイルを開き、ループ処理
    with h5py.File(args.output_hdf5, 'w') as h5_file:
        for pdb_path in tqdm(pdb_files, desc="Processing samples"):
            protein_id = pdb_path.stem
           
            fasta_path = base_dir / "fasta" / f"{protein_id}.fasta"
            interface_path = base_dir / "interface" / f"{protein_id}.json"

            if not (fasta_path.exists() and interface_path.exists()):
                tqdm.write(f"Warning: Missing FASTA or JSON for {protein_id}. Skipping.")
                continue
           
            try:
                # 4. 計算を実行 (モデルは再利用)
                results = process_single_sample(
                    model, batch_converter, str(pdb_path), str(fasta_path), str(interface_path), args.max_masks_per_step
                )
               
                # 5. HDF5にグループとして結果を追記
                protein_group = h5_file.create_group(protein_id)
                save_results_to_hdf5_group(results, protein_group)

            except Exception as e:
                tqdm.write(f"ERROR processing {protein_id}: {e}. Skipping.")
           
            # --- MEMORY LEAK FIX: 未使用のGPUキャッシュを強制的に解放 ---
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nBatch processing complete. Results saved to '{args.output_hdf5}'.")
