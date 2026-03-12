"""
Interface residue analysis utility
タンパク質複合体の界面残基を分析するためのユーティリティ
FreeSASAによる埋没面積計算も含む
"""

try:
    import freesasa
    FREESASA_AVAILABLE = True
except ImportError:
    FREESASA_AVAILABLE = False
    print("Warning: FreeSASA not available. SASA calculations will be skipped.")

import tempfile
import os
import json


def extract_chain_pdb(pdb_path, chain_ids, output_path):
    """
    指定したチェインのみを抽出してPDBファイルを作成
    """
    with open(pdb_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            if line.startswith('ATOM') and line[21] in chain_ids:
                outfile.write(line)
            elif line.startswith('END'):
                outfile.write(line)
                break


def calculate_residue_sasa(pdb_path, chain_ids=None):
    """
    FreeSASAを使って残基ごとのSASAを計算
    
    Return:
        dict: {(chain_id, residue_id): sasa_value, ...}
    """
    if not FREESASA_AVAILABLE:
        return {}
    
    try:
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)
        
        # 残基ごとのSASAを取得する正しい方法
        residue_sasa = {}
        
        # PDBファイルを再度読み込んで原子情報を取得
        with open(pdb_path, 'r') as file:
            atom_index = 0
            for line in file:
                if line.startswith('ATOM'):
                    try:
                        chain_id = line[21]
                        residue_id = int(line[22:26].strip())
                        
                        # 指定されたチェインのみ処理
                        if chain_ids is None or chain_id in chain_ids:
                            key = (chain_id, residue_id)
                            if key not in residue_sasa:
                                residue_sasa[key] = 0
                            
                            # 原子のSASAを取得して残基レベルで累積
                            atom_area = result.atomArea(atom_index)
                            residue_sasa[key] += atom_area
                        
                        atom_index += 1
                    except (ValueError, IndexError):
                        atom_index += 1
                        continue
        
        return residue_sasa
    
    except Exception as e:
        print(f"Error calculating SASA: {e}")
        return {}


def get_interface_residues_with_sasa(pdb_path, chain_ids_1=['A'], chain_ids_2=['B'], 
                                   distance_cutoff=8.0, sasa_cutoff=1.0):
    """
    距離とSASA変化量の両方を使って界面残基を定義
    
    Input:
        pdb_path: PDBファイルのパス
        chain_ids_1: レセプターチェインIDリスト
        chain_ids_2: リガンドチェインIDリスト
        distance_cutoff: 距離閾値（Å）
        sasa_cutoff: SASA変化量の最小値（Å²）
    
    Return:
        interface_residues: {
            'receptor': [{'chain_id': 'A', 'residue_id': 123, 'residue_name': 'ALA', 
                         'sasa_complex': 10.5, 'sasa_isolated': 25.3, 'sasa_buried': 14.8}, ...],
            'ligand': [...]
        }
    """
    # 基本的な距離ベースの界面残基取得
    distance_based_interface = get_interface_residues(pdb_path, chain_ids_1, chain_ids_2, distance_cutoff)
    
    if not FREESASA_AVAILABLE:
        print("FreeSASA not available. Returning distance-based interface only.")
        return distance_based_interface
    
    print("Calculating SASA changes...")
    
    # 一時ファイルを作成
    with tempfile.TemporaryDirectory() as temp_dir:
        complex_pdb = os.path.join(temp_dir, "complex.pdb")
        receptor_pdb = os.path.join(temp_dir, "receptor.pdb")
        ligand_pdb = os.path.join(temp_dir, "ligand.pdb")
        
        # 複合体、レセプター単体、リガンド単体のPDBファイルを作成
        # 複合体（全チェイン）
        extract_chain_pdb(pdb_path, chain_ids_1 + chain_ids_2, complex_pdb)
        # レセプター単体
        extract_chain_pdb(pdb_path, chain_ids_1, receptor_pdb)
        # リガンド単体
        extract_chain_pdb(pdb_path, chain_ids_2, ligand_pdb)
        
        # SASA計算
        complex_sasa = calculate_residue_sasa(complex_pdb, chain_ids_1 + chain_ids_2)
        receptor_sasa = calculate_residue_sasa(receptor_pdb, chain_ids_1)
        ligand_sasa = calculate_residue_sasa(ligand_pdb, chain_ids_2)
    
    # SASA変化量を計算して界面残基を絞り込み
    def filter_by_sasa(residues, isolated_sasa, is_receptor=True):
        filtered_residues = []
        
        for residue in residues:
            chain_id = residue['chain_id']
            residue_id = residue['residue_id']
            key = (chain_id, residue_id)
            
            complex_area = complex_sasa.get(key, 0)
            isolated_area = isolated_sasa.get(key, 0)
            buried_area = isolated_area - complex_area
            
            # SASA変化量が閾値以上の場合のみ界面残基とする
            if buried_area >= sasa_cutoff:
                residue_with_sasa = residue.copy()
                residue_with_sasa.update({
                    'sasa_complex': round(complex_area, 2),
                    'sasa_isolated': round(isolated_area, 2),
                    'sasa_buried': round(buried_area, 2)
                })
                filtered_residues.append(residue_with_sasa)
        
        return filtered_residues
    
    # SASA条件でフィルタリング
    filtered_receptor = filter_by_sasa(distance_based_interface['receptor'], receptor_sasa, True)
    filtered_ligand = filter_by_sasa(distance_based_interface['ligand'], ligand_sasa, False)
    
    print(f"SASA filtering: {len(distance_based_interface['receptor'])} -> {len(filtered_receptor)} receptor residues")
    print(f"SASA filtering: {len(distance_based_interface['ligand'])} -> {len(filtered_ligand)} ligand residues")
    
    return {
        'receptor': filtered_receptor,
        'ligand': filtered_ligand
    }


def get_interface_residues(pdb_path, chain_ids_1=['A'], chain_ids_2=['B'], cut_off=8.0):
    """
    距離のみを使った界面残基のリストを取得する関数（既存関数）
    """
    receptor_residues = []
    ligand_residues = []
    rlist = []
    llist = []
    
    # 残基をグループ化するための状態変数
    tmp_r_list = []
    pre_r_residue_id = None
    pre_r_chain_id = None
    pre_r_residue_name = None
    
    tmp_l_list = []
    pre_l_residue_id = None
    pre_l_chain_id = None
    pre_l_residue_name = None

    with open(pdb_path, 'r') as file:
        line = file.readline()
        while line:
            if line.startswith('ATOM'):
                current_chain_id = line[21]
                
                try:
                    current_residue_id = int(line[22:26].strip())
                    current_residue_name = line[17:20].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    atom_type = line[13:16].strip()
                except (ValueError, IndexError):
                    line = file.readline()
                    continue

                if current_chain_id in chain_ids_1:
                    # 残基IDが変わったら、前の残基を保存
                    if pre_r_residue_id is not None and current_residue_id != pre_r_residue_id:
                        if tmp_r_list:
                            rlist.append(tmp_r_list)
                            receptor_residues.append({
                                'chain_id': pre_r_chain_id,
                                'residue_id': pre_r_residue_id,
                                'residue_name': pre_r_residue_name
                            })
                        tmp_r_list = []
                    
                    tmp_r_list.append([x, y, z, atom_type])
                    pre_r_residue_id = current_residue_id
                    pre_r_chain_id = current_chain_id
                    pre_r_residue_name = current_residue_name

                elif current_chain_id in chain_ids_2:
                    # 残基IDが変わったら、前の残基を保存
                    if pre_l_residue_id is not None and current_residue_id != pre_l_residue_id:
                        if tmp_l_list:
                            llist.append(tmp_l_list)
                            ligand_residues.append({
                                'chain_id': pre_l_chain_id,
                                'residue_id': pre_l_residue_id,
                                'residue_name': pre_l_residue_name
                            })
                        tmp_l_list = []

                    tmp_l_list.append([x, y, z, atom_type])
                    pre_l_residue_id = current_residue_id
                    pre_l_chain_id = current_chain_id
                    pre_l_residue_name = current_residue_name
            
            line = file.readline()

    # 最後の残基を追加
    if tmp_r_list:
        rlist.append(tmp_r_list)
        receptor_residues.append({
            'chain_id': pre_r_chain_id,
            'residue_id': pre_r_residue_id,
            'residue_name': pre_r_residue_name
        })
    if tmp_l_list:
        llist.append(tmp_l_list)
        ligand_residues.append({
            'chain_id': pre_l_chain_id,
            'residue_id': pre_l_residue_id,
            'residue_name': pre_l_residue_name
        })

    print(f"Total residues: {len(receptor_residues)} receptor, {len(ligand_residues)} ligand")

    # 界面残基の判定
    cut_off_sq = cut_off ** 2
    r_interface_indices = set()
    l_interface_indices = set()
    
    for rindex, r_residue_atoms in enumerate(rlist):
        for lindex, l_residue_atoms in enumerate(llist):
            min_distance = float('inf')
            
            for r_atom in r_residue_atoms:
                for l_atom in l_residue_atoms:
                    distance_sq = 0
                    for k in range(3):  # x, y, z座標
                        distance_sq += (r_atom[k] - l_atom[k]) ** 2
                    
                    if distance_sq < min_distance:
                        min_distance = distance_sq
            
            if min_distance <= cut_off_sq:
                r_interface_indices.add(rindex)
                l_interface_indices.add(lindex)

    # 界面残基の情報を抽出
    interface_receptor_residues = [receptor_residues[i] for i in r_interface_indices]
    interface_ligand_residues = [ligand_residues[i] for i in l_interface_indices]
    
    print(f"Interface residues (distance-based): {len(interface_receptor_residues)} receptor, {len(interface_ligand_residues)} ligand")
    
    return {
        'receptor': interface_receptor_residues,
        'ligand': interface_ligand_residues
    }


def save_interface_residues(interface_residues, output_path):
    """
    界面残基の結果をJSONファイルに保存
    """
    with open(output_path, 'w') as f:
        json.dump(interface_residues, f, indent=2)
    print(f"Interface residues saved to {output_path}")

def load_interface_residues(input_path):
    """
    保存された界面残基の結果を読み込み
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def analyze_interface_residues(pdb_path, chain_ids_1=['A'], chain_ids_2=['B'], 
                             distance_cutoff=8.0, sasa_cutoff=1.0, use_sasa=True,
                             save_path=None):
    """
    界面残基を分析して結果を表示する便利関数
    """
    print(f"Analyzing interface residues for {pdb_path}")
    
    if use_sasa and FREESASA_AVAILABLE:
        interface_residues = get_interface_residues_with_sasa(
            pdb_path, chain_ids_1, chain_ids_2, distance_cutoff, sasa_cutoff)
        
        # SASA情報付きで結果を表示
        print(f"\nReceptor interface residues (distance < {distance_cutoff}Å, buried SASA > {sasa_cutoff}Å²):")
        for residue in interface_residues['receptor']:
            print(f"  Chain {residue['chain_id']}: {residue['residue_name']}{residue['residue_id']} "
                  f"(buried: {residue['sasa_buried']}Å²)")
        
        print(f"\nLigand interface residues (distance < {distance_cutoff}Å, buried SASA > {sasa_cutoff}Å²):")
        for residue in interface_residues['ligand']:
            print(f"  Chain {residue['chain_id']}: {residue['residue_name']}{residue['residue_id']} "
                  f"(buried: {residue['sasa_buried']}Å²)")
    else:
        interface_residues = get_interface_residues(pdb_path, chain_ids_1, chain_ids_2, distance_cutoff)
        
        # 距離情報のみで結果を表示
        print(f"\nReceptor interface residues (distance < {distance_cutoff}Å):")
        for residue in interface_residues['receptor']:
            print(f"  Chain {residue['chain_id']}: {residue['residue_name']}{residue['residue_id']}")
        
        print(f"\nLigand interface residues (distance < {distance_cutoff}Å):")
        for residue in interface_residues['ligand']:
            print(f"  Chain {residue['chain_id']}: {residue['residue_name']}{residue['residue_id']}")
    
    # 結果を保存
    if save_path:
        save_interface_residues(interface_residues, save_path)
    
    return interface_residues


# 使用例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze and identify interface residues in a PDB file based on distance and/or SASA changes."
    )
    parser.add_argument(
        "pdb_file",
        help="Path to the input PDB file."
    )
    parser.add_argument(
        "--chains1",
        nargs='+',
        default=['A'],
        help="Receptor chain ID(s) (e.g., A B)."
    )
    parser.add_argument(
        "--chains2",
        nargs='+',
        default=['B'],
        help="Ligand chain ID(s) (e.g., C)."
    )
    parser.add_argument(
        "--distance_cutoff",
        type=float,
        default=8.0,
        help="Distance cutoff in Angstroms for defining interface residues."
    )
    parser.add_argument(
        "--sasa_cutoff",
        type=float,
        default=1.0,
        help="Buried SASA cutoff in Angstrom^2 for refining interface residues."
    )
    parser.add_argument(
        "--output_file",
        default="interface_residues.json",
        help="Path to save the final SASA-based interface residues as a JSON file."
    )
    
    args = parser.parse_args()

    # 距離のみ
    print("=== Distance-based interface ===")
    result1 = analyze_interface_residues(
        args.pdb_file, 
        args.chains1, 
        args.chains2, 
        distance_cutoff=args.distance_cutoff,
        use_sasa=False
    )
    
    # 距離 + SASA
    if FREESASA_AVAILABLE:
        print("\n=== Distance + SASA-based interface ===")
        result2 = analyze_interface_residues(
            args.pdb_file, 
            args.chains1, 
            args.chains2, 
            distance_cutoff=args.distance_cutoff, 
            sasa_cutoff=args.sasa_cutoff, 
            use_sasa=True,
            save_path=args.output_file
        )
        
        print(f"\nComparison:")
        print(f"Distance only: {len(result1['receptor'])} receptor, {len(result1['ligand'])} ligand")
        print(f"Distance + SASA: {len(result2['receptor'])} receptor, {len(result2['ligand'])} ligand")
    else:
        print("\nFreeSASA not available, skipping SASA-based analysis.")
        # SASAが使えない場合、距離ベースの結果を保存する
        if args.output_file:
            save_interface_residues(result1, args.output_file)