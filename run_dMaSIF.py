import sys
import os
import torch
from dMaSIF.model import dMaSIF
from dMaSIF.geometry_processing import atoms_to_points_normals
from Bio.PDB import PDBParser, Structure, Model, Chain, Residue
from Bio.PDB import Atom, PDBIO
import numpy as np
import pathlib

def write_pts_as_pdb(pts, out_pdb="surface_points.pdb", chunk=999):
    """
    pts   : (N,3) Tensor / ndarray
    chunk : 1 Residue に入れる点数上限 (<=9999 が安全)
    """
    if torch.is_tensor(pts):
        pts = pts.cpu().numpy()

    struct = Structure.Structure("surf")
    model  = Model.Model(0)
    chain  = Chain.Chain("A")

    resseq = 1          # Residue number
    atom_id = 1         # PDB atom serial

    for i, xyz in enumerate(pts):
        # ========== Residue 切り替え ==========
        if (i % chunk) == 0:       # 新しい Residue
            residue = Residue.Residue((' ', resseq, ' '), 'PTS', '')
            chain.add(residue)
            resseq += 1

        # ========== 一意な 4 文字の原子名 ==========
        # 例: H000, H001 ... H9998
        local_idx  = i % chunk     # 0–chunk-1
        atom_name  = f"H{local_idx:03d}"[:4]   # 念のため 4 文字制限

        atom = Atom.Atom(
            name        = atom_name,
            coord       = xyz.astype('f4'),
            bfactor     = 0.0,
            occupancy   = 1.0,
            altloc      = ' ',
            fullname    = f"{atom_name:<4}",
            serial_number = atom_id,
            element     = 'H'
        )
        residue.add(atom)
        atom_id += 1

    model.add(chain)
    struct.add(model)

    io = PDBIO()
    io.set_structure(struct)
    io.save(out_pdb)
    print(f"Wrote {out_pdb}  (N = {len(pts)} points, {resseq-1} residues)")

def save_chain_npy(pdbid, sid, pts_xyz, feats16, out_dir="chains_npy"):
    """
    pts_xyz  : (N,3) tensor/ndarray
    feats16  : (N,16)
    保存先   : out_dir/chain_<ID>.npy
    """
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(exist_ok=True)
    combo = torch.cat([pts_xyz, feats16], dim=1).cpu().numpy()   # (N,19)
    np.save(out_dir / f"{pdbid}_chain_{sid}.npy", combo)
    print(f"[saved] {pdbid}_chain_{sid}.npy  shape={combo.shape}")

# -----------------------
# prep
sys.path.append(os.path.abspath("."))
resolution   = "0.7"
radius       = "9"
sup_sampling = "150"
device       = "cuda:0" if torch.cuda.is_available() else "cpu"
seed         = "42"

# -----------------------
# 0. PDB から原子座標を取り出す簡易ヘルパ
# -----------------------
def load_chain_atoms(pdb_file, chain_id="A"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)
    coords, types = [], []
    ele2num = {"C":0,"H":1,"O":2,"N":3,"S":4,"SE":5}
    for atom in structure[0][sid].get_atoms():
        if atom.element not in ele2num:      # 不明元素はスキップ
            continue
        coords.append(atom.coord)
        t = np.zeros(6); t[ele2num[atom.element]] = 1.0
        types.append(t)
    return torch.tensor(coords, dtype=torch.float32), \
           torch.tensor(types , dtype=torch.float32)

# -----------------------
# 1. ネットワークをロード
# -----------------------
from Arguments import parser        # dMaSIF の純正 ArgumentParser
parser.add_argument(
    "--pdb", type=str, help="Name of PDB", required=True
)
parser.add_argument(
    "--outdir", type=str, help="Directory to save the feature vectors", required=True
)
parser.add_argument(
    "--pcout", type=str, help="output of point cloud coords for debugging.", required=False
)

# コマンドライン引数で指定しなかった場合のデフォルト値をセット
parser.set_defaults(
    resolution=resolution,
    radius=radius,
    sup_sampling=sup_sampling,
    device=device,
    seed=seed,
    in_channels=16,
    emb_dims=16,
    n_layers=3,
    use_mesh=False,
)
args = parser.parse_args([
    "--experiment_name", "dMaSIF_site_3layer_16dims_9A_0.7res_150sup_epoch85",
    "--site", "True"
] + sys.argv[1:])  # コマンドライン引数を追加

model_path = os.path.join("dMaSIF", "models", args.experiment_name)
net = dMaSIF(args)
net.load_state_dict(torch.load(model_path, map_location=args.device)["model_state_dict"])
#net = net.to(args.device)
dev = torch.device(args.device)        # or args.device
net = net.to(dev).eval()

# -----------------------
# 2. PDBファイル内の全チェインを順に処理
# -----------------------
# PDBパーサーで構造を読み込み、チェインIDのリストを取得
parser = PDBParser(QUIET=True)
structure = parser.get_structure("prot", args.pdb)
model = structure[0]
all_chain_ids = [chain.id for chain in model]
print(f"Found chains in {args.pdb}: {all_chain_ids}")

# PDBヘッダーからPDB IDを取得します。存在しない場合はファイル名をフォールバックとして使用します。
pdbid = structure.header.get("idcode")
if not pdbid:
    pdbid = os.path.splitext(os.path.basename(args.pdb))[0]

# 各チェインについてループ処理
for sid in all_chain_ids:
    print(f"\n--- Processing chain: {sid} ---")
    # -----------------------
    # PDB → 表面点・法線
    # -----------------------
    try:
        atoms_xyz, atom_types = load_chain_atoms(args.pdb, chain_id=sid)
        if len(atoms_xyz) == 0:
            print(f"Skipping chain {sid}: No valid atoms found.")
            continue
    except KeyError:
        print(f"Chain {sid} not found in PDB model, skipping.")
        continue

    atoms_xyz  = atoms_xyz.to(dev)          # ★ GPU へ
    atom_types = atom_types.to(dev)         # ★ GPU へ
    batch_atoms = torch.zeros(len(atoms_xyz), dtype=torch.long, device=dev)

    pts, normals, batch_pts = atoms_to_points_normals(
            atoms       = atoms_xyz,
            batch       = batch_atoms,
            atomtypes   = atom_types,
            distance    = args.distance,    # 1.05 by default
            resolution  = args.resolution,   # 0.7 A by default
            sup_sampling= args.sup_sampling  # 150 by default
    )

    # -----------------------
    # 3. dMaSIF.features() に渡す dict を組み立て
    # -----------------------
    P = dict(
        xyz         = pts,
        normals     = normals,
        batch       = batch_pts,
        atoms       = atoms_xyz,
        batch_atoms = batch_atoms,
        atom_xyz    = atoms_xyz,             # 同じ
        atomtypes   = atom_types,
        triangles   = None,                  # use_mesh=False
    )

    # -----------------------
    # 4. N×16 の特徴テンソルを取得
    # -----------------------
    with torch.no_grad():
        feats = net.features(P)              # shape = (N_points, 16)

    """
    #print(feats.shape)   # 例: torch.Size([2400, 16])
    #print(pts.shape)

    # Concatenate x,y,z  coordinates with features
    # pts.shape = (N_points, 3), feats.shape = (N_points, 16)
    # combo.shape = (N_points, 19)
    combo = torch.cat([pts, feats], dim=1)
    print(combo.shape)

    np_combo = combo.cpu().numpy()              # GPU → CPU, Torch → NumPy
    header = ",".join(
        ["x","y","z"] + [f"feat{d}" for d in range(feats.shape[1])]
    )                                           # x,y,z,feat0..feat15

    np.savetxt("surface_points_19d.csv",
               np_combo,
               delimiter=",",
               header=header,
               comments="")
    """

    save_chain_npy(pdbid, sid, pts, feats, out_dir=args.outdir)

    if (args.pcout is not None):
        # チェインごとにファイルが上書きされないよう、ファイル名にチェインIDを追加
        pcout_base, pcout_ext = os.path.splitext(args.pcout)
        out_pdb_filename = f"{pcout_base}_{sid}{pcout_ext}"
        write_pts_as_pdb(pts, out_pdb=out_pdb_filename)
