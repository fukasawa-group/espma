import sys
import argparse
import h5py
import numpy as np
import os
import torch
import importlib
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score, confusion_matrix
from torch.utils.data import Dataset
from functools import partial # ★ functools.partial をインポート

# train_piaco2.pyから必要なクラスと関数をインポート
from train_piaco2 import InterfaceDataset, collate_fn, move_esms_to
import utils.provider as provider

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Piaco2 Testing') # ★ 名前を修正
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing [default: 24]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    # ★ モデル名を pc_plm_esm_architecture に変更
    parser.add_argument('--model', default='piaco2_architecture', help='model name [default: pc_plm_esm_architecture]')
    parser.add_argument('--num_point', type=int, default=1000, help='Point Number [default: 1000]')
    
    # データとモデルのパスに関する引数
    parser.add_argument('--data_dir', type=str, default='preprocessed_data', help='experiment root')
    parser.add_argument('--dataset_dir', type=str, default='dc')
    parser.add_argument('--esm_pooling', type=bool, default=True, help='use ESM-2 pooled embeddings')
    parser.add_argument('--esm_crossattn', type=bool, default=True, help='use per-residue ESM-2 cross-attention')
    
    # 読み込むモデルのチェックポイントを指定
    # ★ checkpoint_dir を checkpoint に変更（train.pyと合わせる）
    parser.add_argument('--checkpoint_par_dir', type=str, default='checkpoint', help='Parent directory of checkpoints')
    parser.add_argument('--checkpoint', type=str, default='checkpoint', help='Directory where checkpoints are saved')
    parser.add_argument('--model_name', type=str, default='best_model.pth', help='name of the saved model file [default: best_model.pth]')
    parser.add_argument('--nullify_points', action='store_true', help='Zero out all point features except xyz (dims 3:-2), matching training nullify_points mode')
    
    return parser.parse_args()

def test(args):
    '''HYPER PARAMETER'''
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    print(f"Using device: {device}")
    
    seed = 42
    provider.set_seed(seed)
    print(f"Set random seed ... {seed}")

    '''DATA LOADING'''
    print('Load dataset ...')
    DC_PATH = os.path.join(ROOT_DIR, 'dataset', args.dataset_dir)
    DATA_PATH = os.path.join(ROOT_DIR, 'data', args.data_dir)

    TEST_DATASET = InterfaceDataset(DATA_PATH, DC_PATH, split='test', use_esm_pooling=args.esm_pooling, use_esm_crossattn=args.esm_crossattn)
    
    # ★ partial を使って collate_fn をラップ
    # deterministic=True for stable inference/evaluation
    collate_fn_wrapped = partial(collate_fn, max_pts=args.num_point, lcap=100)

    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn_wrapped)

    '''MODEL LOADING'''
    print('Load model ...')
    MODEL = importlib.import_module(args.model)

    model_path = os.path.join(ROOT_DIR, args.checkpoint_par_dir, args.checkpoint, args.model_name)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        sd = checkpoint['model_state_dict']

        # checkpointからin_channelsとuse_esmを自動検出
        in_channels = sd['encoder.feat_embed.net.0.weight'].shape[1]
        use_esm     = 'encoder.cross_r2l.q_proj.weight' in sd
        #print(f"[DEBUG] Detected from checkpoint: in_channels={in_channels}, use_esm={use_esm}")

        classifier = MODEL.Piaco2(in_channels=in_channels, use_esm=use_esm).to(device)
        classifier.load_state_dict(sd)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    '''TESTING'''
    print('Start testing...')
    
    pred_probs = [] # ★ 確率を保存するように変更
    target_list = []
    
    with torch.no_grad():
        classifier = classifier.eval()
        for j, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            # ★ 新しい collate_fn の返り値に合わせてアンパック
            points, atom_mask, target, plms, esms = data
            
            points = points.to(device)
            target = target.to(device)
            if plms is not None:
                plms = plms.to(device)
            
            # esms をデバイスに送る
            esms = move_esms_to(esms, device)
            
            points = points.transpose(2, 1)
            if getattr(args, 'nullify_points', False):
                # Zero out all features except xyz (dims 0:3); matches training with --nullify_points
                points[:, 3:, :] = 0
            
            # ★ モデルに esms を渡す
            pred_logits = classifier(points, plms, esms)
            
            # ★ シグモイドを適用して確率に変換
            probs = torch.sigmoid(pred_logits)
            
            target_list.extend(target.cpu().numpy())
            pred_probs.extend(probs.cpu().numpy().flatten())

    # 評価指標の計算
    # ★ 確率から0/1の予測を生成
    pred_list = (np.array(pred_probs) > 0.5).astype(int)

    accuracy = accuracy_score(target_list, pred_list)
    # ★ AUCは確率で計算
    auc = roc_auc_score(target_list, pred_probs)
    precision = precision_score(target_list, pred_list)
    recall = recall_score(target_list, pred_list)
    cm = confusion_matrix(target_list, pred_list)

    print('---- Test Results ----')
    print(f'Accuracy:  {accuracy:.6f}')
    print(f'AUC:       {auc:.6f}')
    print(f'Precision: {precision:.6f}')
    print(f'Recall:    {recall:.6f}')
    print('Confusion Matrix:')
    print(cm)
    print('----------------------')

    # Per-entry probability output
    names = [os.path.basename(p) for p in TEST_DATASET._paths]
    print('\n---- Per-entry Probabilities ----')
    print(f'{"Name":<50}  {"Label":>5}  {"Prob":>8}')
    for name, label, prob in zip(names, target_list, pred_probs):
        print(f'{name:<50}  {int(label):>5}  {prob:>8.4f}')

    # 結果を辞書として返す
    return {
        'Accuracy': accuracy,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall,
        'Confusion Matrix': cm,
        'names': names,
        'labels': target_list,
        'probs': pred_probs,
    }

def main(args):
    test(args)

if __name__ == '__main__':
    args = parse_args()
    main(args)