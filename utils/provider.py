import numpy as np
import random
import torch

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        self.val_loss_min = val_loss


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def random_point_dropout(pc: torch.Tensor, max_dropout_ratio: float = 0.875) -> torch.Tensor:
    """
    pc: [B,N,F]（Fの先頭3がxyzでもOK）
    落とした点は各バッチの先頭点で置換（cloneでメモリ重複回避）
    """
    assert pc.dim() == 3, f"expected [B,N,F], got {pc.shape}"
    B, N, F = pc.shape
    device = pc.device
    drop_ratio = torch.rand(B, device=device) * max_dropout_ratio           # [B]
    keep = (torch.rand(B, N, device=device) > drop_ratio.unsqueeze(1))      # [B,N]
    first = pc[:, :1, :].clone()                                           # [B,1,F]
    return torch.where(keep.unsqueeze(-1), pc, first)


@torch.no_grad()
def shift_point_cloud(pc_xyz: torch.Tensor, shift_range: float = 0.1) -> torch.Tensor:
    """
    pc_xyz: [B,N,3]
    """
    assert pc_xyz.shape[-1] == 3
    B = pc_xyz.shape[0]
    shift = (torch.rand(B, 1, 3, device=pc_xyz.device) - 0.5) * 2 * shift_range
    return pc_xyz + shift


@torch.no_grad()
def random_point_jitter(pc_xyz: torch.Tensor, sigma: float = 0.01, clip: float = 0.05) -> torch.Tensor:
    """
    pc_xyz: [B,N,3]
    """
    noise = torch.clamp(sigma * torch.randn_like(pc_xyz), -clip, clip)
    return pc_xyz + noise


def _rand_rotation_matrices(batch: int, device: torch.device) -> torch.Tensor:
    u1 = torch.rand(batch, device=device)
    u2 = torch.rand(batch, device=device) * 2.0 * torch.pi
    u3 = torch.rand(batch, device=device) * 2.0 * torch.pi
    sqrt1 = torch.sqrt(1.0 - u1); sqrt2 = torch.sqrt(u1)
    q = torch.stack([torch.cos(u2)*sqrt1, torch.sin(u2)*sqrt1,
                     torch.cos(u3)*sqrt2, torch.sin(u3)*sqrt2], dim=-1)
    w,x,y,z = q.unbind(-1)
    R = torch.empty((batch,3,3), device=device)
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-z*w);   R[:,0,2]=2*(x*z+y*w)
    R[:,1,0]=2*(x*y+z*w);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-x*w)
    R[:,2,0]=2*(x*z-y*w);   R[:,2,1]=2*(y*z+x*w);   R[:,2,2]=1-2*(x*x+y*y)
    return R


@torch.no_grad()
def rotate_point_cloud_so3(pc_xyz: torch.Tensor) -> torch.Tensor:
    """
    pc_xyz: [B,N,3] を各サンプル独立にSO(3)一様回転
    """
    assert pc_xyz.shape[-1] == 3
    B,N,_ = pc_xyz.shape
    R = _rand_rotation_matrices(B, pc_xyz.device)         # [B,3,3]
    return torch.bmm(pc_xyz, R.transpose(1,2))            # [B,N,3]