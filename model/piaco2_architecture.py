"""
Protein Interface Aware Complex Oracle 2 (PIACO2)
Hierarchical point cloud encoder for protein-protein interface prediction.

Input point layout (per point):
  dims  0..2  : xyz coordinates  (3d)
  dims  3..   : per-point features passed as `in_channels`
                Default: receptor/ligand identity flags only (2d).
                Combined input tensor shape: [B, 3+in_channels, N]

ESM-2 branch (optional):
  Pooled ESM-2 embeddings (mean, 1280d) are concatenated to the
  structure encoder output before the final classifier.
  Cross-attention from ESM-2 residue tokens into point cloud anchors
  (CrossMHA) is available, as it requires more training data than
  typical interface datasets provide.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def pairwise_sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between every pair of rows in a and b.

    Args:
        a: [B, N, C]
        b: [B, M, C]
    Returns:
        dist: [B, N, M]  where dist[b,i,j] = ||a[b,i] - b[b,j]||^2
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b^T
    a_sq = a.pow(2).sum(dim=-1, keepdim=True)          # [B, N, 1]
    b_sq = b.pow(2).sum(dim=-1, keepdim=True)          # [B, M, 1]
    ab   = torch.bmm(a, b.transpose(1, 2))              # [B, N, M]
    dist = a_sq + b_sq.transpose(1, 2) - 2.0 * ab
    return dist.clamp(min=0.0)


def gather_by_index(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather rows from `points` at positions given by `idx`.

    Args:
        points: [B, N, C]
        idx:    [B, S]  (or [B, S, K] for neighbourhood gathering)
    Returns:
        gathered: [B, S, C]  (or [B, S, K, C])
    """
    B = points.shape[0]
    flat_idx    = idx.reshape(B, -1)                    # [B, S*K]
    row_ids     = torch.arange(B, device=points.device).view(B, 1).expand_as(flat_idx)
    gathered    = points[row_ids, flat_idx]             # [B, S*K, C]
    return gathered.reshape(*idx.shape, points.shape[-1])


def knn_query(k: int, ref_xyz: torch.Tensor, query_xyz: torch.Tensor) -> torch.Tensor:
    """Return indices of the k nearest neighbours in ref_xyz for each query.

    Args:
        k:         number of neighbours
        ref_xyz:   [B, N, 3]
        query_xyz: [B, S, 3]
    Returns:
        idx: [B, S, k]
    """
    dists = pairwise_sq_dist(query_xyz, ref_xyz)        # [B, S, N]
    _, idx = torch.topk(dists, k, dim=-1, largest=False, sorted=False)
    return idx


def fps(xyz: torch.Tensor, n_samples: int, deterministic: bool = False) -> torch.Tensor:
    """Farthest point sampling.

    Args:
        xyz:          [B, N, 3]
        n_samples:    number of points to select
        deterministic: if True, start from the point farthest from the origin
    Returns:
        sampled_idx: [B, n_samples]  long tensor
    """
    B, N, _ = xyz.shape
    device   = xyz.device

    selected  = torch.zeros(B, n_samples, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), fill_value=1e10, device=device)

    if deterministic:
        cur = xyz.pow(2).sum(-1).argmax(dim=1)          # point farthest from origin
    else:
        cur = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for s in range(n_samples):
        selected[:, s] = cur
        centroid  = xyz[batch_idx, cur, :].unsqueeze(1)              # [B, 1, 3]
        sq_d      = (xyz - centroid).pow(2).sum(-1)                  # [B, N]
        min_dists = torch.minimum(min_dists, sq_d)
        cur       = min_dists.argmax(dim=1)
    return selected


# ---------------------------------------------------------------------------
# Cross-attention: ESM-2 residues → point-cloud anchors  (optional)
# ---------------------------------------------------------------------------

class StructureSeqCrossAttn(nn.Module):
    """Cross-attention from per-residue ESM-2 tokens into 3-D point anchors.

    Query  = 3-D anchor features  [B, G, d]
    Key/V  = ESM-2 token features [B, L, 1280]

    A distance-based attention bias encourages spatially nearby residues
    to contribute more (bias = -dist / tau).

    Note: this module may be instantiated even when ESM is disabled so that
    model initialization matches legacy architectures; it is only *used* when
    ESM tensors are provided and `use_esm=True`.
    """

    def __init__(
        self,
        d_model:  int,
        plm_dim:  int   = 1280,
        n_heads:  int   = 7,
        dropout:  float = 0.1,
        tau:      float = 3.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.tau     = tau
        self.scale   = math.sqrt(self.d_head)

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(plm_dim,  d_model, bias=False)
        self.v_proj   = nn.Linear(plm_dim,  d_model, bias=False)
        self.out_proj = nn.Linear(d_model,  d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop  = nn.Dropout(dropout)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B, S, D] → [B, H, S, D/H]"""
        B, S, _ = t.shape
        return t.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        anchor_xyz:  torch.Tensor,
        anchor_feat: torch.Tensor,
        seq_xyz:     torch.Tensor,
        seq_esm:     torch.Tensor,
        seq_mask:    torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        anchor_xyz:  [B, G, 3]
        anchor_feat: [B, G, d]
        seq_xyz:     [B, L, 3]
        seq_esm:     [B, L, 1280]
        seq_mask:    [B, L]  True = valid token, False = padding
        Returns:
            ctx: [B, G, d]
        """
        B, G, _ = anchor_xyz.shape

        Q = self._split_heads(self.q_proj(anchor_feat))   # [B, H, G, d_h]
        K = self._split_heads(self.k_proj(seq_esm))       # [B, H, L, d_h]
        V = self._split_heads(self.v_proj(seq_esm))       # [B, H, L, d_h]

        # scaled dot-product logits
        logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, G, L]

        # spatial proximity bias
        dist_bias = -torch.cdist(anchor_xyz, seq_xyz) / self.tau     # [B, G, L]
        logits = logits + dist_bias.unsqueeze(1)

        # mask padding tokens
        if seq_mask is not None:
            pad = ~seq_mask.bool()                                    # [B, L]  True = pad
            logits = logits.masked_fill(pad[:, None, None, :], float('-inf'))

        attn = F.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)

        ctx = torch.matmul(attn, V)                                  # [B, H, G, d_h]
        ctx = ctx.transpose(1, 2).contiguous().view(B, G, self.d_model)
        ctx = self.out_proj(ctx)
        ctx = self.out_drop(ctx)
        return ctx


# ---------------------------------------------------------------------------
# Point-cloud building blocks
# ---------------------------------------------------------------------------

class PointEmbed(nn.Module):
    """1-D convolution → BN → ReLU  (operates on [B, C_in, N] tensors)."""

    def __init__(self, in_ch: int, out_ch: int, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FeatureTransform(nn.Module):
    """Bottleneck residual block on [B, C, G, K] neighbourhood tensors."""

    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        mid = channels // 2
        self.down = nn.Sequential(
            nn.Conv2d(channels, mid,      kernel_size=1, bias=bias),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Conv2d(mid,      channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.up(self.down(x)) + x)


# ---------------------------------------------------------------------------
# Gaussian radial-basis position encoding
# ---------------------------------------------------------------------------

class GaussianPosEnc(nn.Module):
    """Weight neighbourhood features by a Gaussian of the relative distance.

    dist_ij = ||knn_xyz[:,:,j,:] - lc_xyz[:,i,:]||
    w_ij    = exp(-dist_ij / (2 * sigma^2))
    output  = knn_feat * w  (broadcast over channel dim)
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def forward(
        self,
        knn_xyz:  torch.Tensor,
        knn_feat: torch.Tensor,
        lc_xyz:   torch.Tensor,
    ) -> torch.Tensor:
        """
        knn_xyz:  [B, 3, G, K]
        knn_feat: [B, C, G, K]
        lc_xyz:   [B, G, 3]
        Returns:
            weighted: [B, C, G, K]
        """
        B, _, G, K = knn_xyz.shape
        center = lc_xyz.permute(0, 2, 1).unsqueeze(-1)             # [B, 3, G, 1]
        rel    = knn_xyz - center.expand_as(knn_xyz)               # [B, 3, G, K]
        dist   = rel.norm(dim=1)                                    # [B, G, K]
        w      = torch.exp(-dist / (2.0 * self.sigma ** 2))        # [B, G, K]
        return knn_feat * w.unsqueeze(1)


# ---------------------------------------------------------------------------
# Sampling + grouping modules
# ---------------------------------------------------------------------------

class InterfaceGroupModule(nn.Module):
    """Stage-0 grouping: receptor anchors kNN into ligand cloud and vice versa.

    Points are assumed to be concatenated as [receptor | ligand] along dim-N.
    After FPS each half independently, cross-kNN groups the opposite chain,
    enabling the encoder to see interface contacts from the very first stage.
    """

    def __init__(self, group_num: int, k: int):
        super().__init__()
        self.group_num = group_num
        self.k         = k

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        """
        xyz:  [B, N, 3]
        feat: [B, N, d]
        Returns: lc_xyz, lc_feat, knn_xyz, knn_feat  (same shapes as FpsKnn)
        """
        B, N, _ = xyz.shape
        half     = N // 2

        xyz_r, xyz_l   = xyz[:, :half],  xyz[:, half:]
        feat_r, feat_l = feat[:, :half], feat[:, half:]
        N_r, N_l       = half, N - half

        r_num = min(self.group_num // 2, max(1, N_r))
        l_num = min(self.group_num // 2, max(1, N_l))

        det = not self.training
        idx_r = fps(xyz_r, r_num, deterministic=det)   # [B, r_num]
        idx_l = fps(xyz_l, l_num, deterministic=det)   # [B, l_num]

        new_xyz_r  = gather_by_index(xyz_r,  idx_r)   # [B, r_num, 3]
        new_feat_r = gather_by_index(feat_r, idx_r)   # [B, r_num, d]
        new_xyz_l  = gather_by_index(xyz_l,  idx_l)
        new_feat_l = gather_by_index(feat_l, idx_l)

        assert feat_r.shape[1] == xyz_r.shape[1]
        assert idx_r.min() >= 0 and idx_r.max() < N_r

        lc_xyz  = torch.cat([new_xyz_r,  new_xyz_l],  dim=1)   # [B, r+l, 3]
        lc_feat = torch.cat([new_feat_r, new_feat_l], dim=1)   # [B, r+l, d]

        # cross-kNN: receptor anchors → ligand neighbours, and vice versa
        k_rl = min(self.k, N_l)
        k_lr = min(self.k, N_r)

        nn_idx_rl  = knn_query(self.k, xyz_l,  new_xyz_r)      # [B, r_num, k]
        grp_xyz_r  = gather_by_index(xyz_l,  nn_idx_rl)        # [B, r_num, k, 3]
        grp_feat_r = gather_by_index(feat_l, nn_idx_rl)        # [B, r_num, k, d]

        nn_idx_lr  = knn_query(self.k, xyz_r,  new_xyz_l)
        grp_xyz_l  = gather_by_index(xyz_r,  nn_idx_lr)
        grp_feat_l = gather_by_index(feat_r, nn_idx_lr)

        # append xyz to features
        grp_feat_r = torch.cat([grp_feat_r, grp_xyz_r],  dim=-1)
        grp_feat_l = torch.cat([grp_feat_l, grp_xyz_l],  dim=-1)

        knn_xyz  = torch.cat([grp_xyz_r,  grp_xyz_l],  dim=1)  # [B, r+l, k, 3]
        knn_feat = torch.cat([grp_feat_r, grp_feat_l], dim=1)  # [B, r+l, k, d+3]
        return lc_xyz, lc_feat, knn_xyz, knn_feat


class FpsKnnGroup(nn.Module):
    """Stages 1+: standard FPS + kNN grouping within a single point cloud."""

    def __init__(self, group_num: int, k: int):
        super().__init__()
        self.group_num = group_num
        self.k         = k

    def forward(self, xyz: torch.Tensor, feat: torch.Tensor):
        B, N, _ = xyz.shape

        det     = not self.training
        idx     = fps(xyz, self.group_num, deterministic=det).long()
        lc_xyz  = gather_by_index(xyz,  idx)   # [B, G, 3]
        lc_feat = gather_by_index(feat, idx)   # [B, G, d]

        nn_idx   = knn_query(self.k, xyz, lc_xyz)             # [B, G, k]
        knn_xyz  = gather_by_index(xyz,  nn_idx)              # [B, G, k, 3]
        knn_feat = gather_by_index(feat, nn_idx)              # [B, G, k, d]
        knn_feat = torch.cat([knn_feat, knn_xyz], dim=-1)     # [B, G, k, d+3]
        return lc_xyz, lc_feat, knn_xyz, knn_feat


# ---------------------------------------------------------------------------
# Local geometry aggregation
# ---------------------------------------------------------------------------

class LocalGeoAgg(nn.Module):
    """Aggregate neighbourhood features with Gaussian positional weighting.

    Followed by feature expansion (concat anchor) and residual bottleneck
    blocks.
    """

    def __init__(
        self,
        out_dim:       int,
        alpha:         float,
        beta:          float,
        n_blocks:      int,
        dim_expansion: int,
        norm_type:     str = "mn40",
    ):
        super().__init__()
        self.norm_type = norm_type
        self.pos_enc   = GaussianPosEnc(sigma=1.0)

        # knn_feat from groupers already has xyz appended: shape (prev_dim + 3).
        # In forward, anchor features (prev_dim) are also concatenated → 2*prev_dim + 3.
        prev_dim = out_dim // dim_expansion
        in_ch    = 2 * prev_dim + 3

        self.linear1 = PointEmbed(in_ch, out_dim, bias=False)
        self.blocks  = nn.Sequential(*[FeatureTransform(out_dim) for _ in range(n_blocks)])

    def forward(
        self,
        lc_xyz:   torch.Tensor,
        lc_feat:  torch.Tensor,
        knn_xyz:  torch.Tensor,
        knn_feat: torch.Tensor,
    ) -> torch.Tensor:
        B, G, K, _ = knn_feat.shape

        # local coordinate normalisation
        if self.norm_type == "mn40":
            anchor_xyz = lc_xyz.unsqueeze(2)                         # [B, G, 1, 3]
            std        = (knn_xyz - anchor_xyz).std() + 1e-5
            knn_xyz    = (knn_xyz - anchor_xyz) / std

        # feature expansion: concat each neighbour with its anchor
        expanded = torch.cat(
            [knn_feat, lc_feat.unsqueeze(2).expand(-1, -1, K, -1)],
            dim=-1,
        )                                                             # [B, G, K, C_exp]

        # reshape to [B, C, G*K] for Conv1d, then back
        C_exp = expanded.shape[-1]
        expanded = expanded.permute(0, 3, 1, 2)                      # [B, C_exp, G, K]
        expanded = self.linear1(expanded.reshape(B, C_exp, G * K)).reshape(B, -1, G, K)

        # positional encoding
        knn_xyz_perm = knn_xyz.permute(0, 3, 1, 2)                   # [B, 3, G, K]
        weighted     = self.pos_enc(knn_xyz_perm, expanded, lc_xyz)  # [B, C, G, K]

        # residual blocks
        return self.blocks(weighted)


class MaxMeanPool(nn.Module):
    """Global pooling by element-wise sum of max and mean over the K dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, G, K]
        return x.max(dim=-1).values + x.mean(dim=-1)                 # [B, C, G]


# ---------------------------------------------------------------------------
# Hierarchical point-cloud encoder
# ---------------------------------------------------------------------------

class PointCloudEncoder(nn.Module):
    """Multi-stage hierarchical encoder for protein interface point clouds.

    Input:
        x: [B, 3 + in_channels, N]
           first 3 dims are xyz; remaining dims are per-point features.
           Default `in_channels=2` corresponds to receptor/ligand identity flags.

    Output:
        [B, structure_dim]  where structure_dim = embed_dim * prod(dim_expansion)
    """

    def __init__(
        self,
        in_channels:   int,
        input_points:  int,
        num_stages:    int,
        embed_dim:     int,
        k_neighbors:   int,
        alpha:         float,
        beta:          float,
        lga_blocks:    list[int],
        dim_expansion: list[int],
        norm_type:     str  = "mn40",
        use_esm:       bool = False,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.embed_dim  = embed_dim
        self.use_esm    = use_esm

        # project per-point features → embed_dim
        assert in_channels > 0
        self.feat_embed = PointEmbed(in_channels, embed_dim, bias=False)

        # Cross-attention modules (ESM-2 → point anchors).
        # Instantiated unconditionally to keep parameter initialization/RNG
        # consumption aligned with other model definitions; gated at runtime.
        self.cross_r2l = StructureSeqCrossAttn(d_model=embed_dim, plm_dim=1280, n_heads=7)
        self.cross_l2r = StructureSeqCrossAttn(d_model=embed_dim, plm_dim=1280, n_heads=7)
        self.ctx_gate  = nn.Linear(embed_dim, embed_dim, bias=False)

        self.groups   = nn.ModuleList()
        self.lga      = nn.ModuleList()
        self.pool     = nn.ModuleList()

        out_dim   = embed_dim
        grp_num   = input_points
        for i in range(num_stages):
            out_dim  = out_dim  * dim_expansion[i]
            grp_num  = grp_num  // 2
            grouper  = InterfaceGroupModule(grp_num, k_neighbors) if i == 0 \
                       else FpsKnnGroup(grp_num, k_neighbors)
            self.groups.append(grouper)
            self.lga.append(
                LocalGeoAgg(out_dim, alpha, beta, lga_blocks[i], dim_expansion[i], norm_type)
            )
            self.pool.append(MaxMeanPool())

    def _valid_esm(self, esms) -> bool:
        if not isinstance(esms, dict):
            return False
        required = ("xyz_r", "xyz_l", "esm_r", "esm_l")
        if not all(k in esms for k in required):
            return False
        xr, xl, er, el = (esms[k] for k in required)
        if not all(torch.is_tensor(t) and t.ndim == 3 for t in (xr, xl, er, el)):
            return False
        if xr.size(-1) != 3 or xl.size(-1) != 3:
            return False
        if er.size(-1) != 1280 or el.size(-1) != 1280:
            return False
        mr, ml = esms.get("mask_r"), esms.get("mask_l")
        if mr is not None and ml is not None:
            if torch.is_tensor(mr) and torch.is_tensor(ml):
                if mr.sum() == 0 and ml.sum() == 0:
                    return False
        return True

    def forward(self, x: torch.Tensor, esms=None) -> torch.Tensor:
        """
        x:    [B, 3+in_channels, N]
        esms: optional dict with keys xyz_r, xyz_l, esm_r, esm_l [, mask_r, mask_l]
        """
        # split coordinates and features
        x_t   = x.transpose(1, 2).contiguous()           # [B, N, 3+F]
        xyz   = x_t[:, :, :3].contiguous()               # [B, N, 3]
        raw_f = x_t[:, :, 3:].contiguous()               # [B, N, F]

        # embed features: [B, F, N] → [B, embed_dim, N] → [B, N, embed_dim]
        feat = self.feat_embed(raw_f.transpose(1, 2)).transpose(1, 2).contiguous()

        use_cross = self.use_esm and self._valid_esm(esms)

        for i in range(self.num_stages):
            xyz, lc_feat, knn_xyz, knn_feat = self.groups[i](xyz, feat)

            # optional cross-attention at stage 0
            if i == 0 and use_cross:
                G     = lc_feat.size(1)
                half  = G // 2
                ctx_r = self.cross_r2l(
                    xyz[:, :half],
                    lc_feat[:, :half],
                    esms["xyz_l"], esms["esm_l"],
                    esms.get("mask_l"),
                )
                ctx_l = self.cross_l2r(
                    xyz[:, half:],
                    lc_feat[:, half:],
                    esms["xyz_r"], esms["esm_r"],
                    esms.get("mask_r"),
                )
                ctx      = torch.cat([ctx_r, ctx_l], dim=1)  # [B, G, d]
                lc_feat  = lc_feat + self.ctx_gate(ctx)

            knn_w = self.lga[i](xyz, lc_feat, knn_xyz, knn_feat)
            pooled = self.pool[i](knn_w)                     # [B, C, G]
            feat   = pooled.transpose(1, 2).contiguous()     # [B, G, C]

        # global max+mean pooling over remaining points
        return feat.max(dim=1).values + feat.mean(dim=1)     # [B, structure_dim]


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class Piaco2(nn.Module):
    """Protein Interface Aware Complex Oracle 2.

    Combines a hierarchical point-cloud encoder with optional ESM-2
    mean+max pooled sequence embeddings for protein-protein interface
    binding prediction.

    Args:
        in_channels:   per-point feature channels EXCLUDING xyz.
                       Default 2 = receptor/ligand identity flags.
        class_num:     output dimension (1 for binary BCEWithLogits).
        input_points:  number of input interface points N.
        num_stages:    number of hierarchical encoding stages.
        embed_dim:     base channel width.
        k_neighbors:   neighbourhood size for kNN grouping.
        alpha, beta:   positional encoding hyperparameters (kept for compat).
        lga_blocks:    residual blocks per stage.
        dim_expansion: channel multiplier per stage.
        norm_type:     coordinate normalisation scheme ("mn40" or "scan").
        use_esm:       enable cross-attention usage at runtime.
    """

    def __init__(
        self,
        in_channels:   int        = 2,
        class_num:     int        = 1,
        input_points:  int        = 1000,
        num_stages:    int        = 4,
        embed_dim:     int        = 49,
        k_neighbors:   int        = 40,
        beta:          float      = 100.0,
        alpha:         float      = 1000.0,
        lga_blocks:    list[int]  = None,
        dim_expansion: list[int]  = None,
        norm_type:     str        = "mn40",
        use_esm:       bool       = False,
    ):
        super().__init__()
        if lga_blocks    is None: lga_blocks    = [2, 1, 1, 1]
        if dim_expansion is None: dim_expansion = [2, 2, 2, 1]

        self.encoder = PointCloudEncoder(
            in_channels   = in_channels,
            input_points  = input_points,
            num_stages    = num_stages,
            embed_dim     = embed_dim,
            k_neighbors   = k_neighbors,
            alpha         = alpha,
            beta          = beta,
            lga_blocks    = lga_blocks,
            dim_expansion = dim_expansion,
            norm_type     = norm_type,
            use_esm       = use_esm,
        )

        # structure_dim = embed_dim × prod(dim_expansion)
        structure_dim = embed_dim
        for m in dim_expansion:
            structure_dim *= m

        # ── classifier: structure only ────────────────────────────────────
        self.clf_struct = self._mlp(structure_dim, class_num)

        # ── classifier: structure + ESM-2 mean+max (2×1280 = 2560) ──────
        plm_dim = 2560
        self.clf_combined = self._mlp(structure_dim + plm_dim, class_num)

    @staticmethod
    def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, out_dim),
        )

    def forward(
        self,
        x:    torch.Tensor,
        plm:  torch.Tensor | None = None,
        esms: dict        | None = None,
    ) -> torch.Tensor:
        """
        x:   [B, 3 + in_channels, N]
        plm: [B, 2560]  optional ESM-2 mean+max pooled features
        esms: optional cross-attention dict (only used when use_esm=True)
        Returns:
            logits: [B]  (use with BCEWithLogitsLoss)
        """
        struct_feat = self.encoder(x, esms=esms)

        if plm is not None:
            logits = self.clf_combined(torch.cat([struct_feat, plm], dim=1))
        else:
            logits = self.clf_struct(struct_feat)

        return logits.view(-1)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N   = 4, 1000

    # xyz (3) + R/L flags (2)  →  5 channels total
    xyz    = torch.randn(B, 3, N)
    flags  = torch.zeros(B, 2, N)
    flags[:, 0, :N//2] = 1.0   # receptor
    flags[:, 1, N//2:] = 1.0   # ligand
    pts    = torch.cat([xyz, flags], dim=1).to(device)  # [B, 5, N]

    plm    = torch.randn(B, 2560).to(device)

    model  = Piaco2(in_channels=2).to(device)
    n_par  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_par / 1e6:.3f} M")

    logits = model(pts, plm)
    print(f"Output shape: {logits.shape}  (expected [{B}])")
