from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


#  hybrid_torso.py
import torch, torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn  import SAGEConv 

from class_gnn import TorsoGCNv1, TorsoGCNv2, TorsoGCNv3, TorsoGATv1, HybridGnnTorsoATT, HybridGnnTorsoLine


# ---------------------------------------------------------------------
def tensor_to_graph_fast(tensor: torch.Tensor, moves_left: float):
    """
    tensor      : (S,S,S) valori in {-1,0,1}
    moves_left  : scalare (float)
    ritorna     : torch_geometric.data.Data con
                  x          (N,5)  [i,j,k,val, moves_left]
                  edge_index (2,E)
                  edge_type  (E,)   0=same-i , 1=same-j , 2=same-k
    """
    S = tensor.size(0)
    device = tensor.device

    nz = tensor.nonzero(as_tuple=False)          # (M,3) celle ≠0
    if nz.numel() == 0:                          # caso tutto zero
        nz = torch.zeros((1,3), dtype=torch.long, device=device)

    i,j,k = nz.T
    N = nz.size(0)

    # nodal features ---------------------------------------------------
    coord = torch.stack([ i/(S-1), j/(S-1), k/(S-1) ], dim=1).float()
    value = tensor[i,j,k].unsqueeze(1).float() / 2.0
    moves = torch.full((N,1), moves_left / S, device=device)
    x = torch.cat([coord, value, moves], dim=1)       # (N,5)

    # edges ------------------------------------------------------------
    mask_i = (i.unsqueeze(0) == i.unsqueeze(1))
    mask_j = (j.unsqueeze(0) == j.unsqueeze(1))
    mask_k = (k.unsqueeze(0) == k.unsqueeze(1))

    def pairs(mask, etype):
        idx = mask.triu(diagonal=1).nonzero(as_tuple=False)
        if idx.numel()==0:
            return (torch.empty((2,0),device=device,dtype=torch.long),
                    torch.empty((0,),device=device,dtype=torch.long))
        src,dst = idx.T
        eid = torch.full((src.numel()*2,), etype, dtype=torch.long, device=device)
        edge = torch.stack([ torch.cat([src,dst]),
                             torch.cat([dst,src]) ], dim=0)
        return edge, eid

    e1,t1 = pairs(mask_i,0)
    e2,t2 = pairs(mask_j,1)
    e3,t3 = pairs(mask_k,2)

    edge_index = torch.cat([e1,e2,e3], dim=1) if e1.numel()+e2.numel()+e3.numel() else e1
    edge_type  = torch.cat([t1,t2,t3])          if t1.numel()+t2.numel()+t3.numel() else t1
    lin_idx = (i * S + j) * S + k          # (N,)  int64  0‥63
    data = Data(x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                lin_idx=lin_idx)  
    return data

# ---------------------------------------------------------------------
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from torch import nn

class DynamicGNN(nn.Module):
    """R-GCN su frame-0, usa direttamente edge_type (0/1/2)."""
    def __init__(self, in_dim=5, out_dim=32, num_layers=3):
        super().__init__()
        self.feat2c = nn.Linear(in_dim, out_dim)
        self.convs  = nn.ModuleList(
            RGCNConv(out_dim, out_dim, num_relations=3)
            for _ in range(num_layers)
        )
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        x = self.feat2c(batch.x)
        for conv in self.convs:
            x = self.ln(F.relu(conv(x, batch.edge_index, batch.edge_type)))
        batch.x = x
        return batch


# ---------------------------------------------------------------------
class HybridGnnTorso(nn.Module):
    """
    • GNN dinamica sul frame-0 (48 nodi dopo un-pool)
    • MLP sulle azioni passate (T-1)  → 1 vettore
    • Linear sullo scalare moves_left → 1 vettore
    Output: (B,50,dim_c)
    """
    #dim_c=32,
    def __init__(self, S=4, T=8, dim_c=16, gnn_layers=3):
        super().__init__()
        self.S, self.T, self.C = S, T, dim_c
        self.gnn = DynamicGNN(in_dim=5, out_dim=dim_c, num_layers=gnn_layers)
        self.act_mlp = nn.Sequential(
            nn.Linear(S**3, dim_c), nn.ReLU(), nn.Linear(dim_c, dim_c)
        )
        self.scalar_fc = nn.Linear(1, dim_c)

    # ---------------------------------------------------------------
    def _pool_48(self, H0):
        """da (S,S,S,C) estrae 16 share-i + 16 share-j + 16 share-k"""
        A = H0.mean(0).reshape(-1, self.C)   # share-i
        B = H0.mean(1).reshape(-1, self.C)   # share-j
        C = H0.mean(2).reshape(-1, self.C)   # share-k
        return torch.cat([A,B,C], dim=0)     # (48,C)

    # ---------------------------------------------------------------
    def forward(self, xx: torch.Tensor, ss: torch.Tensor):
        """
        xx : (B,T,S,S,S)
        ss : (B,1)
        """
        B, dev = xx.size(0), xx.device
        graphs=[]
        for b in range(B):
            g = tensor_to_graph_fast(xx[b,0], ss[b].item())
            graphs.append(g)
        batch_graph = Batch.from_data_list(graphs)
        data_out = self.gnn(batch_graph)

        # split batch → pool 48
        ptr = batch_graph.ptr
        
        gnn_emb = []
        ptr = batch_graph.ptr
        for b in range(B):
            h  = data_out.x[ptr[b]:ptr[b+1]]              # (N_b,C)
            idx = batch_graph.lin_idx[ptr[b]:ptr[b+1]]    # (N_b,)
            full = torch.zeros(self.S**3, self.C, device=dev)
            full[idx] = h                                 # nessun out-of-bound
            H0 = full.view(self.S, self.S, self.S, self.C)
            gnn_emb.append(self._pool_48(H0))
        gnn_emb = torch.stack(gnn_emb, dim=0)              # (B,48,C)


        # azioni passate
        acts = xx[:,1:].reshape(B, -1, self.S**3).float()           # (B,T-1,64)
        act_emb = self.act_mlp(acts).mean(1)                # (B,C)

        # scalare
        mv_emb = torch.relu(self.scalar_fc(ss)).squeeze(1)  # (B,C)

        ee = torch.cat([gnn_emb,
                        act_emb.unsqueeze(1),
                        mv_emb.unsqueeze(1)], dim=1)        # (B,50,C)
        return ee


from torch_geometric.data import Data, Batch
from torch_geometric.nn   import SAGEConv
import torch, torch.nn.functional as F
from torch import nn, Tensor

# ===============================================================
#  tensor  ➜  PyG Data  (single‐relation complete graph, O(N^2) torch ops)
# ===============================================================
def tensor_to_graph_fast_v2(t: Tensor,
                            moves_left: float,
                            S: int = 4) -> Data:
    """
    t : (T,S,S,S) or (S,S,S) with values in {-1,0,1}
    Returns a PyG Data with:
      x          (N,5)    features [i/S, j/S, k/S, val0/2, moves_left/S]
      edge_index (2,E)    a single‐relation complete graph (directed edges)
      lin_idx    (N,)     mapping linear index 3D->0..S^3-1
    """
    dev = t.device
    if t.dim() == 3:
        t = t.unsqueeze(0)                   # make (T,S,S,S)
    # find active nodes (any nonzero across time)
    active = (t != 0).any(0)                 # (S,S,S) bool
    if not active.any():
        i = j = k = torch.tensor([0], device=dev)
    else:
        i, j, k = active.nonzero(as_tuple=True)
    N = i.numel()
    lin_idx = (i * S + j) * S + k            # linear index in [0..S^3-1]

    # build node features
    coord = torch.stack((i, j, k), dim=1).float() / (S - 1)    # (N,3)
    val0  = t[0, i, j, k].float().unsqueeze(1) / 2.0           # (N,1)
    mv    = torch.full((N,1), moves_left / S, device=dev)      # (N,1)
    x     = torch.cat((coord, val0, mv), dim=1)                # (N,5)

    # build complete directed graph edges (u->v for all u!=v)
    idx = torch.arange(N, device=dev)
    u = idx.repeat(N)                       # (N*N,)
    v = idx.repeat_interleave(N)            # (N*N,)
    mask = u != v
    edge_index = torch.stack((u[mask], v[mask]), dim=0)  # (2, N*(N-1))

    return Data(x=x, edge_index=edge_index, lin_idx=lin_idx)

# ===============================================================
#  DynamicGNNv2   (GraphSAGE, single edge_index)
# ===============================================================
class DynamicGNNv2(nn.Module):
    def __init__(self, in_dim=5, out_dim=32, num_layers=3):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, out_dim)
        self.convs  = nn.ModuleList(
            SAGEConv(out_dim, out_dim, aggr='mean')
            for _ in range(num_layers)
        )
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, batch: Batch):
        x  = self.lin_in(batch.x)
        ei = batch.edge_index
        for conv in self.convs:
            x = self.norm(F.relu(conv(x, ei)))
        batch.x = x
        return batch

# ===============================================================
#  HybridGnnTorsoV2  (GNN + transformer on past actions + scalar)
# ===============================================================
class HybridGnnTorsoV2(nn.Module):
    """
      • GNN (GraphSAGE) on current frame → 48 embeddings
      • Transformer on past actions (T-1 frames) → 1 embedding
      • Linear on moves_left scalar → 1 embedding
      ⇒ Concatenate → (B, 50, C)
    """
    def __init__(self, S=4, T=8, dim_c=32,gnn_layers=1, n_tf_layers=2, n_heads=4):
        super().__init__()
        self.S, self.T, self.C = S, T, dim_c

        # dynamic GNN
        self.gnn = DynamicGNNv2(in_dim=5, out_dim=dim_c,
                                num_layers=gnn_layers)

        # transformer for past actions
        tf_layer = nn.TransformerEncoderLayer(
            d_model=dim_c, nhead=n_heads, dim_feedforward=4*dim_c,
            batch_first=True, norm_first=True
        )
        self.act_tf  = nn.TransformerEncoder(tf_layer, n_tf_layers)
        self.act_lin = nn.Linear(S**3, dim_c)   # flatten 64 → C

        # scalar moves_left embedding
        self.scalar = nn.Linear(1, dim_c)

        # pooling matrix for 48 share‐i/j/k embeddings
        grid = torch.arange(S**3)
        i = grid // (S*S)
        j = (grid // S) % S
        k = grid % S
        mask = torch.cat([
            (i[:,None] == torch.arange(S)).float().T,  # share‐i  → (S, S^3)
            (j[:,None] == torch.arange(S)).float().T,  # share‐j
            (k[:,None] == torch.arange(S)).float().T   # share‐k
        ], dim=0)  # (3S, S^3) = (48,64)
        self.register_buffer("poolW", mask)

    def forward(self, xx: Tensor, ss: Tensor):
        """
        xx : (B, T, S, S, S)   current + past frames
        ss : (B, 1)           moves_left scalar
        """
        B, dev = xx.size(0), xx.device

        # --- GNN on current frame -------------------------------
        graphs = [
            tensor_to_graph_fast_v2(xx[b,0], ss[b].item(), self.S)
            for b in range(B)
        ]
        batch = Batch.from_data_list(graphs).to(dev)
        out   = self.gnn(batch)                  # batch.x shape: (sum N_b, C)

        # dense unpool + pooling to 48 dims
        full = torch.zeros(B, self.S**3, self.C, device=dev)
        full[batch.batch, batch.lin_idx] = out.x
        # (48 × 64) · (B × 64 × C) → (B × 48 × C)
        gnn_emb = torch.matmul(self.poolW, full) / self.poolW.sum(1, keepdim=True)

        # --- transformer on past actions -------------------------
        acts = xx[:,1:].reshape(B, self.T-1, -1).float()   # (B, T-1, 64)
        acts = self.act_lin(acts)                         # (B, T-1, C)
        act_emb = self.act_tf(acts).mean(1)               # (B, C)

        # --- scalar embedding ------------------------------------
        mv_emb = torch.relu(self.scalar(ss)).squeeze(1)   # (B, C)

        # --- concatenate → (B, 50, C) ----------------------------
        return torch.cat([
            gnn_emb,
            act_emb.unsqueeze(1),
            mv_emb.unsqueeze(1)
        ], dim=1)




def create_fixed_positional_encoding(n_position: int, n_embedding: int, device: str):
    pe = torch.zeros(n_position, n_embedding, device=device)
    positions = torch.arange(n_position)
    denominators = 10000 ** (-torch.arange(0, n_embedding, 2) / n_embedding)
    pe[:, 0::2] = torch.outer(positions, denominators).sin()
    pe[:, 1::2] = torch.outer(positions, denominators).cos()
    return pe


class Head(nn.Module):
    def __init__(self, c1: int, c2: int, d: int, causal_mask=False, **kwargs):
        super().__init__()
        self.d = d
        self.causal_mask = causal_mask
        self.query = nn.Linear(c1, d, bias=False)
        self.key = nn.Linear(c2, d, bias=False)
        self.value = nn.Linear(c2, d, bias=False)


    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x (*, nx, c1)
        # y (*, ny, c2)
        q = self.query(x)  # (*, nx, d)
        k = self.key(y)  # (*, ny, d)
        v = self.value(y)  # (*, ny, d)
        a = q @ k.transpose(-2, -1) / (self.d**0.5)  # (*, nx, ny)
        if self.causal_mask:
            b = torch.tril(torch.ones_like(a))
            a = a.masked_fill(b == 0, float("-inf"))
        a = F.softmax(a, dim=-1)  # (*, nx, ny)
        out = a @ v  # (*, nx, d)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, c1: int, c2: int, n_heads=16, d=32, w=4, **kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(c1)
        self.ln2 = nn.LayerNorm(c2)
        self.heads = nn.ModuleList([Head(c1, c2, d, **kwargs) for _ in range(n_heads)])
        self.li1 = nn.Linear(n_heads * d, c1)
        self.ln3 = nn.LayerNorm(c1)
        self.li2 = nn.Linear(c1, c1 * w)
        self.li3 = nn.Linear(c1 * w, c1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # x (*, nx, c1)
        # y (*, ny, c2)
        x_norm = self.ln1(x)  # (*, nx, c1)
        y_norm = self.ln2(y)  # (*, ny, c2)
        x_out = torch.cat(
            [h(x_norm, y_norm) for h in self.heads], dim=-1
        )  # (*, nx, n_heads*d)
        x_out = x + self.li1(x_out)  # (*, nx, c1)
        out = self.ln3(x_out)  # (*, nx, c1)
        out = self.li2(out)  # (*, nx, c1*w)
        out = F.gelu(out)  # (*, nx, c1*w)
        out = self.li3(out)  # (*, nx, c1)
        return x_out + out


class AttentiveModeBatch(nn.Module):
    def __init__(self, dim_3d: int, c1: int, **kwargs):
        super().__init__()
        self.dim_3d = dim_3d
        self.mha = MultiHeadAttention(c1, c1, **kwargs)

    def forward(self, g: list[torch.Tensor]):
        for m1, m2 in [(0, 1), (1, 2), (2, 0)]:
            a = torch.cat((g[m1], g[m2]), dim=-2)  # (*, dim_3d,2*dim_3d,c)
            cc = self.mha(a, a)
            g[m1] = cc[:, :, : self.dim_3d, :]
            g[m2] = cc[:, :, self.dim_3d :, :]
        return g  # [(*, dim_3d, dim_3d, c)]*3


class Torso(nn.Module):
    def __init__(
        self, dim_3d: int, dim_t: int, dim_s: int, dim_c: int, n_layers=8, **kwargs
    ):
        super().__init__()
        self.dim_3d = dim_3d
        self.dim_t = dim_t
        self.dim_c = dim_c
        self.li1 = nn.ModuleList([nn.Linear(dim_s, dim_3d**2) for _ in range(3)])
        self.li2 = nn.ModuleList(
            [nn.Linear(dim_3d * dim_t + 1, dim_c) for _ in range(3)]
        )
        self.blocks = nn.Sequential(
            *[AttentiveModeBatch(dim_3d, dim_c, **kwargs) for _ in range(n_layers)]
        )

    def forward(self, xx: torch.Tensor, ss: torch.Tensor):
        # xx (*, dim_t, dim_3d, dim_3d, dim_3d)
        # ss (*, dim_s)
        x1 = xx.permute(0, 2, 3, 4, 1).reshape(
            -1, self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )  # (*, dim_3d, dim_3d, dim_3d*dim_t)
        x2 = xx.permute(0, 4, 2, 3, 1).reshape(
            -1, self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )  # (*, dim_3d, dim_3d, dim_3d*dim_t)
        x3 = xx.permute(0, 3, 4, 2, 1).reshape(
            -1, self.dim_3d, self.dim_3d, self.dim_3d * self.dim_t
        )  # (*, dim_3d, dim_3d, dim_3d*dim_t)
        g = [x1, x2, x3]  # [(*, dim_3d, dim_3d, dim_3d*dim_t)] * 3
        for i in range(3):
            p = self.li1[i](ss)  # (*, dim_3d**2)
            p = p.reshape(-1, self.dim_3d, self.dim_3d, 1)  # (*, dim_3d, dim_3d, 1)
            g[i] = torch.cat([g[i], p], dim=-1)  # (*, dim_3d, dim_3d, dim_3d*dim_t+1)
            g[i] = self.li2[i](g[i])  # (*, dim_3d, dim_3d, dim_c)
        g = self.blocks(g)  # [(*, dim_3d, dim_3d, dim_c)] * 3
        ee = torch.stack(g, dim=2)  # (*, 3, dim_3d, dim_3d, dim_c)
        ee = ee.reshape(-1, 3 * self.dim_3d**2, self.dim_c)  # (*, 3*dim_3d**2, dim_c)
        return ee


# Algorithm A.4.a
class PredictBlock(nn.Module):
    def __init__(self, n_feats: int, n_heads: int, dim_c: int, dropout_p=0.5, **kwargs):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_feats * n_heads)
        self.att1 = MultiHeadAttention(
            n_feats * n_heads,
            n_feats * n_heads,
            n_heads=n_heads,
            causal_mask=True,
        )
        self.dropout1 = nn.Dropout(dropout_p)
        self.ln2 = nn.LayerNorm(n_feats * n_heads)
        self.att2 = MultiHeadAttention(
            n_feats * n_heads,
            dim_c,
            n_heads=n_heads,
        )
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, xx: torch.Tensor, ee: torch.Tensor):
        xx = self.ln1(xx)  # (*, n_steps, n_feats*n_heads)
        # Self attention
        cc = self.att1(xx, xx)  # (*, n_steps, n_feats*n_heads)
        cc = self.dropout1(cc)
        xx = xx + cc  # (*, n_steps, n_feats*n_heads)
        xx = self.ln2(xx)  # (*, n_steps, n_feats*n_heads)
        # Cross attention
        cc = self.att2(xx, ee)  # (*, n_steps, n_feats*n_heads)
        cc = self.dropout2(cc)
        xx = xx + cc  # (*, n_steps, n_feats*n_heads)
        return xx


# Algorithm A.4
class PredictActionLogits(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_logits: int,
        dim_c: int,
        n_feats=64,
        n_heads= 32,
        n_layers=2,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.emb1 = nn.Embedding(n_logits + 1, n_feats * n_heads)  # +1 for START token
        self.pos_enc = nn.Parameter(torch.rand(n_steps, n_feats * n_heads))
        pos_enc_fix = create_fixed_positional_encoding(
            n_steps, n_feats * n_heads, device
        )
        self.register_buffer("pos_enc_fix", pos_enc_fix)
        self.blocks = nn.Sequential(
            *[PredictBlock(n_feats, n_heads, dim_c, **kwargs) for _ in range(n_layers)]
        )
        self.li1 = nn.Linear(n_feats * n_heads, n_logits)

    def forward(self, aa: torch.Tensor, ee: torch.Tensor, **kwargs):
        # aa (n_steps, n_logits) ; ee (dim_m, dim_c)
        xx = self.emb1(aa)  # (n_steps, n_feats*n_heads)
        xx = (
            xx + self.pos_enc[: xx.shape[1]] + self.pos_enc_fix[: xx.shape[1]]
        )  # (n_steps, n_feats*n_heads)
        for block in self.blocks:
            xx = block(xx, ee)
        oo = F.relu(xx)  # (n_steps, n_feats*n_heads)
        oo = self.li1(oo)  # (n_steps, n_logits)
        return oo, xx


class PolicyHead(nn.Module):
    def __init__(
        self,
        n_steps: int,
        n_logits: int,
        n_samples: int,
        dim_c: int,
        device="cpu",
        **kwargs
    ):
        super().__init__()
        self.n_steps = n_steps
        self.n_logits = n_logits
        self.n_samples = n_samples
        self.device = device
        self.predict_action_logits = PredictActionLogits(
            n_steps,
            n_logits,
            dim_c,
            **kwargs,
        )

    def fwd_train(self, ee: torch.Tensor, gg: torch.Tensor):
        # ee (B, dim_m, dim_c) ; gg (B, n_steps)
        if self.device == "mps":
            gg_shifted = self.n_logits * torch.ones_like(gg, dtype=torch.long)
            gg_shifted[:, 1:] = gg[:, :-1]
            gg = gg_shifted
        else:
            gg = gg.long().roll(shifts=1, dims=1)  # (n_steps)
            gg[:, 0] = self.n_logits  # START token

        oo, zz = self.predict_action_logits(
            gg, ee
        )  # oo (*, n_steps, n_logits) ; zz (*, n_steps, n_feats*n_heads)
        return oo, zz[:, 0, :]

    def fwd_infer(self, ee: torch.Tensor):
        batch_size = ee.shape[0]
        aa = torch.zeros(
            batch_size,
            self.n_samples,
            self.n_steps + 1,
            dtype=torch.long,
            device=self.device,
        )
        aa[:, :, 0] = self.n_logits  # start w/ SOS token
        pp = torch.ones(batch_size, self.n_samples, device=self.device)
        ee = ee.unsqueeze(1).repeat(
            1, self.n_samples, 1, 1
        )  # (1, n_samples, dim_m, dim_c)
        aa = aa.view(-1, self.n_steps + 1)  # (1*n_samples, n_steps)
        pp = pp.view(-1)  # (1*n_samples)
        ee = ee.view(-1, ee.shape[-2], ee.shape[-1])  # (1*n_samples, dim_m, dim_c)
        for i in range(self.n_steps):
            oo_s, zz_s = self.predict_action_logits(aa[:, : i + 1], ee)
            distrib = Categorical(logits=oo_s[:, i])
            aa[:, i + 1] = distrib.sample()  # allow to sample 0, but reserve for START
            p_i = distrib.probs[
                torch.arange(batch_size * self.n_samples), aa[:, i + 1]
            ]  # (batch_size)
            pp = torch.mul(pp, p_i)
        return (
            aa[:, 1:].view(batch_size, self.n_samples, self.n_steps),
            pp.view(batch_size, self.n_samples),
            zz_s[:, 0].view(batch_size, self.n_samples, *zz_s.shape[2:]).mean(1),
        )  # (b, n_samples, n_steps), (b, n_samples), (b, n_feats*n_heads)


class ValueHead(nn.Module):
    # 64, 32, 512
    def __init__(self, n_feats=64, n_heads=32, n_hidden=512, n_quantile=8, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * n_heads, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_quantile),
        )

    def forward(self, xx: torch.Tensor):
        return self.mlp(xx)  # (n_quantile)

def quantile_loss(qq: torch.Tensor, gg: torch.Tensor, delta=1, device="cpu"):
    # qq (n) ; gg (*)
    n = qq.shape[-1]
    tau = (torch.arange(n, dtype=torch.float32, device=device) + 0.5) / n  # (n)
    hh = F.huber_loss(gg.expand(-1, n), qq, reduction="none", delta=delta)  # (n)
    dd = gg - qq  # (n)
    kk = torch.abs(tau - (dd > 0).float())  # (n)
    return torch.mean(torch.mul(hh, kk))  # ()


from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv


class GNNTorso(nn.Module):
    """
    Nodo = cella (i,j,k) del frame t
    Feature nodo =  [i/S , j/S , k/S , t/T , value(-1‥1) , moves_left/R]
    R-GCN con 3 relazioni  →  embedding (B,48,dim_c), cioè
      16 vettori A(j,k) + 16 vettori B(i,k) + 16 vettori C(i,j)
    compatibili con la policy/value head di AlphaTensor.
    """

    def __init__(self, dim_3d: int = 4, dim_t: int = 8, dim_s: int = 1, dim_c: int = 32, num_layers: int = 4):
        super().__init__()
        self.S, self.T, self.C = dim_3d, dim_t, dim_c

        # 6 scalari → dim_c
        self.feat2c = nn.Linear(6, dim_c)

        # R-GCN
        self.convs = nn.ModuleList(
            RGCNConv(dim_c, dim_c, num_relations=3)
            for _ in range(num_layers)
        )
        self.ln = nn.LayerNorm(dim_c)

        # grafo fisso (in buffer)
        self.register_buffer("edge_index", None)
        self.register_buffer("edge_type",  None)

    # ------------------------------------------------------------------ #
    def build_template_graph(self, device):
        """Costruisce edge_index / edge_type per le tre relazioni."""
        S, T = self.S, self.T
        nid = lambda i, j, k, t: t * S**3 + (i*S + j)*S + k
        src, dst, et = [], [], []

        # share-i
        for i in range(S):
            for j1 in range(S):
                for k1 in range(S):
                    for j2 in range(j1 + 1, S):
                        for k2 in range(S):
                            for t in range(T):
                                a, b = nid(i, j1, k1, t), nid(i, j2, k2, t)
                                src += [a, b]; dst += [b, a]; et += [0, 0]
        # share-j
        for j in range(S):
            for i1 in range(S):
                for k1 in range(S):
                    for i2 in range(i1 + 1, S):
                        for k2 in range(S):
                            for t in range(T):
                                a, b = nid(i1, j, k1, t), nid(i2, j, k2, t)
                                src += [a, b]; dst += [b, a]; et += [1, 1]
        # share-k
        for k in range(S):
            for i1 in range(S):
                for j1 in range(S):
                    for i2 in range(i1 + 1, S):
                        for j2 in range(S):
                            for t in range(T):
                                a, b = nid(i1, j1, k, t), nid(i2, j2, k, t)
                                src += [a, b]; dst += [b, a]; et += [2, 2]

        ei = torch.tensor([src, dst], dtype=torch.long, device=device)
        et = torch.tensor(et, dtype=torch.long, device=device)
        return ei, et
    # ------------------------------------------------------------------ #

    def forward(self, xx: torch.Tensor, ss: torch.Tensor):
        """
        xx : (B,T,S,S,S)   valori {-1,0,1}
        ss : (B,1)         mosse rimanenti   (0‥R_limit)
        """
        B, device = xx.size(0), xx.device

        # grafo statico solo la 1ª volta / caricato da checkpoint
        if self.edge_index is None:
            ei, et = self.build_template_graph(device)
            self.edge_index, self.edge_type = ei, et

        # coordinate normalizzate pre-calcolate
        grid_t, grid_i, grid_j, grid_k = torch.meshgrid(
            torch.arange(self.T, device=device),
            torch.arange(self.S, device=device),
            torch.arange(self.S, device=device),
            torch.arange(self.S, device=device),
            indexing="ij"
        )
        t = grid_t.flatten().float() / (self.T - 1)
        i = grid_i.flatten().float() / (self.S - 1)
        j = grid_j.flatten().float() / (self.S - 1)
        k = grid_k.flatten().float() / (self.S - 1)

        embeddings = []
        for b in range(B):
            v  = xx[b].flatten().float()              # -1/0/1
            mv = ss[b].item() / self.T               # normalizza mosse
            m  = torch.full_like(v, mv)

            feats = torch.stack([i, j, k, t, v, m], dim=1)  # (N,6)
            x = self.feat2c(feats)                          # (N,dim_c)

            data = Data(x=x,
                        edge_index=self.edge_index,
                        edge_type=self.edge_type)
            for conv in self.convs:
                data.x = self.ln(F.relu(
                    conv(data.x, data.edge_index, data.edge_type)))

            # --- unpooling 48 = 16+16+16 ----------------------------------
            H = data.x.view(self.T, self.S, self.S, self.S, self.C)
            H0 = H[0]                             # frame corrente (S,S,S,C)

            A = H0.mean(0).reshape(-1, self.C)    # share-i   → (16,C)
            B = H0.mean(1).reshape(-1, self.C)    # share-j   → (16,C)
            C = H0.mean(2).reshape(-1, self.C)    # share-k   → (16,C)

            embeddings.append(torch.cat([A, B, C], dim=0))   # (48,C)

        return torch.stack(embeddings, dim=0)   # (B,48,dim_c)


class TensorModel(nn.Module):
    def __init__(self,dim_3d=4,dim_t=8,dim_s=1,dim_c=16,n_steps=12,n_logits=3,n_samples=4,device="cpu",**kwargs):
        super().__init__()
        self.dim_3d = dim_3d
        self.n_steps = n_steps
        self.n_logits = n_logits
        self.n_samples = n_samples
        #self.torso = Torso(dim_3d, dim_t, dim_s, dim_c, **kwargs)
        #self.torso = GNNTorso(dim_3d, dim_t, dim_s, dim_c, **kwargs)
        #self.torso = HybridGnnTorso(S=4, T=8, dim_c=dim_c)
        #self.torso = HybridGnnTorsoV2(S=4, T=8, dim_c=dim_c)
        #self.torso = TorsoGCNv1(S=4, T=8, dim_c=dim_c)
        #self.torso = TorsoGCNv2(input_dim=3+dim_t, hidden_dim=128, S=2, c=dim_c)
        #self.torso = TorsoGCNv3(S=4, hidden_dim=128, dim_c=dim_c)
        self.torso = TorsoGATv1(input_dim=3+dim_t, hidden_dim=128, S=2, c=dim_c)
        #self.torso = HybridGnnTorsoATT(S=4, T=8, dim_c=dim_c)
        #self.torso = HybridGnnTorsoLine(S=4, T=8, dim_c=dim_c)
        self.policy_head = PolicyHead(
            n_steps, n_logits, n_samples, dim_c, device=device, **kwargs
        )
        self.value_head = ValueHead(**kwargs)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def value_risk_mgmt(qq: torch.Tensor, uq=0.75):
        jj = ceil(uq * qq.shape[-1]) - 1
        return torch.mean(qq[:, jj:], dim=-1)

    def fwd_train(self,xx: torch.Tensor,ss: torch.Tensor,g_action: torch.Tensor,g_value: torch.Tensor,):
        """Compare single action predictions to target actions and values.
        Returns policy (action) and value losses.
        """
        ee = self.torso(xx, ss)  # (3*dim_3d**2, dim_c)
        oo, zz = self.policy_head.fwd_train(
            ee, g_action
        )  # oo (*, n_steps, n_logits) ; zz (*, n_feats*n_heads)
        l_pol = F.cross_entropy(
            oo.view(-1, self.n_logits), g_action.view(-1), reduction="sum"
        )
        qq = self.value_head(zz)  # (n)
        l_val = quantile_loss(qq, g_value, device=self.device)
        return l_pol, l_val

    def fwd_infer(self, xx: torch.Tensor, ss: torch.Tensor):
        """Generate trajectories from input state.
        Returns trajectory, probability, and value.
        """
        ee = self.torso(xx, ss)  # (3*dim_3d**2, dim_c)
        aa, pp, z1 = self.policy_head.fwd_infer(ee)
        # aa (*, n_samples, n_steps) ; pp (*, n_samples) ; z1 (*, n_feats*n_heads)
        qq = self.value_head(z1)  # (n)
        qq = self.value_risk_mgmt(qq)  # (1)
        return aa, pp, qq
