import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from utils_graph import tensor_batch_to_graphs, tensor_to_graph_hybrid, graph_to_line_graph_fast, visualize_hybrid_graph
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.nn  import SAGEConv 



class TorsoGCNv1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, S, c):
        super(TorsoGCNv1, self).__init__()

        self.output_dim = 3 * (S ** 2) * c  # 48 * 8 = 384 nel caso di S=4, c=8
        self.dim1 = 3 * (S ** 2)
        self.dim2 = c

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)

        self.lin = torch.nn.Linear(hidden_dim // 4, self.output_dim)
        print(input_dim, hidden_dim // 4, self.output_dim)

    def forward(self, data, ss=None):
        data = tensor_batch_to_graphs(data, False)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        data.x = x  # aggiorna le feature nel grafo
        data.edge_index = edge_index
        #visualize_graph_2d(data, False)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  
        x = self.lin(x) # Proiezione nello spazio di embedding desiderato (48*8 = 384)
    
        batch_size = data.num_graphs  # Numero di grafi nel batch
        ee = x.view(batch_size, self.dim1, self.dim2)

        return ee


class TorsoGCNv2(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, S, c):
        super(TorsoGCNv2, self).__init__()

        self.output_dim = 3 * (S ** 2) * c  # 48 * 8 = 384 nel caso di S=4, c=8
        self.dim1 = 3 * (S ** 2)
        self.dim2 = c

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)

        self.lin = torch.nn.Linear(hidden_dim // 4, self.output_dim)

    def forward(self, data, ss=None):
        data = tensor_batch_to_graphs(data, False)

        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        if edge_attr is not None and edge_attr.numel() > 0:  # Verifica se edge_attr esiste
            edge_weights = edge_attr.float()
            edge_weights = edge_weights / edge_weights.max()  # Normalizza tra 0 e 1
        else:
            edge_weights = torch.ones(edge_index.size(1), device=edge_index.device)  # fallback: tutti pesi = 1.0
            print('[TorsoGCNv2] Warning: Edge weights not provided. Using uniform weights.')

        x = self.conv1(x, edge_index, edge_weights)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weights)
        x = F.relu(x)

        x = self.conv3(x, edge_index, edge_weights)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        
        x = self.lin(x)
        batch_size = data.num_graphs  
        ee = x.view(batch_size, self.dim1, self.dim2)

        return ee


class TorsoGCNv3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, S, c):
        super(TorsoGCNv3, self).__init__()

        self.output_dim = 3 * (S ** 2) * c  # 48 * 8 = 384 nel caso di S=4, c=8
        self.dim1 = 3 * (S ** 2)
        self.dim2 = c

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.conv3 = GCNConv(hidden_dim // 2, hidden_dim // 4)

        self.lin = torch.nn.Linear(hidden_dim // 4, self.output_dim)
        print(input_dim, hidden_dim // 4, self.output_dim)

    def forward(self, data, ss=None):
        data = tensor_batch_to_graphs(data, True)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        data.x = x  # aggiorna le feature nel grafo
        data.edge_index = edge_index
        #visualize_graph_2d(data, False)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)  
        x = self.lin(x) # Proiezione nello spazio di embedding desiderato (48*8 = 384)
    
        batch_size = data.num_graphs  # Numero di grafi nel batch
        ee = x.view(batch_size, self.dim1, self.dim2)

        return ee
    

class TorsoGATv1(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, S, c, heads=4):
        super().__init__()
        self.output_dim = 3 * (S ** 2) * c
        self.dim1 = 3 * (S ** 2)
        self.dim2 = c

        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=True)
        self.gat3 = GATConv(hidden_dim, hidden_dim // 2, heads=1, concat=True)


        self.lin = torch.nn.Linear(hidden_dim // 2, self.output_dim)

    def forward(self, data, ss=None):
        data = tensor_batch_to_graphs(data, False)

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = F.relu(self.gat3(x, edge_index))

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        batch_size = data.num_graphs
        ee = x.view(batch_size, self.dim1, self.dim2)

        return ee
    

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
class HybridGnnTorsoLine(nn.Module):
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

    def forward(self, xx: torch.Tensor, ss: torch.Tensor):
        """
        xx : (B, T, S, S, S)   current + past frames
        ss : (B, 1)           moves_left scalar
        """
        B, dev = xx.size(0), xx.device
        # --- GNN on current frame -------------------------------
        graphs = [
            tensor_to_graph_hybrid(xx[b,0], ss[b].item(), self.S)
            for b in range(B)
        ]
        
        graphs = [graph_to_line_graph_fast(g) for g in graphs]
        visualize_hybrid_graph(graphs[1])
        batch = Batch.from_data_list(graphs).to(dev)
        out   = self.gnn(batch)                  # batch.x shape: (sum N_b, C)


        # fallback pooling: mean over all line-graph nodes per sample
        gnn_emb = torch.zeros(B, 48, self.C, device=dev)
        for b in range(B):
            mask = batch.batch == b
            if mask.any():
                mean_emb = out.x[mask].mean(0)
                gnn_emb[b] = mean_emb.repeat(48, 1)  # broadcast to 48
        

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

