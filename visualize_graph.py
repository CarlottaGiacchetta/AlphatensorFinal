import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

# --- tiny utility that wraps GNNTorso for a single sample -------------
from math import ceil
from torch import nn
from torch_geometric.nn import RGCNConv

class MiniTorso(nn.Module):
    """minimal GNNTorso with S=2, T=2 so the graph is tiny and plottable"""
    def __init__(self, S=2, T=2, C=16):
        super().__init__()
        self.S, self.T, self.C = S, T, C
        self.feat2c = nn.Linear(6, C)
        self.conv   = RGCNConv(C, C, num_relations=3)
        self.register_buffer("edge_index", None)
        self.register_buffer("edge_type",  None)

    # generate clique edges per relation once
    def _init_graph(self, device):
        S, T = self.S, self.T
        nid = lambda i,j,k,t: t*S**3 + (i*S + j)*S + k
        src,dst,et=[],[],[]
        for t in range(T):
            for i in range(S):
                for j in range(S):
                    for k in range(S):
                        u=nid(i,j,k,t)
                        # same-i
                        for jj in range(j+1,S):
                            v=nid(i,jj,k,t); src+= [u,v]; dst+=[v,u]; et+=[0,0]
                        # same-j
                        for ii in range(i+1,S):
                            v=nid(ii,j,k,t);  src+= [u,v]; dst+=[v,u]; et+=[1,1]
                        # same-k
                        for ii in range(i+1,S):
                            v=nid(ii,j,k,t);  src+= [u,v]; dst+=[v,u]; et+=[2,2]
        ei=torch.tensor([src,dst],dtype=torch.long,device=device)
        et=torch.tensor(et,dtype=torch.long,device=device)
        self.edge_index, self.edge_type = ei,et

    def forward(self, xx, ss):
        device = xx.device

        # 1) inizializzo grafo e coordinate (una sola volta)
        if self.edge_index is None:
            self._init_graph(device)
            grid = torch.meshgrid(
                torch.arange(self.T, device=device),
                torch.arange(self.S, device=device),
                torch.arange(self.S, device=device),
                torch.arange(self.S, device=device),
                indexing="ij",
            )
            t, i, j, k = [g.flatten().float() for g in grid]
            # coord_feats: shape (T*S^3, 4)
            self.register_buffer(
                "coord_feats",
                torch.stack([i/(self.S-1),
                            j/(self.S-1),
                            k/(self.S-1),
                            t/(self.T-1)], dim=1),
            )

        # 2) preparazione feature dinamiche
        # a) valori del tensor su T*S^3 nodi
        v = xx.flatten(start_dim=0, end_dim=1).flatten().float().unsqueeze(1)
        #    ↑ prima flatten T dimensione, poi S³ → shape (T*S^3, 1)
        # b) mosse rimaste replicate per ogni nodo
        mv = ss[0].item() / self.T
        m  = torch.full_like(v, mv)

        # 3) concateniamo tutte le 6 feature → (T*S^3,6)
        feats = torch.cat([self.coord_feats, v, m], dim=1)

        # 4) embedding + conv
        x = self.feat2c(feats)  # (T*S^3, C)
        x = torch.relu(self.conv(x, self.edge_index, self.edge_type))

        # 5) ritorniamo il Data oggetto con embedding e topologia
        return Data(x=x,
                    edge_index=self.edge_index,
                    edge_type=self.edge_type)



# ---------------- create a dummy sample -------------------------------
S,T=2,2
state = torch.randint(-1,2,(T,S,S,S))
moves_left = torch.tensor([[1.0]])

model = MiniTorso(S,T)
data_graph = model(state, moves_left)

# ------------- visualise ------------------------------------------------
g_nx = to_networkx(data_graph, node_attrs=[], edge_attrs=['edge_type'])
pos = nx.spring_layout(g_nx, seed=0)
edge_colors = [['red','green','blue'][et] for *_,et in g_nx.edges.data('edge_type')]
nx.draw(g_nx, pos, node_size=300, edge_color=edge_colors, with_labels=False)
plt.title("Tiny tensor graph (relations: red=same-i, green=same-j, blue=same-k)")
plt.show()
