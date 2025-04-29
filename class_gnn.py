import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from utils_graph import tensor_batch_to_graphs, visualize_graph_2d


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