import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data, Batch
import torch
from torch_scatter import scatter_add
from torch_geometric.data import Data

# ---------------------------------------------------------------------
# 1. 100 % torch – converte UN tensore (T,S,S,S) in Data
# ---------------------------------------------------------------------
def tensor_to_graph_fast(tensor: torch.Tensor) -> Data:
    """
    Versione vectorized di tensor_to_graph.
    - Nessuna dipendenza da NetworkX
    - Tutto su GPU se `tensor` è già su GPU
    """
    device = tensor.device
    if tensor.dim() == 3:            # (S,S,S)  →  (1,S,S,S)
        tensor = tensor.unsqueeze(0)

    T, S = tensor.size(0), tensor.size(1)

    # --------------------- nodi presenti ------------------------------
    mask = tensor.any(dim=0)                             # (S,S,S) bool
    coords = mask.nonzero(as_tuple=False)                # (N,3) long
    if coords.numel() == 0:                              # cubo vuoto?
        coords = torch.zeros((1, 3), dtype=torch.long, device=device)

    N = coords.size(0)
    i, j, k = coords.t().float()                         # (N,)

    # --------------------- feature nodi -------------------------------
    values = tensor[:, coords[:, 0], coords[:, 1], coords[:, 2]].T  # (N,T)
    x = torch.cat([i[:, None], j[:, None], k[:, None], values], dim=1)  # (N, 3+T)

    # --------------------- archi --------------------------------------
    # due nodi collegati se condividono ≥1 coordinata
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)     # (N,N,3)
    num_matches = (diff == 0).sum(-1)                    # (N,N)
    mask_edges = (num_matches > 0) & ~torch.eye(N, dtype=torch.bool, device=device)

    edge_index = mask_edges.nonzero(as_tuple=False).t()  # (2,E)

    k_eq = (coords[:, 2].unsqueeze(1) == coords[:, 2].unsqueeze(0))
    edge_weight = torch.where(k_eq & mask_edges, 1.0, 0.5)
    edge_attr = edge_weight[mask_edges].unsqueeze(1)     # (E,1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------
# 2. Batch di tensori  →  Batch di Data
# ---------------------------------------------------------------------
def tensor_batch_to_graphs(tensor_batch: torch.Tensor,
                           use_line_graph: bool = False) -> Batch:
    """
    Converte un batch di tensori  (B,T,S,S,S)  in  torch_geometric Batch.
    Usa la versione fast; `line_graph` è mantenuto per compatibilità,
    ma se ti serve davvero il line-graph dovrai implementarlo anch’esso in torch.
    """
    graphs = [tensor_to_graph_fast(t) for t in tensor_batch]
    if use_line_graph:
        graphs = [graph_to_line_graph_fast(g) for g in graphs]
    return Batch.from_data_list(graphs)




###############################################################################
# 1. Line‑graph ultra‑veloce (tutto tensor, 0 cicli Python in hot‑path)
###############################################################################

def graph_to_line_graph_fast(data: Data) -> Data:
    """Crea il line-graph su GPU senza cicli Python.
    L'idea è usare la matrice d'incidenza sparsa    
        inc  ∈ {0,1}^{N×E}
    e calcolare  L = inc^T ⋅ inc  (E×E),  che conta
    quante estremità due archi condividono. Lij>0 ⇒ archi adiacenti.

    Complessità:  O(E + |L|)  (tutto in C++/CUDA sparse)."""
    device = data.x.device
    src, dst = data.edge_index            # (2, E)
    E = src.size(0)
    d = data.x.size(1)
    N = int(torch.max(torch.stack([src, dst])) + 1)

    # ------------------------------------------------------------------
    # 1. Feature dei nodi del line‑graph
    #    x_line[e] = x[src[e]] + x[dst[e]]
    # ------------------------------------------------------------------
    x_line = data.x[src] + data.x[dst]    # (E, d)

    # ------------------------------------------------------------------
    # 2. Matrice d'incidenza sparsa  inc  (N×E)
    # ------------------------------------------------------------------
    row = torch.cat([src, dst], dim=0)               # (2E,)
    col = torch.arange(E, device=device).repeat(2)   # (2E,)
    val = torch.ones(2 * E, device=device)
    inc = torch.sparse_coo_tensor(
        torch.stack([row, col]), val, (N, E), device=device)

    # ------------------------------------------------------------------
    # 3. Adiacenza del line‑graph   L = incᵀ ⋅ inc   (E×E  sparsa)
    # ------------------------------------------------------------------
    L = torch.sparse.mm(inc.transpose(0, 1), inc).coalesce()
    idx_i, idx_j = L.indices()                       # (2, |L|)
    mask = idx_i != idx_j                            # rimuovi diagonale
    edge_index_L = torch.stack([
        idx_i[mask], idx_j[mask]
    ], dim=0)                                        # (2, E_L)

    # ------------------------------------------------------------------
    # 4. Feature degli archi del line‑graph
    #    media delle 4 estremità (come nel codice originale)
    # ------------------------------------------------------------------
    if edge_index_L.numel() == 0:
        edge_attr = torch.empty((0, d), device=device)
    else:
        e1, e2 = edge_index_L
        a1, b1 = src[e1], dst[e1]
        a2, b2 = src[e2], dst[e2]
        edge_attr = (data.x[a1] + data.x[b1] + data.x[a2] + data.x[b2]) * 0.25

    return Data(x=x_line, edge_index=edge_index_L, edge_attr=edge_attr)



## OLD VERSION ##
'''
def tensor_to_graph(tensor):

    device = tensor.device
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    T = tensor.size(0)
    S = tensor.size(1)

    G = nx.Graph()
    node_exists = False

    for i in range(S):
        for j in range(S):
            for k in range(S):
                values = tensor[:, i, j, k]  # shape (T,)
                if torch.any(values != 0):
                    node_id = f"{i},{j},{k}"
                    node_exists = True
                    G.add_node(node_id, i=i, j=j, k=k, values=values)

    if not node_exists:
        node_id = "dummy"
        G.add_node(node_id, i=0, j=0, k=0, values=torch.zeros(T, device=device))

    # Creazione degli archi
    nodes = list(G.nodes(data=True))
    for idx_u, u in enumerate(nodes):
        u_id, u_attr = u
        i_u, j_u, k_u = u_attr['i'], u_attr['j'], u_attr['k']
        for idx_v in range(idx_u + 1, len(nodes)):
            v_id, v_attr = nodes[idx_v]
            i_v, j_v, k_v = v_attr['i'], v_attr['j'], v_attr['k']
            num_matches = sum([i_u == i_v, j_u == j_v, k_u == k_v])
            
            if num_matches > 0:  # Aggiungi solo archi con almeno un indice uguale
                weight = 1 if k_u == k_v else 0.5 
                G.add_edge(u_id, v_id, weight=weight)

    data = from_networkx(G)

    # Edge attributes
    if hasattr(data, 'weight'):
        edge_weights = torch.tensor([w for w in data.weight], dtype=torch.float, device=device)
        data.edge_attr = edge_weights.unsqueeze(1)  # Converti in tensore PyTorch
        del data.weight  # Rimuovi il vecchio attributo per evitare duplicati
    else:
        data.edge_attr = torch.zeros((data.edge_index.size(1), 1), dtype=torch.float, device=device)

    # Feature dei nodi: [i_norm, j_norm, k_norm, val_t0, val_t1, ..., val_t_(T-1)]
    node_features = []
    for node_id in G.nodes():
        node_attr = G.nodes[node_id]
        i_n = node_attr['i']
        j_n = node_attr['j']
        k_n = node_attr['k']
        values = node_attr['values']

        i_norm = i_n #/ S
        j_norm = j_n #/ S
        k_norm = k_n #/ S
        values_norm = values #/ T

        # Crea i tensori sul dispositivo corretto
        feat = torch.cat([
            torch.tensor([i_norm, j_norm, k_norm], dtype=torch.float, device=device),
            values_norm.to(device=device)
        ])

        # Verifica che ogni nodo abbia esattamente 11 feature
        assert feat.shape[0] == 3 + T, f"Dimensione feature errata: attesa {3+T}, trovata {feat.shape[0]}"

        node_features.append(feat.tolist())

    # Crea il tensore delle feature sul dispositivo corretto
    data.x = torch.tensor(node_features, dtype=torch.float, device=device)

    # **Sposta tutto su GPU se il tensore di input era su GPU**
    if device.type == 'cuda':
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.edge_attr = data.edge_attr.to(device)
    return data





def tensor_batch_to_graphs(tensor_batch, line_graph):
    graphs = []
    batch_size = tensor_batch.shape[0]  # Numero di tensori nel batch
    
    for i in range(batch_size):
        tensor = tensor_batch[i]  # Estrai il singolo tensore
        graph = tensor_to_graph(tensor)  # Converti in grafo
        if line_graph:
            graph = graph_to_line_graph(graph)
        graphs.append(graph)

    # Crea un batch di grafi
    return Batch.from_data_list(graphs)


def graph_to_line_graph(data):
    """Trasforma un grafo in un line graph mantenendo la stessa dimensionalità delle feature.
    
    Per ogni arco (v1, v2) del grafo originale, viene creato un nodo nel line graph con feature:
        X_v1 + X_v2
    (invece di concatenare min(X_v1, X_v2), max(X_v1, X_v2) e (X_v1+X_v2), che porterebbe a dimensione 3*d).
    
    Gli archi nel line graph sono creati se due archi originali condividono un nodo,
    e l'attributo dell'arco viene calcolato come la media delle feature dei quattro nodi originali coinvolti.
    
    Nota: Se desideri utilizzare la trasformazione completa proposta nel paper (equazione (2))
    che genera feature di dimensione 3*d, dovrai aggiornare il modello per accettare input di dimensione 3*d.
    """
    import torch
    import networkx as nx
    from torch_geometric.utils import from_networkx

    L = nx.Graph()
    edge_to_node = {}

    # Verifica che il grafo originale abbia feature sui nodi
    use_node_features = data.x is not None

    # Creazione dei nodi del line graph: ogni nodo corrisponde a un arco (v1, v2) del grafo originale.
    # Per garantire l'invarianza rispetto all'ordine, usiamo sorted([v1, v2]).
    # La nuova feature viene calcolata come: X_v1 + X_v2.
    for u, v in zip(data.edge_index[0], data.edge_index[1]):
        u_idx = u.item()
        v_idx = v.item()
        u_idx, v_idx = sorted([u_idx, v_idx])
        edge_id = (u_idx, v_idx)
        if edge_id not in edge_to_node:
            if use_node_features:
                f1 = data.x[u_idx]
                f2 = data.x[v_idx]
                new_attr = f1 + f2  # Manteniamo la dimensione invariata (5, ad esempio)
            else:
                new_attr = torch.zeros(1, device=data.edge_index.device)
            L.add_node(edge_id, attr=new_attr)
            edge_to_node[edge_id] = edge_id

    # Creazione degli archi nel line graph: due nodi (cioè due archi originali) sono connessi se condividono almeno un vertice.
    for (u1, v1), node1 in edge_to_node.items():
        for (u2, v2), node2 in edge_to_node.items():
            if node1 != node2:
                if u1 in (u2, v2) or v1 in (u2, v2):
                    # Estrai le feature dei nodi originali
                    f1 = data.x[u1]
                    f2 = data.x[v1]
                    f3 = data.x[u2]
                    f4 = data.x[v2]
                    # Calcola la media delle feature dei quattro nodi
                    edge_feature = (f1 + f2 + f3 + f4) / 4
                    L.add_edge(node1, node2, weight=edge_feature)

    # Conversione del grafo NetworkX in formato PyTorch Geometric
    line_graph = from_networkx(L)
    
    # Se esiste una proprietà 'weight' negli archi, la convertiamo in edge_attr
    if 'weight' in line_graph:
        edge_attrs = []
        for e in L.edges:
            edge_attrs.append(L.edges[e]['weight'])
        line_graph.edge_attr = torch.stack(edge_attrs).to(data.x.device)
        if 'weight' in line_graph:
            del line_graph.weight

    num_edges = line_graph.edge_index.shape[1]
    if not hasattr(line_graph, 'edge_attr') or line_graph.edge_attr is None:
        line_graph.edge_attr = torch.zeros((num_edges, 1), dtype=torch.float, device=data.edge_index.device)

    # Assegna feature ai nodi del line graph
    num_nodes = len(L.nodes)
    if use_node_features:
        node_features = torch.stack([L.nodes[n]['attr'] for n in L.nodes]).to(data.x.device)
        line_graph.x = node_features
    else:
        feature_dim = data.edge_attr.shape[1] if data.edge_attr is not None else 1
        line_graph.x = torch.zeros((num_nodes, feature_dim), dtype=torch.float, device=data.edge_index.device)

    line_graph = line_graph.to(data.edge_index.device)
    return line_graph
'''


def visualize_graph_2d(graph_data):
    """Visualizza un grafo con le feature dei nodi e i pesi degli archi stampati accanto agli archi."""
    
    # Converte il grafo torch_geometric in NetworkX
    G = to_networkx(graph_data, to_undirected=True)

    # Genera una disposizione dei nodi
    pos = nx.spring_layout(G, seed=42)

    # Disegna il grafo
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, font_size=10, edge_color="gray")

    # Etichette dei nodi con feature
    if hasattr(graph_data, 'x') and graph_data.x is not None:
        node_labels = {}
        for i, node in enumerate(G.nodes()):
            features = graph_data.x[i].tolist()
            features_str = ", ".join([f"{f:.2f}" for f in features])
            
            node_labels[node] = f"{node}\n({features_str})"
        
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_color="red")

    # Etichette degli archi con i pesi
    if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None:
        edge_labels = {}
        edge_index = graph_data.edge_index
        edge_weights = graph_data.edge_attr.squeeze().tolist()

        for idx, (u, v) in enumerate(zip(edge_index[0], edge_index[1])):
            u = u.item()
            v = v.item()
            edge = tuple(sorted((u, v)))  # Per evitare duplicati
            weight = edge_weights[idx]
            edge_labels[edge] = f"{weight:.2f}"

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue", font_size=9)

    plt.title("Visualizzazione 2D del Grafo con Feature dei Nodi e Pesi degli Archi")
    plt.show()

