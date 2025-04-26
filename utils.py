import torch
import numpy as np

def reset_game(device):
    n = 2 # dimensione della matrice da moltiplicare --> tensore poi fa n**2
    Tn = torch.zeros((n * n, n * n, n * n), dtype=torch.float32, device=device)

    for i in range(n):
        for j in range(n):
            # Calcola l'indice per l'elemento C[i, j] nella matrice risultante
            c_index = i * n + j
            
            for k in range(n):
                # Calcola gli indici per A[i, k] e B[k, j]
                a_index = i * n + k
                b_index = k * n + j
                
                # Imposta Tn[a_index, b_index, c_index] a 1 per rappresentare il contributo
                Tn[a_index, b_index, c_index] = 1.0
    return Tn

def is_zero_tensor(tensor):
    tensor = np.asarray(tensor.cpu())
    return np.all(tensor == 0)

def outer_product(x, y, z):
    return np.einsum('i,j,k->ijk', x, y, z, dtype=np.int32, casting="same_kind")
    
def rango_tensore_finale(tensor: torch.Tensor) -> int:
    """
    Restituisce la somma dei ranghi (approssimati) delle matrici
    che compongono il tensore 3‑D sugli assi (S,S,S).
    """
    assert tensor.ndim == 3
    # convertiamo una sola volta su CPU, in int32
    arr = tensor.to('cpu', torch.int32).numpy()

    rank_sum = 0
    for z in range(arr.shape[-1]):
        rank_sum += np.linalg.matrix_rank(arr[..., z])
    return rank_sum


#def tokens_to_action_tensor(tokens: torch.Tensor, S=4):
#    """
#    tokens  : 1‑D Tensor [3*S]  con valori {0,1,2}  (0→‑1, 1→0, 2→+1)
#              ─ 0..S‑1   → coeff di u
#              ─ S..2S‑1  → coeff di v
#              ─ 2S..3S‑1 → coeff di w
#    ritorna : Tensor S×S×S  (int8) con l’outer‑product u⊗v⊗w
#    """
#    coeff = tokens.to(torch.int8) - 1          # 0,1,2 → -1,0,1
#    u = coeff[0:S]
#    v = coeff[S:2*S]
#    w = coeff[2*S:3*S]
#    return outer_product(u, v, w)              # S×S×S int8

def tokens_to_action_tensor(tokens: torch.Tensor, S=4, device=None):
    """
    tokens : 1‑D Tensor (3*S) con valori {0,1,2}.
    Ritorna: Tensor S×S×S int8   u⊗v⊗w
    """
    coeff = tokens.to(torch.int8) - 1          # 0,1,2 → -1,0,1
    u, v, w = coeff[0:S], coeff[S:2*S], coeff[2*S:3*S]

    # prodotto esterno puro PyTorch
    act = torch.einsum('i,j,k->ijk', u, v, w).to(torch.int8)

    if device is not None:
        act = act.to(device)
    return act
