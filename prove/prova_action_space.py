def generate_action_space_vecchio(S, coefficients=[-1, 0, 1]):
    indices = list(range(S))
    coefficient_list = coefficients
    action_space = []
    for u_idx in indices:
        for v_idx in indices:
            for w_idx in indices:
                for u_coeff in coefficient_list:
                    for v_coeff in coefficient_list:
                        for w_coeff in coefficient_list:
                            if not (u_coeff == v_coeff == w_coeff == 0):
                                action = {
                                    'u': (u_idx, u_coeff),
                                    'v': (v_idx, v_coeff),
                                    'w': (w_idx, w_coeff)
                                }
                                action_space.append(action)
    return action_space


from itertools import product

def all_coeff_vectors(S=4, coeffs=(-1,0,1), max_nonzero=None):
    # genera tutti i vettori di lunghezza S con valori in coeffs
    # esclude il vettore zero; opzionalmente limita il numero di non-zero
    out = []
    for vec in product(coeffs, repeat=S):
        if not any(vec):
            continue
        if max_nonzero and sum(c!=0 for c in vec) > max_nonzero:
            continue
        out.append(vec)
    return out


import numpy as np
from itertools import product
import torch

def build_action_table(S=4, coeff=(-1,0,1)):
    # tutte le triple (c_u,c_v,c_w) tranne (0,0,0), 26 combinazioni
    coeff_triple = [c for c in product(coeff, repeat=3) if any(c)]
    coeff_triple = np.array(coeff_triple, dtype=np.int8)          # (26,3)

    # tutte le triple d’indice (u,v,w) ∈ [0..S-1]^3
    idx_triple   = np.array(list(product(range(S), repeat=3)), dtype=np.int8) # (64,3)

    # prodotto cartesiano → 64×26 = 1664 righe
    idx_rep   = np.repeat(idx_triple, len(coeff_triple), axis=0)  # (1664,3)
    coeff_rep = np.tile  (coeff_triple,   (len(idx_triple),1))    # (1664,3)

    table = np.concatenate([idx_rep, coeff_rep], axis=1)          # (1664,6)
    # colonne: u_idx,v_idx,w_idx, c_u,c_v,c_w
    return table

ACTION_TABLE = build_action_table(S=4)          # np.array (1664,6)
U_IDX, V_IDX, W_IDX, CU, CV, CW = ACTION_TABLE.T


def index_to_vectors(idx_batch, S=4):
    # idx_batch: 1‑D o 2‑D tensor/array con indici azione
    idx = np.asarray(idx_batch).ravel()
    # one‑hot con broadcasting
    u = np.eye(S, dtype=np.int8)[U_IDX[idx]] * CU[idx,None]
    v = np.eye(S, dtype=np.int8)[V_IDX[idx]] * CV[idx,None]
    w = np.eye(S, dtype=np.int8)[W_IDX[idx]] * CW[idx,None]
    # shape (batch, S)
    return u.reshape(idx_batch.shape + (S,)), \
           v.reshape(idx_batch.shape + (S,)), \
           w.reshape(idx_batch.shape + (S,))


def encode_action(u_idx, v_idx, w_idx, c_u, c_v, c_w, S=4):
    # map coeff ‑1,0,1 ➜ digit 0,1,2
    d_u = c_u + 1
    d_v = c_v + 1
    d_w = c_w + 1
    coeff_id = d_u*9 + d_v*3 + d_w      # 0..26 (27 combo)
    if coeff_id == 13:                  # (0,0,0) → salta
        raise ValueError("zero coeff triple not allowed")
    # comprimi ←–– base‑S per gli indici, base‑27 per coeff
    idx_id   =  (u_idx*S + v_idx)*S + w_idx      # 0..63
    action_id = idx_id*26 + coeff_id - (coeff_id>13)  # 1664 mappa continua
    return action_id




for i in range(1664):
    u, v, w = index_to_vectors(torch.tensor([i]))
    print(u, v, w)


# ho capito perché andare autoregressive è la via :) 