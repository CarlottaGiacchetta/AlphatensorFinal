import numpy as np
from itertools import product

# -------------- helpers --------------
def build_action_table(S=4, coeff=(-1,0,1)):
    coeff_triple = np.array([c for c in product(coeff, repeat=3) if any(c)], dtype=np.int8)
    idx_triple   = np.array(list(product(range(S), repeat=3)), dtype=np.int8)
    idx_rep  = np.repeat(idx_triple, len(coeff_triple), axis=0)
    coef_rep = np.tile(coeff_triple, (len(idx_triple),1))
    return np.concatenate([idx_rep, coef_rep], axis=1)

S=4
ACTION_TABLE = build_action_table(S)
U_IDX, V_IDX, W_IDX, CU, CV, CW = ACTION_TABLE.T
HEAD_IDX = (U_IDX*S + V_IDX)*S + W_IDX
PRODUCT  = CU*CV*CW

# given head1 tensor
head1 = np.array([[[1,0,0,0],
                   [0,1,0,0],
                   [0,0,0,0],
                   [0,0,0,0]],

                  [[0,0,0,0],
                   [0,0,0,0],
                   [1,0,0,0],
                   [0,1,0,0]],

                  [[0,0,1,0],
                   [0,0,0,1],
                   [0,0,0,0],
                   [0,0,0,0]],

                  [[0,0,0,0],
                   [0,0,0,0],
                   [0,0,1,0],
                   [0,0,0,1]]], dtype=np.int8)


final = np.array([[[1, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 1]],

                    [[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]],

                    [[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]],

                    [[1, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 1]]], dtype=np.int8)
# ----------- compute mask for head1 ----------
head_flat = head1.reshape(-1)           # (64,)
residuals = head_flat[HEAD_IDX]         # (1664,)

valid_mask = (PRODUCT!=0) & (residuals*PRODUCT>0)
valid_ids  = np.where(valid_mask)[0]

print("Numero di azioni valide:", len(valid_ids))
print("Prime 10 id valide:", valid_ids[:10])

# Mostriamo la prima azione valida
aid = valid_ids[7]
print("\nAzione id", aid, "→ riga tabella:", ACTION_TABLE[aid])

# Genero i vettori e controllo outer product
u_idx, v_idx, w_idx, c_u, c_v, c_w = ACTION_TABLE[aid]
u = np.zeros(4, dtype=np.int8); u[u_idx] = c_u
v = np.zeros(4, dtype=np.int8); v[v_idx] = c_v
w = np.zeros(4, dtype=np.int8); w[w_idx] = c_w
outer = np.einsum('i,j,k->ijk', u, v, w)

# Verifica che outer riduce head1
new_head = head1 - outer
print("\nNumero di 1 scomparsi grazie a questa mossa:", (head1!=0).sum() - (new_head!=0).sum())

print(outer == final)

