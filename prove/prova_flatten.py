import numpy as np
from itertools import product

# 1. reset_game (numpy version)
def reset_game_np():
    n = 2
    Tn = np.zeros((n * n, n * n, n * n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            c_index = i * n + j
            for k in range(n):
                a_index = i * n + k
                b_index = k * n + j
                Tn[a_index, b_index, c_index] = 1.0
    return Tn

# 2. Build ACTION_TABLE
def build_action_table_np(S=4, coeff=(-1,0,1)):
    # coefficient triples excluding (0,0,0)
    coeff_triple = np.array([c for c in product(coeff, repeat=3) if any(c)], dtype=np.int8)
    # index triples
    idx_triple = np.array(list(product(range(S), repeat=3)), dtype=np.int8)
    # Cartesian product
    idx_rep = np.repeat(idx_triple, len(coeff_triple), axis=0)
    coeff_rep = np.tile(coeff_triple, (len(idx_triple), 1))
    table = np.concatenate([idx_rep, coeff_rep], axis=1)
    return table  # shape (1664, 6)

# prepare
state = reset_game_np()  # shape (4,4,4)
head_flat = state.reshape(-1)  # (64,)



S = 4
ACTION_TABLE = build_action_table_np(S)
U_IDX = ACTION_TABLE[:, 0]
V_IDX = ACTION_TABLE[:, 1]
W_IDX = ACTION_TABLE[:, 2]
CU    = ACTION_TABLE[:, 3]
CV    = ACTION_TABLE[:, 4]
CW    = ACTION_TABLE[:, 5]

# HEAD_IDX and PRODUCT
HEAD_IDX = (U_IDX * S + V_IDX) * S + W_IDX      # (1664,)

print(HEAD_IDX)
print(len(HEAD_IDX))
exit()
PRODUCT  = CU * CV * CW                        # (1664,)

# compute mask
residuals = head_flat[HEAD_IDX]                # (1664,)
m1 = PRODUCT != 0
m2 = (residuals * PRODUCT) > 0  # Va nella direzione giusta residuals[i] * PRODUCT[i] > 0 (se il residuo è +1, puoi solo sottrarre +1; se è –1, solo –1)
m3 = np.abs(PRODUCT) <= np.abs(residuals)
mask = m1 & m2 & m3                            # bool array

# count valid moves
valid_count = np.sum(mask)

# display
print("HEAD_IDX[:10]:", HEAD_IDX[:10])
print("PRODUCT[:10]:", PRODUCT[:10])
print("Mask[:10]:", mask[:10])
print("Number of valid actions:", valid_count)
