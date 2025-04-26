import numpy as np
import matplotlib.pyplot as plt

# --- carica i dati del torso “classico”
torso_loss = np.load("results_torso/figures/pretrain_loss.npy")

# --- placeholder per l’hybrid (36 epoche completate finora)
hybrid_loss = np.array([6577.9424,
6041.5284,
6048.6845,
5961.8269,
5904.9453,
5885.0747,
5675.2148,
5455.2993,
5292.8666,
5148.9410,
5065.9363,
5028.5740,
5005.9822,
5053.3843,
4920.0184,
4754.1473,
4725.3788,
4711.3235,
4578.6533,
4494.1750,
4807.3336,
5007.1617,
5110.2226,
4925.8579,
4912.5052,
4904.9898,
4801.6200,
4717.3495,
4596.1179,
4540.5664,
4543.1471,
4765.6962,
4635.8954,
4547.1713,
4524.4439])

# --- asse x
x_torso  = np.arange(1, len(torso_loss)+1)
x_hybrid = np.arange(1, len(hybrid_loss)+1)

plt.figure(figsize=(6,4))
plt.plot(x_torso,  torso_loss,  label="Torso baseline")
plt.plot(x_hybrid, hybrid_loss, label="Hybrid GNN", marker='o', ls='--')
plt.xlabel("Epoca")
plt.ylabel("Loss media batch")
plt.title("Pre-training loss: Torso vs Hybrid")
plt.legend()
plt.tight_layout()
plt.show()
