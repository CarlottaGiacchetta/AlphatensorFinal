import numpy as np
import matplotlib.pyplot as plt

# --- carica i dati del torso “classico”
gat = np.load("outputs/output_GAT%_j.npy")
gcn = np.load("outputs/outputGCNV2_5007.npy")
line = np.load("outputs/outputHybridLine_4652.npy")
baseline = np.load("outputs/baseline.npy")


# --- asse x
x_gat  = np.arange(1, len(gat)+1)
x_gcn = np.arange(1, len(gcn)+1)
x_line  = np.arange(1, len(line)+1)
x_baseline  = np.arange(1, len(baseline)+1)

plt.figure(figsize=(6,4))
plt.plot(x_gat,  gat,  label="GAT")
plt.plot(x_gcn, gcn, label="HGCN")
plt.plot(x_line, line, label="Hybrid line")
plt.plot(x_baseline, baseline, label="Baseline")
plt.xlabel("Epoca")
plt.ylabel("Loss media batch")
plt.title("Pre-training loss: Torso vs Hybrid")
plt.legend()
plt.tight_layout()
plt.show()
