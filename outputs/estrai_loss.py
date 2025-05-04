import re
import numpy as np

name = "outputHybridLine_4652"
# Sostituisci 'log.txt' col percorso al tuo file
with open(name+".txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

loss_values = []

# Cerca "loss medio" seguito da due punti e un numero
for line in lines:
    match = re.search(r"loss\s*medio\s*[:=]\s*([0-9]*\.?[0-9]+)", line, re.IGNORECASE)
    if match:
        loss = float(match.group(1))
        loss_values.append(loss)

loss_array = np.array(loss_values)

# Salva come file .npy
np.save(name+".npy", loss_array)

print(f"Salvati {len(loss_array)} valori in '{name}.npy'")
