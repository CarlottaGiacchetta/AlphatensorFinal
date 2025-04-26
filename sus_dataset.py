
import torch
from model import TensorModel
from alpha import AlphaZero
from tensor_game import TensorGame
from dataset import SyntheticDemoDataset, StrassenAugDataset
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True


args = {
    'C': 2,
    'num_searches': 3,
    'num_iterations': 10, # quanti modelli salva fa x iterazioni ed ad ogni iterazione fa y self_play e salva il modello
    'num_selfPlay_iterations': 10,
    'num_epochs': 10, # data la memoria del self-play aggiorna il modello con queste epoche
    'num_epochs_pretrain': 10, # questo parte all'inizio
    'batch_size': 32, #32
    'models_path': 'models',
    'device':'mps',
    'T':0,
    'R_limit' : 8    # da controllare bene R_limit perché forse da qualche parte è sballato
}


gugu = StrassenAugDataset(1, args["R_limit"], "cpu")


print(gugu[-1])

print("\n\n")


gugu = SyntheticDemoDataset(1, args["R_limit"], "cpu")

print(gugu[-1])
print(gugu[-2])
print(len(gugu))
exit()
for gaga in gugu:
    print(gaga)