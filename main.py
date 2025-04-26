
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
    'device':'cuda',
    'T':0,
    'R_limit' : 8    # da controllare bene R_limit perché forse da qualche parte è sballato
}

#!TODO: da guardare anche la ucb perché non mi quadra il 1- 

game = TensorGame(args)

model = TensorModel(dim_3d=4, dim_t=8, dim_s=1, dim_c=16, n_steps=12, n_logits=3, n_samples=4, device=args["device"]).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
alphaZero = AlphaZero(model, optimizer, game, args)

alphaZero.learn(
    sup_datasets=[
        SyntheticDemoDataset(50, args["R_limit"], "cpu"),
        StrassenAugDataset(500, args["R_limit"], "cpu"),
    ],
    pretrain_epochs=args["num_epochs_pretrain"],
)

