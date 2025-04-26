import torch
import numpy as np

# -------------- minimal dummy game --------------
class DummyGame:
    def __init__(self, S=4, Tlim=8):
        self.S = S
        self.R_limit = Tlim
        self.history = []
    def get_initial_state(self):
        # stato 4×4×4 tutto zero
        return torch.zeros(self.S, self.S, self.S, dtype=torch.int8)
    def get_encoded_state(self, state):
        # appiattisco + padding a R_limit
        return state.unsqueeze(0).repeat(self.R_limit,1,1,1)
    def get_scalar(self):
        return torch.tensor([self.R_limit], dtype=torch.float32)
    def get_next_state(self, state, action_tokens):
        # non cambia lo stato (per test)
        return state.clone()
    def get_value_and_terminated(self, state, node=None):
        # mai terminale
        return 0.0, False

# -------------- carico il tuo modello --------------
from model import TensorModel  # importa la tua classe

model = TensorModel(
    dim_3d=4, dim_t=8, dim_s=1, dim_c=8,
    n_steps=12, n_logits=3, n_samples=16,
    device="cpu"
).to("cpu")

# -------------- MCTS --------------
from mcts import MCTS      # il file dove hai definito class MCTS

args = {
    'C': 2,
    'num_searches': 10,
    'num_iterations': 8,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'models_path': 'models',
    'device' : 'mps',
    'T' : 0,
    'R_limit' : 7    
}
game = DummyGame(S=4, Tlim=8)
mcts = MCTS(game, args, model)

# -------------- test --------------
state = game.get_initial_state()
action = mcts.search(state)

print("Returned action tokens:", action)
print("Type:", type(action), "Shape:", action.shape)
print("Values ∈ {0,1,2}?", np.unique(action.numpy()))
