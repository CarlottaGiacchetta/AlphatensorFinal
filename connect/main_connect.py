
import torch
torch.manual_seed(0)
from model import ResNet
from connect.alpha_connect import AlphaZero
import numpy as np
from connect.mcts_connect import MCTS
from connect.connect_four import ConnectFour




args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 500,
    'num_parallel_games': 100,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3,
    'models_path': 'models'    
}

game = ConnectFour()

device = "mps"

model = ResNet(game, 9, 128, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()





game = ConnectFour()
player = 1

args = {
    'C': 2,
    'num_searches': 600,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3,
    'models_path': 'models'
}


model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load(f"{args['models_path']}/model_7_ConnectFour.pt", map_location=device))
model.eval()

mcts = MCTS(game, args, model)
state = game.get_initial_state()
while True:
    print(state)
    
    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
            
    else:
        neutral_state = game.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
        
    state = game.get_next_state(state, action, player)
    
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = game.get_opponent(player)