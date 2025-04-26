from connect.mcts_connect import MCTS
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from tqdm import trange
import os


class AlphaZero:

    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)


    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True: # fino a quando non finisce
            neutral_state = self.game.change_perspective(state, player) # fa state * player
            
            # action_probs è quello che mi ritorna il mcts che erano le action probs per quel determinato stato (--> fatti con i visit counts)
            action_probs = self.mcts.search(neutral_state)
            
            memory.append((neutral_state, action_probs, player))
            temperature_action_probs = action_probs ** (1 / self.args['temperature']) # una riscalata dei valori
            temperature_action_probs = temperature_action_probs / np.sum(temperature_action_probs) # RINORMALIZZAZIONE POST SCALATA PERCHè NON SOMMANO A UNO
            action = np.random.choice(self.game.action_size, p=temperature_action_probs) # scelta randomica pesata

            # lo stato cambia! 
            state = self.game.get_next_state(state, action, player)
            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                # questo serve solo per far capire alla rete chi gioca e che risolutati ci sono e per cambiare neutral_state --> encoded(neutral_state) 
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    # la vittoria dipende da che play gioca
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))

                return returnMemory
            
            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])] # prendiamo un batch
            state, policy_targets, value_targets = zip(*sample)

            #reshape e cambio
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        
        # itera per il numero di iterazioni
        # ad ogni iterazione crea da zero una memoria di selfPlay giocate che si è fatto da solo
        for iteration in range(self.args["num_iterations"]):
            memory = []

            self.model.eval() # non cambio i pesi 
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()
        
            self.model.train() # cambio i pesi
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            if not os.path.exists(self.args["models_path"]):
                os.makedirs(self.args["models_path"])
            
            torch.save(self.model.state_dict(), f"{self.args['models_path']}/model_{iteration}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"{self.args['models_path']}/optimizer_{iteration}_{self.game}.pt")
        