import torch
from node import Node
import numpy as np




class MCTS:
    def __init__(self, game, args, model):

        self.game = game
        self.args = args
        self.model = model


    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        # riscaliamo la policy per questa formula
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        root.expand(policy) # data la radice genera tutti i childs

        for search in range(self.args['num_searches']): # sarebbe quindi quanti nod
            
            # ad ogni search resetta il nodo a quello di radice e scende di nuovo al primo (poi secondo, terzo, ecc) best e lo espande
            node = root

            while node.is_fully_expanded(): # questo arriva fino alla MIGLIOR foglia
                node = node.select() # node.select prende il best child

            # adesso node è la miglior foglia

            # value è 1 se vinciamo, altrimenti 0, is_terminal è bool se si vince e si pareggia è 1 altrimenti 0
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)  # cambio il valore

            # prendiamo il miglior nodo, vediamo la policy e il value e espandiamo
            if not is_terminal:
                policy, value = self.model(torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                # non riscaliamo la policy in questo giro? BHO vedi sopra
                # policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)

                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item() #perché item() ? 
                node.expand(policy)
            
            # back propagiamo fino alla radice questo valore
            node.backpropagate(value) 


        action_probs = np.zeros(self.game.action_size) # faccio un array di zero grande come il numero di azioni disponibili
        for child in root.children:
            # ricordo che action_taken è l'indice dell'azione nella policy  e il visit count e quante volte sono andato giuù per quel nodo figlio
            action_probs[child.action_taken] = child.visit_count

        action_probs /= np.sum(action_probs) # normalizzazione

        return action_probs # ritorniamo le action probs per quel determinato stato