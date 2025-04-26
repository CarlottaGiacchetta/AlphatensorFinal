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
        root = Node(self.game, self.args, state, visit_count=1, numb_parents=0)

        # ricorda:
        # stato_gioco e scalari sono gli input “batchati” per fwd_infer.
        # aa = campioni di azione (16 possibili sequenze di 12 token). possiamo aumentarne cambiando n_samples
        # pp = probabilità di ciascuna di quelle sequenze.
        # qq = valutazione (value) predetta dal modello per lo stato iniziale che al momento non ci serve
        stato_gioco = self.game.get_encoded_state(state).unsqueeze(0).to(self.model.device)
        scalari = self.game.get_scalar().unsqueeze(0).to(self.model.device)
        aa, pp, qq = self.model.fwd_infer(stato_gioco, scalari)
        
        # qui aa.shape=(1, N, 12), pp.shape=(1,N)
        candidates = aa[0]    # (N,12)
        priors     = pp[0].cpu().numpy()
        priors    /= priors.sum()

        # 4) espando la radice con quei N candidates
        for tok_vec, p in zip(candidates, priors):
            child_state = self.game.get_next_state(state.clone(), tok_vec)
            child = Node(self.game, self.args, child_state, parent=root, action_taken=tok_vec, visit_count=0, numb_parents=root.numb_parents+1)  # vettore di 12 token prior=p,
            root.children.append(child)

        for search in range(self.args['num_searches']): # sarebbe quindi quanti nod
            
            # ad ogni search resetta il nodo a quello di radice e scende di nuovo al primo (poi secondo, terzo, ecc) best e lo espande
            node = root

            while node.is_fully_expanded(): # questo arriva fino alla MIGLIOR foglia
                node = node.select() # node.select prende il best child
            # adesso node è la miglior foglia

            # value è 1 se vinciamo, altrimenti 0, is_terminal è bool se si vince e si pareggia è 1 altrimenti 0
            node.state = self.game.get_next_state(node.state, node.action_taken)
            get_reward, done = self.game.get_value_and_terminated(node.state, node_num_parents=node.numb_parents) #! TODO: anche quiesta da cambiare perché R_limit deve essere accessibile anche in via fittizzia al nodo (da contare quanti genitori ha?)

            # prendiamo il miglior nodo, vediamo la policy e il value e espandiamo
            if not done:
                stato_gioco = self.game.get_encoded_state(node.state).unsqueeze(0).to(self.model.device)
                scalari = self.game.get_scalar().unsqueeze(0).to(self.model.device)
                aa, pp, qq = self.model.fwd_infer(stato_gioco, scalari)

                # qui aa.shape=(1, N, 12), pp.shape=(1,N)
                candidates = aa[0]    # (N,12)
                priors     = pp[0].cpu().numpy()
                priors    /= priors.sum()

                for tok_vec, p in zip(candidates, priors):
                    child_state = self.game.get_next_state(node.state.clone(), tok_vec)
                    child = Node(self.game, self.args, child_state, parent=node, action_taken=tok_vec.clone(), prior=p, visit_count=0, numb_parents=node.numb_parents+1)  # vettore di 12 token prior=p,
                    node.children.append(child)
            
            # back propagiamo fino alla radice questo valore
            node.backpropagate(get_reward)

        best = max(root.children, key=lambda c: c.visit_count)
        return best.action_taken