import numpy as np
from utils import reset_game, is_zero_tensor, tokens_to_action_tensor, rango_tensore_finale
import torch

class TensorGame:
    def __init__(self, args):
        self.args = args
        self.device = self.args["device"]
        self.initial_state = reset_game(self.args["device"]) #così me lo ritorna subito al posto di crearlo da zero ogni volta
        self.state = reset_game(self.args["device"])
        
        # variabili per il tensor game
        self.T = self.args["T"] # timestamp corrente. immagino zero all'inizio? 
        self.R_limit = self.args["R_limit"] # limitiamo il gioco a R_limit steps
        self.history = [] # la storia delle azioni --> se la appendo ad ogni next_state è inutile inizializzarla già con i zeri per poi eliminarli? 
        self.S = 4 # per strassen
        # resta il problema delle azioni --> che alla fino non ho da enumerare perché faccio autoregressione

    def __repr__(self):
        return "TensorGame"
    
    # questo per resettare tutte le variabili!!
    def get_initial_state(self):
        self.T       = 0
        self.history = []
        self.reward  = 0
        self.state = self.initial_state.clone()
        return self.state
    
    # REMINDER CHE DEVE ESSERE UN TENSORE ACTION
    # e che l'action è un vettore 3*4 con le azioni u, v, w in range [0,1,2] ho fatto il +1 !!!!
    def get_next_state(self, state, action:torch.Tensor):
        assert action.numel() == 3*self.S
        # voglio che il tensore azione venga sottratto al tensore stato, salvo lo stato e la storia delle azioni
        tensore_azione = tokens_to_action_tensor(action, self.S)
        # performo l'azione
        new_state = state - tensore_azione
        return new_state # antiside effects jsut in case
    
    def perform_action(self, action:torch.Tensor):
        assert action.numel() == 3*self.S
        # voglio che il tensore azione venga sottratto al tensore stato, salvo lo stato e la storia delle azioni
        tensore_azione = tokens_to_action_tensor(action, self.S)
        # performo l'azione
        self.state -= tensore_azione
        self.reward -= 1
        self.T +=1
        self.history.append(tensore_azione)
        return self.state
    
    def get_scalar(self):
        # quanto manca al limite R_limit
        remaining = self.R_limit - self.T
        return torch.tensor([remaining], dtype=torch.float32).to(self.device)
    
    def check_win(self, state): #passo solo lo stato perché voglio che contolli solo lo stato
        return is_zero_tensor(state)

    def get_value_and_terminated(self, state, node_num_parents=None):
        if node_num_parents:
            #print("Sono un valore fittizio")
            if self.check_win(state):
                return self.R_limit - 1, True
            elif node_num_parents >= self.R_limit - 1:
                # reward finale: –n_passi - bonus rank
                tmp_reward = node_num_parents + 1
                tmp_reward -= rango_tensore_finale(state)
                return tmp_reward, True
            return 0, False
        else:
            #print("The real thing")
            if self.check_win(state):
                return self.reward, True
            elif self.T >= self.R_limit:
                # reward finale: –n_passi - bonus rank
                self.reward -= rango_tensore_finale(state)
                return self.reward, True
            return 0, False
    
    def get_encoded_state(self, state):
        frames = [state]                                              # stato corrente
        # azioni precedenti (max dim_t‑1), ordine inverso
        for at in reversed(self.history):
            frames.append(at)

        # padding con zeri
        while len(frames) < self.R_limit:
            frames.append(torch.zeros_like(state))

        stack = torch.stack(frames)                             # (dim_t,S,S,S)
        return stack.to(torch.float32).to(self.device)