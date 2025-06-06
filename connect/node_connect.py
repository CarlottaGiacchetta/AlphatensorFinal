import numpy as np
import math

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self): # prospettiva del nodo in cui sceglie qual'è il miglior suo nodo figlio --> questa comunque è una ricerca shallow!! sono il primo layer viene considerato
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2 # da ricordare di controllare la forumula originale perché quel 1- non mi torna
        
        #combina quante volte ho visitato me stesso e il child che sto cercando di calcolare l'ucb
        return q_value + self.args["C"] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):

        for action, prob in enumerate(policy): 
            # per ogni azione e prob relativa all'azione, se la prob e > 0  
            # crea un figlio basandolo sul parent, fai next action e come il min-max tree devi cambiare prospettiva
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1) # cambio di prospettiva

                child = Node(self.game, self.args, child_state, self, action_taken=action, prior=prob)
                self.children.append(child)

        return child # --> necessario ritornare il children?? così in teoria io ritorno l'ultimo
    
    def backpropagate(self, value):
        
        self.value_sum += value # sarebbe la sexiness del nodo stesso
        self.visit_count += 1 # quante volte ho visitato questo nodo in particolare

        value = self.game.get_opponent_value(value)

        if self.parent is not None:
            self.parent.backpropagate(value) # il value sarebbe il valore trovato nel child che viene backpropagato fino alla radice 