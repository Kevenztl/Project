from template import Agent
import time,random,heapq, json
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule

from collections import deque

THINKTIME   = 0.9
NUM_PLAYERS = 2
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.5

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)
        self.weight = [0,0,0,0,0,0]
        with open("agents/t_069/RL_weight/weight_init.json",'r',encoding='utf-8')as w:
            self.weight = json.load(w)['weight']
        print(self.weight)

    # Generates actions from this state.
    def GetActions(self, state, _id):
        return self.game_rule.getLegalActions(state, _id)
    
    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action, _id):
        state = self.game_rule.generateSuccessor(state, action, _id)