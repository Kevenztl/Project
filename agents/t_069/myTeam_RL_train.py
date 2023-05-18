from template import Agent
import time,random,heapq
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

    def GetAction(self, state, _id):
        actions = self.game_rule.getLegalActions(state,_id)
