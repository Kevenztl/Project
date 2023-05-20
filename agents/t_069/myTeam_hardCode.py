from template import Agent
import time,random,heapq
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule

from collections import deque

THINKTIME   = 0.9
NUM_PLAYERS = 2


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)

    # Generates actions from this state.
    def GetActions(self, state, _id):
        return self.game_rule.getLegalActions(state, _id)
    
    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action, _id):
        state = self.game_rule.generateSuccessor(state, action, _id)

    def bestRandomAction(self,state,actions):
        best_action = random.choice(actions)
        game_state = deepcopy(state)
        alternative_actions = []
        for action in actions:
            if not isinstance(action, str):
                if action[2].num_to_floor_line == 0:
                    alternative_actions.append(action)
                elif best_action[2].num_to_floor_line > action[2].num_to_floor_line:
                    best_action = action
        if (len(alternative_actions) > 0):
            best_action = random.choice(alternative_actions)
            matched_line = -1

            for action in alternative_actions:
                cur_line = action[2].pattern_line_dest
                if cur_line >= 0 and game_state.agents[self.id].lines_number[cur_line] + action[2].num_to_pattern_line == cur_line+1:
                    matched_line = max(matched_line, cur_line)
                    best_action = action
                    
        return best_action
    
    def SelectAction(self,actions,rootstate):
        best_action = self.bestRandomAction(rootstate,actions)
        return best_action







