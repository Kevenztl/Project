from template import Agent
import time, random, math
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
from Azul.azul_utils import *
import math

THINKTIME = 0.9
NUM_PLAYER = 2
GAMMA = 0.9
EPS = 0.6

class MctNode:
    def __init__(self,state,parent,_id,agent):
        self.state = state
        self.children = []
        self.parent = parent
        self.id =_id
        self.agent = agent
        self.num_visits = 0
        self.total_reward = 0

    def SimplifyAction(self,actions):
        answer = []
        for action in actions:
            if len(actions) > 50:
                 if not isinstance(action,str):
                    if action[2].pattern_line_dest > 2 and action[2].num_to_pattern_line == 1:
                        continue
            
            if len(actions) > 30:
                 if not isinstance(action,str):
                     if action[2].num_to_floor_line == action[2].number:
                        continue
            answer.append(action)
        return answer

    def ExpandChildren(self, actions = None):
        opponent_id = 1 - self.id

        actions = self.SimplifyAction(actions)

        if actions is None:
            actions = self.agent.GetActions(self.state,self.id)

        for action in actions:
            next_state = deepcopy(self.state)
            self.agent.DoAction(self.state,action,self.id)
            self.children.append(MctNode(next_state,self,opponent_id,self.agent))
    
    def UCB1(self):
        if self.num_visits == 0:
            return float('inf')  # To handle the "cold start" problem
        else:
            total_visits = sum([child.num_visits for child in self.parent.children])
            return self.total_reward / self.num_visits + math.sqrt(2 * math.log(total_visits) / self.num_visits)

class Mcts:
    def __init__(self, _id, game_state, actions,agent):
        self.id = _id
        self.game_state = game_state
        self.actions = actions
        self.agent = agent

        # Root
        self.root_node = MctNode(game_state,None,self.id,self.agent)
        self.root_node.ExpandChildren(actions)

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

    def Select(self):
        """ Select the leaf node with the highest UCB to expand"""
        node = self.root_node
        while len(node.children) > 0:
            node = self.findBestChildNodeUCT(node)
        return node

    def findBestChildNodeUCT(self, node):
        """ Find the best child for the given node based on UCT """
        # Choose the child of the node that has the maximum UCT value
        best_score = -float('inf')
        best_children = []

        # Calculate total visit count of all children
        total_visit_count = sum(child.visited_count for child in node.children)
        log_total_visit_count = math.log(total_visit_count)

        for child in node.children:
            # Calculate the UCT value
            exploit = child.total_reward[child.id] / child.visited_count
            explore = math.sqrt(2.0 * log_total_visit_count / child.visited_count)
            uct_value = exploit + self.C_puct * explore

            if uct_value == best_score:
                best_children.append(child)
            elif uct_value > best_score:
                best_children = [child]
                best_score = uct_value
        return random.choice(best_children)
    
    def Expand(self,node):
        node.ExpandChildren()
        return node.children
    
    def Simulation(self, child):
        gs_copy = deepcopy(child.state)
        current_player_id = child.id
        move_count = 0
        while gs_copy.TilesRemaining():
            # Update move
            plr_state = gs_copy.players[current_player_id]
            moves = plr_state.GetAvailableMoves(gs_copy)
            # Strategy for simulation the future
            selected = self.bestRandomAction(plr_state, moves)
            gs_copy.ExecuteMove(current_player_id, selected)
            # Change to the opponent
            current_player_id = 1 - current_player_id
            move_count += 1
        # reward calculation can be improved
        reward0 = gs_copy.players[0].ScoreRound()[0] + CalculateFutureReward(gs_copy, 0)
        reward1 = gs_copy.players[1].ScoreRound()[0] + CalculateFutureReward(gs_copy, 1)
        return [reward0, reward1], move_count
    def BackPropergation(self):

class myAgent(Agent):
    def __init__(self, _id):
        self.id = _id
        self.count = 0
        self.game_rule = GameRule(NUM_PLAYER)

    def GetActions(self, state, _id):
        actions = self.game_rule.getLegalActions(state, _id)
        if len(actions) == 0:
            actions = self.game_rule.getLegalActions(state, NUM_PLAYER)
        return actions

    def DoAction(self, state, action, _id):
        state = self.game_rule.generateSuccessor(state, action, _id)

    def SelectAction(self, actions, game_state):
        return super().SelectAction(actions, game_state)