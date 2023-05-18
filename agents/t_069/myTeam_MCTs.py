from template import Agent
import time,random,heapq
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule
from typing import Union, List
from collections import deque

THINKTIME   = 0.9
NUM_PLAYERS = 2
GAMMA = 0.9
ALPHA = 0.6
EPSILON = 0.5

class MctNode(object):
    def __init__(self,_id,state,parent,current_action,game_rule):
        self.q_value = 0
        self.parent = parent
        self.visit_times = 0
        self.children: List[MctNode] = []
        self.current_move = current_action
        self.g_state = state
        self.player_id = _id
        self.game_rule = game_rule
        self.actions = self.GetActions(state, _id)
    # Generates actions from this state.
    def GetActions(self, state, _id):
        return self.game_rule.getLegalActions(state, _id)
    
    def isExpanded(self):
        for child in self.children:
            if child.visit_times == 0:
                return False
        return True
    
    def get_unexpanded_child(self):
        for child in self.children:
            if child.visit_times == 0:
                return child
        return None
    def get_best_value_child(self):
        if len(self.children) == 0:
            return None
        best_child = self.children[0]
        for child in self.children:
            if child.q_value > best_child.q_value:
                best_child = child
        return best_child

    # def get_best_value_child(self):
    #     best_child = self.children[0]
    #     for child in self.children:
    #         if child.q_value > best_child.q_value:
    #             best_child = child
    #     return best_child
    #1111111

    def get_best_uct_child(self):
        uct_values = [child.calculate_uct() for child in self.children]
        best_index = uct_values.index(max(uct_values))
        return self.children[best_index]

    # Mean value backward-propergation 
    def update(self):
        self.visit_times += 1
        s1, s2 = 0, 0
        for child in self.children:
            s1 += child.visit_times * child.q_value
            s2 += child.visit_times
        self.q_value = s1 / s2

    def is_round_end(self):
        return not self.g_state.TilesRemaining()

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)
        self.count = 0

    # Generates actions from this state.
    def GetActions(self, state, _id):
        return self.game_rule.getLegalActions(state, _id)
    
    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action, _id):
        # state = self.game_rule.generateSuccessor(state, action, _id)
        return self.game_rule.generateSuccessor(state, action, _id)

    def bestRandomAction(self,actions):
        best_action = random.choice(actions)
        alternative_actions = []

        for action in actions:
            if not isinstance(action,str):
                if action[2].num_to_floor_line == 0:
                    alternative_actions.append(action)
                
                elif action[2].num_to_floor_line < best_action[2].num_to_floor_line:
                    best_action = action
        
        if len(alternative_actions) > 0:
            best_action = random.choice(alternative_actions)

        return best_action
    
    def floor_line_penalty(self,agent_state):
        penalty = 0
        for i in range(len(agent_state.floor)):
            penalty += agent_state.floor[i] * agent_state.FLOOR_SCORES[i]
        return penalty
    

    # MCTs
    # def SelectAction(self, actions, game_state):
    #     best_action = self.bestRandomAction(actions)
        
    #     start_time = time.time()
    #     initial_state = deepcopy(game_state)


    #     return best_action
    
    def selectState(self, node):
        while node and node.isExpanded():
            if node.get_best_value_child():
                node = node.get_best_value_child()
            else:
                break
        return node

    # def isExpanded(self, state):

    def Expand(self, node):
        actions = node.GetActions(node.g_state, node.player_id)
        for action in actions:
            new_state = self.DoAction(deepcopy(node.g_state), action, node.player_id)
            child_node = MctNode(node.player_id, new_state, node, action, self.game_rule)
            node.children.append(child_node)

    # def calculateReward(self,state):
    #     return state.
    # def Simulation():
    # def Simulation(self, state):
    #     state_copy = deepcopy(state.g_state)
    #     while not state.is_round_end():
    #         actions = self.GetActions(state_copy, nstateode.player_id)
    #         random_action = random.choice(actions)
    #         self.DoAction(state_copy, random_action, state.player_id)
    #     reward = self.calculateReward(state_copy)  # You need to implement calculateReward function
    #     return reward
    def Simulation(self, node: MctNode):
        state = deepcopy(node.g_state)
        while state.TilesRemaining():
            for p in state.agents:
                if not state.TilesRemaining():
                    break
                if p.id == self.id:
                    actions = self.GetActions(state,self.id)
                    # my_action = self.SelectAction(actions,state)
                    # self.DoAction(state, my_action, self.id)
                    my_action = self.bestRandomAction(actions)
                    self.DoAction(state, my_action, self.id)
                else:
                    opponent_id = 1-self.id
                    opponent_actions = self.GetActions(state,opponent_id)
                    opponent_action = self.bestRandomAction(opponent_actions)
                    self.DoAction(state,opponent_action,opponent_id)
                # state.ExecuteMove(p.id, player.SelectMove(p.GetAvailableMoves(state), state))

            state.ExecuteEndOfRound()

        agent_reward = {}
        for p in state.agents:
            agent_reward[p.id] = p.score
        reward = agent_reward[self.id]-agent_reward[1-self.id]
        # reward = agent_reward[self.id]
        return reward
        # reward = self.calculateReward(state)
        # return reward

    # def BackPropagation():
    def BackPropagation(self, node, reward):
        print("bp")
        while node is not None:
            node.visit_times += 1
            node.q_value += reward  # Assuming reward is the same for all nodes in the path
            node = node.parent

    def SelectAction(self, actions, game_state):
        self.count += 1
        root = MctNode(self.id, deepcopy(game_state), None, None, self.game_rule)
        start_time = time.time()
        best_action = self.bestRandomAction(actions)

        while time.time() - start_time < THINKTIME:
            if self.count < 20:
                best_action = self.bestRandomAction(actions)
                return best_action
            else:
                v = self.selectState(root)
                if not v.is_round_end():
                    self.Expand(v)
                reward = self.Simulation(v)
                self.BackPropagation(v, reward)
                best_child = root.get_best_value_child()
                best_action = best_child.current_move
        if time.time() - start_time >= THINKTIME:
            print("time out!!!!!!!!!!!!!!!!!!!")
        return best_action

    

