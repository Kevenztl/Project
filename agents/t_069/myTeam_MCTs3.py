from template import Agent
import time, random
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
from Azul.azul_utils import *

THINKTIME = 0.9
NUM_PLAYER = 2
GAMMA = 0.9
EPS = 0.6

class myAgent(Agent):
    def __init__(self, _id):
        self.id = _id
        self.count = 0
        self.game_rule = GameRule(NUM_PLAYER)

        
        self.vs = dict()
        self.ns = dict()
        self.best_action_s = dict()
        self.expanded_action_s = dict()
        self.start_time = None
        self.THINKTIME = 1  # Set your think time
        self.GAMMA = 0.9  # Set your gamma
        self.EPS = 0.2  # Set your epsilon
        self.game_state = None
        self.actions = None
        self.best_action = None
        self.queue = deque([])
        self.t_root_state = self.TransformState(self.game_state, self.id)
    def GetActions(self, state, _id):
        actions = self.game_rule.getLegalActions(state, _id)
        if len(actions) == 0:
            actions = self.game_rule.getLegalActions(state, NUM_PLAYER)
        return actions

    def DoAction(self, state, action, _id):
        state = self.game_rule.generateSuccessor(state, action, _id)

    def GetScore(self, state, _id):
        return self.game_rule.calScore(state, _id)-self.game_rule.calScore(state,1-_id)

    def GameEnd(self, state):
        for plr_state in state.agents:
            completed_rows = plr_state.GetCompletedRows()
            if completed_rows> 0:
                return True
        return False


    def my_AgentToString(self,agent_id, ps):
        grid_order = [
            ['B', 'Y', 'R', 'K', 'W'],
            ['W', 'B', 'Y', 'R', 'K'],
            ['K', 'W', 'B', 'Y', 'R'],
            ['R', 'K', 'W', 'B', 'Y'],
            ['Y', 'R', 'K', 'W', 'B']
        ]

        desc = f"Agent {agent_id} score {ps.score}\n"

        for i in range(ps.GRID_SIZE):
            if ps.lines_tile[i] != -1:
                tt = ps.lines_tile[i]
                ts = TileToShortString(tt)
                filled = " ".join([ts] * ps.lines_number[i] + ["_"] * (i + 1 - ps.lines_number[i])) + " " * (5 - i - 1) * 2
            else:
                assert ps.lines_number[i] == 0
                filled = "_ " * (i + 1) + " " * (5 - i - 1) * 2

            filled += " ".join([f"{B2S(ps.grid_state[i][j])}/{grid_order[i][j]}" for j in range(5)]) + "\n"

            desc += f"    Line {i+1} {filled}\n"

        desc += "\nFloor line " + "".join(['x ' if i == 1 else '_ ' for i in ps.floor]) + "\n\n"

        return desc


    def my_BoardToString(self,game_state):
        factories_desc = "\n".join([f"Factory {i+1} has {TileDisplayToString(fd)}" for i, fd in enumerate(game_state.factories)])
        
        centre_desc = f"Centre has {TileDisplayToString(game_state.centre_pool)}"
        if not game_state.first_agent_taken:
            centre_desc += " + first agent token (-1)"
        
        return f"{factories_desc}\n{centre_desc}\n"

    def TransformState(self, state, _id):
        return self.my_AgentToString(_id, state.agents[_id]) + self.my_BoardToString(state)

    def ActionInList(self, action, action_list):
        if not isinstance(action, str):
            if not ValidAction(action, action_list):
                return False
        else:
            if action not in action_list:
                return False
        return True


    def FullyExpanded(self, t_state, actions):
        if t_state in self.expanded_action_s:
            return [action for action in actions if not self.ActionInList(action, self.expanded_action_s[t_state])]
        else:
            return actions

    def Select(self,count):
        while len(self.FullyExpanded(self.t_root_state, self.actions)) == 0 and not self.GameEnd(self.game_state):
            if time.time() - self.start_time >= self.THINKTIME:
                print("MCT:", count)
                return self.best_action
            self.t_root_state = self.TransformState(self.game_state, self.agent_id)
            if (random.uniform(0,1) < self.EPS) and (self.t_root_state in self.best_action_s):
                cur_action = self.best_action_s[self.t_root_state]
            else:
                cur_action = random.choice(self.actions)
            self.queue.append((self.t_root_state, cur_action))

            self.DoAction(self.game_state, cur_action, self.agent_id)
            self.actions = self.GetActions(self.game_state, self.agent_id)

    def Expand(self):
        self.t_root_state = self.TransformState(self.game_state, self.agent_id)
        available_actions = self.FullyExpanded(self.t_root_state, self.actions)
        if len(available_actions) != 0:
            action = random.choice(available_actions)
            if self.t_root_state in self.expanded_action_s:
                self.expanded_action_s[self.t_root_state].append(action)
            else:
                self.expanded_action_s[self.t_root_state] = [action]
            self.queue.append((self.t_root_state, action))
            self.DoAction(self.game_state, action, self.agent_id)
            self.actions = self.GetActions(self.game_state, self.agent_id)

    def Simulation(self):
        length = 0
        while not self.GameEnd(self.game_state):
            length += 1
            cur_action = random.choice(self.actions)
            self.DoAction(self.game_state, cur_action, self.agent_id)
            self.actions = self.GetActions(self.game_state, self.agent_id)
        reward = self.GetScore(self.game_state, self.agent_id)
        return reward, length

    def Backpropagate(self, reward, length):
        cur_value = reward * (self.GAMMA ** length)
        while len(self.queue) and time.time() - self.start_time < self.THINKTIME:
            t_state, cur_action = self.queue.pop()
            if t_state in self.vs:
                if cur_value > self.vs[t_state]:
                    self.vs[t_state] = cur_value
                    self.best_action_s[t_state] = cur_action
                self.ns[t_state] += 1
            else:
                self.vs[t_state] = cur_value
                self.ns[t_state] = 1
                self.best_action_s[t_state] = cur_action
            cur_value *= self.GAMMA
        if self.t_root_state in self.best_action_s:
            self.best_action = self.best_action_s[self.t_root_state]

    def MCTS(self):
        count = 0
        while time.time() - self.start_time < self.THINKTIME:
            count += 1
            if self.Select(count):
                return self.best_action
            self.Expand()
            reward, length = self.Simulation()
            self.Backpropagate(reward, length)
        return self.best_action

    def SelectAction(self, actions, game_state):
        self.actions = actions
        self.game_state = game_state
        self.start_time = time.time()
        best_action = random.choice(self.actions)
        alternative_actions = []
        for action in self.actions:
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
                if cur_line >= 0 and self.game_state .agents[self.id].lines_number[cur_line] + action[2].num_to_pattern_line == cur_line+1:
                    matched_line = max(matched_line, cur_line)
                    best_action = action
        if self.count <= 20:
            return best_action
        else:
            self.MCTS()
        return best_action





