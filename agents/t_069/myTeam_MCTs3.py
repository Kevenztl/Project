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
class MCTS:
    def __init__(self, agent,_id):
        self.agent = agent
        self.id = _id
        self.vs = dict()
        self.ns = dict()
        self.best_action_s = dict()
        self.expanded_action_s = dict()

    def FullyExpanded(self, t_state, actions):
        if t_state in self.expanded_action_s:
            available_actions = []
            for action in actions:
                if not self.agent.ActionInList(action, self.expanded_action_s[t_state]):
                    available_actions.append(action)
            return available_actions
        else:
            return actions

    def select(self, t_state, actions, state, queue, start_time):
        while len(self.FullyExpanded(t_state, actions)) == 0 and not self.agent.GameEnd(state):
            if time.time() - start_time >= THINKTIME:
                print("MCT")
                return None,None,None
            t_cur_state = self.agent.TransformState(state, self.id)
            if (random.uniform(0,1) < EPS) and (t_cur_state in self.best_action_s):
                cur_action = self.best_action_s[t_cur_state]
            else:
                cur_action = random.choice(actions)
            queue.append((t_cur_state, cur_action))
            next_state = deepcopy(state)
            self.agent.DoAction(next_state, cur_action, self.id)
            actions = self.agent.GetActions(next_state, self.id)
            state = next_state
        return state, actions,queue

    def expand(self, actions, state, queue):
        t_cur_state = self.agent.TransformState(state, self.id)
        available_actions = self.FullyExpanded(t_cur_state, actions)
        if len(available_actions) != 0:
            action = random.choice(available_actions)
            if t_cur_state in self.expanded_action_s:
                self.expanded_action_s[t_cur_state].append(action)
            else:
                self.expanded_action_s[t_cur_state] = [action]
            queue.append((t_cur_state, action))
            next_state = deepcopy(state)
            self.agent.DoAction(next_state, action, self.id)
            actions = self.agent.GetActions(next_state, self.id)
            state = next_state
        return state, actions,queue

    def simulate(self, state, new_actions,start_time):
        length = 0
        while not self.agent.GameEnd(state):
            length += 1
            if time.time() - start_time >= THINKTIME:
                print("MCT")
                return None,None
            cur_action = random.choice(new_actions)
            next_state = deepcopy(state)
            self.agent.DoAction(next_state,cur_action,self.id)
            op_actions = self.agent.GetActions(next_state,1-self.id)
            op_action = op_actions[0]
            for action in op_actions:
                if not isinstance(action,str):
                    if action[2].num_to_floor_line == 0:
                        op_action = action
                        break
                    elif op_action[2].num_to_floor_line > action[2].num_to_floor_line:
                        op_action = action
            self.agent.DoAction(next_state,op_action,1-self.id)
            new_actions = self.agent.GetActions(next_state,self.id)
            state = next_state
        reward = self.agent.GetScore(state, self.id)
        return reward, length

    def backpropagate(self, reward, length, queue, start_time):
        cur_value = reward * (GAMMA ** length)
        while len(queue) != 0 and time.time() - start_time < THINKTIME:
            t_state, cur_action = queue.pop()
            if t_state in self.vs:
                if cur_value > self.vs[t_state]:
                    self.vs[t_state] = cur_value
                    self.best_action_s[t_state] = cur_action
                self.ns[t_state] += 1
            else:
                self.vs[t_state] = cur_value
                self.ns[t_state] = 1
                self.best_action_s[t_state] = cur_action
            cur_value *= GAMMA
    




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

    def GetScore(self, state, _id):
        return self.game_rule.calScore(state, _id)-self.game_rule.calScore(state,1-_id)

    def GameEnd(self, state):
        for plr_state in state.agents:
            completed_rows = plr_state.GetCompletedRows()
            if completed_rows> 0:
                return True
        return False

    def TransformState(self, state, _id):
        return AgentToString(_id, state.agents[_id]) + BoardToString(state)

    def ActionInList(self, action, action_list):
        if not isinstance(action, str):
            if not ValidAction(action, action_list):
                return False
        else:
            if action not in action_list:
                return False
        return True


    def SelectAction(self, actions, game_state):
        start_time = time.time()
        self.count += 1
        # hand code?? hard code
        best_action = random.choice(actions)
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
        if self.count <= 20:
            return best_action
        else:
            mcts = MCTS(self,self.id)
            t_root_state = self.TransformState(game_state, self.id)
            while time.time() - start_time < THINKTIME:
                # count += 1
                state = deepcopy(game_state)
                new_actions = actions
                # t_cur_state = t_root_state
                queue = deque([])
                reward = 0
                t_cur_state = self.TransformState(state, self.id)
                state, new_actions,queue = mcts.select(t_cur_state, new_actions, state, queue, start_time)
            
                if not state and not new_actions:
                    return best_action
                state, new_actions,queue = mcts.expand(new_actions, state, queue)
                if not state and not new_actions:
                    return best_action
                reward, length = mcts.simulate(state,new_actions, start_time)
                if not reward and not length:
                    return best_action
                mcts.backpropagate(reward, length, queue, start_time)
                if t_root_state in mcts.best_action_s:
                    best_action = mcts.best_action_s[t_root_state]

        return best_action
    






