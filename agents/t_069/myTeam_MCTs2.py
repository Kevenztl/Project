from template import Agent
import time, random
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
from Azul.azul_utils import *

THINKTIME = 0.9
NUM_PLAYER = 2
GAMMA = 0.5
EPS = 0.6

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
            # MCT
            vs = dict()
            ns = dict()
            best_action_s = dict()
            expanded_action_s = dict()
            t_root_state = 'r'
            count = 0

            def FullyExpanded(t_state, actions):
                if t_state in expanded_action_s:
                    available_actions = []
                    for action in actions:
                        if not self.ActionInList(action, expanded_action_s[t_state]):
                            available_actions.append(action)
                    return available_actions
                else:
                    return actions

            while time.time() - start_time < THINKTIME:
                count += 1
                state = deepcopy(game_state)
                new_actions = actions
                # t_cur_state = t_root_state
                queue = deque([])
                reward = 0
                t_cur_state = self.TransformState(state, self.id)
                # Select
                while len(FullyExpanded(t_cur_state, new_actions)) == 0 and not self.GameEnd(state):
                    if time.time() - start_time >= THINKTIME:
                        print("MCT:", count)
                        return best_action
                    t_cur_state = self.TransformState(state, self.id)
                    if (random.uniform(0,1) < EPS) and (t_cur_state in best_action_s):
                        cur_action = best_action_s[t_cur_state]
                    else:
                        cur_action = random.choice(new_actions)
                    # !!!
                    queue.append((t_cur_state, cur_action))

                    next_state = deepcopy(state)
                    self.DoAction(next_state, cur_action, self.id)
                    new_actions = self.GetActions(next_state, self.id)
                    state = next_state
                #oppo move
                    op_actions = self.GetActions(next_state,1-self.id)
                    op_action = op_actions[0]
                    t_cur_state = self.TransformState(state, self.id)
                    for action in op_actions:
                        if not isinstance(action,str):
                            if action[2].num_to_floor_line == 0:
                                op_action = action
                                break
                            elif op_action[2].num_to_floor_line>action[2].num_to_floor_line:
                                op_action = action
                    self.DoAction(next_state,op_action,1-self.id)
                    # t_cur_state = self.TransformState(state,self.id)
                    new_actions = self.GetActions(next_state,self.id)
                    state = next_state
                # Expand
                t_cur_state = self.TransformState(state, self.id)
                available_actions = FullyExpanded(t_cur_state, new_actions)
                if len(available_actions) == 0:
                    continue
                else:
                    action = random.choice(available_actions)
                if t_cur_state in expanded_action_s:
                    expanded_action_s[t_cur_state].append(action)
                else:
                    expanded_action_s[t_cur_state] = [action]
                # !!!
                queue.append((t_cur_state, action))
                # t_cur_state = self.TransformState(state, self.id)
                next_state = deepcopy(state)
                self.DoAction(next_state, action, self.id)

                new_actions = self.GetActions(next_state, self.id)
                state = next_state

                # Simulation
                length = 0
                while not self.GameEnd(state):
                    length += 1
                    if time.time() - start_time >= THINKTIME:
                        print("MCT",count)
                        return best_action
                    cur_action = random.choice(new_actions)
                    next_state = deepcopy(state)
                    self.DoAction(next_state,cur_action,self.id)
                    op_actions = self.GetActions(next_state,1-self.id)
                    op_action = op_actions[0]
                    t_cur_state = self.TransformState(state, self.id)
                    for action in op_actions:
                        if not isinstance(action,str):
                            if action[2].num_to_floor_line == 0:
                                op_action = action
                                break
                            elif op_action[2].num_to_floor_line > action[2].num_to_floor_line:
                                op_action = action
                    self.DoAction(next_state,op_action,1-self.id)

                    new_actions = self.GetActions(next_state,self.id)
                    state = next_state
                reward = self.GetScore(state, self.id)

                # Backpropagate
                cur_value = reward * (GAMMA ** length)
                while len(queue) and time.time() - start_time <THINKTIME:
                    t_state, cur_action = queue.pop()
                    if t_state in vs:
                        if cur_value > vs[t_state]:
                            vs[t_state] = cur_value
                            best_action_s[t_state] = cur_action
                        ns[t_state] += 1
                    else:
                        vs[t_state] = cur_value
                        ns[t_state] = 1
                        best_action_s[t_state] = cur_action
                    cur_value *= GAMMA
                if t_root_state in best_action_s:
                    best_action = best_action_s[t_root_state]

        return best_action
