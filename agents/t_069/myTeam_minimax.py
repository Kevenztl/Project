import time, random
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
import numpy as np
import math
class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.count = 0
        self.game_rule = GameRule(2)

    def GetActions(self, state, _id):
        actions = self.game_rule.getLegalActions(state, _id)
        if len(actions) == 0:
            actions = self.game_rule.getLegalActions(state, 2)
        return actions

    # def DoAction(self, state, action, _id):
    #     new_state = deepcopy(state)
    #     self.game_rule.generateSuccessor(new_state, action, _id)
    #     return new_state
    # def DoAction(self, state, action, _id):
    #     state = self.game_rule.generateSuccessor(state, action, _id)
    def DoAction(self, state, action, _id):
        new_state = deepcopy(state)
        self.game_rule.generateSuccessor(new_state, action, _id)
        return new_state



    # def GetScore(self, state, _id):
    #     return self.game_rule.calScore(state, _id) - self.game_rule.calScore(state, 1 - _id)
    def GetScore(self, state, _id):
        # base_score = self.game_rule.calScore(state, _id) - self.game_rule.calScore(state, 1 - _id)
        base_score = self.game_rule.calScore(state, _id)
        # # tile_factor = sum([line.count(1) for line in state.agents[_id].grid_state])
        # tile_factor = np.count_nonzero(state.agents[_id].grid_state == 1)
        tile_factor = np.count_nonzero(state.agents[_id].grid_state == 1)
        

        return base_score+tile_factor


    def GameEnd(self, state):
        for plr_state in state.agents:
            completed_rows = plr_state.GetCompletedRows()
            if completed_rows > 0:
                return True
        return False

    def bestRandomAction(self, state, actions):
        best_action = random.choice(actions)
        game_state = deepcopy(state)
        alternative_actions = []
        for action in actions:
            if not isinstance(action, str):
                if action[2].num_to_floor_line  ==0:
                    alternative_actions.append(action)
                elif best_action[2].num_to_floor_line > action[2].num_to_floor_line:
                    best_action = action
        if len(alternative_actions) > 0:
            best_action = random.choice(alternative_actions)
            matched_line = -1

            for action in alternative_actions:
                cur_line = action[2].pattern_line_dest
                if cur_line >= 0 and game_state.agents[self.id].lines_number[cur_line] + action[2].num_to_pattern_line == cur_line + 1:
                    matched_line = max(matched_line, cur_line)
                    best_action = action
        return best_action
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        best_action = actions[0]
        best_score = float('-inf')

        for action in actions:
            next_state = self.DoAction(deepcopy(rootstate), action, self.id)
            score = self.GetScore(next_state, self.id)
            if score > best_score:
                best_score = score
                best_action = action

        minimax = Minimax(2, start_time, best_action)
        minimax.adjust_depth(rootstate, rootstate.agents, self.id)
        best_action = minimax.get_best_action(rootstate, self, self.id, True)

        return best_action





class Minimax:
    def __init__(self, depth, start_time, best_action):
        self.depth = depth
        self.start_time = start_time
        self.best_random = best_action
    def adjust_depth(self, state, agent, id):
        completed_rows = sum([agent[id].GetCompletedColumns() for id in range(2)])
        if completed_rows < 2:
            self.depth = 2
        elif completed_rows < 4:
            self.depth = 3
        else:
            self.depth = 4
            
    def minimax(self, state, depth, is_maximizing, agent, id, alpha, beta, max_time=0.9):
        if depth == 0 or agent.GameEnd(state):
            return agent.GetScore(state, id)

        actions = agent.GetActions(state, id)
        if is_maximizing:
            max_value = float('-inf')
            for action in actions:
                if time.time() - self.start_time >= max_time:
                    return max_value
                next_state = deepcopy(state)
                agent.DoAction(next_state, action, id)
                value = self.minimax(next_state, depth - 1, False, agent, id, alpha, beta)
                max_value = max(value, max_value)
                alpha = max(alpha, max_value)
                if beta <= alpha:
                    break
            return max_value
        else:
            min_value = float('inf')
            for action in actions:
                if time.time() - self.start_time >= max_time:
                    return min_value
                next_state = deepcopy(state)
                agent.DoAction(next_state, action, id)
                value = self.minimax(next_state, depth - 1, True, agent, id, alpha, beta)
                min_value = min(value, min_value)
                beta = min(beta, min_value)
                if beta <= alpha:
                    break
            return min_value

    def get_best_action(self, state, agent, id, is_maximizing, max_time=0.9):
        actions = agent.GetActions(state, id)
        actions = sorted(actions, key=lambda action: agent.GetScore(agent.DoAction(deepcopy(state), action, id), id), reverse=True)
        best_value = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')
        for action in actions:
            next_state = deepcopy(state)
            agent.DoAction(next_state, action, id)

            # 考虑对手的得分
            opponent_value = self.minimax(next_state, self.depth - 1, not is_maximizing, agent, 1 - id, alpha, beta)
            score = agent.GetScore(next_state, id)
            opponent_score = agent.GetScore(next_state, 1 - id)
            # value = agent.GetScore(next_state, id) -  1 / (1 + math.exp(-opponent_value)) * opponent_value
            value = score -  1 / (1 + math.exp(-opponent_score)) * opponent_score 

            if action[2].num_to_floor_line ==0:
                value += 0.5  # 优先选择不放到Floor Line的操作

            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= max_time:
                return self.best_random

        return best_action

