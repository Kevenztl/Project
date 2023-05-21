import time, random
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
import numpy as np
import json
import math
class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.count = 0
        self.game_rule = GameRule(2)
        self.weight = [0, 0, 0, 0, 0, 0]
        with open("agents/t_069/RL_weight/weight.json", "r", encoding='utf-8') as fw:
            self.weight = json.load(fw)['weight']
    def CalQValue(self, state, action,_id):
        """
        Calculates the Q-value of an action, given a state.

        Args:
            state (State): The current game state.
            action (Action): The action to be performed.
            _id (int): The ID of the agent.

        Returns:
            ans (float): The calculated Q-value.
        """
        features = self.CalFeature(state,action,_id)
        if len(features) != len(self.weight):
            return -float('inf')
        else: 
            ans = 0
            for i in range(len(features)):
                ans += features[i] * self.weight[i]
        return ans
    
    def CalFeature(self, state, action, _id):
        """
        Calculates the feature vector for a given state-action pair.

        Args:
            state (State): The current game state.
            action (Action): The action to be performed.
            _id (int): The ID of the agent.

        Returns:
            features (list): The calculated feature vector.
        """
        features = []
        next_state = deepcopy(state)
        self.DoAction(next_state, action, _id)

        # Floor line
        floor_tiles = len(next_state.agents[_id].floor_tiles)
        features.append(floor_tiles / 7)

        # Line 1-5
        for i in range(5):
            if next_state.agents[_id].lines_number[i] == i + 1:
                features.append(1)
            else:
                features.append(0)

        # Feature: Number of completed rows
        completed_rows = sum([1 for row in next_state.agents[_id].grid_state if sum(row) == len(row)])
        features.append(completed_rows / 5)

        # Feature: Number of completed columns
        completed_columns = sum([1 for col in zip(*next_state.agents[_id].grid_state) if sum(col) == len(col)])
        features.append(completed_columns / 5)

        # Feature: Number of completed sets (a set is a collection of all 5 colors in the grid_state)
        grid_state_flat = [item for sublist in next_state.agents[_id].grid_state for item in sublist]
        min_tile_count = min([grid_state_flat.count(tile) for tile in set(grid_state_flat)])
        features.append(min_tile_count / 5)

        return features
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
    def GetScore(self,state, _id):
        next_state = deepcopy(state)
        self.DoAction(next_state, 'ENDROUND', _id)
        oppoent_id = 1 - _id
        return (self.game_rule.calScore(next_state, _id) - self.game_rule.calScore(state, _id)) \
            - (self.game_rule.calScore(next_state, oppoent_id)) - (self.game_rule.calScore(state, oppoent_id))


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
        start_time = time.time()
        self.count += 1
        best_action = self.bestRandomAction(rootstate,actions)
        best_Q_value = -float('inf')

        if len(actions) > 1:
            for action in actions:
                if time.time() - start_time > 0.9:
                    print("timeout")
                    break
                Q_value = self.CalQValue(rootstate, action,self.id)
                if Q_value > best_Q_value:
                    best_Q_value = Q_value
                    best_action = action

        minimax = Minimax(2, start_time, best_action)
        minimax.adjust_depth(rootstate, rootstate.agents, self.id)
        best_action = minimax.get_best_action(rootstate, self, self.id, True)

        return best_action

    # def SelectAction(self, actions, rootstate):
    #     start_time = time.time()
    #     best_action = random.choice(actions)
    #     alternative_actions = []
    #     for action in actions:
    #         if not isinstance(action, str):
    #             if action[2].num_to_floor_line == 0:
    #                 alternative_actions.append(action)
    #             elif best_action[2].num_to_floor_line > action[2].num_to_floor_line:
    #                 best_action = action
    #     if len(alternative_actions) > 0:
    #         best_action = random.choice(alternative_actions)
    #         matched_line = -1

    #         for action in alternative_actions:
    #             cur_line = action[2].pattern_line_dest
    #             if cur_line >= 0 and rootstate.agents[self.id].lines_number[cur_line] + action[2].num_to_pattern_line == cur_line + 1:
    #                 matched_line = max(matched_line, cur_line)
    #                 best_action = action

    #     minimax = Minimax(2, start_time, best_action)

    #     best_action = minimax.get_best_action(rootstate, self, self.id, True)


    #     return best_action




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
        if is_maximizing:
            actions = agent.GetActions(state, id)
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
            actions = agent.GetActions(state, 1 - id)
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
        best_value = float('-inf')
        best_action = None
        alpha = float('-inf')
        beta = float('inf')
        for action in actions:
            next_state = deepcopy(state)
            agent.DoAction(next_state, action, id)

            # 使用Q函数对动作价值进行估计
            q_value = agent.CalQValue(next_state, action, id)

            # 调用minimax函数进行搜索，计算动作的值
            value = self.minimax(next_state, self.depth - 1, False, agent, 1 - id, alpha, beta)

            # 综合Q值和对手得分进行动作价值的计算
            value = q_value - value + 1 / (1 + math.exp(-agent.GetScore(next_state, 1 - id))) * agent.GetScore(next_state, 1 - id)

            if action[2].num_to_floor_line == 0:
                value += 0.5  # 优先选择不放到Floor Line的操作
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= max_time:
                print(1)
                return self.best_random

        return best_action

