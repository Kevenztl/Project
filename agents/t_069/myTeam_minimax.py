import time, random
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy

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



    def GetScore(self, state, _id):
        return self.game_rule.calScore(state, _id) - self.game_rule.calScore(state, 1 - _id)

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
                if action[2].num_to_floor_line == 0:
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
        best_action = random.choice(actions)
        alternative_actions = []
        for action in actions:
            if not isinstance(action, str):
                if action[2].num_to_floor_line == 0:
                    alternative_actions.append(action)
                elif best_action[2].num_to_floor_line > action[2].num_to_floor_line:
                    best_action = action
        if len(alternative_actions) > 0:
            best_action = random.choice(alternative_actions)
            matched_line = -1

            for action in alternative_actions:
                cur_line = action[2].pattern_line_dest
                if cur_line >= 0 and rootstate.agents[self.id].lines_number[cur_line] + action[2].num_to_pattern_line == cur_line + 1:
                    matched_line = max(matched_line, cur_line)
                    best_action = action

        minimax = Minimax(3, start_time, best_action)
        op_state = deepcopy(rootstate)
        op_actions = self.GetActions(op_state, 1 - self.id)
        op_best_action = minimax.get_best_action(op_state, self, 1 - self.id, False)

        # 模拟对手的最佳行动，并记录对手行动之前的游戏状态
        op_state_before_action = deepcopy(op_state)
        self.DoAction(op_state, op_best_action, 1 - self.id)

        best_action = minimax.get_best_action(rootstate, self, self.id, True)

        # 恢复对手行动之前的游戏状态
        op_state = deepcopy(op_state_before_action)

        return best_action





class Minimax:
    def __init__(self, depth, start_time, best_action):
        self.depth = depth
        self.start_time = start_time
        self.best_random = best_action

    def minimax(self, state, depth, is_maximizing, agent, id, max_time=0.9):
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
                value = self.minimax(next_state, depth - 1, False, agent, id)
                max_value = max(value, max_value)
            return max_value
        else:
            min_value = float('inf')
            for action in actions:
                if time.time() - self.start_time >= max_time:
                    return min_value
                next_state = deepcopy(state)
                agent.DoAction(next_state, action, id)
                value = self.minimax(next_state, depth - 1, True, agent, id)
                min_value = min(value, min_value)
            return min_value

    def get_best_action(self, state, agent, id, is_maximizing, max_time=0.9):
        actions = agent.GetActions(state, id)
        best_value = float('-inf')
        best_action = None
        for action in actions:
            next_state = deepcopy(state)
            agent.DoAction(next_state, action, id)
            value = self.minimax(next_state, self.depth - 1, is_maximizing, agent, id)
            if value > best_value:
                best_value = value
                best_action = action
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= max_time:
                return self.best_random

        return best_action
