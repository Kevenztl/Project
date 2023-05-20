import time, random
import Azul.azul_utils as utils
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
import numpy as np
import math
from .myTeam_MCTsnew import MCTS
from .myTeam_hardCode import myAgent 
from Azul.azul_utils import *
NUM_PLAYER = 2
random = myAgent
class myAgent():
    def __init__(self, _id):
        self.id = _id
        self.count = 0
        self.game_rule = GameRule(NUM_PLAYER)
    def bestRandomAction(self,game_state,actions,id):
        """
        Chooses a random action from the possible actions with consideration for actions that would not result in negative points. 
        This function implements a complex selection strategy:
        1. It finds the missing tiles on each line (gaps).
        2. It tries to fill these gaps.
        3. It calculates a score for each action taking into account the state after the action and the state before the action.
        4. It also calculates the availability of each tile type in the factories.
        5. It then selects actions that would ensure having enough tiles for completion.
        6. It avoids actions that would result in penalties.
        7. Finally, it ranks actions based on the score calculated previously, selecting the action with the highest score.

        Args:
            game_state (State): The current game state.
            actions (list): A list of possible actions.

        Returns:
            best_action (Action): The chosen action based on the strategy explained above.

        Functions:
            getTileGap: Returns a dictionary with tile colour as keys and two lists as values. 
                The first list contains the number of missing tiles for each line, and the second list contains the line numbers.

            fillHole: Returns a list of actions that can fill the gaps on the game board.

            GetScore: Returns the difference in scores of the agent and the opponent after performing the 'ENDROUND' action.

            getFactoryStatistics: Returns a dictionary with the count of each tile type in all the factories and the centre pool.

            getEnoughTileActions: Returns actions that ensure having enough tiles of the same colour to complete a line.

            getNoPenality: Returns actions that will not result in a penalty (no tiles are sent to the floor line).

            sortActions: Returns a list of actions sorted in descending order based on the score returned by GetScore function.
        """
        def getTileGap(game_state):
            tile_gap = {}
            for each_line in range(5):
                tile_colour = game_state.agents[id].lines_tile[each_line]
                tile_number = game_state.agents[id].lines_number[each_line]
                if tile_number != 0 and tile_number != each_line + 1:
                    missing_num = each_line + 1 - tile_number
                    if tile_colour not in tile_gap:
                        tile_gap[tile_colour] = [[missing_num],[each_line]]
                    else:
                        tile_gap[tile_colour][0].append(missing_num)
                        tile_gap[tile_colour][1].append(each_line)
            return tile_gap

        def fillHole(actions, tile_gap):
            fill_actions = []
            for action in actions:
                if not isinstance(action,str):
                    if action[2].tile_type in tile_gap.keys():
                        if action[2].number >= min(tile_gap[action[2].tile_type][0]):
                            if action[2].pattern_line_dest in tile_gap[action[2].tile_type][1]:
                                fill_actions.append(action)
            return fill_actions

        def GetScore(state, _id):
            next_state = deepcopy(state)
            self.DoAction(next_state, 'ENDROUND', _id)
            oppoent_id = 1 - id
            return (self.game_rule.calScore(next_state, _id) - self.game_rule.calScore(state, _id)) \
                - (self.game_rule.calScore(next_state, oppoent_id)) - (self.game_rule.calScore(state, oppoent_id))

        def getFactoryStatistics(game_state):
            tile_statistics = {}
            for tile in utils.Tile:
                tile_statistics[tile] = 0
            factories = game_state.factories
            for factory in factories:
                for tile in utils.Tile:
                    tile_statistics[tile] += factory.tiles[tile]
            for tile in utils.Tile:
                tile_statistics[tile] += game_state.centre_pool.tiles[tile]
            return tile_statistics

        def getEnoughTileActions(actions, game_state, tile_statistics):
            enough_action = []
            for action in actions:
                if not isinstance(action,str):
                    current_colour = action[2].tile_type
                    current_num = game_state.agents[id].lines_number[action[2].pattern_line_dest]
                    gap_num = action[2].pattern_line_dest + 1 - current_num
                    if tile_statistics[current_colour] - gap_num >= 2:
                        enough_action.append(action)
            return enough_action
        def getNoPenality(actions):
            no_penality_action = []
            for action in actions:
                if not isinstance(action,str):
                    if action[2].num_to_floor_line == 0:
                        no_penality_action.append(action)
            return no_penality_action

        def sortActions(state, actions):
            empty_line = []
            for each_line in range(5):
                tile_number = state.agents[id].lines_number[each_line]
                if tile_number == 0:
                    empty_line.append(each_line + 1)
                else:
                    empty_line.append(-1)
            priority_list = []
            complete_row = []
            for action in actions:
                if not isinstance(action,str):
                    score = GetScore(state,id)
                    priority_list.append((action, score))
                    if action[2].number in empty_line:
                        index = empty_line.index(action[2].number)
                        if action[2].pattern_line_dest == index:
                            complete_row.append((action, score))
            if complete_row == []:
                sorted_list = sorted(priority_list, key=lambda x: x[1], reverse=True)
            else:
                sorted_list = sorted(complete_row, key=lambda x: x[1], reverse=True)
            return sorted_list

        best_action = actions[0]
        tile_gap = getTileGap(game_state)
        available_tile_statistics = getFactoryStatistics(game_state)
        filling_action = fillHole(actions, tile_gap)
        if filling_action != []:
            no_penality_action = getNoPenality(filling_action)
            if no_penality_action == []:
                no_penality_action = filling_action
        else:
            enough_tile_action = getEnoughTileActions(actions, game_state, available_tile_statistics)
            if enough_tile_action != []:
                no_penality_action = getNoPenality(enough_tile_action)
                if no_penality_action == []:
                    no_penality_action = getNoPenality(actions)
            else:
                no_penality_action = getNoPenality(actions)
        priority_list = sortActions(game_state, no_penality_action)

        if priority_list != []:
            best_action = priority_list[0][0]
        return best_action
    

    def GetActions(self, state, _id):
        actions = self.game_rule.getLegalActions(state, _id)
        if len(actions) == 0:
            actions = self.game_rule.getLegalActions(state, NUM_PLAYER)
        return actions


    def FullyExpanded(self, t_state, actions):
        if t_state in self.expanded_action_s:
            available_actions = []
            for action in actions:
                if not self.agent.ActionInList(action, self.expanded_action_s[t_state]):
                    available_actions.append(action)
            return available_actions
        else:
            return actions

    def TransformState(self, state, _id):
        return AgentToString(_id, state.agents[_id]) + BoardToString(state)

    def UCTValue(self, t_state, action, total_visits, c=1.0):
        action_visits = self.ns.get((t_state, action), 0)
        action_value = self.vs.get((t_state, action), 0)
        if action_visits == 0:
            return float('inf')
        return action_value / action_visits + c * math.sqrt(math.log(total_visits) / action_visits)
    def DoAction(self, state, action, _id):
        new_state = deepcopy(state)
        self.game_rule.generateSuccessor(new_state, action, _id)
        return new_state


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


    
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        best_action = self.bestRandomAction(rootstate,actions,self.id)
        best_score = float('-inf')

        for action in actions:
            next_state = self.DoAction(deepcopy(rootstate), action, self.id)
            score = self.GetScore(next_state, self.id)
            if score > best_score:
                best_score = score
                best_action = action

        minimax = Minimax(self,2, start_time, best_action)
        minimax.adjust_depth(rootstate, rootstate.agents, self.id)
        best_action = minimax.get_best_action(rootstate, self, self.id, True)

        return best_action




class Minimax:
    def __init__(self,agent, depth, start_time, best_action):
        self.agent = agent
        self.depth = depth
        self.start_time = start_time
        self.best_random = best_action
        self.mcts = MCTS(self.agent, self.agent.id)

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
        # if depth > 0 and not agent.GameEnd(state):
        #     mcts_result = self.mcts.run(state, id, max_time, self.start_time)  # Run MCTS simulation
        #     mcts_value = mcts_result[0]  # Get MCTS value
        #     # Combine MCTS value with Minimax value
        #     value = agent.GetScore(state, id) + mcts_value


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

    # def get_best_action(self, state, agent, id, is_maximizing, max_time=0.9):
    #     actions = agent.GetActions(state, id)
    #     actions = sorted(actions, key=lambda action: agent.GetScore(agent.DoAction(deepcopy(state), action, id), id),
    #                      reverse=True)
    #     best_value = float('-inf')
    #     best_action = None
    #     alpha = float('-inf')
    #     beta = float('inf')
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
