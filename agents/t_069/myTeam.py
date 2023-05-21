# from template import Agent
# import time,random,heapq
# from copy import deepcopy
# from Azul.azul_model import AzulGameRule as GameRule

# from collections import deque

# THINKTIME   = 0.9
# NUM_PLAYERS = 2

# class myAgent(Agent):
#     def __init__(self,_id):
#         super().__init__(_id)
#         self.game_rule = GameRule(NUM_PLAYERS)

#     # Generates actions from this state.
#     def GetActions(self, state, _id):
#         return self.game_rule.getLegalActions(state, _id)
    
#     # Carry out a given action on this state and return True if goal is reached received.
#     def DoAction(self, state, action, _id):
#         state = self.game_rule.generateSuccessor(state, action, _id)

#     def bestRandomAction(self,actions):
#         best_action = random.choice(actions)
#         alternative_actions = []

#         for action in actions:
#             if not isinstance(action,str):
#                 if action[2].num_to_floor_line == 0:
#                     alternative_actions.append(action)
                
#                 elif action[2].num_to_floor_line < best_action[2].num_to_floor_line:
#                     best_action = action
        
#         if len(alternative_actions) > 0:
#             best_action = random.choice(alternative_actions)

#         return best_action
    
#     def floor_line_penalty(self,agent_state):
#         penalty = 0
#         for i in range(len(agent_state.floor)):
#             penalty += agent_state.floor[i] * agent_state.FLOOR_SCORES[i]
#         return penalty
    
#     def SelectAction(self,actions,rootstate):
#         best_action = self.bestRandomAction(actions)
#         return best_action


from template import Agent
import time, random, math,json
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
from Azul.azul_utils import *
from .myTeam_hardCode import myAgent 
import Azul.azul_utils as utils
import numpy as np
THINKTIME = 0.9
NUM_PLAYER = 2
GAMMA = 0.9
EPS = 0.6

class MCTS:
    def __init__(self, agent, _id):
        self.agent = agent
        self.id = _id
        self.vs = dict()
        self.ns = dict()
        self.best_action_s = dict()
        self.expanded_action_s = dict()
        self.best_avg_reward = -float('inf')  # New attribute
        self.best_avg_reward_action = None


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

    # def UCTValue(self,state, t_state, action, total_visits, c=1.0):
    #     action_visits = self.ns.get((t_state, action), 0)
    #     action_value = self.agent.CalQValue(state, action)  # Use Q-value as the action value
    #     if action_visits == 0:
    #         return float('inf')
    #     return action_value / action_visits + c * math.sqrt(math.log(total_visits) / action_visits)
    def UCTValue(self, state, t_state, action, total_visits, c=0.3, q_weight=0.85):
        action_visits = self.ns.get((t_state, action), 0)
        action_value = self.agent.CalQValue(state, action,self.id)  # Use Q-value as the action value
        if action_visits == 0:
            return float('inf')
        ucb_value = action_value / action_visits + c * math.sqrt(math.log(total_visits) / action_visits)
        q_value = q_weight * action_value  # Multiply Q-value with a weight
        uct_value = ucb_value + q_value  # Combine UCB value and Q-value
        return uct_value

    
    def heuristic_score(self, action, state,id):
        next_state = deepcopy(state)
        self.agent.DoAction(next_state, action, id)

        score = 0
        # Add a bonus if the action completes a row or column
        if any(np.count_nonzero(next_state.agents[id].grid_state, axis=1) == next_state.agents[id].GRID_SIZE): # check rows
            score += 10
        if any(np.count_nonzero(next_state.agents[id].grid_state, axis=0) == next_state.agents[id].GRID_SIZE): # check columns
            score += 10

        # Subtract a penalty if the action causes tiles to fall on the floor
        if np.count_nonzero(next_state.agents[id].floor) > np.count_nonzero(state.agents[id].floor):
            score -= 5 * (np.count_nonzero(next_state.agents[id].floor) - np.count_nonzero(state.agents[id].floor))

        # Check if the action fills a pattern line beyond its capacity
        # if any(next_state.agents[id].lines_number[line] > line + 1 for line in range(next_state.agents[id].PATTERN_LINES)):
        #     score -= 100  # Apply a penalty for filling a pattern line beyond its capacity

        return score



    def select(self, t_state, actions, state, queue, start_time):
        while len(self.FullyExpanded(t_state, actions)) == 0 and not self.agent.GameEnd(state):
            if time.time() - start_time >= THINKTIME:
                print("MCT1")
                return None, None, None
            t_cur_state = self.agent.TransformState(state, self.id)
            total_visits = sum(self.ns.get((t_state, action), 0) for action in actions)
            uct_values = {action: self.UCTValue(state,t_state, action, total_visits) for action in actions}
            t_cur_state = self.agent.TransformState(state, self.id)
            cur_action = max(uct_values, key=uct_values.get)
            queue.append((t_cur_state, cur_action))
            next_state = deepcopy(state)
            self.agent.DoAction(next_state, cur_action, self.id)
            actions = self.agent.GetActions(next_state, self.id)
            state = next_state
            op_actions = self.agent.GetActions(next_state, 1 - self.id)
            op_action = self.agent.bestRandomAction(state, op_actions, 1 - self.id)
            self.agent.DoAction(next_state, op_action, 1 - self.id)
            actions = self.agent.GetActions(next_state, self.id)
            state = next_state
        return state, actions, queue

    def expand(self, actions, state, queue):
        t_cur_state = self.TransformState(state, self.id)
        available_actions = self.FullyExpanded(t_cur_state, actions)
        if len(available_actions) != 0:
            action = max(available_actions, key=lambda a: self.agent.CalQValue(state, a,self.id)+self.heuristic_score(a, state,self.id))  # Choose the action with the highest Q-value
            if t_cur_state in self.expanded_action_s:
                self.expanded_action_s[t_cur_state].append(action)
            else:
                self.expanded_action_s[t_cur_state] = [action]
            queue.append((t_cur_state, action))
            next_state = deepcopy(state)
            self.agent.DoAction(next_state, action, self.id)
            actions = self.agent.GetActions(next_state, self.id)
            state = next_state
        return state, actions, queue
    def simulate(self, state, new_actions, start_time, max_depth=5):
        length = 0
        depth = 0
        while not self.agent.GameEnd(state) and depth < max_depth:
            length += 1
            depth += 1
            if time.time() - start_time >= THINKTIME:
                print("MC2T")
                return None, None
            # Choose the action with the highest Q-value + heuristic score
            cur_action = max(new_actions, key=lambda a: self.agent.CalQValue(state, a,self.id)+self.heuristic_score(a, state,self.id))
            next_state = deepcopy(state)
            self.agent.DoAction(next_state, cur_action, self.id)
            op_actions = self.agent.GetActions(next_state, 1 - self.id)
            # Change here: instead of choosing a random action for the opponent, choose the one with the lowest Q-value + heuristic score
            op_action = min(op_actions, key=lambda a: self.agent.CalQValue(next_state, a,1-self.id)+self.heuristic_score(a, next_state,1-self.id), default=None)
            self.agent.DoAction(next_state, op_action, 1 - self.id)
            new_actions = self.agent.GetActions(next_state, self.id)
            state = next_state
        reward = self.agent.GetScore(state, self.id)
        return reward, length

    # def simulate(self, state, new_actions, start_time, max_depth=20):
    #     length = 0
    #     depth = 0
    #     while not self.agent.GameEnd(state) and depth < max_depth:
    #         length += 1
    #         depth += 1
    #         if time.time() - start_time >= THINKTIME:
    #             print("MC2T")
    #             return None, None
    #         # Choose the action with the highest Q-value + heuristic score
    #         cur_action = max(new_actions, key=lambda a: self.agent.CalQValue(state, a)+ self.heuristic_score(a, state))
    #         next_state = deepcopy(state)
    #         self.agent.DoAction(next_state, cur_action, self.id)
    #         op_actions = self.agent.GetActions(next_state, 1 - self.id)
    #         op_action = self.agent.bestRandomAction(state, op_actions, 1 - self.id)
    #         self.agent.DoAction(next_state, op_action, 1 - self.id)
    #         new_actions = self.agent.GetActions(next_state, self.id)
    #         state = next_state
    #     reward = self.agent.GetScore(state, self.id)
    #     return reward, length


    def backpropagate(self, reward, length, queue, start_time):
        cur_value = reward * (GAMMA ** length)
        while len(queue) != 0 and time.time() - start_time < THINKTIME:
            t_state, cur_action = queue.pop()
            if t_state in self.vs:
                if cur_value > self.vs[t_state]:
                    self.vs[t_state] = cur_value
                    self.best_action_s[t_state] = cur_action
                self.ns[t_state] += 1
                # Compare average reward of this action with current best
                avg_reward = self.vs[t_state] / self.ns[t_state]
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    self.best_avg_reward_action = cur_action
            else:
                self.vs[t_state] = cur_value
                self.ns[t_state] = 1
                self.best_action_s[t_state] = cur_action
            cur_value *= GAMMA

    def run(self, game_state, id, max_time=0.9, start_time=0,best_action = None):
        actions = self.agent.GetActions(game_state, id)
        best_action = best_action
        t_root_state = self.TransformState(game_state, self.id)
        while time.time() - start_time < max_time:
            state = deepcopy(game_state)
            new_actions = actions
            queue = deque([])
            reward = 0
            t_cur_state = self.agent.TransformState(state, id)
            state, new_actions, queue = self.select(t_cur_state, new_actions, state, queue, start_time)
            if not state and not new_actions:
                return best_action
            state, new_actions, queue = self.expand(new_actions, state, queue)
            if not state and not new_actions:
                return best_action
            reward, length = self.simulate(state, new_actions, start_time)
            if not reward and not length:
                return best_action
            self.backpropagate(reward, length, queue, start_time)
            if t_root_state in self.best_action_s:
                best_action = self.best_action_s[t_root_state]
        return self.best_avg_reward_action


class myAgent(Agent):
    def __init__(self, _id):
        self.id = _id
        self.count = 0
        self.game_rule = GameRule(NUM_PLAYER)
        self.weight = [0, 0, 0, 0, 0, 0]
        with open("agents/t_069/RL_weight/weight.json", "r", encoding='utf-8') as fw:
            self.weight = json.load(fw)['weight']

    def GetActions(self, state, _id):
        actions = self.game_rule.getLegalActions(state, _id)
        if len(actions) == 0:
            actions = self.game_rule.getLegalActions(state, NUM_PLAYER)
        return actions
    def DoAction(self, state, action, _id):
        self.game_rule.generateSuccessor(state, action, _id)
    # def DoAction(self, state, action, _id):
    #     if not isinstance(action, str):
    #         available_tiles = self.game_rule.getAvailableTiles(state, action.tile_type)  # 可用的瓷砖数量
    #         if action.number > available_tiles:
    #             action.number = available_tiles  # 如果减去的数量超过了可用的数量，将减去的数量设置为可用的数量
    #     self.game_rule.generateSuccessor(state, action, _id)

    #             self.game_rule.generateSuccessor(state, action, _id)
    def GetScore(self, state, _id):
        # base_score = self.game_rule.calScore(state, _id) - self.game_rule.calScore(state, 1 - _id)
        base_score = self.game_rule.calScore(state, _id)
        # # tile_factor = sum([line.count(1) for line in state.agents[_id].grid_state])
        # tile_factor = np.count_nonzero(state.agents[_id].grid_state == 1)
        tile_factor = np.count_nonzero(state.agents[_id].grid_state == 1)
        

        return base_score+tile_factor
    def CalFeatures(self, state, action,id):
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
        self.DoAction(next_state, action,id)
        # F1 Floor line
        floor_tiles = len(next_state.agents[id].floor_tiles)
        features.append(floor_tiles / 7)
        # F2-6 complete line 1-5
        for i in range(5):
            if next_state.agents[id].lines_number[i] == i+1:
                features.append(1)
            else:
                features.append(0)
        return features

    def CalQValue(self, state, action,id):
        """
        Calculates the Q-value of an action, given a state.

        Args:
            state (State): The current game state.
            action (Action): The action to be performed.
            _id (int): The ID of the agent.

        Returns:
            ans (float): The calculated Q-value.
        """
        features = self.CalFeatures(state, action,id)
        if len(features) != len(self.weight):
            print("F ansd W length not matched")
            return -float('inf')
        else:
            ans = 0
            for i in range(len(features)):
                ans += features[i] * self.weight[i]
            return ans
    def GameEnd(self, state):
        for plr_state in state.agents:
            completed_rows = plr_state.GetCompletedRows()
            if completed_rows > 0:
                return True
        return False
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
                oppoent_id = 1 - _id
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
    def GetScore(self,state, _id):
        next_state = deepcopy(state)
        self.DoAction(next_state, 'ENDROUND', _id)
        oppoent_id = 1 - _id
        return (self.game_rule.calScore(next_state, _id) - self.game_rule.calScore(state, _id)) \
            - (self.game_rule.calScore(next_state, oppoent_id)) - (self.game_rule.calScore(state, oppoent_id))


    def SelectAction(self, actions, game_state):
        start_time = time.time()
        self.count += 1
        best_action = self.bestRandomAction(game_state,actions,self.id)
        best_Q_value = -float('inf')

        if len(actions) > 1:
            for action in actions:
                if time.time() - start_time > THINKTIME:
                    print("timeout")
                    break
                Q_value = self.CalQValue(game_state, action,self.id)
                if Q_value > best_Q_value:
                    best_Q_value = Q_value
                    best_action = action
        # if self.calculate_score(game_state, best_action)<=self.calculate_score(game_state, best_random):
        #     print(self.calculate_score(game_state, best_action),self.calculate_score(game_state, best_random))
        #     best_action = best_random
        if self.count <= 20:
            return best_action
        else:
            mct = MCTS(self, self.id)
            best_action = mct.run(game_state, self.id, THINKTIME, start_time,best_action)
            # if self.GetScore(self.DoAction(game_state,best_action,self.id),self.id)<=self.GetScore(self.DoAction(game_state,best_select,self.id),self.id):
            #     best_action = best_select
            return best_action





