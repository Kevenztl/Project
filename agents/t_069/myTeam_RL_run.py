from template import Agent
from Azul.azul_model import AzulGameRule as GameRule
import Azul.azul_utils as utils
import random, time, json
from copy import deepcopy

THINKTIME = 0.9
NUM_PLAYERS = 2
LINES = 5

class myAgent(Agent):
    def __init__(self, _id):
        """
        Initializes the agent.

        Args:
            _id (int): The ID of the agent.
        """
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)
        self.weight = [0, 0, 0, 0, 0, 0]
        with open("agents/t_069/RL_weight/weight.json", "r", encoding='utf-8') as fw:
            self.weight = json.load(fw)['weight']
        print(self.weight)

    def bestRandomAction(self,game_state,actions):
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
            for each_line in range(LINES):
                tile_colour = game_state.agents[self.id].lines_tile[each_line]
                tile_number = game_state.agents[self.id].lines_number[each_line]
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
            oppoent_id = 1 - self.id
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
                    current_num = game_state.agents[self.id].lines_number[action[2].pattern_line_dest]
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
            for each_line in range(LINES):
                tile_number = state.agents[self.id].lines_number[each_line]
                if tile_number == 0:
                    empty_line.append(each_line + 1)
                else:
                    empty_line.append(-1)
            priority_list = []
            complete_row = []
            for action in actions:
                if not isinstance(action,str):
                    score = GetScore(state, self.id)
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


    def DoAction(self, state, action):
        """
        Performs a given action on the state, altering it.

        Args:
            state (State): The current game state.
            action (Action): The action to be performed.
            _id (int): The ID of the agent.
        """
        state = self.game_rule.generateSuccessor(state, action, self.id)

    def CalFeatures(self, state, action):
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
        self.DoAction(next_state, action)
        # F1 Floor line
        floor_tiles = len(next_state.agents[self.id].floor_tiles)
        features.append(floor_tiles / 7)
        # F2-6 complete line 1-5
        for i in range(5):
            if next_state.agents[self.id].lines_number[i] == i+1:
                features.append(1)
            else:
                features.append(0)
        return features

    def CalQValue(self, state, action):
        """
        Calculates the Q-value of an action, given a state.

        Args:
            state (State): The current game state.
            action (Action): The action to be performed.
            _id (int): The ID of the agent.

        Returns:
            ans (float): The calculated Q-value.
        """
        features = self.CalFeatures(state, action)
        if len(features) != len(self.weight):
            print("F ansd W length not matched")
            return -float('inf')
        else:
            ans = 0
            for i in range(len(features)):
                ans += features[i] * self.weight[i]
            return ans

    def SelectAction(self, actions, game_state):
        """
        Selects the action the agent will take.

        Args:
            actions (list): A list of possible actions.
            game_state (State): The current game state.

        Returns:
            best_action (Action): The selected action.
        """
        start_time = time.time()
        best_action = self.bestRandomAction(game_state,actions)
        best_Q_value = -float('inf')
        if len(actions) > 1:
            for action in actions:
                if time.time() - start_time > THINKTIME:
                    print("timeout")
                    break
                Q_value = self.CalQValue(game_state, action)
                if Q_value > best_Q_value:
                    best_Q_value = Q_value
                    best_action = action
        return best_action
