from template import Agent
import time,random, json
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule
import Azul.azul_utils as utils


THINKTIME   = 0.9
NUM_PLAYERS = 2
ALPHA = 0.1
GAMMA = 0.5
EPSILON = 0.3
LINES = 5

class myAgent(Agent):
    def __init__(self,_id):
        """
        Initializes the agent.

        Args:
            _id (int): The ID of the agent.
        """
        super().__init__(_id)
        self.count = 0
        self.game_rule = GameRule(NUM_PLAYERS)
        self.weight = [0,0,0,0,0,0]

    def GetActions(self, state, _id):
        """
        Generates possible actions for the agent from the current state.

        Args:
            state (State): The current game state.
            _id (int): The ID of the agent.

        Returns:
            actions (list): A list of possible actions.
        """
        actions = self.game_rule.getLegalActions(state, _id)
        if len(actions) == 0:
            actions = self.game_rule.getLegalActions(state,NUM_PLAYERS)
        return actions
    
    def DoAction(self, state, action, _id):
        """
        Performs a given action on the state, altering it.

        Args:
            state (State): The current game state.
            action (Action): The action to be performed.
            _id (int): The ID of the agent.
        """
        state = self.game_rule.generateSuccessor(state, action, _id)

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

    def GetScore(self, state, next_state, _id):
        """
        Calculates the score difference between current state and next state.

        Args:
            state (State): The current game state.
            next_state (State): The next game state.
            _id (int): The ID of the agent.

        Returns:
            score_difference (int): The difference in score.
        """
        next_state = deepcopy(state)
        self.DoAction(next_state, 'ENDROUND', _id)
        opponent_id = 1 - self.id
        return (self.game_rule.calScore(next_state, _id)-self.game_rule.calScore(state,_id)) \
                -(self.game_rule.calScore(next_state,opponent_id)) - (self.game_rule.calScore(state,opponent_id))
    
    def e_greedy(self):
        """
        Implements ε-greedy strategy for action selection.

        Returns:
            bool: True if a random number is less than 1-EPSILON, False otherwise.
        """
        if random.uniform(0,1) < 1- EPSILON:
            return True
        return False

    def time_out(self,start_time):
        """
        Checks if the agent has exceeded its allotted think time.

        Args:
            start_time (float): The start time of the thinking process.

        Returns:
            bool: True if the agent has exceeded its think time, False otherwise.
        """
        if time.time() - start_time > THINKTIME:
            return True
        return False
        
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
        features.append(floor_tiles/7)

        # Line 1-5
        for i in range(5):
            if next_state.agents[_id].lines_number[i] == i+1:
                features.append(1)
            else:
                features.append(0)
        return features
    
    def bestActionPlayer(self, game_state, actions, _id, start_time, best_Q, best_action):
        """
        Finds the best action and its Q-value for the player from the given actions.

        Args:
            game_state (State): The current game state.
            actions (list): A list of possible actions.
            _id (int): The ID of the agent.
            start_time (float): The start time of the thinking process.
            best_Q (float): The current best Q-value.
            best_action (Action): The current best action.

        Returns:
            best_action (Action): The updated best action.
            best_Q (float): The updated best Q-value.
        """

        if self.e_greedy():
            for action in actions:
                # Check for Time out
                if self.time_out(start_time):
                    break
                Q_value = self.CalQValue(game_state,action, _id)

                # Update Q
                if Q_value > best_Q:
                    best_Q = Q_value
                    best_action = action
        # Using Randomn action
        else:
            Q_value = self.CalQValue(game_state,best_action,_id)
            best_Q =  Q_value
        return best_action, best_Q

    def bestActionOpponent(self, game_state, actions, _id, opponent_best_Q, opponent_best_action):
        """
        Finds the best action and its Q-value for the opponent from the given actions.

        Args:
            game_state (State): The current game state.
            actions (list): A list of possible actions.
            _id (int): The ID of the agent.
            opponent_best_Q (float): The opponent's current best Q-value.
            opponent_best_action (Action): The opponent's current best action.

        Returns:
            opponent_best_action (Action): The opponent's updated best action.
            opponent_best_Q (float): The opponent's updated best Q-value.
        """
        if len(actions) > 1:
                for action in actions:
                    opponent_Q = self.CalQValue(game_state, action, _id)
                    if opponent_Q > opponent_best_Q:
                        opponent_best_Q = opponent_Q
                        opponent_best_action = action
        return opponent_best_action, opponent_best_Q

    def SelectAction(self, actions, game_state):
        """
        Selects the action the agent will take.

        Args:
            actions (list): A list of possible actions.
            game_state (State): The current game state.

        Returns:
            best_action (Action): The selected action.
        """

        with open("agents/t_069/RL_weight/weight.json",'r',encoding='utf-8')as w:
            self.weight = json.load(w)['weight']
        # print(self.weight)
    
        start_time = time.time()
        best_action = self.bestRandomAction(game_state,actions)
        best_Q = -float('inf')

        # More than one actions are available 
        if len(actions) > 1:
            # Player's best action & best Q
            best_action, best_Q = self.bestActionPlayer(game_state,actions,self.id,start_time,best_Q,best_action)

            # Next state (Opponent)
            next_state = deepcopy(game_state)
            self.DoAction(next_state,best_action,self.id)

            opponent_id = 1 - self.id
            opponent_actions = self.GetActions(next_state, opponent_id)
            opponent_best_action = self.bestRandomAction(next_state,opponent_actions)
            opponent_best_Q = -float('inf')

            # Opponent best action
            opponent_best_action, opponent_best_Q = self.bestActionOpponent(next_state,opponent_actions,opponent_id,opponent_best_Q,opponent_best_action)

            # Do opponent action:
            self.DoAction(next_state,opponent_best_action,opponent_id)

            # Reward
            reward = self.GetScore(game_state,next_state,self.id)

            # Next available actions
            next_actions = self.GetActions(next_state,self.id)

            best_next_Q = -float('inf')
            for action in next_actions:
                Q_value = self.CalQValue(next_state,action,self.id)
                best_next_Q = max(Q_value,best_next_Q)

            # Feature
            features = self.CalFeature(game_state,best_action,self.id)
            delta = reward + GAMMA * best_next_Q - best_Q

            # Update weight
            for i in range(len(features)):
                self.weight[i] += ALPHA * delta * features[i]

            # Write to weight
            with open("agents/t_069/RL_weight/weight.json",'w',encoding='utf-8') as w:
                json.dump({"weight": self.weight},w,indent = 4, ensure_ascii=False)
        return best_action