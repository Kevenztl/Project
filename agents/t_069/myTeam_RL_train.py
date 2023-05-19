from template import Agent
import time,random,heapq, json
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule

from collections import deque

THINKTIME   = 0.9
NUM_PLAYERS = 2
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.6

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.count = 0
        self.game_rule = GameRule(NUM_PLAYERS)
        self.weight = [0,0,0,0,0,0]

    # Generates actions from this state.
    def GetActions(self, state, _id):
        actions = self.game_rule.getLegalActions(state, _id)
        if len(actions) == 0:
            actions = self.game_rule.getLegalActions(state,NUM_PLAYERS)
        return actions
    
    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action, _id):
        state = self.game_rule.generateSuccessor(state, action, _id)

    def bestRandomAction(self,actions):
        best_action = random.choice(actions)
        alternative_actions = []
        for action in actions:
            if not isinstance(action,str):
                if action[2].num_to_floor_line == 0:
                    alternative_actions.append(action)
                elif action[2].num_to_floor_line < best_action[2].num_to_floor_line:
                    best_action = action
        if len(alternative_actions) > 0:
            best_action = random.choice(alternative_actions)

        # # Select move that involves placing the most number of tiles
        # # in a pattern line. Tie break on number placed in floor line.
        # most_to_line = -1
        # corr_to_floor = 0

        # for mid,fid,tgrab in actions:
        #     if most_to_line == -1:
        #         best_action = (mid,fid,tgrab)
        #         most_to_line = tgrab.num_to_pattern_line
        #         corr_to_floor = tgrab.num_to_floor_line
        #         continue

        #     if tgrab.num_to_pattern_line > most_to_line:
        #         best_action = (mid,fid,tgrab)
        #         most_to_line = tgrab.num_to_pattern_line
        #         corr_to_floor = tgrab.num_to_floor_line
        #     elif tgrab.num_to_pattern_line == most_to_line and \
        #         tgrab.num_to_pattern_line < corr_to_floor:
        #         best_action = (mid,fid,tgrab)
        #         most_to_line = tgrab.num_to_pattern_line
        #         corr_to_floor = tgrab.num_to_floor_line
        return best_action

    def GetScore(self, state, next_state, _id):
        next_state = deepcopy(state)
        self.DoAction(next_state, 'ENDROUND', _id)
        opponent_id = 1 - self.id
        return (self.game_rule.calScore(next_state, _id)-self.game_rule.calScore(state,_id)) \
                -(self.game_rule.calScore(next_state,opponent_id)) - (self.game_rule.calScore(state,opponent_id))
    
    def e_greedy(self):
        if random.uniform(0,1) < 1- EPSILON:
            return True
        return False

    def time_out(self,start_time):
        if time.time() - start_time > THINKTIME:
            print("Timeout!!!!!!")
            return True
        return False
        
    def CalQValue(self, state, action,_id):
        features = self.CalFeature(state,action,_id)
        if len(features) != len(self.weight):
            print("Length not matched!!!!!!!!!")
            return -99999
        else: 
            ans = 0
            for i in range(len(features)):
                ans += features[i] * self.weight[i]
        return ans
    
    def CalFeature(self, state, action, _id):
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
        # More than one actions are available 
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
        if len(actions) > 1:
                for action in actions:
                    opponent_Q = self.CalQValue(game_state, action, _id)
                    if opponent_Q > opponent_best_Q:
                        opponent_best_Q = opponent_Q
                        opponent_best_action = action
        return opponent_best_action, opponent_best_Q

    def SelectAction(self, actions, game_state):
        with open("agents/t_069/RL_weight/weight.json",'r',encoding='utf-8')as w:
            self.weight = json.load(w)['weight']
        # print(self.weight)
    
        start_time = time.time()
        best_action = self.bestRandomAction(actions)
        best_Q = -99999

        # More than one actions are available 
        if len(actions) > 1:
            # Player's best action & best Q
            best_action, best_Q = self.bestActionPlayer(game_state,actions,self.id,start_time,best_Q,best_action)

            # Next state (Opponent)
            next_state = deepcopy(game_state)
            self.DoAction(next_state,best_action,self.id)

            opponent_id = 1 - self.id
            opponent_actions = self.GetActions(next_state, opponent_id)
            opponent_best_action = self.bestRandomAction(opponent_actions)
            opponent_best_Q = -99999

            # Opponent best action
            opponent_best_action, opponent_best_Q = self.bestActionOpponent(next_state,opponent_actions,opponent_id,opponent_best_Q,opponent_best_action)

            # Do opponent action:
            self.DoAction(next_state,opponent_best_action,opponent_id)

            # Reward
            reward = self.GetScore(game_state,next_state,self.id)

            # Next available actions
            next_actions = self.GetActions(next_state,self.id)

            best_next_Q = -99999
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