from template import Agent
import time,random,heapq, json
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

from collections import deque

THINKTIME = 0.9
NUM_PLAYERS = 2
ALPHA = 0.1
GAMMA = 0.6
EPSILON = 0.1

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.count = 0
        self.game_rule = GameRule(NUM_PLAYERS)
        self.model = self.build_model()
    
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=6, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(1, activation='linear'))  # the output is the Q-value
        model.compile(loss='mse', optimizer=Adam())
        return model
    
     # Generates actions from this state.
    def GetActions(self, state, _id):
        actions = self.game_rule.getLegalActions(state, _id)
        if len(actions) == 0:
            actions = self.game_rule.getLegalActions(state,NUM_PLAYERS)
        return actions
    
    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action, _id):
        state = self.game_rule.generateSuccessor(state, action, _id)

    def bestRandomAction(self,state,actions):
        best_action = random.choice(actions)
        game_state = deepcopy(state)
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
    
    def CalQValue(self, state, action, _id, start_time):
        while self.time_out(start_time):
            features = np.array(self.CalFeature(state, action, _id)).reshape(-1, 6)
            return self.model.predict(features)[0]
        return self.model.predict(features)[0]
    
    def CalFeature(self, state, action, _id):
        features = []
        next_state = deepcopy(state)
        self.DoAction(next_state, action, _id)

        floor_tiles = len(next_state.agents[_id].floor_tiles)
        features.append(floor_tiles/7)

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
                    return None, None
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
    
    def save_model(self, filename="model_weights.h5"):
        self.model.save_weights(filename)

    def load_model(self, filename="model_weights.h5"):
        self.model.load_weights(filename)

    
    def SelectAction(self, actions, game_state):
        start_time = time.time()
        best_action = self.bestRandomAction(game_state,actions)
        best_Q = -float('inf')

        # More than one actions are available 
        if len(actions) > 1:
            # Player's best action & best Q
            best_action, best_Q = self.bestActionPlayer(game_state,actions,self.id,start_time,best_Q,best_action)
            if best_action is None:
                return self.bestRandomAction(game_state,actions)
            
            # Next state (Opponent)
            next_state = deepcopy(game_state)
            self.DoAction(next_state,best_action,self.id)

            opponent_id = 1 - self.id
            opponent_actions = self.GetActions(next_state, opponent_id)
            opponent_best_action = self.bestRandomAction(next_state,opponent_actions)
            opponent_best_Q = -float('inf')

            # Opponent best action
            opponent_best_action, opponent_best_Q = self.bestActionOpponent(next_state,opponent_actions,opponent_id,opponent_best_Q,opponent_best_action)
            if opponent_best_action is None:
                return self.bestRandomAction(game_state,actions)
            

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
            target = self.model.predict(np.array(self.CalFeature(game_state, best_action, self.id)).reshape(-1, 6))
            Q_values = [self.model.predict(np.array(self.CalFeature(next_state, action, self.id)).reshape(-1, 6)) for action in self.GetActions(next_state, self.id)]
            max_Q = max(Q_values)
            target[0] += ALPHA * (reward + GAMMA * max_Q[0])


            self.model.fit(np.array(self.CalFeature(game_state, best_action, self.id)).reshape(-1, 6), target, epochs=1, verbose=0)
        
            self.save_model()

        return best_action
