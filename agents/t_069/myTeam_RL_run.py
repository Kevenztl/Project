from template import Agent
from Azul.azul_model import AzulGameRule as GameRule
import random, time, json
from copy import deepcopy

THINKTIME = 0.9
NUM_PLAYERS = 2


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)
        self.weight = [0, 0, 0, 0, 0, 0]
        with open("agents/t_069/RL_weight/weight.json", "r", encoding='utf-8') as fw:
            self.weight = json.load(fw)['weight']
        print(self.weight)

    def DoAction(self, state, action):
        state = self.game_rule.generateSuccessor(state, action, self.id)

    def CalFeatures(self, state, action):
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
        start_time = time.time()
        best_action = random.choice(actions)
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
