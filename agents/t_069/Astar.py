from template import Agent
import time,random,heapq
from copy import deepcopy
from Azul.azul_model import AzulGameRule as GameRule

from collections import deque

THINKTIME   = 0.9
NUM_PLAYERS = 2


class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYERS)

    # Generates actions from this state.
    def GetActions(self, state, _id):
        return self.game_rule.getLegalActions(state, _id)
    
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

        return best_action
    
    # Get the max score different
    def heuristic(self, state):
        my_score = self.game_rule.calScore(state,self.id)
        return my_score


    def SelectAction(self,actions,rootstate):
        best_action = self.bestRandomAction(actions)

        start_time = time.time()

        initial_state = deepcopy(rootstate)
        initial_priority = self.heuristic(rootstate)

        priority_queue = [(initial_priority, 0, initial_state, [])]
        state_counter = 1  # Initialize state counter
        bestScore = self.game_rule.calScore(rootstate,self.id)

        while len(priority_queue) and time.time() - start_time < THINKTIME:
            _, _ ,state, path = heapq.heappop(priority_queue)

            newActions = self.GetActions(state,self.id)

            if len(newActions) == 0:
                score = self.game_rule.calScore(rootstate,self.id)
                if score > bestScore:
                    bestScore = score
                    best_action = path[0]
                continue

            for action in newActions:
                if time.time() - start_time >= THINKTIME:
                    return best_action
                
                nextState = deepcopy(state)
                nextPath = path + [action]
                nextPriority = self.heuristic(state)

                self.DoAction(nextState, action, self.id)

                # Opponent:
                opponent_actions = self.GetActions(nextState,1-self.id)

                if len(opponent_actions)==0:
                    score = self.game_rule.calScore(rootstate,self.id)
                    if score > bestScore:
                        bestScore = score
                        best_action = path[0]
                    continue
                opponent_action = self.bestRandomAction(opponent_actions)

                self.DoAction(nextState, opponent_action, 1-self.id)

                # Push the successor state into the priority queue
                heapq.heappush(priority_queue, (nextPriority, state_counter, nextState, nextPath))
                state_counter += 1  # Increment the state counter
        return best_action






