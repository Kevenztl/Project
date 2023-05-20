from template import Agent
import time,myTeam_hardcode2,heapq
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
    
    def floor_line_penalty(self,agent_state):
        penalty = 0
        for i in range(len(agent_state.floor)):
            penalty += agent_state.floor[i] * agent_state.FLOOR_SCORES[i]
        return penalty
    
    # def heuristic(self, state):
    #     my_score = self.game_rule.calScore(state,self.id)
    #     floorline_penalty = self.floor_line_penalty(state.agents[self.id])
    #     my_score = -1 * my_score
    #     return my_score + floorline_penalty

    def heuristic(self, state):
        agent_state = state.agents[self.id]

        penalty = self.floor_line_penalty(agent_state)

        completed_rows = agent_state.GetCompletedRows()
        completed_columns = agent_state.GetCompletedColumns()
        completed_sets = agent_state.GetCompletedSets()

        heuristic_value = (5 * completed_rows) + (7 * completed_columns) + (10 * completed_sets) - penalty
        return -1 * heuristic_value


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







