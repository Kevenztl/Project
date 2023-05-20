import Azul.azul_utils as utils
from template import Agent
import time, random
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
from Azul.azul_utils import *

THINKTIME = 0.9
NUM_PLAYER = 2
GAMMA = 0.9
EPS = 0.6
LINES = 5


class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.game_rule = GameRule(NUM_PLAYER)

    def GetActions(self, state, _id):
        actions = self.game_rule.getLegalActions(state, _id)
        if len(actions) == 0:
            actions = self.game_rule.getLegalActions(state, NUM_PLAYER)
        return actions

    def DoAction(self, state, action, _id):
        state = self.game_rule.generateSuccessor(state, action, _id)

    def SelectAction(self, actions, game_state):

        # print(game_state.agents[0].lines_tile)
        # print(game_state.agents[0].lines_number)

        def getTileGap(game_state):
            tile_gap = {}
            # 格式：颜色:[[缺了几个][第几行缺的]]
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
                current_colour = action[2].tile_type
                current_num = game_state.agents[self.id].lines_number[action[2].pattern_line_dest]
                gap_num = action[2].pattern_line_dest + 1 - current_num
                # print(current_colour, current_num, gap_num, action[2].pattern_line_dest)
                if tile_statistics[current_colour] - gap_num >= 2:
                    enough_action.append(action)
            return enough_action
        def getNoPenality(actions):
            no_penality_action = []
            for action in actions:
                if action[2].num_to_floor_line == 0:
                    no_penality_action.append(action)
            return no_penality_action

        def sortActions(state, actions):
            tile_gap = {}
            empty_line = []
            # [-1,2,-1,4,-1] -1:非空, 其它:有几个空
            for each_line in range(LINES):
                tile_number = game_state.agents[self.id].lines_number[each_line]
                if tile_number == 0:
                    empty_line.append(each_line + 1)
                else:
                    empty_line.append(-1)
            print(empty_line)
            priority_list = []
            complete_row = []
            for action in actions:
                score = GetScore(state, self.id)
                priority_list.append((action, score))
                if action[2].number in empty_line:
                    index = empty_line.index(action[2].number)
                    if action[2].pattern_line_dest == index:
                        complete_row.append((action, score))
            if complete_row == []:
                sorted_list = sorted(priority_list, key=lambda x: x[1], reverse=True)
                print("a")
            else:
                sorted_list = sorted(complete_row, key=lambda x: x[1], reverse=True)
                print("b")
            return sorted_list

        print()
        best_action = actions[0]
        tile_gap = getTileGap(game_state)
        available_tile_statistics = getFactoryStatistics(game_state)
        filling_action = fillHole(actions, tile_gap)
        if filling_action != []:
            no_penality_action = getNoPenality(filling_action)
            if no_penality_action == []:
                no_penality_action = filling_action
            print(1, len(filling_action), len(no_penality_action))
        else:
            enough_tile_action = getEnoughTileActions(actions, game_state, available_tile_statistics)
            # print(enough_tile_action)
            if enough_tile_action != []:
                no_penality_action = getNoPenality(enough_tile_action)
                if no_penality_action == []:
                    no_penality_action = getNoPenality(actions)
                print(2, len(enough_tile_action), len(no_penality_action))
            else:
                no_penality_action = getNoPenality(actions)
                print(3, len(enough_tile_action), len(no_penality_action))
        # print(no_penality_action)
        priority_list = sortActions(game_state, no_penality_action)


        if priority_list != []:
            print(4)
            best_action = priority_list[0][0]
        print()
        return best_action




