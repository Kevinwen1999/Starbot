from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import numpy as np
import pandas as pd
import os
import math
import time


# Available actions
ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

ACTIONS = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]
for x in range(0, 64):
    for y in range(0, 64):
        if (x + 1) % 32 == 0 and (y + 1) % 32 == 0:
            ACTIONS.append(ACTION_ATTACK + '_' + str(x - 16) + '_' + str(y - 16))


# Greedy coefficient
EPSILON = 0.9
# Learning rate
ALPHA = 0.1
# Discount factor
GAMMA = 0.9
# Reward constants
KILL_UNIT = 1
KILL_BUILDING = 3


class TerranAgent(base_agent.BaseAgent):

    def __init__(self):
        super(TerranAgent, self).__init__()
        print("     XXXXXXXX    ")
        self.attack_coordinates = None
        self.base_top_left = False
        self.last_unit_score = 0
        self.last_building_score = 0
        self.previous_state = None
        self.previous_action = None
        self.step_count = 0
        self.ACTIONS_INVALID = {}
        if os.path.isfile('./sc2tp.csv'):
            self.q_table = pd.read_csv("~/PycharmProjects/starbot/sc2tp.csv", sep='|', index_col=[0])
        else:
            self.q_table = pd.DataFrame(columns=ACTIONS, dtype=np.float64)

    def append_state(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(ACTIONS),
                                                         index=self.q_table.columns,
                                                         name=state))

    def update_table(self, state, next_state, action, r):
        self.append_state(state)
        self.append_state(next_state)
        if not state == next_state:
            q_predict = self.q_table.ix[state, action]
            next_state_actions = self.q_table.ix[next_state, :]

            if next_state in self.ACTIONS_INVALID:
                for ex in self.ACTIONS_INVALID[next_state]:
                    del next_state_actions[ex]

            q_target = r + GAMMA * next_state_actions.max() if next_state != 'terminal' else r
            self.q_table.ix[state, action] += ALPHA * (q_target - q_predict)

    def get_next_action(self, state):
        self.append_state(state)
        state_actions = self.q_table.loc[state, :]

        if state in self.ACTIONS_INVALID:
            for ex in self.ACTIONS_INVALID[state]:
                del state_actions[ex]

        if (np.random.uniform() > EPSILON) or not (state_actions.any()):
            action_name = np.random.choice(state_actions.index)
        else:
            action_name = state_actions.idxmax()  # Act according to q_table
        return action_name

    def q_learn(self, obs, current_state):
        if self.previous_action is not None:
            r = 0

            if obs.observation.score_cumulative.killed_value_units > self.last_unit_score:
                r += KILL_UNIT

            if obs.observation.score_cumulative.killed_value_structures > self.last_building_score:
                r += KILL_BUILDING

            if current_state == 'terminal':
                r = 1 if obs.reward == 0 else obs.reward

            self.update_table(str(self.previous_state), str(current_state), self.previous_action, r)

    def get_unit_list(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

    def unit_selected(self, obs, unit):
        if len(obs.observation.single_select) > 0 and obs.observation.single_select[0].unit_type == unit:
            return True
        elif len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0].unit_type == unit:
            return True
        return False

    def possible(self, obs, action):
        return action in obs.observation.available_actions

    def transform_relative(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def transform_coord(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def step(self, obs):
        super(TerranAgent, self).step(obs)

        if obs.last():
            self.q_learn(obs, 'terminal')
            self.q_table.to_csv("sc2tp.csv", sep='|')
            return actions.FUNCTIONS.no_op()

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            x_mean = player_x.mean()
            y_mean = player_y.mean()
            if x_mean <= 31 and y_mean <= 31:
                self.attack_coordinates = (39, 45)
                self.base_top_left = True
            else:
                self.attack_coordinates = (21, 24)
                self.base_top_left = False
            self.previous_state = None
            self.previous_action = None
            self.step_count = 0

        if self.step_count == 0:
            self.step_count += 1
            scv_count = len(self.get_unit_list(obs, units.Terran.SCV))
            barracks_cnt = len(self.get_unit_list(obs, units.Terran.Barracks))
            supply_cnt = len(self.get_unit_list(obs, units.Terran.SupplyDepot))
            food_cap = obs.observation.player.food_cap
            food_army = obs.observation.player.food_army
            minerals = obs.observation.player.minerals
            army_count = obs.observation.player.army_count

            current_state = np.zeros(13)
            current_state[0] = barracks_cnt
            current_state[1] = supply_cnt
            current_state[2] = food_cap
            current_state[3] = food_army
            current_state[4] = army_count

            enemy_block = np.zeros(4)
            enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative ==
                                features.PlayerRelative.ENEMY).nonzero()
            for i in range(0, len(enemy_x)):
                y = int(math.ceil((enemy_y[i] + 1) / 32.0))
                x = int(math.ceil((enemy_x[i] + 1) / 32.0))
                enemy_block[(y - 1) * 2 + (x - 1)] = 1

            enemy_block = enemy_block if self.base_top_left else enemy_block[::-1]

            for i in range(0, 4):
                current_state[i + 5] = enemy_block[i]

            friendly_block = np.zeros(4)
            friendly_y, friendly_x = (obs.observation.feature_minimap.player_relative ==
                                features.PlayerRelative.SELF).nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 32.0))
                x = int(math.ceil((friendly_x[i] + 1) / 32.0))
                friendly_block[(y - 1) * 2 + (x - 1)] = 1

            friendly_block = friendly_block if self.base_top_left else friendly_block[::-1]

            for i in range(0, 4):
                current_state[i + 9] = friendly_block[i]

            self.q_learn(obs, current_state)

            action_invalid = []
            if supply_cnt == 2 or scv_count == 0:
                action_invalid.append(ACTION_BUILD_SUPPLY_DEPOT)
            if supply_cnt == 0 or barracks_cnt == 2 or scv_count == 0:
                action_invalid.append(ACTION_BUILD_BARRACKS)
            if food_cap - food_army == 0 or barracks_cnt == 0:
                action_invalid.append(ACTION_BUILD_MARINE)
            if army_count == 0:
                for x in range(0, 64):
                    for y in range(0, 64):
                        if (x + 1) % 32 == 0 and (y + 1) % 32 == 0:
                            action_invalid.append(ACTION_ATTACK + '_' + str(x - 16) + '_' + str(y - 16))

            self.ACTIONS_INVALID[str(current_state)] = action_invalid

            action = self.get_next_action(str(current_state))
            self.last_unit_score = obs.observation.score_cumulative.killed_value_units
            self.last_building_score = obs.observation.score_cumulative.killed_value_structures
            self.previous_action = action
            self.previous_state = current_state

            if '_' in action:
                action, cx, cy = action.split('_')

            if action == ACTION_BUILD_SUPPLY_DEPOT or action == ACTION_BUILD_BARRACKS:
                scvs = self.get_unit_list(obs, units.Terran.SCV)
                if len(scvs) > 0:
                    scv = random.choice(scvs)
                    return actions.FUNCTIONS.select_point("select", [scv.x, scv.y])

            elif action == ACTION_BUILD_MARINE:
                brs = self.get_unit_list(obs, units.Terran.Barracks)
                if len(brs) > 0:
                    br = random.choice(brs)
                    return actions.FUNCTIONS.select_point("select_all_type", [br.x, br.y])

            elif action == ACTION_ATTACK:
                mrs = self.get_unit_list(obs, units.Terran.Marine)
                if len(mrs) > 0 and self.possible(obs, actions.FUNCTIONS.select_army.id):
                    return actions.FUNCTIONS.select_army("select")

        elif self.step_count == 1:
            self.step_count += 1
            action = self.previous_action
            if '_' in action:
                action, cx, cy = action.split('_')

            if action == ACTION_BUILD_SUPPLY_DEPOT:
                depots = self.get_unit_list(obs, units.Terran.SupplyDepot)
                if len(depots) < 2 and self.possible(obs, actions.FUNCTIONS.Build_SupplyDepot_screen.id):
                    ccs = self.get_unit_list(obs, units.Terran.CommandCenter)
                    if len(ccs) > 0:
                        cc = random.choice(ccs)
                        if len(depots) == 0:
                            coord = self.transform_coord(cc.x, 10, cc.y, 0)
                        else:
                            coord = self.transform_coord(cc.x, 20, cc.y, 0)

                        return actions.FUNCTIONS.Build_SupplyDepot_screen("now", coord)

            elif action == ACTION_BUILD_BARRACKS:
                brs = self.get_unit_list(obs, units.Terran.Barracks)
                if len(brs) < 2 and self.possible(obs, actions.FUNCTIONS.Build_Barracks_screen.id):
                    ccs = self.get_unit_list(obs, units.Terran.CommandCenter)
                    if len(ccs) > 0:
                        cc = random.choice(ccs)
                        if len(brs) == 0:
                            coord = self.transform_coord(cc.x, 0, cc.y, 10)
                        else:
                            coord = self.transform_coord(cc.x, 0, cc.y, 20)

                        return actions.FUNCTIONS.Build_Barracks_screen("now", coord)

            elif action == ACTION_BUILD_MARINE:
                if self.possible(obs, actions.FUNCTIONS.Train_Marine_quick.id):
                    return actions.FUNCTIONS.Train_Marine_quick("queued")

            elif action == ACTION_ATTACK:
                if self.unit_selected(obs, units.Terran.Marine):
                    if self.possible(obs, actions.FUNCTIONS.Attack_minimap.id):
                        return actions.FUNCTIONS.Attack_minimap("now", self.transform_relative(int(cx), int(cy)))

        elif self.step_count == 2:
            self.step_count = 0
            action = self.previous_action
            if '_' in action:
                action, cx, cy = action.split('_')

            if action == ACTION_BUILD_SUPPLY_DEPOT or action == ACTION_BUILD_BARRACKS:
                if self.possible(obs, actions.FUNCTIONS.Harvest_Gather_screen.id):
                    mines = self.get_unit_list(obs, units.Neutral.MineralField)
                    if len(mines) > 0:
                        mine = random.choice(mines)
                        return actions.FUNCTIONS.Harvest_Gather_screen("queued", [mine.x, mine.y])

        return actions.FUNCTIONS.no_op()


def main(argv):
    agent = TerranAgent()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_easy)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=64, minimap=64),
                    use_feature_units=True),
                step_mul=16,
                game_steps_per_episode=0,
                    visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
