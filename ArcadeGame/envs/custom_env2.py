from gym import Env
from gym import spaces
import numpy as np
from Games.game8 import ArcadeGame
from config import all_configs

SCREEN_HEIGHT = all_configs["screen_height"]
SCREEN_WIDTH = all_configs["screen_width"]
COLUMN_COUNT = all_configs["number_of_slots"]
SCREEN_COLUMN_COUNT = all_configs["screen_number_of_slots"]
K = all_configs["K"]
NUMBER_OF_SLOTS = all_configs["number_of_slots"]


class CustomEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    num_envs = 1

    def __init__(self):
        self.game = ArcadeGame()
        self.action_space = spaces.Discrete(NUMBER_OF_SLOTS * K + K - 1)
        self.observation_space = spaces.Box(
            shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3), low=0, high=255, dtype=np.uint8
        )

    def step(self, action):
        reward, done, info = 0, False, {}
        self.game.first_slot = action
        reward, done = self.game.update_spec_grid()
        if not done:
            reward, done = self.game.check_solution()
        observation = self.game.draw_screen()
        return observation, reward, done, info

    def reset(self):
        self.game.new_game()
        observation = self.game.draw_screen()
        return observation

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return self.game.draw_screen()  # return RGB frame suitable for video
        else:
            super(CustomEnv, self).render(mode=mode)  # just raise an exceptionset

    def close(self):
        self.game.exit()
