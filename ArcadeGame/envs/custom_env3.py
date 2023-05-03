from gym import Env
from gym import spaces
import numpy as np
from Games.game7 import ArcadeGame, grid_width, grid_height, FULL_GRID_REWARD, SOLUTION_REWARD, GAP_REJECTION_REWARD, num_columns, num_rows, REJECTION_REWARD, SCREEN_HEIGHT, SCREEN_WIDTH
from config import all_configs

class CustomEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    num_envs = 1

    def __init__(self, mode):
        self.game = ArcadeGame(mode=mode)
        self.mode = mode
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            shape=(grid_width, grid_height, 3), low=0, high=255, dtype=np.uint8
        )

    def step(self, action):
        reward, done, info = self.game.reward, False, {}
        if action == 0:  # RIGHT
            if not self.game.block_slot_selection:
                if self.game.current_position[1] < num_columns - self.game.slot_width:
                    self.game.current_position[1] += 1
                else:
                    self.game.reward += GAP_REJECTION_REWARD # negative reward ?
                observation = self.game.draw_screen()
            else:
                self.game.reward += GAP_REJECTION_REWARD # negative reward ?
        elif action == 1:  # LEFT
            if not self.game.block_slot_selection:
                if self.game.current_position[1] > 0:
                    self.game.current_position[1] -= 1
                else:
                    self.game.reward += GAP_REJECTION_REWARD # negative reward ?
                observation = self.game.draw_screen()
            else:
                self.game.reward += GAP_REJECTION_REWARD # negative reward ?
        elif action == 3:  # UP
            if self.game.current_position[0] > 0:
                self.game.current_position[0] -= 1
            else:
                self.game.reward += GAP_REJECTION_REWARD # negative reward ?
            observation = self.game.draw_screen()
        elif action == 3:  # DOWN
            if self.game.current_position[0] < num_rows - 1:
                self.game.current_position[0] += 1
            else:
                self.game.reward += GAP_REJECTION_REWARD # negative reward ?
            observation = self.game.draw_screen()
        elif action == 4:  # ENTER
            if self.game.allow_slot_allocation():
                self.game.allocate_slot()
                self.game.block_slot_selection = True
                # self.game.draw_screen()
                if self.game.curr_node == self.game.dst_node:
                    self.game.reward += SOLUTION_REWARD
                    if self.game.check_if_full_grid():
                        self.game.reward += FULL_GRID_REWARD
                        self.game.reset_game()
                        done = True
                    else:
                        self.game.new_game()
                else:
                    # negative reward ?
                    self.game.reward += GAP_REJECTION_REWARD
                    self.game.new_round()
                observation = self.game.draw_screen()
            else:
                self.game.reward += GAP_REJECTION_REWARD # negative reward ?
        elif action == 5:  # SPACE
            # large negative reward ?
            self.game.reward += REJECTION_REWARD
            self.game.reset_game()
            done = True

        observation = self.game.draw_screen()
        return observation, reward, done, info

    def reset(self):
        self.game.reset_game()
        observation = self.game.draw_screen()
        return observation

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            return self.game.draw_screen()  # return RGB frame suitable for video
        else:
            super(CustomEnv, self).render(mode=mode)  # just raise an exceptionset

    def close(self):
        self.game.exit()

    def set_mode(self, mode):
        self.game.set_mode(mode)
