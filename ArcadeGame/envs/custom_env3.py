from gym import Env
from gym import spaces
import numpy as np
from Games.game7 import ArcadeGame, grid_width, grid_height, num_rows, FULL_GRID_REWARD, SOLUTION_REWARD, GAP_REJECTION_REWARD, num_columns, num_rows, REJECTION_REWARD, SCREEN_HEIGHT, SCREEN_WIDTH
from config import all_configs

class CustomEnv(Env):
    metadata = {"render.modes": ["human", "rgb_array"]}
    num_envs = 1

    def __init__(self, mode):
        self.game = ArcadeGame(mode=mode)
        self.mode = mode
        self.action_space = spaces.Discrete(num_rows*num_columns + 1) # extra for exit
        self.observation_space = spaces.Box(
            shape=(grid_width, grid_height, 3), low=0, high=255, dtype=np.uint8
        )

    def step(self, action):
        episode_reward, done, info = 0, False, {}
        # Action represents position on the grid
        if action == num_rows*num_columns: # SPACE / exit
            episode_reward = REJECTION_REWARD
            self.game.score += episode_reward
            self.game.reset_game()
            done = True
        if not self.game.block_slot_selection:  # selecting first slot of route
            blocked_actions = self.game.get_blocked_actions()
            if action not in blocked_actions:
                if self.game.allow_slot_allocation():
                    self.game.allocate_slot()
                    # self.game.draw_screen()
                    if self.game.curr_node == self.game.dst_node:  # arrived to destination node
                        episode_reward = SOLUTION_REWARD
                        self.game.score += episode_reward
                        if self.game.check_if_full_grid():
                            episode_reward += FULL_GRID_REWARD
                            self.game.score += episode_reward
                            self.game.reset_game()
                            done = True
                        else:
                            self.game.new_game()
                    else:  # not at destination node
                        self.game.block_slot_selection = True
                        episode_reward = 0
                        self.game.score += episode_reward
                        self.game.new_round()
                else:  # route selection invalid
                    episode_reward = GAP_REJECTION_REWARD
                    self.game.score += episode_reward
            else:  # action in blocked actions
                episode_reward = GAP_REJECTION_REWARD
                self.game.score += episode_reward
        else: # selecting subsequent slots of route
            allowed_actions = self.game.get_allowed_actions()
            if action in allowed_actions:
                if self.game.allow_slot_allocation():
                    self.game.allocate_slot()
                    if self.game.curr_node == self.game.dst_node:  # arrived to destination node
                        episode_reward = SOLUTION_REWARD
                        if self.game.check_if_full_grid():
                            episode_reward += FULL_GRID_REWARD
                            self.game.score += episode_reward
                            self.game.reset_game()
                            done = True
                        else:
                            self.game.score += episode_reward
                            self.game.new_game()
                        self.game.block_slot_selection = False
                    else:  # not at destination node
                        episode_reward = 0
                        self.game.score += episode_reward
                        self.game.new_round()
                else:  # route selection invalid
                    episode_reward = GAP_REJECTION_REWARD
                    self.game.score += episode_reward
            else:  # action in blocked actions
                episode_reward = GAP_REJECTION_REWARD
                self.game.score += episode_reward

        observation = self.game.draw_screen()
        return observation, episode_reward, done, info

        # if action == 0:  # RIGHT
        #     if not self.game.block_slot_selection:
        #         if self.game.current_position[1] < num_columns - self.game.slot_width:
        #             self.game.current_position[1] += 1
        #             episode_reward = 0
        #         else:
        #             episode_reward = GAP_REJECTION_REWARD # negative reward ?
        #     else:
        #         episode_reward = GAP_REJECTION_REWARD # negative reward ?
        #     self.game.score += episode_reward
        # elif action == 1:  # LEFT
        #     if not self.game.block_slot_selection:
        #         if self.game.current_position[1] > 0:
        #             self.game.current_position[1] -= 1
        #             episode_reward = 0
        #         else:
        #             episode_reward = GAP_REJECTION_REWARD # negative reward ?
        #     else:
        #         episode_reward = GAP_REJECTION_REWARD # negative reward ?
        #     self.game.score += episode_reward
        # elif action == 2:  # UP
        #     if self.game.current_position[0] > 0:
        #         self.game.current_position[0] -= 1
        #         episode_reward = 0
        #     else:
        #         episode_reward = GAP_REJECTION_REWARD # negative reward ?
        #     self.game.score += episode_reward
        # elif action == 3:  # DOWN
        #     if self.game.current_position[0] < num_rows - 1:
        #         self.game.current_position[0] += 1
        #         episode_reward = 0
        #     else:
        #         episode_reward = GAP_REJECTION_REWARD # negative reward ?
        #     self.game.score += episode_reward
        # elif action == 4:  # ENTER
        #     if self.game.allow_slot_allocation():
        #         self.game.allocate_slot()
        #         self.game.block_slot_selection = True
        #         # self.game.draw_screen()
        #         if self.game.curr_node == self.game.dst_node:  # arrived at destination node
        #             episode_reward = SOLUTION_REWARD
        #             if self.game.check_if_full_grid():
        #                 episode_reward += FULL_GRID_REWARD
        #                 self.game.score += episode_reward
        #                 self.game.reset_game()
        #                 done = True
        #             else:
        #                 self.game.new_game()
        #         else:
        #             episode_reward = 0
        #             self.game.score += episode_reward
        #             self.game.new_round()  # not a final link 
        #     else:
        #         episode_reward = GAP_REJECTION_REWARD # negative reward for trying wrong slot
        #         self.game.score += episode_reward
        # elif action == 5:  # SPACE
        #     # large negative reward ?
        #     episode_reward = REJECTION_REWARD
        #     self.game.score += episode_reward
        #     self.game.reset_game()
        #     done = True
        # observation = self.game.draw_screen()
        # return observation, episode_reward, done, info

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
