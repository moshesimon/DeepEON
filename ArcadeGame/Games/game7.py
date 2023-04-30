from itertools import islice
import pygame
import numpy as np
import networkx as nx
import sys
import os
import time

from config import all_configs, logger
logging = False

NUMBER_OF_SLOTS = all_configs["number_of_slots"]
# SCREEN_NUMBER_OF_SLOTS = all_configs["screen_number_of_slots"]
K = all_configs["K"]
WIDTH = all_configs["width"]
HEIGHT = all_configs["height"]
SCREEN_WIDTH = all_configs["screen_width"]
SCREEN_HEIGHT = all_configs["screen_height"]
LEFT_SIDE_OFFSET = all_configs["screen_side_offset"]
PATH_ROWS = all_configs["path_rows"]
SPECTRUM_SLOTS_ROWS_FROM_TOP = all_configs["spectrum_slots_rows_from_top"]
# WHITE = all_configs["self.WHITE"]
# BLACK = all_configs["self.BLACK"]
# GREEN = all_configs["self.GREEN"]
# RED = all_configs["self.RED"]
SOLUTION_REWARD = all_configs["solution_reward"]
REJECTION_REWARD = all_configs["rejection_reward"]
GAP_REJECTION_REWARD = all_configs["gap_rejection_reward"]
LEFT_REWARD = all_configs["left_reward"]
RIGHT_REWARD = all_configs["right_reward"]
SEED = all_configs["seed"]
END_LIMIT = all_configs["end_limit"]
EPISODE_END = all_configs["episode_end"]

FULL_GRID_REWARD = all_configs["full_grid_reward"]
NUMBER_OF_NODES = all_configs["number_of_nodes"]

num_rows = int(0.5 * NUMBER_OF_NODES * (NUMBER_OF_NODES - 1))
row_labels = [i for i in range(1, num_rows + 1)]
num_columns = NUMBER_OF_SLOTS
column_labels = [i for i in range(0, num_columns)]
block_size = 20
block_padding_all = 1
padding_top = 50
padding_left = 30
padding_right = 30
padding_bottom = 30
font_padding_left = 5
font_padding_top = 1

grid_width = (block_size + block_padding_all) * num_columns + padding_left + padding_right
grid_height = (block_size + block_padding_all) * num_rows + padding_top + padding_bottom

link_names = [i for i in range(1, NUMBER_OF_NODES+1)]
links = []
for i in range(len(link_names)):
    for j in range(i+1, len(link_names)):
        links.append(f"{link_names[i]}-{link_names[j]}")


class ArcadeGame:
    def __init__(self, mode):
        pygame.init()
        if mode != "human" and mode != "rgb_array":
            raise Exception("Invalid mode!")
        if mode == "rgb_array":
            self.BLACK = (0, 0, 0)
            self.WHITE = (0, 0, 1)
            self.GRAY = (0, 0, 2)
            self.GREEN = (0, 0, 3)
            self.self.YELLOW = (0, 0, 4)
            self.self.ORANGE = (0, 0, 5)
            self.RED = (0, 0, 6)
        else:
            self.RED = (255, 0, 0)
            self.BLACK = (0, 0, 0)
            self.GREEN = (0, 255, 0)
            self.WHITE = (255, 255, 255)
            self.YELLOW = (255, 255, 0)
            self.GRAY = (128, 128, 128)
            self.ORANGE = (255, 165, 0)
        self.myfont = pygame.font.SysFont("monospace", 14)
        self.window = (grid_width, grid_height)
        self.background = pygame.Surface(self.window)
        self.highscore = -100
        self.seed()

    def draw_screen(self):
        self.background.fill(self.GRAY)

        top = padding_top
        for i in range(num_rows):
            left = padding_left
            label = self.myfont.render(str(links[i]), 1, self.YELLOW)
            self.background.blit(label, (left-padding_left+font_padding_left, top+font_padding_top))
            for j in range(num_columns):
                if self.grid[i, j] == 0:
                    pygame.draw.rect(self.background,self.GREEN,(left,top,block_size,block_size))
                elif self.grid[i, j] == 1:
                    pygame.draw.rect(self.background,self.RED,(left,top,block_size,block_size))
                left = left + block_size + block_padding_all
            top = top + block_size + block_padding_all
        for i in range(self.slot_width):
            pygame.draw.rect(self.background,self.ORANGE, (padding_left+(self.current_position[1]+i)*(block_size+block_padding_all), padding_top+self.current_position[0]*(block_size+block_padding_all), block_size, block_size))
        
        label = self.myfont.render("SRC", 1, self.BLACK)
        self.background.blit(label, (padding_left/2, font_padding_top))
        label = self.myfont.render(str(self.src_node), 1, self.BLACK)
        self.background.blit(label, (padding_left/2, font_padding_top + padding_top/2))

        label = self.myfont.render("CURR", 1, self.BLACK)
        self.background.blit(label, ((block_size+block_padding_all)*(2*NUMBER_OF_SLOTS/5) + padding_left, font_padding_top))
        label = self.myfont.render(str(self.curr_node), 1, self.BLACK)
        self.background.blit(label, ((block_size+block_padding_all)*(2*NUMBER_OF_SLOTS/5) + padding_left, font_padding_top + padding_top/2))

        label = self.myfont.render("DST", 1, self.BLACK)
        self.background.blit(label, (padding_left + (block_size+block_padding_all)*(NUMBER_OF_SLOTS) - padding_right/2, font_padding_top))
        label = self.myfont.render(str(self.dst_node), 1, self.BLACK)
        self.background.blit(label, (padding_left + (block_size+block_padding_all)*(NUMBER_OF_SLOTS) - padding_right/2, font_padding_top + padding_top/2))

        label = self.myfont.render("ROUND", 1, self.BLACK)
        self.background.blit(label, (padding_left, num_rows*(block_size+block_padding_all) + padding_top + padding_bottom/2))
        label = self.myfont.render(str(self.rounds), 1, self.BLACK)
        self.background.blit(label, ((block_size+block_padding_all)*(3*NUMBER_OF_SLOTS/9) + padding_left, num_rows*(block_size+block_padding_all) + padding_top + padding_bottom/2))

        label = self.myfont.render("SCORE", 1, self.BLACK)
        self.background.blit(label, ((block_size+block_padding_all)*(6*NUMBER_OF_SLOTS/10) + padding_left, num_rows*(block_size+block_padding_all) + padding_top + padding_bottom/2))
        label = self.myfont.render(str(self.reward), 1, self.BLACK)
        self.background.blit(label, ((block_size+block_padding_all)*(9*NUMBER_OF_SLOTS/10) + padding_left, num_rows*(block_size+block_padding_all) + padding_top + padding_bottom/2))

        self.surfarr = pygame.surfarray.array3d(self.background)
        return self.surfarr

    def render(self):
        self.screen = pygame.display.set_mode(self.window)
        pygame.display.set_caption("DeepEON Arcade")
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()

    def draw_box(self, col, row, colour):
        pygame.draw.rect(
            self.background, colour, (col * WIDTH, row * HEIGHT, WIDTH, HEIGHT)
        )
    
    def reset_game(self):
        self.score = 0
        self.reward = 0
        self.blocks = 0
        self.rounds = 0
        self.grid = np.zeros((num_rows, num_columns), dtype=np.uint8)
        self.new_game()

    def new_game(self):
        self.rounds += 1
        self.block_slot_selection = False
        self.selected_slot = 0
        self.src_node = np.random.randint(1, NUMBER_OF_NODES + 1)
        self.dst_node = np.random.randint(1, NUMBER_OF_NODES + 1)
        while self.dst_node == self.src_node:
            self.dst_node = np.random.randint(1, NUMBER_OF_NODES + 1)
        self.curr_node = self.src_node
        # self.slot_width = np.random.randint(2, 5)
        self.slot_width = 2
        self.new_round()

    def new_round(self):
        self.current_position = [0, self.selected_slot]  # [row, column]

    def allow_slot_allocation(self):
        selected_node1 = int(links[self.current_position[0]].split('-')[0])
        selected_node2 = int(links[self.current_position[0]].split('-')[1])
        if self.curr_node != selected_node1 and self.curr_node != selected_node2:
            if logging:
                logger.info(f"[WRONG PATH] Slot allocation not allowed at {self.current_position[0], self.current_position[1]}")
            return False
        for i in range(self.slot_width):
            if self.grid[self.current_position[0]][self.current_position[1] + i] == 1:
                if logging:
                    logger.info(f"[TAKEN] Slot allocation not allowed at {self.current_position[0], self.current_position[1] + i}")
                return False
        return True

    def allocate_slot(self):
        selected_node1 = int(links[self.current_position[0]].split('-')[0])
        selected_node2 = int(links[self.current_position[0]].split('-')[1])
        if self.curr_node == selected_node1:
            self.curr_node = selected_node2
        elif self.curr_node == selected_node2:
            self.curr_node = selected_node1
        else:
            if logging:
                logger.error(f"Something went wrong. Current node: {self.curr_node} Selected node 1: {selected_node1} Selected node 2: {selected_node2}")
        for i in range(self.slot_width):
            self.grid[self.current_position[0]][self.current_position[1] + i] = 1
        self.selected_slot = self.current_position[1]

    def check_solution(self):
        done = False
        reward = self.get_solution_reward()
        if reward == SOLUTION_REWARD:
            self.reward += reward
            self.update_link_grid()
        else:
            self.blocks += 1
            self.reward += reward

        if EPISODE_END == 1 and self.blocks >= END_LIMIT:
            done = True
        elif EPISODE_END == 2 and self.rounds >= END_LIMIT:
            done = True

        if self.score > self.highscore:
            self.highscore = self.score

        self.new_round()

        return reward, done

    def check_if_full_grid(self):
        if np.count_nonzero(self.grid) == num_rows * num_columns:
            return True
        return False

    def seed(self):
        np.random.seed(SEED)

    def exit(self):
        pygame.quit()
        sys.exit()


def main():  # only used for human mode
    game = ArcadeGame(mode="human")
    game.reset_game()
    game.draw_screen()
    game.render()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    if not game.block_slot_selection:
                        if game.current_position[1] < num_columns - game.slot_width:
                            game.current_position[1] += 1
                        game.draw_screen()
                        game.render()
                elif event.key == pygame.K_LEFT:
                    if not game.block_slot_selection:
                        if game.current_position[1] > 0:
                            game.current_position[1] -= 1
                        game.draw_screen()
                        game.render()
                elif event.key == pygame.K_UP:
                    if game.current_position[0] > 0:
                        game.current_position[0] -= 1
                    game.draw_screen()
                    game.render()
                elif event.key == pygame.K_DOWN:
                    if game.current_position[0] < num_rows - 1:
                        game.current_position[0] += 1
                    game.draw_screen()
                    game.render()
                elif event.key == pygame.K_RETURN:
                    if game.allow_slot_allocation():
                        game.allocate_slot()
                        game.block_slot_selection = True
                        game.draw_screen()
                        game.render()
                        if game.curr_node == game.dst_node:
                            game.reward += SOLUTION_REWARD
                            if logging:
                                logger.info(f"Round {game.rounds} over. Reward: {game.reward}")
                            if game.check_if_full_grid():
                                game.reward += FULL_GRID_REWARD
                                if logging:
                                    logger.info(f"[FULL GRID] Game with {game.rounds} rounds finished. Reward: {game.reward}")
                                game.reset_game()
                            else:
                                game.new_game()
                        else:
                            game.new_round()
                        game.draw_screen()
                        game.render()
                elif event.key == pygame.K_SPACE:
                    game.reset_game()
                    game.draw_screen()
                    game.render()


if __name__ == "__main__":
    main()
