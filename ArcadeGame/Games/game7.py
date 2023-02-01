from itertools import islice
import pygame
import numpy as np
import networkx as nx
import sys
import os
import time

from config import all_configs


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
WHITE = all_configs["white"]
BLACK = all_configs["black"]
GREEN = all_configs["green"]
RED = all_configs["red"]
SOLUTION_REWARD = all_configs["solution_reward"]
REJECTION_REWARD = all_configs["rejection_reward"]
GAP_REJECTION_REWARD = all_configs["gap_rejection_reward"]
LEFT_REWARD = all_configs["left_reward"]
RIGHT_REWARD = all_configs["right_reward"]
SEED = all_configs["seed"]
END_LIMIT = all_configs["end_limit"]
EPISODE_END = all_configs["episode_end"]

NUMBER_OF_NODES = 5
rows = int(0.5 * NUMBER_OF_NODES * (NUMBER_OF_NODES - 1))
row_labels = [i for i in range(1, rows + 1)]
columns = NUMBER_OF_SLOTS + 1
column_labels = [i for i in range(0, columns)]
block_size = 20
block_padding_all = 1
padding_top = 30
padding_left = 30
padding_right = 30
padding_bottom = 30
grid_width = (block_size + block_padding_all) * columns + padding_left + padding_right
grid_height = (block_size + block_padding_all) * rows + padding_top + padding_bottom

font_padding_left = 5
font_padding_top = 1

link_names = [i for i in range(1, NUMBER_OF_NODES+1)]
links = []
for i in range(len(link_names)):
    for j in range(i+1, len(link_names)):
        links.append(f"{link_names[i]}-{link_names[j]}")
print(links)
print(len(links))

class ArcadeGame:
    def __init__(self):
        pygame.init()
        self.myfont = pygame.font.SysFont("monospace", 14)
        self.window = (grid_width, grid_height)
        self.background = pygame.Surface(self.window)
        self.highscore = -100
        self.seed()

    def draw_screen(self):
        self.background.fill(RED)

        top = padding_top
        for i in range(rows):
            left = padding_left
            label = self.myfont.render(str(links[i]), 1, (255, 255, 0))
            self.background.blit(label, (left-padding_left+font_padding_left, top+font_padding_top))
            for j in range(columns):
                pygame.draw.rect(self.background,(0,255,255),(left,top,block_size,block_size))
                left = left + block_size + block_padding_all
            top = top + block_size + block_padding_all
        self.surfarr = pygame.surfarray.array3d(self.background)
        return self.surfarr

    def render(self):
        self.screen = pygame.display.set_mode(self.window)
        pygame.display.set_caption("DeepEON Arcade")
        self.screen.blit(self.background, (0, 0))
        pygame.display.flip()
        print(
            f"Round: {self.rounds} Number of blocks: {self.blocks} Reward: {self.reward} High Score: {self.highscore}"
        )

    def draw_box(self, col, row, colour):
        pygame.draw.rect(
            self.background, colour, (col * WIDTH, row * HEIGHT, WIDTH, HEIGHT)
        )

    def new_game(self):
        self.score = 0
        self.reward = 0
        self.blocks = 0
        self.rounds = 0
        self.link_grid = {}
        # self.new_round()

    def new_round(self):
        """
        Sets up all parameters for a new round
        """
        self.first_slot = 0
        self.rounds += 1
        self.target = np.random.randint(2, 7)
        self.source = np.random.randint(1, self.target)
        self.slots = np.random.randint(2, 5)
        self.update_spec_grid()  # populate spectrum grid

    def update_spec_grid(self):
        # self.spec_grid = np.zeros(NUMBER_OF_SLOTS * K + K - 1, dtype=int)
        try:
            for i in range(self.slots):
                self.spec_grid[self.first_slot + i] = 1
            return 0, False
        except:
            return REJECTION_REWARD, True

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

    def get_solution_reward(self, first_slot=-1):
        """
        Checks for solution
        """
        if first_slot == -1:
            first_slot = self.first_slot
        self.path_selected = first_slot // (NUMBER_OF_SLOTS + 1)
        self.ans_grid = self.path_grid(self.paths[self.path_selected])
        self.temp_first_slot = first_slot - self.path_selected * (NUMBER_OF_SLOTS + 1)
        try:
            for row in self.ans_grid.values():  # for spectrum of each link
                for i in range(self.slots):  # for each slot
                    if (
                        row[self.temp_first_slot + i] != 0
                    ):  # if slot in spectrum is occupied
                        return REJECTION_REWARD
        except IndexError:  # if one of slots is a gap
            return GAP_REJECTION_REWARD
        return SOLUTION_REWARD

    def path_grid(self, path):
        i = 0
        all_edges = []
        while i < len(path) - 1:  # prepare all edges in path
            if path[i] < path[i + 1]:
                all_edges.append((path[i], path[i + 1]))
            else:
                all_edges.append((path[i + 1], path[i]))
            i += 1

        temp_path_grid = {}
        for edge in all_edges:  # populate answer grid with edges
            temp_path_grid[edge] = self.link_grid[edge]
        return temp_path_grid

    def update_link_grid(self):
        for edge in self.ans_grid.keys():
            grid = self.link_grid[edge]
            for i in range(self.slots):
                grid[self.temp_first_slot + i] = 1
            self.link_grid[edge] = grid  #

    def seed(self):
        np.random.seed(SEED)

    def exit(self):
        pygame.quit()
        sys.exit()


def main():  # only used for human mode
    done = False
    game = ArcadeGame()
    game.new_game()
    game.draw_screen()
    game.render()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    pygame.quit()
    exit(0)
    while True:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                game.exit()
            if event.type == pygame.KEYDOWN:
                if (
                    event.key == pygame.K_RIGHT
                    and game.first_slot < NUMBER_OF_SLOTS * K + K - 1 - game.slots
                ):
                    if game.first_slot + game.slots in game.gaps:
                        game.first_slot += game.slots + 1
                    else:
                        game.first_slot += 1
                    game.update_spec_grid()
                    game.draw_screen()
                    game.render()
                elif event.key == pygame.K_LEFT and game.first_slot > 0:
                    if game.first_slot - 1 in game.gaps:
                        game.first_slot -= game.slots + 1
                    else:
                        game.first_slot -= 1
                    game.update_spec_grid()
                    game.draw_screen()
                    game.render()
                elif event.key == pygame.K_RETURN:
                    reward, done = game.check_solution()
                    if done:
                        game.new_game()
                    game.draw_screen()
                    game.render()


if __name__ == "__main__":
    main()
