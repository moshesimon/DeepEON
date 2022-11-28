from itertools import islice
import pygame
import numpy as np
import networkx as nx
import sys
import os
from ArcadeGame.config import all_configs


NUMBER_OF_SLOTS = all_configs["number_of_slots"]
# SCREEN_NUMBER_OF_SLOTS = all_configs["screen_number_of_slots"]
K = all_configs["K"]
WIDTH = all_configs["width"]
HEIGHT = all_configs["height"]
SCREEN_WIDTH = all_configs["screen_width"]
SCREEN_HEIGHT = all_configs["screen_height"]
SCREEN_SIDE_OFFSET = all_configs["screen_side_offset"]
PATH_ROWS = all_configs["path_rows"]
SPECTRUM_SLOTS_ROWS_FROM_TOP = all_configs["spectrum_slots_rows_from_top"]
WHITE = all_configs["white"]
BLACK = all_configs["black"]
GREEN = all_configs["green"]
RED = all_configs["red"]
SOLUTION_REWARD = all_configs["solution_reward"]
REJECTION_REWARD = all_configs["rejection_reward"]
LEFT_REWARD = all_configs["left_reward"]
RIGHT_REWARD = all_configs["right_reward"]
SEED = all_configs["seed"]
END_LIMIT = all_configs["end_limit"]


class ArcadeGame:
    def __init__(self, config):
        self.config = config
        self.window = (SCREEN_WIDTH, SCREEN_HEIGHT)
        self.background = pygame.Surface(self.window)
        self.highscore = 0
        self.edges = [(1, 2), (2, 3), (1, 4), (3, 5), (2, 5), (4, 5), (3, 6), (4, 6)]
        self.G = nx.Graph()
        self.G.add_edges_from(self.edges)
        self.gaps = []
        for i in range(K):
            self.gaps.append(NUMBER_OF_SLOTS + (NUMBER_OF_SLOTS + 1) * i)
        self.seed()

    def draw_screen(self):
        self.background.fill(RED)
        for k, path in enumerate(self.paths):
            for i, row in enumerate(self.path_grid(path).values()):  # print links grid
                for column in range(NUMBER_OF_SLOTS):
                    if row[column] == 0:
                        self.draw_box(
                            column + SCREEN_SIDE_OFFSET + k * (NUMBER_OF_SLOTS + 1),
                            PATH_ROWS - i,
                            WHITE,
                        )
                    else:
                        self.draw_box(
                            column + SCREEN_SIDE_OFFSET + k * (NUMBER_OF_SLOTS + 1),
                            PATH_ROWS - i,
                            BLACK,
                        )

        for column in range(NUMBER_OF_SLOTS * K + K - 1):  # print slots
            if self.spec_grid[column] == 0:
                self.draw_box(
                    column + SCREEN_SIDE_OFFSET, SPECTRUM_SLOTS_ROWS_FROM_TOP, RED
                )
            else:
                self.draw_box(
                    column + SCREEN_SIDE_OFFSET, SPECTRUM_SLOTS_ROWS_FROM_TOP, GREEN
                )

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

    def new_game(self):
        self.score = 0
        self.reward = 0
        self.blocks = 0
        self.link_grid = {}
        for edge in self.edges:  # populate link grid
            self.link_grid[edge] = np.zeros(NUMBER_OF_SLOTS, dtype=int)
        self.new_round()

    def new_round(self):
        """
        Sets up all parameters for a new round
        """
        self.first_slot = 0
        self.target = np.random.randint(2, 7)
        self.source = np.random.randint(1, self.target)
        p = nx.shortest_simple_paths(self.G, self.source, self.target)
        self.paths = list(islice(p, K))
        self.slots = np.random.randint(1, 5)
        self.update_spec_grid()  # populate spectrum grid

    def update_spec_grid(self):
        self.spec_grid = np.zeros(NUMBER_OF_SLOTS * K + K - 1, dtype=int)
        try:
            for i in range(self.slots):
                self.spec_grid[self.first_slot + i] = 1
            return 0, False
        except:
            return REJECTION_REWARD, True

    def check_solution(self):
        done = False
        if self.is_solution():
            reward = SOLUTION_REWARD
            self.reward += reward
            self.score += 10
            self.update_link_grid()
            self.new_round()
        else:
            reward = REJECTION_REWARD
            self.blocks += 1
            self.reward += reward
            if self.blocks >= END_LIMIT:
                if self.score > self.highscore:
                    self.highscore = self.score
                done = True
        return reward, done

    def is_solution(self, first_slot=-1):
        """
        Checks for solution
        """
        if first_slot == -1:
            first_slot = self.first_slot
        self.path_selected = first_slot // (NUMBER_OF_SLOTS + 1)
        self.ans_grid = self.path_grid(self.paths[self.path_selected])
        self.temp_first_slot = first_slot - self.path_selected * (NUMBER_OF_SLOTS + 1)
        for row in self.ans_grid.values():  # for spectrum of each link
            for i in range(self.slots):  # for each slot
                if (
                    row[self.temp_first_slot + i] != 0
                ):  # if slot in spectrum is occupied
                    return False
        return True

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
                elif event.key == pygame.K_LEFT and game.first_slot > 0:
                    if game.first_slot - 1 in game.gaps:
                        game.first_slot -= game.slots + 1
                    else:
                        game.first_slot -= 1
                    game.update_spec_grid()
                elif event.key == pygame.K_RETURN:
                    reward, done = game.check_solution()
                    if done:
                        game.new_game()

        game.draw_screen()
        game.render()


if __name__ == "__main__":
    main()
