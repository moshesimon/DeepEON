from gym import Env
from gym import spaces
import numpy as np
from game5 import ArcadeGame

SCREEN_WIDTH = 920
SCREEN_HEIGHT = 150

class CustomEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}


    def __init__(self):
        self.game = ArcadeGame()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(shape= (SCREEN_WIDTH, SCREEN_HEIGHT,3),low=0,high= 255,dtype=np.uint8)
        
    def step(self, action):
        reward, done, info = 0, False, {} 

        if action == 0 and self.game.first_slot < 44-self.game.slots: # RIGHT
            self.game.first_slot +=1
            self.game.update_spec_grid()
            reward = 1
        elif action == 1 and self.game.first_slot > 0: #LEFT
            self.game.first_slot -=1
            self.game.update_spec_grid()
            reward = 1
        elif action == 2: # ENTER
            reward, done = self.game.check_solution()
        
        observation = self.game.draw_screen()

        return observation, reward, done, info

    def reset(self):
        self.game.new_game()
        observation = self.game.draw_screen()
        return observation 

    def render(self):
        self.game.render()

    def close (self):
        self.game.exit()
        
    