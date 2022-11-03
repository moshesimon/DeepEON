from gym import Env
from gym import spaces
import numpy as np
from Games.game6 import ArcadeGame, SCREEN_HEIGHT,SCREEN_WIDTH,COLUMN_COUNT,K

class CustomEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    num_envs = 1
    
    def __init__(self, config):
        self.config = config 
        self.game = ArcadeGame(self.config)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(shape= (SCREEN_WIDTH, SCREEN_HEIGHT, 3),low=0,high=255,dtype=np.uint8)
        
    def step(self, action):
        reward, done, info = 0, False, {} 
        if action == 0 and self.game.first_slot < COLUMN_COUNT*K + K-1 - self.game.slots: # RIGHT
            if self.game.first_slot + self.game.slots in self.game.gaps:
                self.game.first_slot += self.game.slots + 1
            else:
                self.game.first_slot +=1
            self.game.update_spec_grid()
            reward = self.config["right_reward"]
        elif action == 1 and self.game.first_slot > 0: #LEFT
            if self.game.first_slot - 1 in self.game.gaps:
                self.game.first_slot -= self.game.slots + 1
            else:
                self.game.first_slot -=1
            self.game.update_spec_grid()
            reward = self.config["left_reward"]
        elif action == 2: # ENTER
            reward, done = self.game.check_solution()
            
        observation = self.game.draw_screen()

        return observation, reward, done, info

    def reset(self):
        self.game.new_game()
        observation = self.game.draw_screen()
        return observation 

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return self.game.draw_screen() # return RGB frame suitable for video
        else:
            super(CustomEnv, self).render(mode=mode) # just raise an exceptionset
        
    def close (self):
        self.game.exit()
        
    