from datetime import datetime
import pygame
from stable_baselines3 import DQN
from envs.custom_env import CustomEnv
from Games.game6 import ArcadeGame, K, COLUMN_COUNT
import numpy as np
import pandas as pd
from datetime import datetime

episode_count_targets = 100
game_config = {
  "solution_reward": 10,
  "rejection_reward": -10,
  "left_reward": 0,
  "right_reward": 0,
  "seed": 0,
  "max_blocks": 1,
}
 
game = ArcadeGame(game_config)
episode_count = 0
episode_rewards = []
while episode_count < episode_count_targets:
    episode_reward = 0
    done = False
    game.new_game()
    game.draw_screen()
    #game.render()
    while not done:
        solution = False
        for k in range(K):
            for i in range(COLUMN_COUNT-game.slots+1): #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                first_slot = k*(COLUMN_COUNT+1) + i
                if game.is_solution(first_slot=first_slot):
                    solution = True
                    game.first_slot = first_slot
                    break
            if solution:
                break
        if not solution:
            game.first_slot = 0
        game.update_spec_grid()
        game.draw_screen()
        #game.render()
        reward, done = game.check_solution()
        episode_reward += reward
        #print(episode_reward)
    
    episode_rewards.append(episode_reward)
    episode_count += 1
    print(episode_count)
    
    
mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print(mean_reward)
index = np.arange(0,episode_count_targets)
df = pd.DataFrame({"index":index,"Episode Rewards":np.array(episode_rewards)})
current_date_time = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
df.to_json(f"Evaluation data\evaluation_KSP_FF_{current_date_time}.json")
    