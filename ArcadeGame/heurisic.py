from datetime import datetime
from Games.game6 import ArcadeGame
import numpy as np
import pandas as pd
from datetime import current_date_time
import os
from config import current_dir, heuristic_config

episode_count_targets = 100
full_name = f"{heuristic_config['number_of_slots_evaluated']}_0_{heuristic_config['K']}_{heuristic_config['solution_reward']}_{heuristic_config['rejection_reward']}_{heuristic_config['seed']}_{heuristic_config['max_blocks']}"

game = ArcadeGame(heuristic_config)
episode_count = 0
episode_rewards = []
while episode_count < episode_count_targets:
    episode_reward = 0
    done = False
    game.new_game()
    game.draw_screen()
    game.render()
    while not done:
        solution = False
        for k in range(K):
            for i in range(
                COLUMN_COUNT - game.slots + 1
            ):  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                first_slot = k * (COLUMN_COUNT + 1) + i
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
        # game.render()
        reward, done = game.check_solution()
        episode_reward += reward
        # print(episode_reward)

    episode_rewards.append(episode_reward)
    episode_count += 1
    print(episode_count)


mean_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)
print(mean_reward)
index = np.arange(0, episode_count_targets)
df = pd.DataFrame({"index": index, "Episode Rewards": np.array(episode_rewards)})
df.to_json(
    os.path.join(current_dir, "Evaluations", f"heurisrtic_evaluation_{full_name}.json")
)
