import os

current_dir = os.path.dirname(os.path.abspath(__file__))


temp_configs = {
    "game": 6,
    "env": 2,
    "episode_end": 1,
    "end_limit": 1,
    "number_of_slots": 24,
    "number_of_episodes_evaluated": 1000,
    "K": 1,
    "solution_reward": 10,
    "rejection_reward": -10,
    "gap_rejection_reward": 0,
    "left_reward": 0,
    "right_reward": 0,
    "seed": 0,
    "seed_eval": 1,
    "total_timesteps": 10000000,
    "buffer_size": 100000,
    "batch_size": 32,
    "exploration_final_eps": 0,
    "exploration_fraction": 0.7,
    "gamma": 0.99,
    "learning_rate": 0.001,
    "learning_starts": 50000,
    "target_update_interval": 10000,
    "train_freq": (4, "step"),
    "width": 20,
    "height": 20,
    "screen_width": 0,
    "screen_height": 0,
    "screen_side_offset": 1,
    "path_rows": 9,
    "spectrum_slots_rows_from_top": 10,
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "green": (255, 255, 0),
    "red": (255, 0, 0),
}

if temp_configs["game"] == 6:
    temp_configs["screen_height"] = 150
    temp_configs["path_rows"] = 5
    temp_configs["spectrum_slots_rows_from_top"] = 6
    temp_configs["green"] = (0, 255, 0)
    temp_configs["screen_width"] = (
    temp_configs["number_of_slots"] * temp_configs["width"]
    * temp_configs["K"]
    + (temp_configs["K"] + 1) * temp_configs["width"]
)
elif temp_configs["game"] == 8:
    temp_configs["screen_height"] = 260
    temp_configs["screen_width"] = (
        temp_configs["number_of_slots"] * temp_configs["width"]
        * temp_configs["K"]
        + (temp_configs["K"] + 2) * temp_configs["width"]
        + 2 * temp_configs["screen_side_offset"]
        + temp_configs["number_of_slots"] * temp_configs["width"]
        * (temp_configs["K"])
    )

all_configs = temp_configs

full_name = f"{all_configs['gamma']}_{all_configs['batch_size']}_{all_configs['learning_rate']}_{all_configs['learning_starts']}_{all_configs['target_update_interval']}_{all_configs['train_freq'][0]}_{all_configs['total_timesteps']}_{all_configs['buffer_size']}_{all_configs['exploration_final_eps']}_{all_configs['exploration_fraction']}_{all_configs['game']}_{all_configs['number_of_slots']}_{all_configs['K']}_{all_configs['solution_reward']}_{all_configs['rejection_reward']}_{all_configs['gap_rejection_reward']}_{all_configs['seed']}_{all_configs['env']}_{all_configs['episode_end']}_{all_configs['end_limit']}"

model_config = {
    "env": all_configs["env"],
    "episode_end": all_configs["episode_end"],
    "number_of_slots": all_configs["number_of_slots"],
    "K": all_configs["K"],
    "solution_reward": all_configs["solution_reward"],
    "rejection_reward": all_configs["rejection_reward"],
    "left_reward": all_configs["left_reward"],
    "right_reward": all_configs["right_reward"],
    "seed": all_configs["seed"],
    "end_limit": all_configs["end_limit"],
    "total_timesteps": all_configs["total_timesteps"],
    "buffer_size": all_configs["buffer_size"],
    "batch_size": all_configs["batch_size"],
    "exploration_final_eps": all_configs["exploration_final_eps"],
    "exploration_fraction": all_configs["exploration_fraction"],
    "gamma": all_configs["gamma"],
    "learning_rate": all_configs["learning_rate"],
    "learning_starts": all_configs["learning_starts"],
    "target_update_interval": all_configs["target_update_interval"],
    "train_freq": all_configs["train_freq"],
}

game_config = {
    "game": all_configs["game"],
    "K": all_configs["K"],
    "solution_reward": all_configs["solution_reward"],
    "rejection_reward": all_configs["rejection_reward"],
    "left_reward": all_configs["left_reward"],
    "right_reward": all_configs["right_reward"],
    "seed": all_configs["seed"],
    "seed_eval": all_configs["seed_eval"],
}