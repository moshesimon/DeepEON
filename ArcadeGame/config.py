import os

current_dir = os.path.dirname(os.path.abspath(__file__))


temp_configs = {
    "env": 2,
    "episode_end": 1,
    "number_of_slots": 8,
    "screen_number_of_slots": 8,
    "number_of_slots_evaluated": 8,
    "number_of_episodes_evaluated": 1000,
    "K": 3,
    "solution_reward": 10,
    "rejection_reward": -10,
    "left_reward": 0,
    "right_reward": 0,
    "seed": 0,
    "end_limit": 3,
    "total_timesteps": 1000000,
    "buffer_size": 100000,
    "batch_size": 32,
    "exploration_final_eps": 0.1,
    "exploration_fraction": 0.5,
    "gamma": 0.999,
    "learning_rate": 0.00025,
    "learning_starts": 50000,
    "target_update_interval": 10000,
    "train_freq": (4, "step"),
    "width": 20,
    "height": 20,
    "screen_width": 0,
    "screen_height": 150,
    "screen_side_offset": 1,
    "path_rows": 5,
    "spectrum_slots_rows_from_top": 6,
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "green": (0, 255, 0),
    "red": (255, 0, 0),
}

temp_configs["screen_width"] = (
    temp_configs["screen_number_of_slots"] * temp_configs["width"]
    * temp_configs["K"]
    + (temp_configs["K"] + 1) * temp_configs["width"]
)

all_configs = temp_configs
