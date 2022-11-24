from random import seed
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.dqn.dqn import DQN
from envs.custom_env2 import CustomEnv, COLUMN_COUNT, SCREEN_COLUMN_COUNT, K
from wandb.integration.sb3 import WandbCallback
import wandb
import argparse
from datetime import datetime
import os
import pathlib
from config import current_dir, all_configs


parse = False
# Build your ArgumentParser however you like
def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size")
    parser.add_argument("--batch_size")
    parser.add_argument("--exploration_final_eps")
    parser.add_argument("--exploration_fraction")
    parser.add_argument("--gamma")
    parser.add_argument("--learning_rate")
    parser.add_argument("--learning_starts")
    parser.add_argument("--target_update_interval")
    parser.add_argument("--train_freq")
    parser.add_argument("--total_timesteps")
    return parser


model_config = {
    "number_of_slots": all_configs["number_of_slots"],
    "screen_number_of_slots": all_configs["screen_number_of_slots"],
    "K": all_configs["K"],
    "solution_reward": all_configs["solution_reward"]
    "rejection_reward": all_configs["rejection_reward"],
    "left_reward": all_configs["left_reward"],
    "right_reward": all_configs["right_reward"],
    "seed": all_configs["seed"],
    "max_blocks": all_configs["max_blocks"],
    "total_timesteps": all_configs["total_timesteps"],
    "save_every_timesteps": all_configs["save_every_timesteps"],
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

config = model_config

if parse:
    parser = setup_parser()

    # Get the hyperparameters
    args = parser.parse_args()

    args_config = {
        "solution_reward": model_config.get("solution_reward"),
        "rejection_reward": model_config.get("rejection_reward"),
        "left_reward": model_config.get("left_reward"),
        "right_reward": model_config.get("right_reward"),
        "seed": model_config.get("seed"),
        "max_blocks": model_config.get("max_blocks"),
        "buffer_size": int(args.buffer_size),
        "batch_size": int(args.batch_size),
        "exploration_final_eps": float(args.exploration_final_eps),
        "exploration_fraction": float(args.exploration_fraction),
        "gamma": float(args.gamma),
        "learning_rate": float(args.learning_rate),
        "learning_starts": int(args.learning_starts),
        "target_update_interval": int(args.target_update_interval),
        "train_freq": int(args.train_freq),
        "total_timesteps": int(args.total_timesteps),
    }

    config = args_config


current_date_time = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")

wandb.init(
    project="DeepEON",
    entity="deepeon",
    name=f"{current_date_time}",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)


env = CustomEnv(config)

model = DQN(
    CnnPolicy,
    env,
    verbose=1,
    tensorboard_log=os.path.join("tensorboardEON", wandb.run.name),
    learning_starts=config["learning_starts"],
    buffer_size=config["buffer_size"],
    batch_size=config["batch_size"],
    exploration_final_eps=config["exploration_final_eps"],
    exploration_fraction=config["exploration_fraction"],
    gamma=config["gamma"],
    learning_rate=config["learning_rate"],
    train_freq=config["train_freq"],
)


for i in range(1, int(config["total_timesteps"] / config["save_every_timesteps"]) + 1):
    model.learn(
        total_timesteps=config["save_every_timesteps"],
        callback=WandbCallback(
            model_save_path=os.path.join(
                "Models", f"{wandb.run.name}", f"{config['save_every_timesteps']*i}"
            ),
            verbose=2,
        ),
        tb_log_name=f"{wandb.run.name}",
        reset_num_timesteps=False,
    )
wandb.run.finish()
env.close()
