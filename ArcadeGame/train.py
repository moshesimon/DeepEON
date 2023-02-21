from random import seed
from stable_baselines3.dqn.policies import CnnPolicy
from stable_baselines3.dqn.dqn import DQN
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
import wandb
import argparse
from datetime import datetime
import os
import numpy as np
import pathlib
from config import current_dir, full_name, model_config, all_configs
from envs.custom_env import CustomEnv as CustomEnv1
from envs.custom_env2 import CustomEnv as CustomEnv2
from envs.custom_env3 import CustomEnv as CustomEnv3


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


NUMBER_OF_NODES = all_configs["number_of_nodes"]
NUMBER_OF_SLOTS = all_configs["number_of_slots"]

full_name = f"env3_{NUMBER_OF_NODES}_nodes_{NUMBER_OF_SLOTS}_slots_test15"

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


config = model_config

if model_config["env"] == 1:
    env = CustomEnv1()
elif model_config["env"] == 2:
    env = CustomEnv2()
elif model_config["env"] == 3:
    env = CustomEnv3()
else:
    print("env not selected correctly in config.py")
    exit(1)

if parse:
    parser = setup_parser()

    # Get the hyperparameters
    args = parser.parse_args()

    args_config = {
        "number_of_slots": config.get("number_of_slots"),
        "screen_number_of_slots": config.get("screen_number_of_slots"),
        "solution_reward": config.get("solution_reward"),
        "rejection_reward": config.get("rejection_reward"),
        "left_reward": config.get("left_reward"),
        "right_reward": config.get("right_reward"),
        "seed": config.get("seed"),
        "end_limit": config.get("end_limit"),
        "K": config.get("K"),
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

run = wandb.init(
    project="DeepEON",
    entity="deepeon",
    name=full_name,
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

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

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

walldb_callback = WandbCallback(
    model_save_path=os.path.join(current_dir, "Models", full_name),
    verbose=2,
)
save_on_best_training_reward_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
callback_list = CallbackList(callbacks=[save_on_best_training_reward_callback, walldb_callback])

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=callback_list,
    tb_log_name=full_name,
    reset_num_timesteps=False,
)
run.finish()
env.close()
