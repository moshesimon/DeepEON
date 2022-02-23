from random import seed
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3 import DQN
from envs.custom_env import CustomEnv
from wandb.integration.sb3 import WandbCallback
import wandb

model_config = {
  "policy_type": CnnPolicy,
  "total_timesteps": 25000,
  "learning_starts": 5000,
}

game_config = {
  "solution_reward": 10,
  "rejection_reward": -10,
  "left_reward": 0,
  "right_reward": 0,
  "seed": 0
}

run = wandb.init(
  project="sb3",
  config=model_config,
  sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
)

env = CustomEnv(game_config)

model = DQN(model_config["policy_type"], 
            env, verbose=1, 
            tensorboard_log=f"./tensorboardEON/{run.id}", 
            seed= model_config["seed"], 
            learning_starts=model_config["learning_starts"])

model.learn(total_timesteps=model_config["total_timesteps"], 
            callback=WandbCallback(model_save_path=f"Models/{run.id}"))