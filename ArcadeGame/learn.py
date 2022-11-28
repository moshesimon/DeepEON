from random import seed
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3 import DQN
from ArcadeGame.envs.custom_env2 import CustomEnv
from wandb.integration.sb3 import WandbCallback
import wandb
import argparse
from datetime import datetime
import os
from ArcadeGame.config import current_dir, game_config

current_date_time = datetime.now().strftime("%m.%d.%Y_%H.%M.%S")

# print("model loaded")
# parms = model.get_parameters()

# run = wandb.init(
#  project = "test2")#,
#  #config = parms)#,
#  sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
# )

env = CustomEnv(game_config)
model = DQN(
    CnnPolicy,
    env,
    verbose=1,
    tensorboard_log=os.path.join(current_dir, "tensorboardEON", current_date_time),
    buffer_size=5000,
)
model.learn(total_timesteps=100000)
model.save(os.path.join(current_dir, "Models", current_date_time))
