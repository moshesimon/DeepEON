from stable_baselines3 import DQN
from ArcadeGame.envs.custom_env import CustomEnv
from ArcadeGame.config import current_dir, all_configs, game_config

env = CustomEnv(game_config)
model = DQN.load("deepq_EON3")
model.set_env(env)
model.learn(
    total_timesteps=all_configs["total_timesteps"],
    tb_log_name="Logs",
    reset_num_timesteps=False,
)
model.save("deepq_EON3_continued_1")
