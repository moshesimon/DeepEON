from stable_baselines3 import DQN
from envs.custom_env import CustomEnv

model_config = {
  "total_timesteps": 25000,
}

game_config = {
  "solution_reward": 10,
  "rejection_reward": -10,
  "left_reward": 0,
  "right_reward": 0,
  "seed": 0
}
env = CustomEnv(game_config)
model = DQN.load("deepq_EON3")
model.set_env(env)
model.learn(total_timesteps=model_config["total_timesteps"],tb_log_name="Logs",reset_num_timesteps=False)
model.save("deepq_EON3_continued_1")