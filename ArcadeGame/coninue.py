from stable_baselines3 import DQN
from envs.custom_env import CustomEnv

env = CustomEnv()
model = DQN.load("deepq_EON3")
model.set_env(env)
model.learn(total_timesteps=1500,tb_log_name="Logs",reset_num_timesteps=False)
model.save("deepq_EON3_continued_1")