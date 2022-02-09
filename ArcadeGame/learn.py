import gym
import stable_baselines3
from stable_baselines3.common import env_checker
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3 import DQN
from envs.custom_env import CustomEnv
#from stable_baselines.common.evaluation import evaluate_policy
env = CustomEnv()

#env_checker.check_env(env, warn=True, skip_render_check=True)

model = DQN(CnnPolicy, env, verbose=1,tensorboard_log="./tensorboardEON/")
model.learn(total_timesteps=1500)
model.save("deepq_EON4")