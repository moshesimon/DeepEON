import gym
import stable_baselines3
from stable_baselines3.common import env_checker
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3 import DQN
from envs.custom_env import CustomEnv
#from stable_baselines.common.evaluation import evaluate_policy
env = CustomEnv()

model = DQN.load("deepq_EON4")
print("loaded")
#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#print(f"Mean Reward: {mean_reward}, STD Reward: {std_reward}")
env.highscore = 0
while True:
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    