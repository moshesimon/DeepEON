import gym
import stable_baselines3
from stable_baselines3.common import env_checker
from stable_baselines3.dqn import CnnPolicy
from stable_baselines3 import DQN
from envs.custom_env import CustomEnv
#from stable_baselines.common.evaluation import evaluate_policy
env = CustomEnv()

#env_checker.check_env(env, warn=True, skip_render_check=True)

model = DQN(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=1500)
model.save("deepq_EON")

del model # remove to demonstrate saving and loading

model = DQN.load("deepq_EON")
print("loaded")

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#print(f"Mean Reward: {mean_reward}, STD Reward: {std_reward}")
env.highscore = 0
episode_rewards = []
while True:
    obs = env.reset()
    reward_sum = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    episode_rewards.append(reward_sum)