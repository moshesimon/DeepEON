from stable_baselines3.common import base_class
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from envs.custom_env import CustomEnv
import numpy as np
import gym
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd





def evaluate(
    model: "base_class.BaseAlgorithm",
    env: gym.Env,
    n_eval_episodes: int = 100,
    deterministic: bool = False,
    render: bool = False,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
):
    episode_counts = 0
    episode_rewards = []
    episode_lengths = []
    current_rewards = 0
    current_lengths = 0
    observations = env.reset()
    states = None
    episode_starts = True
    episode_count_targets = n_eval_episodes
    while episode_counts < episode_count_targets:
        actions, states = model.predict(observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        if dones:
            episode_rewards.append(current_rewards)
            episode_lengths.append(current_lengths)
            episode_counts += 1
            current_rewards = 0
            current_lengths = 0
            observations = env.reset()
            print(episode_counts)

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward

n_episodes = 10
env = CustomEnv()
env.seed(0)
model = DQN.load("Models\deepq_EON5")
model.set_env(env)
episode_rewards, episode_lengths = evaluate(model,env,n_eval_episodes=n_episodes,return_episode_rewards=True)
index = np.arange(0,n_episodes)
df = pd.DataFrame({"index":index,"Episode Rewards":np.array(episode_rewards), "Episode Lengths": np.array(episode_lengths)})
df.to_json("Evaluation data\evaluation_date_2.json")
