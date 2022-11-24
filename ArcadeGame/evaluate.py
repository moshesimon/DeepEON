from stable_baselines3.common import base_class
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn.dqn import DQN
from envs.custom_env import CustomEnv
import numpy as np
import gym
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import config
from config import current_dir, all_configs

DEEPEON_NAME = "11.04.2022_01.02.31"
TIMESTEPS = 1300000


def evaluate(
    model: "base_class.BaseAlgorithm",
    env: gym.Env,
    n_eval_episodes: int = 100,
    deterministic: bool = False,
    render: bool = False,
):
    print("Starting..")
    episode_count = 0
    episode_rewards = []
    episode_lengths = []
    while episode_count < n_eval_episodes:
        current_reward = 0
        current_length = 0
        done = False
        observation = env.reset()
        while not done:
            action, state = model.predict(observation, deterministic=deterministic)
            observation, reward, done, info = env.step(action)
            current_reward += reward
            current_length += 1
            if render:
                env.render()

        episode_rewards.append(current_reward)
        episode_lengths.append(current_length)
        episode_count += 1

        if render:
            env.render()
        else:
            print(episode_count)

    return episode_rewards, episode_lengths


n_episodes = 100
game_config = {
    "solution_reward": all_configs["solution_reward"],
    "rejection_reward": all_configs["rejection_reward"],
    "left_reward": all_configs["left_reward"],
    "right_reward": all_configs["right_reward"],
    "seed": all_configs["seed"],
    "max_blocks": all_configs["max_blocks"],
}

env = CustomEnv(game_config)
env.seed(game_config["seed"])
model = DQN.load(
    os.path.join(current_dir, "Models", f"{DEEPEON_NAME}", f"{TIMESTEPS}", "model")
)
print("Loaded")
model.set_env(env)
episode_rewards, episode_lengths = evaluate(model, env, n_episodes, render=False)
index = np.arange(0, n_episodes)
df = pd.DataFrame(
    {
        "index": index,
        "Episode Rewards": np.array(episode_rewards),
        "Episode Lengths": np.array(episode_lengths),
    }
)
df.to_json(
    os.path.join(
        current_dir,
        "Evaluations",
        f"evaluation_{DEEPEON_NAME}_seed_{game_config['seed']}.json",
    )
)
