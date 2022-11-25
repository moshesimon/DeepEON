import pygame
from stable_baselines3.dqn.dqn import DQN
from envs.custom_env import CustomEnv
import os
from config import current_dir, game_config

env = CustomEnv(game_config)

model = DQN.load(os.path.join(current_dir, "Models", "400-model"))
print("loaded")

env.game.highscore = 0
while True:

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            env.close()
