import pygame
from stable_baselines3.dqn.dqn import DQN
from envs.custom_env import CustomEnv

game_config = {
    "solution_reward": 10,
    "rejection_reward": -100,
    "left_reward": -1,
    "right_reward": -10,
    "seed": 0,
}

env = CustomEnv(game_config)

model = DQN.load("Models/silver-glade-5/model")
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
