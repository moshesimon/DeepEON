import pygame
from stable_baselines3 import DQN
from envs.custom_env import CustomEnv, SCREEN_HEIGHT, SCREEN_WIDTH
import cv2
import os

DEEPEON_NAME = "11.09.2022_10.05.53"
TIMESTEPS = 5800000
current_dir = os.path.dirname(os.path.abspath(__file__))


def record():
    print("saving..")
    height, width, layers = frame_array[0].shape
    out = cv2.VideoWriter(
        os.path.join(
            current_dir, "Recordings", f"video_{DEEPEON_NAME}_timestep_{TIMESTEPS}.mp4"
        ),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (width, height),
    )
    for i in range(len(frame_array)):
        out.write(frame_array[i])
    out.release()


game_config = {
    "solution_reward": 10,
    "rejection_reward": -10,
    "left_reward": 0,
    "right_reward": 0,
    "seed": 0,
    "max_blocks": 1,
}


env = CustomEnv(game_config)
model = DQN.load(
    os.path.join(current_dir, "Models", f"{DEEPEON_NAME}", f"{TIMESTEPS}", "model")
)
print("model loaded")

env.highscore = 0
frame_array = []
episodes = 0
while episodes < 10:
    episodes += 1
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        frame = env.render(mode="rgb_array")
        frame_array.append(frame)

record()
env.close()
