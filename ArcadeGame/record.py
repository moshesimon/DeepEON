import pygame
from stable_baselines3 import DQN
from envs.custom_env import CustomEnv as CustomEnv1
from envs.custom_env2 import CustomEnv as CustomEnv2
from envs.custom_env3 import CustomEnv as CustomEnv3
import cv2
import os
import numpy as np
from config import current_dir, full_name, all_configs, logger


if all_configs["env"] == 1:
    env = CustomEnv1()
elif all_configs["env"] == 2:
    env = CustomEnv2()
elif all_configs["env"] == 3:
    env = CustomEnv3(mode="rgb_array")
    from envs.custom_env3 import SCREEN_WIDTH, SCREEN_HEIGHT
else:
    print("env not selected correctly in config.py")
    exit(1)

# NUMBER_OF_NODES = all_configs["number_of_nodes"]
# NUMBER_OF_SLOTS = all_configs["number_of_slots"]

# NUMBER_OF_NODES = 3
# NUMBER_OF_SLOTS = 8
# num = '13'
# env = CustomEnv3()

# full_name = f"env3_{NUMBER_OF_NODES}_nodes_{NUMBER_OF_SLOTS}_slots_test{num}"

# SCREEN_HEIGHT = all_configs["height"]
# SCREEN_WIDTH = all_configs["width"]

# if all_configs["env"] == 1:
#     env = CustomEnv1()
# elif all_configs["env"] == 2:
#     env = CustomEnv2()
# elif all_configs["env"] == 3:
#     env = CustomEnv3()
# else:
#     print("env not selected correctly in config.py")
#     exit(1)

logger.info(f"full name: {full_name}")

def record(frame_array):
    print("saving..")
    height, width, layers = frame_array[0].shape
    out = cv2.VideoWriter(
        os.path.join(current_dir, "Recordings", f"video_{full_name}.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (height, width),
    )
    for i in range(len(frame_array)):
        rotated=cv2.rotate(frame_array[i], cv2.ROTATE_90_CLOCKWISE)
        flipped_horizontally = cv2.flip(rotated, 1)
        out.write(flipped_horizontally)
    out.release()


model = DQN.load(os.path.join(current_dir, "Models", f"{full_name}", "model"))
print("model loaded")

env.highscore = 0
frame_array = []
games = 0
while games < 100:
    obs = env.reset()
    done = False
    record_ep = False
    old_rew = 0
    temp_frame_array = []
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        # print(reward)
        if reward > 0 and not record_ep: # start recoding this episode
            record_ep = True
            episode_reward = 0

        if record_ep:
            episode_reward += reward
            record_ep = True
            env.set_mode("human")
            frame = env.render()
            # env.set_mode("rgb_array")
            temp_frame_array.append(frame)

        if record_ep and done:
            print(f"game {games} done, reward: {episode_reward}")
            record_ep = False
            games += 1
            for frame in temp_frame_array:
                frame_array.append(frame)
            frame = np.ones((temp_frame_array[0].shape[0], temp_frame_array[0].shape[1], temp_frame_array[0].shape[2]), dtype=np.uint8) * 255
            for i in range(10):
                frame_array.append(frame)

record(frame_array)
env.close()
