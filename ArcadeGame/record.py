import pygame
from stable_baselines3 import DQN
from envs.custom_env import CustomEnv as CustomEnv1
from envs.custom_env2 import CustomEnv as CustomEnv2
from envs.custom_env3 import CustomEnv as CustomEnv3
import cv2
import os
from config import current_dir, full_name, all_configs

NUMBER_OF_NODES = all_configs["number_of_nodes"]
NUMBER_OF_SLOTS = all_configs["number_of_slots"]

NUMBER_OF_NODES = 3
NUMBER_OF_SLOTS = 8
num = '13'
env = CustomEnv3()

full_name = f"env3_{NUMBER_OF_NODES}_nodes_{NUMBER_OF_SLOTS}_slots_test{num}"

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
while games < 10:
    obs = env.reset()
    done = False
    record_ep = False
    old_rew = 0
    temp_frame_array = []
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print(reward)
        if reward > 0 or record_ep:
            record_ep = True
            frame = env.render(mode="rgb_array")
            temp_frame_array.append(frame)
        if done and record_ep:
            print(f"game {games} done")
            games += 1
            for frame in temp_frame_array:
                frame_array.append(frame)

record(frame_array)
env.close()
