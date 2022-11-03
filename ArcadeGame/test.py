# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
# from stable_baselines3 import DQN
# from envs.custom_env import CustomEnv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from torch import layout

# HEURISTIC_NAME = "KSP_FF"
# DEEPEON_NAME = "silver-glade-5"
# SEED = "1"

# df1 = pd.read_json(f"Evaluation data\evaluation_{HEURISTIC_NAME}_seed_{SEED}.json") #baseline
# df2 = pd.read_json(f"Evaluation data\evaluation_{DEEPEON_NAME}_seed_{SEED}.json") #DQL
# larger = []
# r1, r2 = list(df1["Episode Rewards"]), list(df2["Episode Rewards"])
# for i in range(100):
#     if r1[i] < r2[i]:
#         larger.append(r2[i]-r1[i])
# print(len(larger), larger)
fig, ax = plt.subplots()
KSP = [32.3,30.9,34.6,33.7,36.0]
DEEP = [29.5,32.1,31.3,31.5,28.4]
plt.title("DeepEON vs KSP-FF")
plt.xlabel("Seed ")
plt.ylabel("Avarage Score")
from matplotlib.ticker import MaxNLocator

seed = [0,1,2,3,4]

ax.scatter(seed,DEEP,label ="DeepEON")
ax.scatter(seed,KSP,label = "KSP-FF")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend()

plt.show()
