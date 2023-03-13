from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from config import current_dir


PLOT_TITLE = "DeepEON vs Heuristic (24 Slots,env=2,ep_end=1,end_lim=1, k=1)"
LEGEND = ["Game 6", "Game 8"]
PLAYER_TYPE = ["agent", "agent"]
GAMMA = [0.99, 0.99]
BATCH_SIZE = [32, 32]
LEARNING_RATE = [0.001, 0.001]
LEARNING_STARTS = [50000, 50000]
TARGET_UPDATE_INTERVAL = [10000, 10000]
TRAINING_FREQUENCY = [4, 4]
MAX_STEPS = [1000000, 10000000]
BUFFER_SIZE = [10000, 100000]
EPXLOREATION_FINAL_EPS = [0, 0.01]
EXPLORATION_FRACTION = [0.5, 0.7]
GAME = [6, 8]
NUMBER_OF_SLOTS = [24,24]
K = [1,1]
SOLUTION_REWARD = [10, 10]
REJECTION_REWARD = [-10, -10]
GAP_REJECTION_REWARD = [0,0]
SEED = [0, 0]
ENV = [2, 2]
EPISODE_END = [1,1]
END_LIMIT = [1,1]
NUMBER_OF_EPISODES_EVALUATED = [1000, 1000]
SEED_EVAL = [1, 1]
AVERAGE_OVER = 50

fig, ax = plt.subplots()
plt.title(PLOT_TITLE)
plt.xlabel("Rounds")
plt.ylabel("Score")

eps = []
for x in range(NUMBER_OF_EPISODES_EVALUATED[0] // AVERAGE_OVER):
    eps.append(x * AVERAGE_OVER + AVERAGE_OVER)

for i in range(len(PLAYER_TYPE)):
    df = pd.read_json(
        os.path.join(
            current_dir,
            "Evaluations",
            f"{PLAYER_TYPE[i]}_evaluation_{GAMMA[i]}_{BATCH_SIZE[i]}_{LEARNING_RATE[i]}_{LEARNING_STARTS[i]}_{TARGET_UPDATE_INTERVAL[i]}_{TRAINING_FREQUENCY[i]}_{MAX_STEPS[i]}_{BUFFER_SIZE[i]}_{EPXLOREATION_FINAL_EPS[i]}_{EXPLORATION_FRACTION[i]}_{GAME[i]}_{NUMBER_OF_SLOTS[i]}_{K[i]}_{SOLUTION_REWARD[i]}_{REJECTION_REWARD[i]}_{GAP_REJECTION_REWARD[i]}_{SEED[i]}_{ENV[i]}_{EPISODE_END[i]}_{END_LIMIT[i]}_{NUMBER_OF_EPISODES_EVALUATED[i]}_{SEED_EVAL[i]}.json",
       )
    )
    mean_reward = np.mean(df["Episode Rewards"])
    std_reward = np.std(df["Episode Rewards"])
    print(f"{LEGEND[i]}: Mean Reward: {mean_reward}, STD Reward: {std_reward}")
    a = 0
    av_rew = []
    for j, er in enumerate(df["Episode Rewards"]):
        a += er
        if (j + 1) % AVERAGE_OVER == 0:
            av = a / AVERAGE_OVER
            av_rew.append(av)
            a = 0
    ax.plot(eps, av_rew, label=LEGEND[i])
    plt.legend()

plt.show()
