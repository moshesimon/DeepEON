from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from config import current_dir


PLOT_TITLE = "16 slots env2 end 1 (max=1) vs 16 slots env2 end 2 (max=27)"
PLAYER_TYPE = ["agent", "agent"]
LEGEND = ["DeepEON end 1", "DeepEON end 2"]
NUMBER_OF_SLOTS_EVALUATED = [16, 16]
NUMBER_OF_EPISODES_EVALUATED = [1000, 1000]
NUMBER_OF_SLOTS_TRAINED = [16, 16]
K = [3, 3]
SOLUTION_REWARD = [10, 10]
REJECTION_REWARD = [-10, -10]
GAP_REJECTION_REWARD = [-15, -15]
SEED = [0, 0]
END_LIMIT = [1, 27]
EPISODE_END = [1, 2]
ENV = [2, 2]
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
            f"{PLAYER_TYPE[i]}_evaluation_{NUMBER_OF_SLOTS_EVALUATED[i]}_{NUMBER_OF_EPISODES_EVALUATED[i]}_{NUMBER_OF_SLOTS_TRAINED[i]}_{K[i]}_{SOLUTION_REWARD[i]}_{REJECTION_REWARD[i]}_{GAP_REJECTION_REWARD[i]}_{SEED[i]}_{ENV[i]}_{EPISODE_END[i]}_{END_LIMIT[i]}.json",
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
