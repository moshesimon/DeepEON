from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from config import current_dir

"solar-microwave-1"
"silver-glade-5"
"DeepEON4"
"deepq_EON5"

PLOT_TITLE = "DeepEON vs Heuristic"
PLAYER_TYPE = ["agent", "heuristic"]
LEGEND = ["DeepEON", "Heuristic"]
NUMBER_OF_SLOTS_EVALUATED = [8, 8]
NUMBER_OF_EPISODES_EVALUATED = [100, 100]
NUMBER_OF_SLOTS_TRAINED = [8, 8]
K = [3, 3]
SOLUTION_REWARD = [10, 10]
REJECTION_REWARD = [-10, -10] 
SEED = [0, 0]
END_LIMIT = [1, 1]
ENV = [1, 1]
AVERAGE_OVER = 50

fig, ax = plt.subplots()
plt.title(PLOT_TITLE)
plt.xlabel("Rounds")
plt.ylabel("Score")

eps= []
for x in range(NUMBER_OF_EPISODES_EVALUATED[0]//AVERAGE_OVER):
    eps.append(x*AVERAGE_OVER + AVERAGE_OVER)

for i in range(len(PLAYER_TYPE)):
    df = pd.read_json(
        os.path.join(
            current_dir,
            "Evaluations",
            f"{PLAYER_TYPE[i]}_evaluation_{NUMBER_OF_SLOTS_EVALUATED[i]}_{NUMBER_OF_EPISODES_EVALUATED[i]}_{NUMBER_OF_SLOTS_TRAINED[i]}_{K[i]}_{SOLUTION_REWARD[i]}_{REJECTION_REWARD[i]}_{SEED[i]}_{END_LIMIT[i]}_{ENV[i]}.json",
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
