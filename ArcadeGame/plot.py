from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

"solar-microwave-1"
"silver-glade-5"
"DeepEON4"
"deepq_EON5"
HEURISTIC_NAME = "1_SP_FF_11.15.2022_14.28.09"
DEEPEON_NAME = "11.09.2022_10.05.53"
SLOTS = "8"
SEED = "0"
current_dir = os.path.dirname(os.path.abspath(__file__))

df1 = pd.read_json(
    os.path.join(
        current_dir, "Evaluations", f"evaluation_{HEURISTIC_NAME}_slots_{SLOTS}.json"
    )
)  # baseline
df2 = pd.read_json(
    os.path.join(
        current_dir, "Evaluations", f"evaluation_{HEURISTIC_NAME}_slots_{SLOTS}.json"
    )
)  # DQL

mean_reward1 = np.mean(df1["Episode Rewards"])
std_reward1 = np.std(df1["Episode Rewards"])
print(f"{HEURISTIC_NAME}: Mean Reward: {mean_reward1}, STD Reward: {std_reward1}")

mean_reward2 = np.mean(df2["Episode Rewards"])
std_reward2 = np.std(df2["Episode Rewards"])
print(f"{DEEPEON_NAME}: Mean Reward: {mean_reward2}, STD Reward: {std_reward2}")


# df1.plot.line("index",y ="Episode Rewards")
# df2.plot.line("index",y ="Episode Rewards")

# df1.plot.line("index",y ="Episode Lengths")
# df2.plot.line("index",y ="Episode Lengths")

fig, ax = plt.subplots()

# ax.plot(df1["index"],df2["Episode Rewards"],label = "DeepEON")
# ax.plot(df1["index"],df1["Episode Rewards"],label = "Baseline")

plt.title(f"DeepEON with {SLOTS} slots vs 1_SP_FF with {SLOTS} slots")
plt.xlabel("Rounds")
plt.ylabel("Score")

eps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

a = 0
av_rew1 = []
for i, er in enumerate(df1["Episode Rewards"]):
    a += er
    if (i + 1) % 10 == 0:
        av = a / 10
        av_rew1.append(av)
        a = 0
a = 0
av_rew2 = []
for i, er in enumerate(df2["Episode Rewards"]):
    a += er
    if (i + 1) % 10 == 0:
        av = a / 10
        av_rew2.append(av)
        a = 0

# ax.plot(eps,av_rew2,label = DEEPEON_NAME)
# ax.plot(eps,av_rew1,label = HEURISTIC_NAME)


ax.plot(df1["index"], df2["Episode Rewards"], label="DEEPEON")
ax.plot(df1["index"], df1["Episode Rewards"], label="HEURISTIC")

plt.legend()

plt.show()
