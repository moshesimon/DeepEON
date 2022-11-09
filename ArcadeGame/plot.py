from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
"solar-microwave-1"
"silver-glade-5"
"DeepEON4"
"deepq_EON5"
HEURISTIC_NAME = "KSP_FF_11.04.2022_09.24.44"
DEEPEON_NAME = "11.04.2022_01.02.31"
SEED = "0"

df1 = pd.read_json(f"Evaluation data\evaluation_{HEURISTIC_NAME}.json") #baseline
df2 = pd.read_json(f"Evaluation data\evaluation_{DEEPEON_NAME}_seed_{SEED}.json") #DQL

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

#ax.plot(df1["index"],df2["Episode Rewards"],label = "DeepEON")
#ax.plot(df1["index"],df1["Episode Rewards"],label = "Baseline")

plt.title(f"DeepEON with 16 slots vs KSP_FF with 16 slots")
plt.xlabel("Rounds")
plt.ylabel("Score")

eps = [10,20,30,40,50,60,70,80,90,100]

a = 0
av_rew1 = []
for i,er in enumerate(df1["Episode Rewards"]):
    a+=er
    if (i+1) %10 == 0:
        av = a /10
        av_rew1.append(av)
        a = 0
a = 0
av_rew2 = []
for i,er in enumerate(df2["Episode Rewards"]):
    a+=er
    if (i+1) %10 == 0:
        av = a /10
        av_rew2.append(av)
        a = 0

# ax.plot(eps,av_rew2,label = DEEPEON_NAME)
# ax.plot(eps,av_rew1,label = HEURISTIC_NAME)



ax.plot(df1["index"],df2["Episode Rewards"],label = "DEEPEON")
ax.plot(df1["index"],df1["Episode Rewards"],label = "HEURISTIC")

plt.legend()

plt.show()


