import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df1 = pd.read_json("Evaluation data\evaluation_date_0.json")
df2 = pd.read_json("Evaluation data\evaluation_date_2.json")

mean_reward1 = np.mean(df1["Episode Rewards"])
std_reward1 = np.std(df1["Episode Rewards"])
print(f"Mean Reward: {mean_reward1}, STD Reward: {std_reward1}")

mean_reward2 = np.mean(df2["Episode Rewards"])
std_reward2 = np.std(df2["Episode Rewards"])
print(f"Mean Reward: {mean_reward2}, STD Reward: {std_reward2}")

# df1.plot.line("index",y ="Episode Rewards")
# df2.plot.line("index",y ="Episode Rewards")

# df1.plot.line("index",y ="Episode Lengths")
# df2.plot.line("index",y ="Episode Lengths")

# plt.plot(df1["index"],df2["Episode Rewards"])
# plt.plot(df1["index"],df1["Episode Rewards"])


plt.show()


