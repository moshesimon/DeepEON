from stable_baselines3 import DQN
from envs.custom_env import CustomEnv as CustomEnv1
from envs.custom_env2 import CustomEnv as CustomEnv2
from config import current_dir, all_configs, game_config

if model_config["env"] == 1:
    env = CustomEnv1(model_config)
elif model_config["env"] == 2:
    env = CustomEnv2(model_config)
else:
    print("env not selected correctly in config.py")
    exit(1)

model = DQN.load("deepq_EON3")
model.set_env(env)
model.learn(
    total_timesteps=all_configs["total_timesteps"],
    tb_log_name="Logs",
    reset_num_timesteps=False,
)
model.save("deepq_EON3_continued_1")
