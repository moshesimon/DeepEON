
def encode(all_configs):
    return f"{all_configs['gamma']}_{all_configs['batch_size']}_{all_configs['learning_rate']}_{all_configs['learning_starts']}_{all_configs['target_update_interval']}_{all_configs['train_freq']}_{all_configs['total_timesteps']}_{all_configs['buffer_size']}_{all_configs['exploration_final_eps']}_{all_configs['exploration_fraction']}_{all_configs['game']}_{all_configs['number_of_slots']}_{all_configs['screen_number_of_slots']}_{all_configs['K']}_{all_configs['solution_reward']}_{all_configs['rejection_reward']}_{all_configs['gap_rejection_reward']}_{all_configs['seed']}_{all_configs['env']}_{all_configs['episode_end']}_{all_configs['end_limit']}"

def decode(full_name):
    temp_configs = {}
    temp_configs["gamma"] = float(full_name.split("_")[0])
    temp_configs["batch_size"] = int(full_name.split("_")[1])
    temp_configs["learning_rate"] = float(full_name.split("_")[2])
    temp_configs["learning_starts"] = int(full_name.split("_")[3])
    temp_configs["target_update_interval"] = int(full_name.split("_")[4])
    temp_configs["train_freq"] = full_name.split("_")[5]
    temp_configs["total_timesteps"] = int(full_name.split("_")[6])
    temp_configs["buffer_size"] = int(full_name.split("_")[7])
    temp_configs["exploration_final_eps"] = float(full_name.split("_")[8])
    temp_configs["exploration_fraction"] = float(full_name.split("_")[9])
    temp_configs["game"] = full_name.split("_")[10]
    temp_configs["number_of_slots"] = int(full_name.split("_")[11])
    temp_configs["screen_number_of_slots"] = int(full_name.split("_")[12])
    temp_configs["K"] = int(full_name.split("_")[13])
    temp_configs["solution_reward"] = float(full_name.split("_")[14])
    temp_configs["rejection_reward"] = float(full_name.split("_")[15])
    temp_configs["gap_rejection_reward"] = float(full_name.split("_")[16])
    temp_configs["seed"] = int(full_name.split("_")[17])
    temp_configs["env"] = full_name.split("_")[18]
    temp_configs["episode_end"] = int(full_name.split("_")[19])
    temp_configs["end_limit"] = int(full_name.split("_")[20])
    print(f"""
    Gamma: {temp_configs["gamma"]} 
    Batch Size: {temp_configs["batch_size"]}
    Learning Rate: {temp_configs["learning_rate"]}
    Learning Starts: {temp_configs["learning_starts"]}
    Target Update Interval: {temp_configs["target_update_interval"]}
    Train Freq: {temp_configs["train_freq"]}
    Total Timesteps: {temp_configs["total_timesteps"]}
    Buffer Size: {temp_configs["buffer_size"]}
    Exploration Final Eps: {temp_configs["exploration_final_eps"]}
    Exploration Fraction: {temp_configs["exploration_fraction"]}
    Game: {temp_configs["game"]}
    Number of Slots: {temp_configs["number_of_slots"]}
    Screen Number of Slots: {temp_configs["screen_number_of_slots"]}
    K: {temp_configs["K"]}
    Solution Reward: {temp_configs["solution_reward"]}
    Rejection Reward: {temp_configs["rejection_reward"]}
    Gap Rejection Reward: {temp_configs["gap_rejection_reward"]}
    Seed: {temp_configs["seed"]}
    Env: {temp_configs["env"]}
    Episode End: {temp_configs["episode_end"]}
    End Limit: {temp_configs["end_limit"]}
    """)
    return temp_configs

