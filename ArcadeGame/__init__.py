from gym.envs.registration import register

register(
    id="EONArcade-v2",
    entry_point="gym_game.envs:CustomEnv",
)
