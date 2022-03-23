# # import matplotlib
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from stable_baselines3 import DQN
# # from envs.custom_env import CustomEnv

# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import cv2

# from Games.game5 import ArcadeGame
# game_config = {
#   "solution_reward": 10,
#   "rejection_reward": -10,
#   "left_reward": 0,
#   "right_reward": 0,
#   "seed": 0
# }
# game = ArcadeGame(game_config)

# game.new_game()
# game.check_solution()
# y = game.draw_screen()


# def make_lut_u():
#     return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

# def make_lut_v():
#     return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)


# #img = cv2.imread('pics/Untitled.jpg')
# # print(type(img), np.shape(img))
# # img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# # y, u, v = cv2.split(img_yuv)

# # print(type(y), np.shape(y))

# # lut_u, lut_v = make_lut_u(), make_lut_v()

# # Convert back to BGR so we can apply the LUT and stack the images
# #y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
# # u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
# # v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
# print(type(y), np.shape(y))
# # u_mapped = cv2.LUT(u, lut_u)
# # v_mapped = cv2.LUT(v, lut_v)

# # result = np.vstack([img, y, u_mapped, v_mapped])
# print(np.max(y))
# cv2.imwrite('shed_combo.png', y)
import numpy as np
pos = 30
batch_size = 100
buffer_size = 10000

batch_inds1 = (np.random.randint(1, buffer_size, size=batch_size) + pos) % buffer_size

batch_inds2 = np.random.randint(0, pos, size=batch_size)

print(batch_inds1,batch_inds2)