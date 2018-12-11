import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import retro                 # Retro Environment
import random
from utils import stack_frames
from models import DQNetwork, Memory
from time import sleep

env = retro.make(game='SpaceInvaders-Atari2600')
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())
actions_n = len(possible_actions)
n_episode = 1
## Single random actions for all the games
# for i in range(100):
#     # If it's the first step
#     state = env.reset()
#     done = False
#     act = env.action_space.sample()
#     while not done:
#         # act = env.action_space.sample()
#         # act_i = i
#         # act = possible_actions[act_i]
#         print("action ", act)
#         sleep(0.001)
#         next_state, reward, done, _ = env.step(act)
#         # print("reward", reward)
#         # print("done", done)
#         env.render()
#     print("Game is over ...............................................")
#     sleep(2)
#     print("info", _)

## random actions for 5 games
for i in range(5):
    # If it's the first step
    state = env.reset()
    done = False
    # act = env.action_space.sample()
    while not done:
        act = env.action_space.sample()
        # act_i = i
        # act = possible_actions[act_i]
        print("action ", act)
        sleep(0.001)
        next_state, reward, done, _ = env.step(act)
        # print("reward", reward)
        # print("done", done)
        env.render()
    print("Game is over ...............................................")
    sleep(2)
    print("info", _)

env.close()