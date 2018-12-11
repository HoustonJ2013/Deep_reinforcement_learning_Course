import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import retro                 # Retro Environment
from skimage import transform # Help us to preprocess the frames
from skimage.color import rgb2gray # Help us to gray our frames
from collections import deque# Ordered collection with ends
import random
from utils import stack_frames
from models import DQNetwork
from time import sleep

env = retro.make(game='SpaceInvaders-Atari2600')
possible_actions = np.array(np.identity(env.action_space.n, dtype=int).tolist())

### MODEL HYPERPARAMETERS
state_size = [110, 84, 4]      # Our input is a stack of 4 frames hence 110x84x4 (Width, height, channels)
action_size = env.action_space.n # 8 possible actions
learning_rate = 0.00025      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes = 50            # Total episodes for training
max_steps = 50000              # Max possible steps in an episode
batch_size = 64                # Batch size

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00001           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9                    # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
memory_size = 1000000          # Number of experiences the Memory can keep

### PREPROCESSING HYPERPARAMETERS
stack_size = 4                 # Number of frames stacked

### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = False


# Initialize deque with zero-images one array for each image
stacked_frames = deque([np.zeros((110, 84), dtype=np.int) for i in range(stack_size)], maxlen=4)


tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

saver = tf.train.Saver()  # Gets all variables in `graph`.


with tf.Session() as sess:
    total_test_rewards = []

    # Load the model
    saver.restore(sess, "./models/model.ckpt")

    for episode in range(3):
        total_rewards = 0

        state = env.reset()
        state, stacked_frames = stack_frames(stacked_frames, state, True)

        print("****************************************************")
        print("EPISODE ", episode)

        while True:
            # Reshape the state
            state = state.reshape((1, *state_size))
            # Get action from Q-network
            # Estimate the Qs values state
            Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state})

            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]
            print(choice)

            # Perform the action and get the next_state, reward, and done information
            next_state, reward, done, _ = env.step(action)
            env.render()
            sleep(0.001)
            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            state = next_state

    env.close()