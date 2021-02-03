import os
import gym
import torch
import time
import numpy as np
import random
import math
import torch.nn.functional as F
import torchvision.transforms as T
import json

from torch import optim
from torch import nn
from model import DQN
from utils import Transition, ReplayMemory, FrameHistory, rgb2gray, id_to_action, ACTION_SPACE
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


RENDER = True
N_EPISODES = 10
SKIP_FRAMES = 2
HISTORY_LENGTH = 3
ACTION_SPACE_SIZE = len(ACTION_SPACE)


def select_action(agent, state):
    """
    This mehtod selects an action based on the state per the policy network.
    """
    agent.eval()
    with torch.no_grad():
        action_id = agent(torch.stack(state.history)[None, ...]).max(1).indices
    return action_id, id_to_action(action_id)


def perform_action(env, action):
    """
    This method performs a number of steps, dictated by SKIP_FRAMES, of the gym environment given an action.
    """
    reward = 0
    for _ in range(SKIP_FRAMES+1):
        next_frame, r, done, info = env.step(action)
        reward += r
        if done: break
    return next_frame, reward, done, info


def run_episode(env, agent, state, transform=None, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    frame = env.reset()
    while True:
        
        # get state history
        state.push(transform_frame(frame))
        if len(state) < state.history_length:
            action = np.array([0.0, 1.0, 0.0])
        else:
            # perform inference
            action_id, action = select_action(agent, state)

        next_frame, reward, done, info = perform_action(env, action)
        episode_reward += reward     
        frame = next_frame
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


def transform_frame(frame):
    """
    This method transforms frames outputed by one step of the gym environment to a grayscale tensor.
    """
    return torch.from_numpy(rgb2gray(frame)).to(device) / 255.


def transform(x): return 


if __name__ == "__main__":
    
    n_test_episodes = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make('CarRacing-v0').unwrapped

    model_dir = "models/"
    model_fn = "20201103-194853-checkpoint-700.pth"
    results_dir = "./results/"

    if not os.path.exists(results_dir):
        os.mkdir(results_dir) 

    nh = 128
    channels_per_layer = [16, 32, 64, 64]
    agent = DQN(channels_per_layer, nh, c_in=HISTORY_LENGTH, c_out=ACTION_SPACE_SIZE).to(device)
    st = torch.load('/'.join([model_dir, model_fn]), map_location=device)['model']
    agent.load_state_dict(st)
    for param in agent.parameters(): param.requires_grad_(False)
    agent.eval()

    state = FrameHistory(HISTORY_LENGTH)
    episode_rewards = []

    # run episodes
    for i in range(N_EPISODES):
        episode_reward = run_episode(env, agent, state, transform, rendering=RENDER)
        print("[EPISODE]: %i, [REWARD]: %i" % (i, episode_reward))
        episode_rewards.append(episode_reward)

    # save results
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    

    fname = "results/results_dqn_agent-%s.json" % "".join(model_fn.split(".")[:-1])
    
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    fh.close()
    print('... finished')