import os
import gym
import torch
import time
import numpy as np
import random
import math
import torch.nn.functional as F
import torchvision.transforms as T

from itertools import count
from torch import optim
from torch import nn
from model import DQN
from utils import Transition, ReplayMemory, FrameHistory, rgb2gray, id_to_action, exponential_moving_average, ACTION_SPACE
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


RENDER = False
RECORD = False
BATCH_SIZE = 64
GAMMA = 0.95
N_EPISODES = 1000
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.9999
REWARD_MULTIPLIER = 1.5
MODEL_SAVE_FREQUENCY = 50
SKIP_FRAMES = 2
BUFFER_SIZE = 10000
LEARNING_RATE = 1e-3
HISTORY_LENGTH = 3
POLYAK_PARAM = 0.01
ACTION_SPACE_SIZE = len(ACTION_SPACE)


def select_action(state):
    """
    This method selects an action based on the state per the policy network.
    """
    global eps_threshold
    sample = random.random()
    if sample > eps_threshold:
        policy_net.eval()
        with torch.no_grad():
            action_id = policy_net(torch.stack(state.history)[None, ...]).max(1).indices
    else:
        action_id = torch.tensor([random.randrange(ACTION_SPACE_SIZE)], device=device, dtype=torch.long)
    return action_id, id_to_action(action_id)


def update_epsilon(epsilon):
    """
    This method updates the epsilon parameter for epsilon greedy such that it decays as per EPS_START, 
    EPS_END, and EPS_DECAY_LENGTH.
    """
    if epsilon > EPS_END:
        epsilon *= EPS_DECAY
    return epsilon


def optimize_policy(replay_buffer, policy_net, target_net, optimizer, loss_function):
    """
    This method optimizes the policy network by minimizing the TD error between the Q from the 
    policy network and the Q calculated through a Bellman backup via the target network.
    """
    global losses
    global eps_threshold
    if len(replay_buffer) < BATCH_SIZE: return
    transitions = replay_buffer.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Manage edge cases 
    non_final_mask = torch.tensor(tuple(map(lambda x: x is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([x for x in batch.next_state if x is not None])
    
    # Create batch
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Get Q value per policy network
    policy_net.train()
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Get Q value per target_network
    next_state_values = torch.zeros(BATCH_SIZE, device=device) 
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch # value at terminal state is reward_batch

    # Compute loss
    loss = loss_function(state_action_values, expected_state_action_values)
    losses.append(loss.item())
    
    # Optimize the policy network
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(policy_net.parameters(), 2.0)
    optimizer.step()

    eps_threshold = update_epsilon(eps_threshold)

    # Record output
    if RECORD:
        grad_norm = torch.stack([params.grad.data.norm() for params in policy_net.parameters()])
        writer.add_scalar('TD Loss', loss.item(), total_iterations)
        writer.add_scalar('Min Gradient Norm', grad_norm.min().item(), total_iterations)
        writer.add_scalar('Max Gradient Norm', grad_norm.max().item(), total_iterations)
        writer.add_scalar('Epsilon', eps_threshold, total_iterations)


def transform_frame(frame):
    """
    This method transforms frames outputed by one step of the gym environment to a grayscale tensor.
    """
    return torch.from_numpy(rgb2gray(frame)).to(device) / 255.


def update_state(env, frame, state, default_action=[0.0, 1.0, 0.0]):
    """
    This method updates the state based on a new frame given by one step of the gym environment.
    """
    global episode_reward
    global total_iterations
    state.push(transform_frame(frame))
    while len(state) < state.history_length:
        state.push(transform_frame(frame))


def store_transition(state, next_state, action_id, reward, replay_buffer):
    """
    This method stores the transition in the replay buffer.
    """
    replay_buffer.push(
            transform(torch.stack(state.history)),
            action_id,
            transform(torch.stack(next_state.history)) if next_state else next_state,
            torch.tensor([reward], device=device),
    )


def get_next_state(state, next_frame, done):
    """
    This method gets the next state given the next frame produces by one step of the gym environment.
    """
    if not done: 
        next_state = state.clone()
        next_state.push(transform_frame(next_frame))
    else:
        next_state = None
    return next_state


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


def polyak_averaging(target_net, policy_net, beta, requires_grad=False):
    """
    This method updated the target network parameters by performing a polyak averaging between 
    the target network and the policy network.
    """
    target_net.eval()
    policy_net.eval()
    target_st = target_net.state_dict()
    policy_st = policy_net.state_dict()
    with torch.no_grad():
        for key, policy_param in policy_st.items():
            target_st[key] = beta * policy_param + (1 - beta) * target_st[key]
        target_net.load_state_dict(target_st)
    for param in target_net.parameters(): param.requires_grad_(requires_grad)    


def transform(x): return x


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "CarRacing-v0"
    env = gym.make(env_name).unwrapped
    model_dir = "./models/"
    model_fn = time.strftime("%Y%m%d-%H%M%S") + "-checkpoint"
    tensorboard_dir = "./runs/"

    if not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir) 
    if not os.path.exists(model_dir):
        os.mkdir(model_dir) 

    nh = 128
    channels_per_layer = [16, 32, 64, 64]
    policy_net = DQN(channels_per_layer, nh, c_in=HISTORY_LENGTH, c_out=ACTION_SPACE_SIZE).to(device)
    target_net = DQN(channels_per_layer, nh, c_in=HISTORY_LENGTH, c_out=ACTION_SPACE_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    for param in target_net.parameters(): param.requires_grad_(False)
    target_net.eval()

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayMemory(BUFFER_SIZE)

    if RECORD: writer = SummaryWriter()
    state = FrameHistory(HISTORY_LENGTH)
    total_iterations = 0
    eps_threshold = EPS_START
    losses =[]
    episode_rewards = []

    for episode in range(N_EPISODES):
        
        # Initialize the environment and states
        episode_reward = 0
        consecutive_negative_rewards = 0
        frame = env.reset()

        for iteration in count():

            # Execute iteration
            update_state(env, frame, state)                                                         # Update state
            action_id, action = select_action(state)                                                # Select action
            next_frame, reward, done, info = perform_action(env, action)                            # Perform action
            if action[1] == 1.0: reward *= REWARD_MULTIPLIER                                        # Engineer reward
            next_state = get_next_state(state, next_frame, done)                                    # Get next state
            store_transition(state, next_state, action_id, reward, replay_buffer)                   # Store transition in replay buffer
            optimize_policy(replay_buffer, policy_net, target_net, optimizer, loss_function)        # Optimize policy

            # Increment episode reward
            if reward < 0 and iteration > 100:  consecutive_negative_rewards += 1
            else:           consecutive_negative_rewards = 0
            episode_reward += reward
            if RECORD: writer.add_scalar('Reward', episode_reward, total_iterations)

            # Evaluate exit criteria
            if done or episode_reward < 0 or consecutive_negative_rewards >= 25:
                break
            
            # Set-up next iteration
            frame = next_frame
            total_iterations += 1
            if RENDER: env.render()

            # Update the target network
            polyak_averaging(target_net, policy_net, POLYAK_PARAM, requires_grad=False)

        # Set-up next episode
        episode_rewards.append(episode_reward)
        consecutive_negative_rewards = 0
        state.history = []
        if RECORD and episode % MODEL_SAVE_FREQUENCY == 0:
            torch.save({
                    'model': policy_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'episode': episode,
                }, '/'.join([model_dir, model_fn + "-" + str(episode) +".pth"]))
        n = max(0, len(episode_rewards)-100)
        avg_rewards = sum(episode_rewards[n:]) / len(episode_rewards[n:])
        print(f"[EPISODE]: %i, [ITERATION]: %i, [EPSILON]: %f,[EPISODE REWARD]: %i, [AVG REWARD PER 100 EPISODES]: %i, [ITERATION PER EPOCH]: %i" % (episode, total_iterations, eps_threshold, episode_reward, avg_rewards, iteration))
        env.close()
        
    print("\n")
    print("[SUM OF REWARD FOR ALL %i EPISODES]: %i" % (N_EPISODES, sum(episode_rewards)))
    if RECORD: writer.close()