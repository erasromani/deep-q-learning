import random
import numpy as np
from collections import namedtuple
import copy


ACTION_SPACE = np.array([
    [0.0, 0.0, 0.0],        # Nothing
    [0.0, 1.0, 0.0],        # Accelerate
    [0.0, 0.0, 0.2],        # Brake
    [-1.0, 0.0, 0.0],       # Left
    [-1.0, 1.0, 0.0],       # Left, Accelerate
    [-1.0, 0.0, 0.2],       # Left, Brake
    [1.0, 0.0, 0.0],        # Right
    [1.0, 1.0, 0.0],        # Right, Accelerate
    [1.0, 0.0, 0.2]         # Right, Brake
])


def id_to_action(id):
    return ACTION_SPACE[id]


def exponential_moving_average(x, beta=0.9):
    average = 0
    ema_x = x.copy()
    for i, o in enumerate(ema_x):
        average = average * beta + (1 - beta) * o
        ema_x[i] = average / (1 - beta**(i+1))
    return ema_x


def simple_moving_average(x, window=100):
    x_avg, N = [], len(x)
    for i in range(N):
        n = max(0, i-window+1)
        x_avg.append(sum(x[n:i+1]) / len(x[n:i+1]))
    return np.array(x_avg)


def rgb2gray(rgb):
    """ 
    This method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32')


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class FrameHistory(object):

    def __init__(self, history_length):
        self.history_length = history_length
        self.history = []

    def push(self, frame):
        """Saves a frame."""
        self.history.append(frame)
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def clone(self):
        new = FrameHistory(self.history_length)
        new.history = self.history.copy()
        return new

    def __len__(self):
        return len(self.history)