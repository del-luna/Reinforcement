import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2gray

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc = nn.Sequential(
            nn.Linear(9*9*32, 256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n),
        )

    def foward(self, x):
        x = F.relu(self.conv1)
        x = F.relu(self.conv2)
        x = x.view(x.size(0), -1)
        return self.fc(x)

transform = T.Compose([T.ToPILImage(),
                       T.ToTensor()])

def preprocess(observe):
    pre_observe = np.uint8(resize(observe, (84,84), mode='constant') * 255)
    return pre_observe

def get_screen():
    screen = env.render(mode='rgb_array')
    screen = preprocess(screen).transpose((2, 0, 1))
    _, screen_h, screen_w = screen.shape
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    return transform(screen).unsqueeze(0).to(device)

if __name__ == "__main__":

    batch_size = 32
    gamma = 0.99
    loss_list = []
    reward_list = []
    episode_reward = 0

    env = gym.make('BreakoutDeterministic-v4')
    model = DQN()
    if torch.cuda.is_availabe():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(1000)
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

    observe = env.reset()
    state = preprocess(observe)
    history = np.stack((state, state, state, state), axis=2)
    history = np.reshape([history], (1, 84, 84, 4))

    for frame_idx in range(1, 100001):
        epsilon = epsilon_by_frame(frame_idx)
        action = model.act(history, epsilon)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            reward_list.append(episode_reward)
            episode_reward = 0
        
        if len(reaply_buffer) > batch_size:
            loss = temporal_difference(batch_size)
            loss_list.append(loss.data[0])

        if frame_idx % 200 == 0:
            plot(frame_idx, reward_list, loss_list)
