import gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, capacity, batch_size):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 8, 4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2),
            nn.ReLU(),
            nn.Linear(x, 256),
            nn.Linear(256, env.action_space.n)
        )

    def foward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.foward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)
        return action

if __name__ == "__main__":
    env = gym.make('BreakoutDeterministic-v4')
    model = DQN()
    if torch.cuda.is_availabe():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    replay_buffer = ReplayBuffer(1000)


