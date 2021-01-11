import gym
import math
import random
from tqdm import tqdm
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
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)

        self.fc = nn.Sequential(
            nn.Linear(9*9*32, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
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

def epsilon_greedy(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            import pdb;pdb.set_trace()
            return model(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def plot_trajectory():
    plt.figure(2)
    plt.clf()
    trajectory_t = torch.tensor(trajectory, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(trajectory_t.numpy())
    # 100 episodes means
    if len(trajectory_t) >= 100:
        means = trajectory_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  #delay
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    import pdb;pdb.set_trace()
    state_action_values = model(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    with torch.no_grad():
        next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if __name__ == "__main__":

    BATCH_SIZE = 32
    GAMMA = 0.999
    EPS_START = 1.
    EPS_END = 0.1
    EPS_DECAY = 200
    env = gym.make('BreakoutDeterministic-v4')
    env.reset()
    trajectory = []
    init_screen = get_screen()
    _, _, screen_height, screen_width = init_screen.shape
    n_actions = env.action_space.n

    transform = T.Compose([T.ToPILImage(),
                       T.ToTensor()])

    model = DQN(n_actions)
    model.to(device)
    optimizer = optim.RMSprop(model.parameters())
    memory = ReplayBuffer(10000)

    steps_done = 0
    num_episodes = 300
    for i_episodes in tqdm(range(num_episodes)):
        env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen

        for t in count():
            action = epsilon_greedy(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            last_screen = current_screen
            current_screen = get_screen()
            if not done:
                next_state = current_screen - last_screen
            else:
                next_state = None
            
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model()
            if done:
                trajectory.append(t + 1)
                plot_trajectory()
                break

    print('Complete')
    env.render(mode='rgb_array')
    env.close()
    plt.ioff()
    plt.show()