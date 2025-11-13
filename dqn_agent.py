# dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('s','a','r','s2','d'))

class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x): return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim, action_space, device='cpu'):
        # action_space: discrete mapping we will create: e.g., open_lanes=1..num_lanes (num_lanes choices) x truck_priority (2)
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.device = device
        self.qnet = QNet(obs_dim, self.n_actions).to(device)
        self.target = QNet(obs_dim, self.n_actions).to(device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.optim = optim.Adam(self.qnet.parameters(), lr=1e-3)
        self.replay = deque(maxlen=20000)
        self.batch_size = 64
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.update_target_every = 500
        self.step_count = 0

    def choose_action(self, obs):
        # obs: numpy array
        if random.random() < self.eps:
            return random.randrange(self.n_actions)
        else:
            s = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q = self.qnet(s)
            return int(torch.argmax(q).item())

    def store(self, s,a,r,s2,d):
        self.replay.append(Transition(s,a,r,s2,d))

    def learn(self):
        if len(self.replay) < self.batch_size: return
        batch = random.sample(self.replay, self.batch_size)
        s = torch.tensor(np.vstack([b.s for b in batch]), dtype=torch.float32).to(self.device)
        a = torch.tensor([b.a for b in batch], dtype=torch.long).unsqueeze(1).to(self.device)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        s2 = torch.tensor(np.vstack([b.s2 for b in batch]), dtype=torch.float32).to(self.device)
        d = torch.tensor([b.d for b in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

        q_vals = self.qnet(s).gather(1,a)
        with torch.no_grad():
            q_next = self.target(s2).max(1)[0].unsqueeze(1)
            q_target = r + self.gamma * q_next * (1 - d)

        loss = nn.functional.mse_loss(q_vals, q_target)
        self.optim.zero_grad(); loss.backward(); self.optim.step()

        # eps decay
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target.load_state_dict(self.qnet.state_dict())
