import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        gamma=0.99,
        lr=3e-4,
        batch_size=128,
        buffer_size=10000,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=1000,
        target_update=1000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.steps_done = 0

        self.policy_net = DQN(state_dim, action_dim).to(device)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state):
        epsilon = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.policy_net(state_t).argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
        )

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition = type(transitions[0])
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch.next_state, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())