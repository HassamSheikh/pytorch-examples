"""Implementation of REINFORCE Algorithm as taught in Berkeley Deep RL courseself.
Special thanks to Justus Schock for answering my questions to PyTorch forum """

import torch
import torch.nn as nn
import torch.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.optim import Adam
import gym
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.network = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax(dim=1))
        self.log_probs = []
        self.rewards = []

    def forward(self, state):
        return self.network(state)

    def train(self):
        exp_return = 0
        returns = []
        policy_loss = []
        for reward in self.rewards[::-1]:
            exp_return= reward + 0.99*exp_return
            returns.insert(0, exp_return)
        for log_prob, q_value in zip(self.log_probs, returns):
            policy_loss.append(log_prob*q_value)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.rewards = []
        self.log_probs = []

    def policy_add_reward(self, reward):
        self.rewards.append(reward)

    def select_action(self, state):
        probs = self(state)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(-m.log_prob(action))
        return action.item()


def main():
    max_episodes = 15000
    episode = 1
    env = gym.make('CartPole-v0')
    env.seed(1)
    torch.manual_seed(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    policy = Policy().to(device)
    optimizer = Adam(policy.parameters(), lr=1e-3)
    policy.optimizer = optimizer
    running_reward = 10
    while episode:
        state = env.reset()
        for t in range(1000):
            temp_state = torch.from_numpy(state).to(torch.float).unsqueeze(0).to(device)
            action = policy.select_action(temp_state)
            state, reward, done, _ = env.step(action)
            policy.policy_add_reward(reward)
            if done:
                break
        running_reward = running_reward * 0.99 + t * 0.01
        policy.train()
        if episode % 100 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        if episode > max_episodes:
            break
        episode += 1

if __name__ == '__main__':
    main()
