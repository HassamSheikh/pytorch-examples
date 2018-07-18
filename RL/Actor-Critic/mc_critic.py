""" It is mostly the same code from Pytorch example of AC """
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from torch.optim import Adam
import gym
import time
import numpy as np
from collections import namedtuple

env = gym.make('CartPole-v0')
env.seed(1)
torch.manual_seed(1)

ActionValue = namedtuple('ActionValue', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc_1 = nn.Linear(4, 128)
        self.action_layer = nn.Linear(128, 2)
        self.value_layer = nn.Linear(128, 1)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        value = self.value_layer(x)
        action_prob = F.softmax(self.action_layer(x), dim=-1)
        return action_prob, value

policy = Policy()
optimizer = Adam(policy.parameters(), lr=1e-3)

def select_action(state):
    state_tensor = torch.from_numpy(state).float()
    probs, value = policy(state_tensor)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_actions.append(ActionValue(-m.log_prob(action), value))
    return action.item()

def train():
    R = 0
    policy_losses = []
    value_losses = []
    q_values = []
    for r in policy.rewards[::-1]:
        R = r + 0.99 * R
        q_values.insert(0, R)
    for (log_prob, value), r in zip(policy.saved_actions, q_values):
        reward = r - value.item()
        policy_losses.append(log_prob * reward)
        value_losses.append(F.mse_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    policy.rewards, policy.saved_actions = [], []

def main():
    running_reward = 10
    for i_episode in range(10000):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        train()
        if i_episode % 10 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
