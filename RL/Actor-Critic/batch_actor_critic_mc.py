""" Implementation Batch Actor-Critic Method as taught in CS 294. The Value Function is fitted using the Monte Carlo Policy Evaluation
"""

#Note: It is probably the most inefficient implementation of the AC method

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


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.network = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2), nn.Softmax())
        self.log_probs = []
        self.states = []
        self.next_states = []
        self.rewards = []

    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc_1 = nn.Linear(4, 128)
        self.value_layer = nn.Linear(128, 1)
        self.value_layer.weight = torch.nn.Parameter(torch.zeros(1, 128))

    def forward(self, state):
        state = F.relu(self.fc_1(state))
        out = self.value_layer(state)
        return out

actor = Actor()
actor_optimizer = Adam(actor.parameters(), lr=1e-3)

critic = Critic()
critic_optimizer = Adam(critic.parameters(), lr=1e-3)


def select_action(state):
     state_tensor = torch.from_numpy(state).float()
     probs = actor(state_tensor)
     m = Categorical(probs)
     action = m.sample()
     actor.log_probs.append(-m.log_prob(action))
     return action.item()

def train():
    R = 0
    policy_losses = []
    value_losses = []
    target_v_values = []
    for r in actor.rewards[::-1]:
        R = r + 0.99 * R
        target_v_values.insert(0, R)
    for state, target_v_value in zip(actor.states, target_v_values):
        state_tensor = torch.from_numpy(state).float()
        state_value = critic(state_tensor)
        value_losses.append(F.mse_loss(state_value, torch.tensor([target_v_value])))
    value_losses = torch.stack(value_losses).sum()
    critic_optimizer.zero_grad()
    value_losses.backward()
    critic_optimizer.step()

    for state, next_state, log_prob, reward in zip(actor.states, actor.next_states, actor.log_probs, actor.rewards):
        state_tensor = torch.from_numpy(state).float()
        state_value = critic(state_tensor)
        next_state_tensor = torch.from_numpy(next_state).float()
        next_state_value = critic(next_state_tensor)
        advantage = reward + 0.99*next_state_value.item() - state_value.item()
        policy_losses.append(log_prob * advantage)
    policy_losses = torch.stack(policy_losses).sum()
    actor_optimizer.zero_grad()
    policy_losses.backward()
    actor_optimizer.step()
    actor.states, actor.next_states, actor.log_probs, actor.rewards = [], [], [], []

def main():
    running_reward = 10
    for i_episode in range(10000):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            next_state, reward, done, _ = env.step(action)
            actor.rewards.append(reward)
            actor.states.append(state)
            actor.next_states.append(next_state)
            if done:
                break
            state = next_state
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
