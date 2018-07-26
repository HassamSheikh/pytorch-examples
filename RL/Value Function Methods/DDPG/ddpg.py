import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
