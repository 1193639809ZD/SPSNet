import torch
from torch import nn


class EntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-08):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prob):
        prob = prob + self.epsilon
        entropy = -torch.mul(prob, prob.log()).mean()
        return entropy
