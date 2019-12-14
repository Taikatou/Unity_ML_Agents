import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, state_size, action_size, fc1, fc2, seed):
        super(Actor, self).__init__()

        # network mapping state to action

        self.seed = torch.manual_seed(seed)

        self.bn = nn.BatchNorm1d(state_size)
        self.bn2 = nn.BatchNorm1d(fc1)
        self.bn3 = nn.BatchNorm1d(fc2)

        self.fc1 = nn.Linear(state_size, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc4 = nn.Linear(fc2, action_size)

        # last layer weight and bias initialization
        torch.nn.init.uniform_(self.fc4.weight, a=-3e-3, b=3e-3)
        torch.nn.init.uniform_(self.fc4.bias, a=-3e-3, b=3e-3)

        # Tanh
        self.tan = nn.Tanh()

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = (self.fc4(x))
        norm = torch.norm(x)

        # h3 is a 2D vector (a force that is applied to the agent)
        # we bound the norm of the vector to be between 0 and 10
        return 10.0 * (F.tanh(norm)) * x / norm if norm > 0 else 10 * x

        # return self.tan(self.fc4(x))

