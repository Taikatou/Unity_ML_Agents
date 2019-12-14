import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1, fc2, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        input_dim = (state_size * 2) + action_size
        self.fc1 = nn.Linear(input_dim, fc1)
        self.fc2 = nn.Linear(fc1 + action_size, fc2)

        self.bn = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(fc1)

        self.fc5 = nn.Linear(fc2, 1)

        # last layer weight and bias initialization
        self.fc5.weight.data.uniform_(-3e-4, 3e-4)
        self.fc5.bias.data.uniform_(-3e-4, 3e-4)

        # torch.nn.init.uniform_(self.fc5.weight, a=-3e-4, b=3e-4)
        # torch.nn.init.uniform_(self.fc5.bias, a=-3e-4, b=3e-4)

    def forward(self, input_, action):
        """Build a network that maps state & action to action values."""

        x = self.bn(input_)
        x = F.relu(self.bn2(self.fc1(x)))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))

        x = self.fc5(x)

        return x
