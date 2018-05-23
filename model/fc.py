
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable



class fc_pytorch_base(nn.Module):
    """
    for a unified API with cnn.py
    """
    def __init__(self):
        super(fc_pytorch_base, self).__init__()
        self.posterior_samples = []
        self.posterior_weights = []
    def state_dict(self):
        return [self.model.state_dict()]

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd[0])

class fc(fc_pytorch_base):
    """docstring for fc"""

    def __init__(self, Hn, input_dim=28*28, output_dim=10, dropout_rate=0):
        super(fc, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, Hn),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(Hn, Hn),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(Hn, output_dim),
        )

    def forward(self, x):
        # Flattens the input image to dimension BS x input_dim and passes it through the model
        return self.model(x.view(x.size(0), -1))


class fc1(fc_pytorch_base):
    def __init__(self, Hn, input_dim=28*28, output_dim=10, dropout_rate=0):
        super(fc1, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = torch.nn.Sequential(
            # Do we want dropout before the first linear layer too?
            torch.nn.Linear(input_dim, Hn),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(Hn, output_dim),
        )

    def forward(self, x):
        # return self.model(input.view(-1, self.input_dim))
        return self.model(x.view(x.size(0), -1))


class fc1_separate(fc_pytorch_base):
    def __init__(self, Hn, input_dim=28*28, output_dim=10, dropout_rate=0):
        super(fc1_separate, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(input_dim, Hn)
        self.fc2 = nn.Linear(Hn, output_dim)

    def forward(self, x, return_activations=False):
        x = x.view(x.size(0), -1)
        a1 = self.fc1(x)
        h1 = self.relu(a1)
        h1 = self.dropout(h1)
        a2 = self.fc2(h1)

        if return_activations:
            return torch.cat([a2, h1], dim=1)
        else:
            return a2

