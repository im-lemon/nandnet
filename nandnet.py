import torch.nn as nn, torch

class NANDnet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ReLU = nn.ReLU()
        self.hidden1 = nn.Linear(2,1)

    def forward(self, x):
        x = self.hidden1(x)

        return x