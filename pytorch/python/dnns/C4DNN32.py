#!/usr/bin/env python
import torch
import torch.nn.functional as F
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.Dense0 = torch.nn.Linear(32, 10)
        self.Dense1 = torch.nn.Linear(10, 10)
        self.Dense2 = torch.nn.Linear(10, 10)
        self.Dense3 = torch.nn.Linear(10, 1)
    def forward(self, x):
        x=self.Dense0(x)
        x=F.relu(self.Dense1(x))
        x=F.relu(self.Dense2(x))
        x=F.relu(self.Dense3(x))
        return x