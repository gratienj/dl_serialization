#!/usr/bin/env python
import torch
import torch.nn.functional as F
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.Dense0 = torch.nn.Linear(8, 10)
        self.Dense1 = torch.nn.Linear(10, 10)
        self.Dense2 = torch.nn.Linear(10, 10)
        self.Dense3 = torch.nn.Linear(10, 10)
        self.Dense4 = torch.nn.Linear(10, 10)
        self.Dense5 = torch.nn.Linear(10, 10)
        self.Dense6 = torch.nn.Linear(10, 10)
        self.Dense7 = torch.nn.Linear(10, 10)
        self.Dense8 = torch.nn.Linear(10, 10)
        self.Dense9 = torch.nn.Linear(10, 10)
        self.Dense10 = torch.nn.Linear(10, 10)
        self.Dense11 = torch.nn.Linear(10, 10)
        self.Dense12 = torch.nn.Linear(10, 10)
        self.Dense13 = torch.nn.Linear(10, 10)
        self.Dense14 = torch.nn.Linear(10, 10)
        self.Dense15 = torch.nn.Linear(10, 1)
    def forward(self, x):
        x=self.Dense0(x)
        x=F.relu(self.Dense1(x))
        x=F.relu(self.Dense2(x))
        x=F.relu(self.Dense3(x))
        x=F.relu(self.Dense4(x))
        x=F.relu(self.Dense5(x))
        x=F.relu(self.Dense6(x))
        x=F.relu(self.Dense7(x))
        x=F.relu(self.Dense8(x))
        x=F.relu(self.Dense9(x))
        x=F.relu(self.Dense10(x))
        x=F.relu(self.Dense11(x))
        x=F.relu(self.Dense12(x))
        x=F.relu(self.Dense13(x))
        x=F.relu(self.Dense14(x))
        x=F.relu(self.Dense15(x))
        return x