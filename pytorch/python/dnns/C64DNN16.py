#!/usr/bin/env python
import torch
import torch.nn.functional as F
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.Dense0 = torch.nn.Linear(16, 10)
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
        self.Dense15 = torch.nn.Linear(10, 10)
        self.Dense16 = torch.nn.Linear(10, 10)
        self.Dense17 = torch.nn.Linear(10, 10)
        self.Dense18 = torch.nn.Linear(10, 10)
        self.Dense19 = torch.nn.Linear(10, 10)
        self.Dense20 = torch.nn.Linear(10, 10)
        self.Dense21 = torch.nn.Linear(10, 10)
        self.Dense22 = torch.nn.Linear(10, 10)
        self.Dense23 = torch.nn.Linear(10, 10)
        self.Dense24 = torch.nn.Linear(10, 10)
        self.Dense25 = torch.nn.Linear(10, 10)
        self.Dense26 = torch.nn.Linear(10, 10)
        self.Dense27 = torch.nn.Linear(10, 10)
        self.Dense28 = torch.nn.Linear(10, 10)
        self.Dense29 = torch.nn.Linear(10, 10)
        self.Dense30 = torch.nn.Linear(10, 10)
        self.Dense31 = torch.nn.Linear(10, 10)
        self.Dense32 = torch.nn.Linear(10, 10)
        self.Dense33 = torch.nn.Linear(10, 10)
        self.Dense34 = torch.nn.Linear(10, 10)
        self.Dense35 = torch.nn.Linear(10, 10)
        self.Dense36 = torch.nn.Linear(10, 10)
        self.Dense37 = torch.nn.Linear(10, 10)
        self.Dense38 = torch.nn.Linear(10, 10)
        self.Dense39 = torch.nn.Linear(10, 10)
        self.Dense40 = torch.nn.Linear(10, 10)
        self.Dense41 = torch.nn.Linear(10, 10)
        self.Dense42 = torch.nn.Linear(10, 10)
        self.Dense43 = torch.nn.Linear(10, 10)
        self.Dense44 = torch.nn.Linear(10, 10)
        self.Dense45 = torch.nn.Linear(10, 10)
        self.Dense46 = torch.nn.Linear(10, 10)
        self.Dense47 = torch.nn.Linear(10, 10)
        self.Dense48 = torch.nn.Linear(10, 10)
        self.Dense49 = torch.nn.Linear(10, 10)
        self.Dense50 = torch.nn.Linear(10, 10)
        self.Dense51 = torch.nn.Linear(10, 10)
        self.Dense52 = torch.nn.Linear(10, 10)
        self.Dense53 = torch.nn.Linear(10, 10)
        self.Dense54 = torch.nn.Linear(10, 10)
        self.Dense55 = torch.nn.Linear(10, 10)
        self.Dense56 = torch.nn.Linear(10, 10)
        self.Dense57 = torch.nn.Linear(10, 10)
        self.Dense58 = torch.nn.Linear(10, 10)
        self.Dense59 = torch.nn.Linear(10, 10)
        self.Dense60 = torch.nn.Linear(10, 10)
        self.Dense61 = torch.nn.Linear(10, 10)
        self.Dense62 = torch.nn.Linear(10, 10)
        self.Dense63 = torch.nn.Linear(10, 1)
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
        x=F.relu(self.Dense16(x))
        x=F.relu(self.Dense17(x))
        x=F.relu(self.Dense18(x))
        x=F.relu(self.Dense19(x))
        x=F.relu(self.Dense20(x))
        x=F.relu(self.Dense21(x))
        x=F.relu(self.Dense22(x))
        x=F.relu(self.Dense23(x))
        x=F.relu(self.Dense24(x))
        x=F.relu(self.Dense25(x))
        x=F.relu(self.Dense26(x))
        x=F.relu(self.Dense27(x))
        x=F.relu(self.Dense28(x))
        x=F.relu(self.Dense29(x))
        x=F.relu(self.Dense30(x))
        x=F.relu(self.Dense31(x))
        x=F.relu(self.Dense32(x))
        x=F.relu(self.Dense33(x))
        x=F.relu(self.Dense34(x))
        x=F.relu(self.Dense35(x))
        x=F.relu(self.Dense36(x))
        x=F.relu(self.Dense37(x))
        x=F.relu(self.Dense38(x))
        x=F.relu(self.Dense39(x))
        x=F.relu(self.Dense40(x))
        x=F.relu(self.Dense41(x))
        x=F.relu(self.Dense42(x))
        x=F.relu(self.Dense43(x))
        x=F.relu(self.Dense44(x))
        x=F.relu(self.Dense45(x))
        x=F.relu(self.Dense46(x))
        x=F.relu(self.Dense47(x))
        x=F.relu(self.Dense48(x))
        x=F.relu(self.Dense49(x))
        x=F.relu(self.Dense50(x))
        x=F.relu(self.Dense51(x))
        x=F.relu(self.Dense52(x))
        x=F.relu(self.Dense53(x))
        x=F.relu(self.Dense54(x))
        x=F.relu(self.Dense55(x))
        x=F.relu(self.Dense56(x))
        x=F.relu(self.Dense57(x))
        x=F.relu(self.Dense58(x))
        x=F.relu(self.Dense59(x))
        x=F.relu(self.Dense60(x))
        x=F.relu(self.Dense61(x))
        x=F.relu(self.Dense62(x))
        x=F.relu(self.Dense63(x))
        return x