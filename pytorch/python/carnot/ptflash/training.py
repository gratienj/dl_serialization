import torch
import warnings
import sys
# import numpy as np
import pandas as pd

# from ptflash.doe import DOE
from ptflash.utils import get_properties
from ptflash.equilibrium import SPTVLE
import torch.nn as nn
import torch.nn.init as init



class PTFlash_model(nn.Module):
    def __init__(self,components, dtype=torch.float64, device="cpu"):
        super(PTFlash_model, self).__init__()
        self.components = components
        self.dtype = dtype
        self.device = device
        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(self.components, self.dtype, self.device)
        self.pcs = pcs
        self.tcs = tcs
        self.omegas = omegas
        self.kij = kij
        self.kijt = kijt
        self.kijt2 = kijt2


    def forward(self, inputs):

        ptvle = SPTVLE(
            self.pcs,
            self.tcs,
            self.omegas,
            self.kij,
            self.kijt,
            self.kijt2,
            dtype=self.dtype,
            cubic_solver="halley",
            device=self.device,
        )
        unstable, theta_v, xi, yi, ki = ptvle(
            inputs
        )

        return unstable, theta_v, xi, yi, ki

class PTVLE_model(nn.Module):
    def __init__(self):
        super(PTVLE_model, self).__init__()

    def forward(self, inputs):

        dtype = torch.float64
        device = "cpu"
        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(components, dtype, device)
        unstable, theta_v, xi, yi, ki = PTVLE(
            pcs,
            tcs,
            omegas,
            kij,
            kijt,
            kijt2,
            dtype=dtype,
            cubic_solver="halley",
            device=device,
        )

        return unstable, theta_v, xi, yi, ki

