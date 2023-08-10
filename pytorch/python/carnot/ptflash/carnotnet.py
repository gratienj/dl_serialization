import torch
import warnings
import sys
import numpy as np

# from ptflash.doe import DOE
from ptflash.utils import get_properties
from ptflash.equilibrium import PTVLE
import torch.nn as nn
import torch.nn.init as init

class CarnotNet(nn.Module):
    def __init__(self,device="cpu",dtype=torch.float64):
        super(CarnotNet, self).__init__()

        # CH4, C2H6, C3H8, C4H10, C5H12, C6H14, C7H16, CO2, N2
        components = [74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379]
        n_components = len(components)
        conditions = {"pmin": 5.0e6, "pmax": 2.5e7, "tmin": 200, "tmax": 600}

        self.device = device
        self.dtype = dtype
        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(components, self.dtype, self.device)

        self.ptvle = PTVLE(
            pcs,
            tcs,
            omegas,
            kij,
            kijt,
            kijt2,
            dtype=self.dtype,
            cubic_solver="halley",
            device=self.device,
        )

    def forward(self, inputs):
        print("CARNOT NET FORWARD")
        unstable, theta_v, xi, yi, ki = self.ptvle(
            inputs,
            sa_max_nit1=9,
            sa_max_nit2=40,
            split_max_nit1=9,
            split_max_nit2=40,
            async_analysis=False,
            debug=False,
        )

        return unstable, theta_v, xi, yi, ki

