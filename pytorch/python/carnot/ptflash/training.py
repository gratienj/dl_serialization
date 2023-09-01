import torch
import warnings
import sys
# import numpy as np
import pandas as pd

# from ptflash.doe import DOE
from ptflash.utils import get_properties
from ptflash.subroutines import PTFlash
from ptflash.equilibrium import SPTVLE1, SPTVLE2, analyser

import torch.nn as nn
import torch.nn.init as init


class PTClassifier_model(nn.Module):
    def __init__(self,
                 components,
                 classifier,
                 dtype=torch.float64,
                 device="cpu"):
        super(PTClassifier_model, self).__init__()
        self.components = components

        self.dtype = dtype
        self.device = device

        self.classifier = classifier
        self.classifier.eval()
        self.classifier.to(self.device)

        self.threshold=(0.05, 0.95)
        self.sa_max_nit1 = 9
        self.sa_max_nit2 = 40
        self.split_max_nit1 = 9
        self.split_max_nit2 = 40

        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(self.components, self.dtype, self.device)
        self.pcs = pcs
        self.tcs = tcs
        self.omegas = omegas
        self.kij = kij
        self.kijt = kijt
        self.kijt2 = kijt2


        self.pcs = pcs.to(dtype=dtype, device=device)
        self.tcs = tcs.to(dtype=dtype, device=device)
        self.omegas = omegas.to(dtype=dtype, device=device)
        self.flasher = PTFlash(pcs, tcs, omegas, kij, kijt, kijt2, "halley", self.dtype, self.device)

    def forward(self, inputs):
        with torch.no_grad():
            prob = self.classifier(inputs).sigmoid()
        stable = (prob >= self.threshold[1]).view(-1)
        unstable = ~stable
        return unstable

class PTInitializer_model(nn.Module):
    def __init__(self,
                 components,
                 initializer,
                 dtype=torch.float64,
                 device="cpu"):
        super(PTInitializer_model, self).__init__()
        self.components = components

        self.dtype = dtype
        self.device = device

        self.initializer = initializer
        self.initializer.eval()
        self.initializer.to(self.device)

        self.threshold=(0.05, 0.95)
        self.sa_max_nit1 = 9
        self.sa_max_nit2 = 40
        self.split_max_nit1 = 9
        self.split_max_nit2 = 40

        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(self.components, self.dtype, self.device)
        self.pcs = pcs
        self.tcs = tcs
        self.omegas = omegas
        self.kij = kij
        self.kijt = kijt
        self.kijt2 = kijt2


        self.pcs = pcs.to(dtype=dtype, device=device)
        self.tcs = tcs.to(dtype=dtype, device=device)
        self.omegas = omegas.to(dtype=dtype, device=device)
        self.flasher = PTFlash(pcs, tcs, omegas, kij, kijt, kijt2, "halley", self.dtype, self.device)

    def forward(self, inputs):

        with torch.no_grad():
            lnki = self.initializer(inputs)

        ki = lnki.exp()

        return ki

class PTStability_model(nn.Module):
    def __init__(self,
                 components, dtype=torch.float64,
                 device="cpu",
                 initializer=None,
                 classifier=None):
        super(PTStability_model, self).__init__()
        self.components = components

        self.dtype = dtype
        self.device = device

        self.initializer = initializer
        self.classifier = classifier

        self.threshold=(0.05, 0.95)
        self.sa_max_nit1 = 9
        self.sa_max_nit2 = 40
        self.split_max_nit1 = 9
        self.split_max_nit2 = 40

        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(self.components, self.dtype, self.device)
        self.pcs = pcs
        self.tcs = tcs
        self.omegas = omegas
        self.kij = kij
        self.kijt = kijt
        self.kijt2 = kijt2


        self.pcs = pcs.to(dtype=dtype, device=device)
        self.tcs = tcs.to(dtype=dtype, device=device)
        self.omegas = omegas.to(dtype=dtype, device=device)
        self.flasher = PTFlash(pcs, tcs, omegas, kij, kijt, kijt2, "halley", self.dtype, self.device)

    def forward(self, inputs):

        #self.classifier.eval()
        #self.classifier.to(self.device)
        with torch.no_grad():
            prob = self.classifier(inputs).sigmoid()
        stable = (prob >= self.threshold[1]).view(-1)
        unstable = ~stable

        prob = prob[~stable]
        inputs = inputs[~stable]
        undetermined = (prob > self.threshold[0]).view(-1)
        sa_indices = torch.where(undetermined)[0]
        sa_inputs = inputs[sa_indices]

        #self.initializer.eval()
        #self.initializer.to(self.device)
        with torch.no_grad():
            lnki = self.initializer(inputs)

        ki = lnki.exp()

        #print("INDICES NUMEL",sa_indices.numel())
        if sa_indices.numel() > 0:
            sa_ki = ki[sa_indices]
            vapour_stable, vapour_wi, vapour_tm = analyser( self.flasher,
                                                            sa_inputs,
                                                            sa_ki,
                                                            None,
                                                            vapour_like=True,
                                                            sa_max_nit1=self.sa_max_nit1,
                                                            sa_max_nit2=self.sa_max_nit2,
                                                            debug=False,
                                                        )

            liquid_stable, liquid_wi, liquid_tm = analyser( self.flasher,
                                                            sa_inputs,
                                                            sa_ki,
                                                            None,
                                                            vapour_like=False,
                                                            sa_max_nit1=self.sa_max_nit1,
                                                            sa_max_nit2=self.sa_max_nit2,
                                                            debug=False,
                                                            )

            sa_stable = (vapour_stable & liquid_stable).view(-1)
            sa_unstable = ~sa_stable
            # reinitialize ki based on stability analysis
            # if liquid_tm < vapour_tm, then ki = zi / liquid_wi
            # if liquid_tm > vapour_tm, then ki = vapour_wi / zi
            cond = liquid_tm[sa_unstable] < vapour_tm[sa_unstable]
            sa_inputs2 = sa_inputs[sa_unstable]
            liquid_ki = sa_inputs2[:, 2:] / liquid_wi[sa_unstable]
            vapour_ki = vapour_wi[sa_unstable] / sa_inputs2[:, 2:]
            estimated_ki = torch.where(cond.unsqueeze(-1), liquid_ki, vapour_ki)
            ki[sa_indices[sa_unstable]] = estimated_ki

            indices = torch.where(~stable)[0]
            stable[indices[sa_indices[sa_stable]]] = True

            unstable = ~stable
            mask = torch.ones(inputs.shape[0], dtype=torch.bool)
            mask[sa_indices[sa_stable]] = False
            inputs = inputs[mask]
            ki = ki[mask]

        return unstable, inputs, ki

class PTFlash_model1(nn.Module):
    def __init__(self,components, dtype=torch.float64, device="cpu"):
        super(PTFlash_model1, self).__init__()
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

        ptvle = SPTVLE1(
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

class PTFlash_model2(nn.Module):
    def __init__(self,components, dtype=torch.float64, device="cpu"):
        super(PTFlash_model2, self).__init__()
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


    def forward(self, inputs, ki):

        ptvle = SPTVLE2(
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
        unstable, theta_v, xi, yi, ki = ptvle(inputs,ki)

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

