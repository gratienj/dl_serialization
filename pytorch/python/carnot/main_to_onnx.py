import torch
import warnings
import sys
import numpy as np
import pandas as pd

#from ptflash.doe import DOE
from ptflash.utils import get_properties
from ptflash.doe import DOE
from ptflash.equilibrium import PTVLE
from ptflash.training import PTFlash_model


torch.set_default_dtype(torch.double)
seed = 321
components = [74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379]
n_components = len(components)
conditions = {"pmin": 5.0e6, "pmax": 2.5e7, "tmin": 200, "tmax": 600}

n_samples = 16
doe = DOE(
    n_samples,
    n_components,
    pmin=conditions["pmin"],
    pmax=conditions["pmax"],
    tmin=conditions["tmin"],
    tmax=conditions["tmax"],
    doe_type="dirichlet",
    random_state=seed,
)
design = doe.create_design()

dtype = torch.float64
torch.set_default_dtype(torch.float64)
device = "cpu"
inputs = torch.tensor(design.to_numpy(), dtype=dtype, device=device)

pyTorch_Module = PTFlash_model()
pyTorch_Module.train(False)

torch.onnx.export(pyTorch_Module,               # model being run
                  (inputs), #(inputs, comparison, sa_max_nit1, sa_max_nit2, split_max_nit1, split_max_nit2, async_analysis, debug),                         # model input (or a tuple for multiple inputs)
                  "ptflash_NoNN.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=False,        # store the trained parameter weights inside the model file
                  opset_version=15)          # the ONNX version to export the model to
                 # do_constant_folding=False,  # whether to execute constant folding for optimization
                 # input_names = ['inputs'],  # ['P', 'T', 'Z1', 'Z2'],  #  ['inputs','sa_max_nit1','sa_max_nit2','split_max_nit1','split_max_nit2','async_analysis','debug'],   # the model's input names
                 # output_names = ['unstable','theta_v','xi','yi','ki'], # the model's output names
                 # dynamic_axes={'input' : {0 : io_axis},    # variable length axes
                               # 'output' : {0 : io_axis}})