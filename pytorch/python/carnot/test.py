import torch
import warnings

from ptflash.doe import DOE
from ptflash.utils import get_properties
from ptflash.equilibrium import PTVLE
from ptflash.networks import PTNet, TrainWrapper
from ptflash import utils

from ptflash.carnotnet import CarnotNet
from ptflash.training import PTFlash_model

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import jit

import pytorch_lightning as pl
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsolutePercentageError
from torchmetrics import Accuracy, F1Score

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import onnx
import onnxruntime as ort

import numpy as np

class RR_Solver(torch.autograd.Function):
    """
    Use the implicit function theorem to directly obtain the partial derivatives
    of theta_v w.r.t. ki instead of differentiation through iterations of `rr_solver`.
    """

    @staticmethod
    def forward(ctx, zi, ki):
        theta = utils.rr_solver(zi, ki)[0]
        ctx.save_for_backward(zi, ki, theta)
        return theta

    @staticmethod
    def backward(ctx, grad_output):
        zi, ki, theta = ctx.saved_tensors
        # calculate the gradient of the Rachford-Rice equation w.r.t. theta_v
        gv = torch.sum(
            -zi * ((ki - 1) / (1 + (ki - 1) * theta)) ** 2, dim=1, keepdim=True
        )
        # calculate the gradient of the Rachford-Rice equation w.r.t. ki
        gk = zi / (1 + (ki - 1) * theta) ** 2
        # the derivatives of theta_V w.r.t. ki
        gvk = -gk / gv
        return None, grad_output * gvk

def train(
    net,
    train_loader,
    valid_loader,
    num_epochs=20,
    device="cpu",
):
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1.0e-5)

    for epoch in range(1, num_epochs + 1):
        train_ki = 0
        train_theta = 0
        valid_ki = 0
        valid_theta = 0

        # Training loop
        for batch in train_loader:
            x = batch[0].to(device)
            ln_ki = batch[1].to(device)
            theta = batch[2].to(device)
            opt.zero_grad()
            pred_ln_ki = net(x)
            pred_theta = RR_Solver.apply(x[:, 2:], pred_ln_ki.exp())
            mae_ki = F.l1_loss(pred_ln_ki, ln_ki)
            mae_theta = F.l1_loss(pred_theta, theta)
            loss = mae_ki + mae_theta
            loss.backward()
            opt.step()
            train_ki += mae_ki
            train_theta += mae_theta
        train_ki /= len(train_loader)
        train_theta /= len(train_loader)

        # Validation loop
        with torch.no_grad():
            net.eval()
            for batch in valid_loader:
                x = batch[0].to(device)
                ln_ki = batch[1].to(device)
                theta = batch[2].to(device)
                pred_ln_ki = net(x)
                pred_theta = RR_Solver.apply(x[:, 2:], pred_ln_ki.exp())
                mae_ki = F.l1_loss(pred_ln_ki, ln_ki)
                mae_theta = F.l1_loss(pred_theta, theta)
                valid_ki += mae_ki
                valid_theta += mae_theta
        valid_ki /= len(valid_loader)
        valid_theta /= len(valid_loader)

        print(
            "Epoch {:2} -- Train losses: ki = {:.4f}  theta = {:.4f} -- Valid losses: ki = {:.4f}  theta = {:.4f}".format(
                epoch, train_ki, train_theta, valid_ki, valid_theta
            )
        )


if __name__ == "__main__":
    # An instance of your model.
    import os
    import argparse

    parser = argparse.ArgumentParser(description='GNNNet test functions')
    parser.add_argument("--test_id",       type=int, default=0,              help="Test query")
    parser.add_argument("--data_id",       type=int, default=0,              help="Data query")
    parser.add_argument("--batch_size",    type=int, default=1,              help="Batch size query")
    parser.add_argument("--data_dir_path", type=str, default="./data",       help="data dir path")
    parser.add_argument("--dir_path",      type=str, default="./",           help="dir path")
    parser.add_argument("--model_name",    type=str, default="gnn1_model",   help="model name")
    parser.add_argument("--device",        type=str, default="cpu",          help="device cpu, cuda")
    args = parser.parse_args()
    test_id = args.test_id
    data_id = args.data_id

    if test_id == 0:
        print("TEST 2 COMPONENTS")
        components = [74828, 110543]
        n_components = len(components)
        conditions = {"pmin": 1.0e5, "pmax": 1.0e7, "tmin": 200, "tmax": 500}
        n_samples = 100
        doe = DOE(
            n_samples,
            n_components,
            pmin=conditions["pmin"],
            pmax=conditions["pmax"],
            tmin=conditions["tmin"],
            tmax=conditions["tmax"],
            doe_type="dirichlet",
            random_state=1,
        )
        design = doe.create_design()

        print(design.head(2))
        dtype = torch.float64
        device = "cpu"
        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(components, dtype, device)
        ptvle = PTVLE(
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
        inputs = torch.tensor(design.to_numpy(), dtype=dtype, device=device)

        unstable, theta_v, xi, yi, ki = ptvle(
            inputs,
            sa_max_nit1=9,
            sa_max_nit2=40,
            split_max_nit1=9,
            split_max_nit2=40,
            async_analysis=False,
            debug=True,
        )
        print("UNSTABLE",unstable)
        print("THETA",theta_v)
        print("XI",xi)
        print("YI",yi)
        print("KI",ki)

        dtype = torch.float64
        torch.set_default_dtype(torch.float64)
        device = "cpu"

        print("PREPARE DATA")
        #X = torch.from_numpy(design.to_numpy())[unstable]
        #y = ki.log()

        dtype = torch.float64
        torch.set_default_dtype(torch.float64)
        #X = X.type(torch.FloatTensor)
        #y = y.type(torch.FloatTensor)


        print("DEFINE CARNOT MODEL")
        net = PTFlash_model(components,dtype,device)
        net.eval()

        print("EVAL CARNOT MODEL")
        unstable, theta_v, xi, yi, ki = net(inputs)
        print("UNSTABLE",unstable)
        print("THETA",theta_v)
        print("XI",xi)
        print("YI",yi)
        print("KI",ki)

        print("JIT SCRIPT EXPORT MODEL")
        snet = torch.jit.script(net)
        snet.save('ptflash.pt')
        np.save(f"data_2comp.npy",design.to_numpy())


        '''
        print("ONNX EXPORT")
        torch.onnx.export(net,  # model being run
                          X,
                          # (inputs, comparison, sa_max_nit1, sa_max_nit2, split_max_nit1, split_max_nit2, async_analysis, debug),                         # model input (or a tuple for multiple inputs)
                          "ptflash_NoNN.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=False,  # store the trained parameter weights inside the model file
                          opset_version=10)  # the ONNX version to export the model to
        # do_constant_folding=False,  # whether to execute constant folding for optimization
        # input_names = ['inputs'],  # ['P', 'T', 'Z1', 'Z2'],  #  ['inputs','sa_max_nit1','sa_max_nit2','split_max_nit1','split_max_nit2','async_analysis','debug'],   # the model's input names
        # output_names = ['unstable','theta_v','xi','yi','ki'], # the model's output names
        # dynamic_axes={'input' : {0 : io_axis},    # variable length axes
        # 'output' : {0 : io_axis}})
        '''


    if test_id == 10:
        print("TEST 2 COMPONENTS")
        components = [74828, 110543]
        n_components = len(components)
        conditions = {"pmin": 1.0e5, "pmax": 1.0e7, "tmin": 200, "tmax": 500}
        n_samples = 100
        doe = DOE(
            n_samples,
            n_components,
            pmin=conditions["pmin"],
            pmax=conditions["pmax"],
            tmin=conditions["tmin"],
            tmax=conditions["tmax"],
            doe_type="dirichlet",
            random_state=1,
        )
        design = doe.create_design()

        print(design.head(2))
        dtype = torch.float64
        device = "cpu"
        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(components, dtype, device)
        ptvle = PTVLE(
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
        inputs = torch.tensor(design.to_numpy(), dtype=dtype, device=device)

        unstable, theta_v, xi, yi, ki = ptvle(
            inputs,
            sa_max_nit1=9,
            sa_max_nit2=40,
            split_max_nit1=9,
            split_max_nit2=40,
            async_analysis=False,
            debug=True,
        )
        print("UNSTABLE",unstable)
        print("THETA",theta_v)
        print("XI",xi)
        print("YI",yi)
        print("KI",ki)

        dtype = torch.float64
        torch.set_default_dtype(torch.float64)
        device = "cpu"

        #print("PREPARE DATA")
        #X = torch.from_numpy(design.to_numpy())[unstable]
        #y = ki.log()

        #dtype = torch.float64
        #torch.set_default_dtype(torch.float64)
        #X = X.type(torch.FloatTensor)
        #y = y.type(torch.FloatTensor)


        print("LOAD CARNOT MODEL FROM PT FILE")
        net = torch.jit.load('ptflash_2comp.pt')
        net.eval()

        print("EVAL CARNOT MODEL")
        unstable, theta_v, xi, yi, ki = net(inputs)
        print("UNSTABLE",unstable)
        print("THETA",theta_v)
        print("XI",xi)
        print("YI",yi)
        print("KI",ki)

    if test_id == 1 :
        print("TEST INITIALIZER 9 COMPONENTS")
        torch.set_default_dtype(torch.double)
        seed = 321

        # CH4, C2H6, C3H8, C4H10, C5H12, C6H14, C7H16, CO2, N2
        components = [74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379]
        n_components = len(components)
        conditions = {"pmin": 5.0e6, "pmax": 2.5e7, "tmin": 200, "tmax": 600}

        n_samples = 100000
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
        device = "cpu"
        pcs, tcs, omegas, kij, kijt, kijt2 = utils.get_properties(components, dtype, device)
        ptvle = PTVLE(
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
        inputs = torch.tensor(design.to_numpy(), dtype=dtype, device=device)
        unstable, theta_v, xi, yi, ki = ptvle(
            inputs, sa_max_nit1=9, sa_max_nit2=40, split_max_nit1=9, split_max_nit2=40
        )

        print("PREPARE DATA")
        X = torch.from_numpy(design.to_numpy())[unstable]
        y = ki.log()

        dtype = torch.float32
        torch.set_default_dtype(torch.float32)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        print("SPLIT DATA")
        X_train, X_test, y_train, y_test, theta_train, theta_test = train_test_split(
            X, y, theta_v, test_size=0.2, random_state=seed
        )
        X_train, X_valid, y_train, y_valid, theta_train, theta_valid = train_test_split(
            X_train, y_train, theta_train, test_size=0.2, random_state=seed
        )

        print("CREATE DATA LOADER")
        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_valid, y_valid)
        test_ds = TensorDataset(X_test, y_test)
        batch_size = 512
        workers = 4
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )

        print("DEFINE NETWORK")
        pl.seed_everything(seed)
        net = PTNet(
            input_dim=X_train.shape[1],
            output_dim=y_train.shape[1],
            mean=X_train.mean(dim=0),
            scale=X_train.std(dim=0),
            units=[64] * 7,
            activation="SiLU",
            concat=True,
            residual=True,
        )

        print("Find Good Trainig RATE")
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else None, logger=False
        )

        tuner = pl.tuner.Tuner(trainer)
        finder = tuner.lr_find(
            TrainWrapper(net, nn.L1Loss()),
            train_loader,
            valid_loader,
            min_lr=1e-6,
            max_lr=0.1,
            num_training=len(train_loader),
        )
        suggestion = finder.suggestion()
        lr = np.array(finder.results["lr"])
        loss = np.array(finder.results["loss"])
        plt.figure(figsize=(6, 3))
        plt.plot(lr, loss)
        plt.plot(suggestion, loss[lr == suggestion], markersize=8, marker="o", color="red")
        plt.plot(lr[loss.argmin()], loss.min(), markersize=8, marker="s", color="black")
        plt.title("Learning rate range test")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True)
        plt.show()

        print("TRAINING STEP")
        wrapper = TrainWrapper(
            net,
            loss_func=nn.L1Loss(),
            metrics=MetricCollection(
                dict(mse=MeanSquaredError(), mape=MeanAbsolutePercentageError())
            ),
            sche_args=dict(
                base_lr=1e-4,
                max_lr=0.01,
                step_size_up=2 * len(train_loader),
                mode="triangular2",
                cycle_momentum=False,
            ),
        )
        trainer = pl.Trainer(
            logger=False, accelerator="gpu", max_epochs=48, enable_checkpointing=False
        )
        trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=valid_loader)


        print("TEST STEP")
        test_res = trainer.test(wrapper, test_loader)

        print("TRIANING STEP 2")

        train_ds2 = TensorDataset(X_train, y_train, theta_train)
        valid_ds2 = TensorDataset(X_valid, y_valid, theta_valid)

        batch_size = 512
        workers = 4
        train_loader2 = DataLoader(
            train_ds2,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader2 = DataLoader(
            valid_ds2, batch_size=batch_size, num_workers=workers, pin_memory=True
        )
        train(net, train_loader2, valid_loader2, num_epochs=10, device="cuda")

        torch.save(net, "initializer.pt")

        net.eval()

        batch = next(iter(test_loader))
        X = batch[0].to('cuda')
        pred = net(X)

        # Export the model
        torch.onnx.export(net,  # model being run
                          X,  # model input (or a tuple for multiple inputs)
                          "initializer.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=15,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})
    if test_id == 11:
        print("TEST 9 COMPONENTS")
        components = [74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379]
        n_components = len(components)
        conditions = {"pmin": 5.0e6, "pmax": 2.5e7, "tmin": 200, "tmax": 600}
        n_samples = 100
        doe = DOE(
            n_samples,
            n_components,
            pmin=conditions["pmin"],
            pmax=conditions["pmax"],
            tmin=conditions["tmin"],
            tmax=conditions["tmax"],
            doe_type="dirichlet",
            random_state=1,
        )
        design = doe.create_design()

        print(design.head(2))
        dtype = torch.float64
        device = "cpu"
        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(components, dtype, device)
        ptvle = PTVLE(
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
        inputs = torch.tensor(design.to_numpy(), dtype=dtype, device=device)

        unstable, theta_v, xi, yi, ki = ptvle(
            inputs,
            sa_max_nit1=9,
            sa_max_nit2=40,
            split_max_nit1=9,
            split_max_nit2=40,
            async_analysis=False,
            debug=True,
        )
        print("UNSTABLE",unstable)
        print("THETA",theta_v)
        print("XI",xi)
        print("YI",yi)
        print("KI",ki)

        dtype = torch.float64
        torch.set_default_dtype(torch.float64)
        device = "cpu"

        print("DEFINE CARNOT MODEL")
        net = PTFlash_model(components,dtype,device)
        net.eval()

        print("EVAL CARNOT MODEL")
        unstable, theta_v, xi, yi, ki = net(inputs)
        print("UNSTABLE",unstable)
        print("THETA",theta_v)
        print("XI",xi)
        print("YI",yi)
        print("KI",ki)

        print("JIT SCRIPT EXPORT MODEL")
        snet = torch.jit.script(net)
        snet.save('ptflash_9comp.pt')
        np.save(f"data_9comp.npy",design.to_numpy())

        #print("PREPARE DATA")
        #X = torch.from_numpy(design.to_numpy())[unstable]
        #y = ki.log()

        #dtype = torch.float64
        #torch.set_default_dtype(torch.float64)
        #X = X.type(torch.FloatTensor)
        #y = y.type(torch.FloatTensor)


        print("LOAD CARNOT MODEL FROM PT FILE")
        net = torch.jit.load('ptflash_9comp.pt')
        net.eval()

        print("EVAL CARNOT MODEL")
        inputs = torch.tensor(design.to_numpy()[data_id:data_id+args.batch_size,:], dtype=dtype, device=device)
        unstable, theta_v, xi, yi, ki = net(inputs)
        print("UNSTABLE",unstable)
        print("THETA",theta_v)
        print("XI",xi)
        print("YI",yi)
        print("KI",ki)

    if test_id == 2 :
        print("TEST CLASSIFIER 9 COMPONENTS")
        seed = 123


        # CH4, C2H6, C3H8, C4H10, C5H12, C6H14, C7H16, CO2, N2
        components = [74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379]
        n_components = len(components)
        conditions = {"pmin": 5.0e6, "pmax": 2.5e7, "tmin": 200, "tmax": 600}

        n_samples = 1000000
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
        device = "cpu"
        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(components, dtype, device)
        ptvle = PTVLE(
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
        inputs = torch.tensor(design.to_numpy(), dtype=dtype, device=device)
        unstable, theta_v, xi, yi, ki = ptvle(
            inputs, sa_max_nit1=9, sa_max_nit2=40, split_max_nit1=9, split_max_nit2=40
        )

        # classifier is used to predict stability
        X = torch.from_numpy(design.to_numpy())
        y = (~unstable).double().reshape(-1, 1)

        dtype = torch.float32
        torch.set_default_dtype(torch.float32)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        print("CLASSIFIER DATA SPLIT STEP")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed
        )
        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_valid, y_valid)
        test_ds = TensorDataset(X_test, y_test)
        batch_size = 512
        workers = 4
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )

        print("CLASSIFIER NETWORK DEFINITION STEP")
        net = PTNet(
            input_dim=X_train.shape[1],
            output_dim=1,
            mean=X_train.mean(dim=0),
            scale=X_train.std(dim=0),
            units=(32, 32, 32),
            activation="SiLU",
        )

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else None, logger=False
        )

        tuner = pl.tuner.Tuner(trainer)
        finder = tuner.lr_find(
            TrainWrapper(net, nn.BCEWithLogitsLoss()),
            train_loader,
            valid_loader,
            min_lr=1e-6,
            max_lr=0.1,
            num_training=len(train_loader),
        )
        suggestion = finder.suggestion()
        lr = np.array(finder.results["lr"])
        loss = np.array(finder.results["loss"])
        plt.figure(figsize=(6, 3))
        plt.plot(lr, loss)
        plt.plot(
            suggestion, loss[lr == suggestion], markersize=8, marker="o", color="red"
        )
        plt.plot(lr[loss.argmin()], loss.min(), markersize=8, marker="s", color="black")
        plt.title("Learning rate range test")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True)
        plt.show()


        print("CLASSIFIER TRAINING STEP")
        wrapper = TrainWrapper(
            net,
            loss_func=nn.BCEWithLogitsLoss(),
            metrics=MetricCollection([Accuracy(task="binary"), F1Score(task="binary")]),
            sche_args=dict(
                base_lr=1e-4,
                max_lr=0.01,
                step_size_up=2 * len(train_loader),
                mode="triangular2",
                cycle_momentum=False,
            ),
        )
        trainer = pl.Trainer(
            logger=False, accelerator="gpu", max_epochs=20, enable_checkpointing=False
        )
        trainer.fit(wrapper, train_dataloaders=train_loader, val_dataloaders=valid_loader)

        print("CLASSIFIER TEST STEP")
        test_res = trainer.test(wrapper, test_loader)

        torch.save(net, "classifier.pt")

        batch = next(iter(test_loader))
        X = batch[0]
        pred = net(X)

        # Export the model
        torch.onnx.export(net,  # model being run
                          X,  # model input (or a tuple for multiple inputs)
                          "classifier.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=15,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})

    if test_id == 22 :
        print("TEST ONNX CLASSIFIER 9 COMPONENTS INFERENCE",ort.__version__)

        seed = 123

        # CH4, C2H6, C3H8, C4H10, C5H12, C6H14, C7H16, CO2, N2
        components = [74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379]
        n_components = len(components)
        conditions = {"pmin": 5.0e6, "pmax": 2.5e7, "tmin": 200, "tmax": 600}

        n_samples = 100
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
        device = "cpu"
        pcs, tcs, omegas, kij, kijt, kijt2 = get_properties(components, dtype, device)
        ptvle = PTVLE(
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
        inputs = torch.tensor(design.to_numpy(), dtype=dtype, device=device)
        unstable, theta_v, xi, yi, ki = ptvle(
            inputs, sa_max_nit1=9, sa_max_nit2=40, split_max_nit1=9, split_max_nit2=40
        )

        # classifier is used to predict stability
        X = torch.from_numpy(design.to_numpy())
        y = (~unstable).double().reshape(-1, 1)

        dtype = torch.float32
        torch.set_default_dtype(torch.float32)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )

        print("CLASSIFIER DATA SPLIT STEP")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed
        )
        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_valid, y_valid)
        test_ds = TensorDataset(X_test, y_test)
        batch_size = 512
        workers = 4
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )

        # Load the ONNX model
        net = onnx.load("classifier.onnx")

        # Check that the model is well formed
        onnx.checker.check_model(net)

        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(net.graph))

        ort_session = ort.InferenceSession("classifier.onnx")

        batch = next(iter(test_loader))
        X = batch[0]

        outputs = ort_session.run(
            None,
            {"input": X.numpy()},
        )
        print(outputs[0])
        print(y_test.numpy())

    if test_id == 21:
        print("TEST INITIALIZER 9 COMPONENTS")
        torch.set_default_dtype(torch.double)
        seed = 321
        # CH4, C2H6, C3H8, C4H10, C5H12, C6H14, C7H16, CO2, N2
        components = [74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379]
        n_components = len(components)
        conditions = {"pmin": 5.0e6, "pmax": 2.5e7, "tmin": 200, "tmax": 600}

        n_samples = 100
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
        device = "cpu"
        pcs, tcs, omegas, kij, kijt, kijt2 = utils.get_properties(components, dtype, device)
        ptvle = PTVLE(
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
        inputs = torch.tensor(design.to_numpy(), dtype=dtype, device=device)
        unstable, theta_v, xi, yi, ki = ptvle(
            inputs, sa_max_nit1=9, sa_max_nit2=40, split_max_nit1=9, split_max_nit2=40
        )

        print("PREPARE DATA")
        X = torch.from_numpy(design.to_numpy())[unstable]
        y = ki.log()

        dtype = torch.float32
        torch.set_default_dtype(torch.float32)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        print("SPLIT DATA")
        X_train, X_test, y_train, y_test, theta_train, theta_test = train_test_split(
            X, y, theta_v, test_size=0.2, random_state=seed
        )
        X_train, X_valid, y_train, y_valid, theta_train, theta_valid = train_test_split(
            X_train, y_train, theta_train, test_size=0.2, random_state=seed
        )

        print("CREATE DATA LOADER")
        train_ds = TensorDataset(X_train, y_train)
        valid_ds = TensorDataset(X_valid, y_valid)
        test_ds = TensorDataset(X_test, y_test)
        batch_size = 512
        workers = 4
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )

        # Load the ONNX model
        net = onnx.load("initializer.onnx")

        # Check that the model is well formed
        onnx.checker.check_model(net)

        # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(net.graph))


        ort_session = ort.InferenceSession("initializer.onnx")

        batch = next(iter(test_loader))
        X = batch[0]

        outputs = ort_session.run(
            None,
            {"input": X.numpy()},
        )
        print(outputs[0])
        print(y_test.numpy())



    if test_id == 5:
        print("TEST CARNOTE 9 COMPONENTS")
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
        pcs, tcs, omegas, kij, kijt, kijt2 = utils.get_properties(components, dtype, device)
        ptvle = PTVLE(
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
        inputs = torch.tensor(design.to_numpy(), dtype=dtype, device=device)
        unstable, theta_v, xi, yi, ki = ptvle(
            inputs, sa_max_nit1=9, sa_max_nit2=40, split_max_nit1=9, split_max_nit2=40
        )

        X = torch.from_numpy(design.to_numpy())[unstable]
        y = ki.log()

        dtype = torch.float32
        torch.set_default_dtype(torch.float32)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        print("CREATE DATA LOADER")
        test_ds = TensorDataset(X, y)
        batch_size = 8
        workers = 1
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, num_workers=workers, pin_memory=True
        )

        print("CREATE CARNOT NET")
        net = CarnotNet(device="cpu", dtype=torch.float32)
        net.eval()


        print("EVAL CARNOT NET")
        batch = next(iter(test_loader))
        X = batch[0]
        pred = net(X)

        # Export the model
        print("ONNX EXPORT CARNOT NET")
        torch.onnx.export(net,  # model being run
                          X,  # model input (or a tuple for multiple inputs)
                          "ptflashcarnot.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=False,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          input_names=['input'],  # the model's input names
                          output_names = ['unstable','theta_v','xi','yi','ki'],
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}}
                          )
                          #do_constant_folding=True,  # whether to execute constant folding for optimization






