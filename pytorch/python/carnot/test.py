import torch
import warnings

from ptflash.doe import DOE
from ptflash.utils import get_properties
from ptflash.equilibrium import PTVLE
from ptflash.networks import PTNet, TrainWrapper
from ptflash import utils

from ptflash.carnotnet import CarnotNet
from ptflash.training import PTClassifier_model, PTInitializer_model
from ptflash.training import PTFlash_model1, PTFlash_model2, PTStability_model

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

from timeit import default_timer

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
    parser.add_argument("--n_samples",     type=int, default=100,            help="nb samples")
    parser.add_argument("--batch_size",    type=int, default=1,              help="Batch size query")
    parser.add_argument("--n_iter",        type=int, default=100,            help="nb samples")
    parser.add_argument("--models",        type=str, default="class:init:stab:flash",  help="model lisy query")
    parser.add_argument("--data_dir_path", type=str, default="./data",       help="data dir path")
    parser.add_argument("--dir_path",      type=str, default="./",           help="dir path")
    parser.add_argument("--model_name",    type=str, default="gnn1_model",   help="model name")
    parser.add_argument("--device",        type=str, default="cpu",          help="device cpu, cuda")
    parser.add_argument("--save_mode",     type=str, default="dict",         help="dict, script")
    parser.add_argument("--export_onnx",   type=int, default=0,              help="enable onnx export")
    parser.add_argument("--output",        type=int, default=0,              help="Output level")
    parser.add_argument("--use_ml",        type=int, default=0,              help="use ml")

    args = parser.parse_args()
    test_id = args.test_id
    data_id = args.data_id
    n_samples = args.n_samples
    device = args.device
    save_mode = args.save_mode
    if test_id == 0:
        print("TEST 2 COMPONENTS")
        components = [74828, 110543]
        n_components = len(components)
        conditions = {"pmin": 1.0e5, "pmax": 1.0e7, "tmin": 200, "tmax": 500}
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
        snet.save(f'ptflash_2comp_{device}.pt')
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

        #print("PREPARE DATA")
        #X = torch.from_numpy(design.to_numpy())[unstable]
        #y = ki.log()

        #dtype = torch.float64
        #torch.set_default_dtype(torch.float64)
        #X = X.type(torch.FloatTensor)
        #y = y.type(torch.FloatTensor)


        print("LOAD CARNOT MODEL FROM PT FILE")
        net = torch.jit.load(f'ptflash_2comp_{device}.pt')
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


        if args.export_onnx == 1 :
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
        if args.export_onnx == 1 :
            torch.save(net, "initializer_x32.pt")
        else:
            torch.save(net, "initializer_x64.pt")

        net.eval()

        batch = next(iter(test_loader))
        X = batch[0].to('cuda')
        pred = net(X)

        # Export the model
        if args.export_onnx == 1 :
            torch.onnx.export(net,  # model being run
                              X,  # model input (or a tuple for multiple inputs)
                              "initializer_x32.onnx",  # where to save the model (can be a file or file-like object)
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

        #print(design.head(2))
        dtype = torch.float64
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

        if args.output == 2:
            runstable = unstable.numpy()
            rtheta_v = theta_v.numpy()
            rxi = xi.numpy()
            ryi = yi.numpy()
            unstable_index = 0
            for i in range(args.batch_size):
                unstable_value = 1 if runstable[i] else 0
                print(f"RES[{i}] UNSTABLE {unstable_value}")
                if runstable[i] :
                    print("             THETA_V : ", rtheta_v[unstable_index])
                    print("             LIQ XI : ", rxi[unstable_index, :])
                    print("             VAP YI : ", ryi[unstable_index, :])
                    unstable_index += 1

        #X = torch.from_numpy(design.to_numpy())[unstable]
        #y_ki = ki.log()
        #y_unstable = (~unstable).double().reshape(-1, 1)

        if args.output == 1 :
            print("UNSTABLE",unstable)
            print("THETA",theta_v)
            print("XI",xi)
            print("YI",yi)
            print("KI",ki)


        dtype = torch.float64
        torch.set_default_dtype(torch.float64)

        if args.export_onnx == 1 :
            dtype = torch.float32
            torch.set_default_dtype(torch.float32)
            inputs = inputs.type(torch.FloatTensor)
            #y_ki = y_ki.type(torch.FloatTensor)
            #y_unstable = y_unstable.type(torch.FloatTensor)

        print("DEFINE CARNOT MODEL")
        if args.use_ml == 0:
            net = PTFlash_model1(components=components,dtype=dtype,device=device)
            net.eval()

            print("EVAL CARNOT MODEL")
            start = default_timer()
            unstable, theta_v, xi, yi, ki = net(inputs)
            time = default_timer() - start
            print(f"TIME FOR PYTORCH INFERENCE:{time:.4f}s \n")

            if args.export_onnx == 1:
                torch.onnx.export(net,  # model being run
                                  (inputs,None),  # model input (or a tuple for multiple inputs)
                                  f"ptflash_9comp_{n_samples}_{device}.onnx",  # where to save the model (can be a file or file-like object)
                                  export_params=True,  # store the trained parameter weights inside the model file
                                  opset_version=15,  # the ONNX version to export the model to
                                  do_constant_folding=True,  # whether to execute constant folding for optimization
                                  input_names=['input','ki'],  # the model's input names
                                  output_names=['unstable', 'theta_v', 'xi', 'yi', 'ski'],  # the model's output names
                                  dynamic_axes={'input': {0: 'batch_size'},
                                                'ki': {0: 'batch_size'},
                                                'unstable': {0: 'batch_size'},
                                                'theta_v': {0: 'batch_size'},
                                                'xi': {0: 'batch_size'},
                                                'yi': {0: 'batch_size'},
                                                'ski': {0: 'batch_size'}})
            else:
                if args.save_mode == "dict":
                    print("TORCH EXPORT MODEL")
                    torch.save(net.state_dict(), f"ptflash_dict_9comp_{n_samples}_{device}.pt")

                if args.save_mode == "sjit":
                    print("JIT SCRIPT EXPORT MODEL")
                    net.eval()
                    snet = torch.jit.script(net)
                    snet.save(f"ptflash_sjit_9comp_{n_samples}_{device}.pt")

                if args.save_mode == "sjitopt":
                    print("JIT SCRIPT OPT EXPORT MODEL")
                    net.eval()
                    torch._C._jit_set_profiling_executor(False)
                    snet = torch.jit.script(net)
                    model = torch.jit.freeze(snet)
                    snet_opt = torch.jit.optimize_for_inference(snet)
                    snet_opt.save(f"ptflash_sjitopt_9comp_{n_samples}_{device}.pt")

                if args.save_mode == "tjit":
                    print("JIT SCRIPT EXPORT MODEL")
                    snet = torch.jit.trace(net,inputs)
                    snet.save(f"ptflash_tjit_9comp_{n_samples}_{device}.pt")

        if args.use_ml == 1:
            '''
            classifier = PTNet( input_dim=X_train.shape[1],
                                output_dim=1,
                                mean=X.mean(dim=0),
                                scale=X.std(dim=0),
                                units=(32, 32, 32),
                                activation="SiLU",
                            )'''

            if args.export_onnx == 1:
                classifier = torch.load("classifier_x32.pt")
            else:
                classifier = torch.load("classifier_x64.pt")

            classifier.to(device)
            classifier.eval()
            cnet = PTClassifier_model(components=components,
                                     classifier=classifier,
                                     dtype=dtype,
                                     device=device)

            '''
            initializer = PTNet(input_dim=X.shape[1],
                                output_dim=y_ki.shape[1],
                                mean=X.mean(dim=0),
                                scale=X.std(dim=0),
                                units=[64] * 7,
                                activation="SiLU",
                                concat=True,
                                residual=True,
                                ) 
                            '''
            if args.export_onnx == 1:
                initializer = torch.load("initializer_x32.pt")
            else:
                initializer = torch.load("initializer_x64.pt")

            initializer.to(device)
            initializer.eval()
            inet = PTInitializer_model(components=components,
                                       initializer=initializer,
                                       dtype=dtype,
                                       device=device)


            snet = PTStability_model(components=components,
                                     dtype=dtype,
                                     device=device,
                                     initializer=initializer,
                                     classifier=classifier)

            net = PTFlash_model2(components=components,
                                dtype=dtype,
                                device=device)


            print("EVAL CARNOT MODEL")

            start = default_timer()
            snet.eval()
            unstable = cnet(inputs)
            time = default_timer() - start
            print(f"TIME FOR PYTORCH CLASSIFIER INFERENCE:{time:.4f}s \n")

            start = default_timer()
            inet.eval()
            ki = cnet(inputs)
            time = default_timer() - start
            print(f"TIME FOR PYTORCH INITIALIZER INFERENCE:{time:.4f}s \n")

            start = default_timer()
            snet.eval()
            unstable, sinputs, ski = snet(inputs)
            time = default_timer() - start
            print(f"TIME FOR PYTORCH STABILITY INFERENCE:{time:.4f}s \n")

            start = default_timer()
            net.eval()
            unstable2, theta_v, xi, yi, ki = net(sinputs,ski)
            time = default_timer() - start
            print(f"TIME FOR PYTORCH FLASH INFERENCE:{time:.4f}s \n")


            if args.export_onnx == 1:
                torch.onnx.export(snet,  # model being run
                                  inputs,  # model input (or a tuple for multiple inputs)
                                  f"ptstabilityml_9comp_{n_samples}_{device}.onnx",  # where to save the model (can be a file or file-like object)
                                  export_params=True,  # store the trained parameter weights inside the model file
                                  opset_version=15,  # the ONNX version to export the model to
                                  do_constant_folding=True,  # whether to execute constant folding for optimization
                                  input_names=['input'],  # the model's input names
                                  output_names=['output'],  # the model's output names
                                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                                'output': {0: 'batch_size'}})

                torch.onnx.export(net,  # model being run
                                  (sinputs,ski),  # model input (or a tuple for multiple inputs)
                                  f"ptflashml_9comp_{n_samples}_{device}.onnx",  # where to save the model (can be a file or file-like object)
                                  export_params=True,  # store the trained parameter weights inside the model file
                                  opset_version=15,  # the ONNX version to export the model to
                                  do_constant_folding=True,  # whether to execute constant folding for optimization
                                  input_names=['input','ki'],  # the model's input names
                                  output_names=['unstable','theta_v','xi','yi','ski'],  # the model's output names
                                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                                'ki': {0: 'batch_size'},
                                                'unstable': {0: 'batch_size'},
                                                'theta_v': {0: 'batch_size'},
                                                'xi': {0: 'batch_size'},
                                                'yi': {0: 'batch_size'},
                                                'ski': {0: 'batch_size'}})
            else:
                if args.save_mode == "dict":
                    print("TORCH EXPORT MODEL ML")
                    torch.save(snet.state_dict(), f"ptstabilityml_dict_9comp_{n_samples}_{device}.pt")

                    torch.save(net.state_dict(), f"ptflashml_dict_9comp_{n_samples}_{device}.pt")

                if args.save_mode == "sjit":
                    print("JIT EXPORT SJIT MODEL ML")
                    jcnet = torch.jit.script(cnet)
                    jcnet.save(f"ptclassifierml_sjit_9comp_{n_samples}_{device}.pt")

                    jinet = torch.jit.script(inet)
                    jinet.save(f"ptinitializerml_sjit_9comp_{n_samples}_{device}.pt")

                    jsnet = torch.jit.script(snet)
                    jsnet.save(f"ptstabilityml_sjit_9comp_{n_samples}_{device}.pt")

                    jnet = torch.jit.script(net)
                    jnet.save(f"ptflashml_sjit_9comp_{n_samples}_{device}.pt")

                if args.save_mode == "tjit":
                    print("JIT EXPORT TJIT MODEL ML")
                    jsnet = torch.jit.trace(snet,inputs)
                    jsnet.save(f"ptstabilityml_tjit_9comp_{n_samples}_{device}.pt")
                    jnet = torch.jit.trace(net,(sinputs,ski))
                    jnet.save(f"ptflashml_tjit_9comp_{n_samples}_{device}.pt")


        if args.output == 1 :
            print("UNSTABLE",unstable)
            print("THETA",theta_v)
            print("XI",xi)
            print("YI",yi)
            print("KI",ki)


        np.save(f"data_9comp_{n_samples}_{device}.npy",design.to_numpy())

        #print("PREPARE DATA")
        #X = torch.from_numpy(design.to_numpy())[unstable]
        #y = ki.log()

        #dtype = torch.float64
        #torch.set_default_dtype(torch.float64)
        #X = X.type(torch.FloatTensor)
        #y = y.type(torch.FloatTensor)



        print("EVAL CARNOT MODEL")
        if save_mode == "dict":
            if args.use_ml == 0:
                print("LOAD CARNOT MODEL FROM DICT PT FILE")
                net = PTFlash_model1(components=components,
                                    dtype=dtype,
                                    device=device)
                net.load_state_dict(torch.load(f"ptflash_dict_9comp_{n_samples}_{device}.pt"))
                net.eval()

                times = []
                for i in range(args.n_iter):
                    inputs = torch.tensor(design.to_numpy()[data_id:data_id+args.batch_size,:], dtype=dtype, device=device)
                    start = default_timer()
                    unstable, theta_v, xi, yi, ki = net(inputs)
                    time = default_timer() - start
                    times.append(time)
                    data_id = (data_id+args.batch_size)%n_samples

                print(f"TIME FOR TORCH DICT INFERENCE FLASH :",[f"{time:.4f}" for time in times])

                if args.output == 1:
                    print("UNSTABLE", unstable)
                    print("THETA", theta_v)
                    print("XI", xi)
                    print("YI", yi)
                    print("KI", ki)


            if args.use_ml == 1:
                print("LOAD CARNOT ML MODEL FROM DICT PT FILE")

                snet = PTStability_model(components=components,
                                        dtype=dtype,
                                        device=device,
                                        initializer=initializer,
                                        classifier=classifier)
                snet.load_state_dict(torch.load(f"ptstabilityml_dict_9comp_{n_samples}_{device}.pt"))

                net.load_state_dict(torch.load(f"ptflashml_dict_9comp_{n_samples}_{device}.pt"))

                inputs = torch.tensor(design.to_numpy()[data_id:data_id + args.batch_size, :], dtype=dtype,
                                      device=device)



                ctimes = []
                if "class" in args.models:
                    for i in range(args.n_iter):
                        start = default_timer()
                        unstable = cnet(inputs)
                        ctime1 = default_timer() - start
                        ctimes.append(ctime1)

                itimes = []
                if "init" in args.models:
                    for i in range(args.n_iter):
                        start = default_timer()
                        ki = inet(inputs)
                        itime1 = default_timer() - start
                        itimes.append(itime1)

                stimes = []
                if "stab" in args.models:
                    for i in range(args.n_iter):
                        start = default_timer()
                        unstable, sinputs, ski = snet(inputs)
                        stime1 = default_timer() - start
                        stimes.append(stime1)

                times = []
                if "flash" in args.models:
                    for i in range(args.n_iter):
                        start = default_timer()
                        unstable, theta_v, xi, yi, ki = net(sinputs,ski)
                        time1 = default_timer() - start
                        times.append(time1)

                print(f"TIME FOR TORCH DICT INFERENCE CLASSIFIER  :",[f"{time:.4f}" for time in ctimes])
                print(f"                              INITIALIZER :",[f"{time:.4f}" for time in itimes])
                print(f"                              STABILITY   :",[f"{time:.4f}" for time in stimes])
                print(f"                              FLASH       :",[f"{time:.4f}" for time in times])
                '''
                print("EVAL CARNOT STABILITY MODEL")
                start = default_timer()
                snet.eval()
                sunstable, sinputs, ski = snet(inputs)
                stime1 = default_timer() - start

                if args.output == 1:
                    print("UNSTABLE", sunstable)


                start = default_timer()
                unstable, theta_v, xi, yi, ki = net(sinputs,ski)
                time1 = default_timer() - start

                if args.output == 1:
                    print("THETA", theta_v)
                    print("XI", xi)
                    print("YI", yi)
                    print("KI", ki)

                start = default_timer()
                sunstable, sinputs, ski = snet(inputs)
                stime2 = default_timer() - start


                start = default_timer()
                unstable, theta_v, xi, yi, ki = net(sinputs,ski)
                time2 = default_timer() - start


                print(f"TIME FOR PYTORCH STABILITY INFERENCE 1:{stime1:.4f}s \n")
                print(f"TIME FOR PYTORCH STABILITY INFERENCE 2:{stime2:.4f}s \n")

                print(f"TIME FOR TORCH DICT INFERENCE 1:{time1:.4f}s \n")
                print(f"TIME FOR TORCH DICT INFERENCE 2:{time2:.4f}s \n")
                '''


        if save_mode == "sjit":
            if args.use_ml == 0:
                print("LOAD CARNOT MODEL FROM SJIT PT FILE")
                net = torch.jit.load(f'ptflash_sjit_9comp_{n_samples}_{device}.pt')
                net.eval()

                for i in range(3):
                    inputs = torch.tensor(design.to_numpy()[0:1024,:], dtype=dtype, device=device)
                    unstable, theta_v, xi, yi, ki = net(inputs)

                times = []
                for i in range(args.n_iter):
                    inputs = torch.tensor(design.to_numpy()[data_id:data_id+args.batch_size,:], dtype=dtype, device=device)
                    start = default_timer()
                    unstable, theta_v, xi, yi, ki = net(inputs)
                    time = default_timer() - start
                    times.append(time)
                    #data_id = (data_id+args.batch_size)%n_samples

                print(f"TIME FOR TORCH SJIT INFERENCE FLASH :",[f"{time:.4f}" for time in times])

        if save_mode == "sjitopt":
            if args.use_ml == 0:
                print("LOAD CARNOT MODEL FROM SJIT OPT PT FILE")
                net = torch.jit.load(f'ptflash_sjitopt_9comp_{n_samples}_{device}.pt')
                net.eval()

                for i in range(3):
                    inputs = torch.tensor(design.to_numpy()[0:1024, :], dtype=dtype, device=device)
                    unstable, theta_v, xi, yi, ki = net(inputs)

                times = []
                for i in range(args.n_iter):
                    inputs = torch.tensor(design.to_numpy()[data_id:data_id + args.batch_size, :], dtype=dtype,
                                          device=device)
                    start = default_timer()
                    unstable, theta_v, xi, yi, ki = net(inputs)
                    time = default_timer() - start
                    times.append(time)
                    # data_id = (data_id+args.batch_size)%n_samples

                print(f"TIME FOR TORCH SJIT OPT INFERENCE FLASH :", [f"{time:.4f}" for time in times])


            if args.use_ml == 1:
                print("LOAD CLASSIFIER MODEL ML FROM SJIT PT FILE")
                cnet = torch.jit.load(f'ptclassifierml_sjit_9comp_{n_samples}_{device}.pt')
                cnet.eval()

                print("LOAD INITIALIZER MODEL ML FROM SJIT PT FILE")
                inet = torch.jit.load(f'ptinitializerml_sjit_9comp_{n_samples}_{device}.pt')
                inet.eval()

                print("LOAD STABILITY MODEL ML FROM SJIT PT FILE")
                snet = torch.jit.load(f'ptstabilityml_sjit_9comp_{n_samples}_{device}.pt')
                snet.eval()

                print("LOAD FLASH MODEL ML FROM SJIT PT FILE")
                net = torch.jit.load(f'ptflashml_sjit_9comp_{n_samples}_{device}.pt')
                net.eval()

                inputs = torch.tensor(design.to_numpy()[data_id:data_id + args.batch_size, :], dtype=dtype,
                                      device=device)

                ctimes = []
                if "class" in args.models:
                    for i in range(args.n_iter):
                        start = default_timer()
                        unstable = cnet(inputs)
                        ctime1 = default_timer() - start
                        ctimes.append(ctime1)

                itimes = []
                if "init" in args.models:
                    for i in range(args.n_iter):
                        start = default_timer()
                        ki = inet(inputs)
                        itime1 = default_timer() - start
                        itimes.append(itime1)

                stimes = []
                if "stab" in args.models:
                    for i in range(args.n_iter):
                        start = default_timer()
                        unstable, sinputs, ski = snet(inputs)
                        stime1 = default_timer() - start
                        stimes.append(stime1)

                times = []
                if "flash" in args.models:
                    for i in range(args.n_iter):
                        start = default_timer()
                        unstable, theta_v, xi, yi, ki = net(sinputs,ski)
                        time1 = default_timer() - start
                        times.append(time1)

                print(f"TIME FOR TORCH SJIT INFERENCE CLASSIFIER  :",[f"{time:.4f}" for time in ctimes])
                print(f"                              INITIALIZER :",[f"{time:.4f}" for time in itimes])
                print(f"                              STABILITY   :",[f"{time:.4f}" for time in stimes])
                print(f"                              FLASH       :",[f"{time:.4f}" for time in times])

        if save_mode == "tjit":
            if args.use_ml == 0:
                print("LOAD CARNOT MODEL FROM TRACE TJIT PT FILE")
                net = torch.jit.load(f'ptflash_tjit_9comp_{n_samples}_{device}.pt')
                net.eval()

                start = default_timer()
                unstable, theta_v, xi, yi, ki = net(inputs)
                time = default_timer() - start
                print(f"TIME FOR TORCH TJIT INFERENCE 1:{time:.4f}s \n")

                start = default_timer()
                #unstable, theta_v, xi, yi, ki = net(inputs)
                time = default_timer() - start
                print(f"TIME FOR TORCH TJIT INFERENCE 2:{time:.4f}s \n")

            if args.use_ml == 1:
                print("LOAD STABILITY MODEL ML FROM TJIT PT FILE")
                snet = torch.jit.load(f'ptstabilityml_tjit_9comp_{n_samples}_{device}.pt')
                snet.eval()

                print("LOAD FLASH MODEL ML FROM TJIT PT FILE")
                net = torch.jit.load(f'ptflashml_tjit_9comp_{n_samples}_{device}.pt')
                net.eval()

                start = default_timer()
                unstable, sinputs, ski = snet(inputs)
                stime1 = default_timer() - start

                start = default_timer()
                unstable, theta_v, xi, yi, ki = net(sinputs,ski)
                time1 = default_timer() - start

                start = default_timer()
                unstable, sinputs, ski = snet(inputs)
                stime2 = default_timer() - start

                start = default_timer()
                unstable, theta_v, xi, yi, ki = net(sinputs,ski)
                time2 = default_timer() - start

                print(f"TIME FOR TORCH TJIT STABILITY INFERENCE 1:{stime1:.4f}s \n")
                print(f"TIME FOR TORCH TJIT STABILITY INFERENCE 2:{stime2:.4f}s \n")

                print(f"TIME FOR TORCH TJIT FLASH INFERENCE 2:{time1:.4f}s \n")
                print(f"TIME FOR TORCH TJIT FLASH INFERENCE 2:{time2:.4f}s \n")


    if test_id == 2 :
        print("TEST CLASSIFIER 9 COMPONENTS")
        torch.set_default_dtype(torch.double)
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

        if args.export_onnx == 1:
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

        if args.export_onnx == 1 :
            torch.save(net, "classifier_x32.pt")
        else:
            torch.save(net, "classifier_x64.pt")


        batch = next(iter(test_loader))
        X = batch[0]
        pred = net(X)

        # Export the model
        if args.export_onnx == 1 :
            torch.onnx.export(net,  # model being run
                              X,  # model input (or a tuple for multiple inputs)
                              "classifier_x32.onnx",  # where to save the model (can be a file or file-like object)
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






