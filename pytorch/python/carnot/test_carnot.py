import pandas as pd
import numpy as np
from carnotxy.generator import Generator

from carnotxy.carnot import carnotpy as cp
from timeit import default_timer

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
    batch_size = args.batch_size
    device = args.device
    save_mode = args.save_mode
    if test_id == 0:
        print("TEST 2 COMPONENTS")
        components = [74828, 110543]
        n_components = len(components)
        conditions = {"pmin": 1.0e5, "pmax": 1.0e7, "tmin": 200, "tmax": 500}

    if test_id == 1 :
        print("TEST 9 COMPONENTS")

        # CH4, C2H6, C3H8, C4H10, C5H12, C6H14, C7H16, CO2, N2
        components = [74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379]
        n_components = len(components)
        conditions = {"pmin": 5.0e6, "pmax": 2.5e7, "tmin": 200, "tmax": 600}

        gen = Generator(
            components=components, n_samples=n_samples, op_cond=conditions, random_state=1
        )
        #print(gen.inputs.head(10))
        np.save(f"data_9comp_{n_samples}_v9.npy",gen.inputs.to_numpy())

        # mode=1, run flash calculation
        start = default_timer()
        gen.generate(mode=1, eos="SRK", n_jobs=1, verbose=10)
        time = default_timer() - start
        print(f"TIME FOR CARNOT FLASH : {time:.4f}")

        #print(gen.outputs.head(10))
        '''
        results = gen.outputs.to_numpy()
        np.save(f"results_9comp_{n_samples}_v9.npy",results)
        xi = results[:,:n_components]
        theta_l = results[:,n_components:n_components+1]
        xki = results[:,n_components+1:2*n_components+1]
        xvol = results[:,2*n_components+1:2*n_components+2]
        yi = results[:,2*n_components+2:3*n_components+2]
        theta_v = results[:,3*n_components+2:3*n_components+3]
        print(results.shape)
        #print("THETA_L",theta_l[:batch_size])
        #print("THETA_V",theta_v[:batch_size])
        #print("XI",xi[:batch_size,:])
        #print("YI",yi[:batch_size,:])
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        for i in range(args.batch_size):
            unstable = 0
            if theta_l[i] > 0 and theta_l[i] < 1 :
                unstable = 1
            print(f"RES[{i}] UNSTABLE {unstable}")
            if unstable == 1:
                print("             THETA_V : ",theta_v[i][0])
                print("             LIQ XI : ",xi[i,:])
                print("             VAP YI : ",yi[i,:])
        '''
    if test_id == 2 :
        print("TEST 9 COMPONENTS")

        # CH4, C2H6, C3H8, C4H10, C5H12, C6H14, C7H16, CO2, N2
        components = [74828, 74840, 74986, 106978, 109660, 110543, 100007, 124389, 7727379]
        n_components = len(components)
        conditions = {"pmin": 5.0e6, "pmax": 2.5e7, "tmin": 200, "tmax": 600}

        gen = Generator(
            components=components, n_samples=n_samples, op_cond=conditions, random_state=1
        )

        inputs = gen.inputs.to_numpy()
        np.save(f"data_9comp_{n_samples}_v9.npy",inputs)

        EoS = cp.eEquilibriumModelType_SoaveRedlichKwong

        bk = cp.thermoArea_create(
            EoS,
            cp.ePhaseModelType_DefaultPhaseModel,
            cp.eViscoModelType_DefaultViscoModel,
        )

        cp.bk_loadBank("carnotxy/carnot/CarnotBank.xml")
        fluid = cp.fluid_create(bk)

        for component in components:
            cp.addComponent(fluid, cp.thermoArea_addComponent(bk, component))

        start = default_timer()
        for i in range(batch_size):
            p = inputs[data_id+i,0]
            t = inputs[data_id+i,1]
            composition = inputs[data_id+i,2:]
            cp.setComposition(fluid, composition)
            cp.setPropertyAsDouble(fluid, cp.ePropertyType_Pressure, p)
            cp.setPropertyAsDouble(fluid, cp.ePropertyType_Temperature, t)
            print(i,"[P, T, Z]:",p,t,composition)

            flash = cp.computeEquilibrium_(fluid, cp.eEquilibriumType_PT)
            theta_liq = np.nan
            # Liquid phase
            try:
                liq = cp.getFluid(flash, cp.eFluidStateType_Liquid)
            except RuntimeError:
                pass
            else:
                xi = cp.getComposition(liq)
                theta_liq = cp.getNbMoles(liq)
                cp.computeProperty(liq, cp.ePropertyType_lnPhi, 0)
                cp.computeProperty(liq, cp.ePropertyType_Volume, 0)
                xki = cp.getProperty(liq, cp.ePropertyType_lnPhi, 0)
                vol_liq = cp.getPropertyAsDouble(
                    liq, cp.ePropertyType_Volume, 0
                )

            print(f"THETA LIQ[{i}]={theta_liq}")

            theta_vap = np.nan
            # Vapor phase
            try:
                vap = cp.getFluid(flash, cp.eFluidStateType_Vapour)
            except RuntimeError:
                pass
            else:
                yi = cp.getComposition(vap)
                theta_vap = cp.getNbMoles(vap)
                cp.computeProperty(vap, cp.ePropertyType_lnPhi, 0)
                cp.computeProperty(vap, cp.ePropertyType_Volume, 0)
                yki = cp.getProperty(
                    vap, cp.ePropertyType_lnPhi, 0
                )
                vol_vap = cp.getPropertyAsDouble(
                    vap, cp.ePropertyType_Volume, 0
                )

            print(f"THETA VAP[{i}]={theta_vap}")

        time = default_timer() - start
        print(f"TIME FOR CARNOT FLASH : {time:.4f}")
