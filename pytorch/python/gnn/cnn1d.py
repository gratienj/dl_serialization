

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, Dropout, ReLU, Conv1d

from typing import Optional, Union, Tuple, List
from torch import Tensor, LongTensor

torch.set_default_tensor_type('torch.DoubleTensor')

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = Conv1d(in_channels,out_channels,3)

    def forward(self, x):
        return self.conv1(x)


    def save(self, dir_name, model_name):
        import os
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        model_name = "{}.pt".format(model_name)
        save_path = os.path.join(dir_name, model_name)
        path = open(save_path, mode="wb")

        state = {'state_dict': self.state_dict() }
        torch.save(state, path)
        path.close()

    def load(self, path):
       # load check point
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['state_dict'])
        return self

if __name__ == "__main__":
    # An instance of your model.
    from dataset import VOFDataSet
    from torch_geometric.data import DataLoader
    import torch_geometric.transforms as T
    import os
    import argparse

    parser = argparse.ArgumentParser(description='GNNNet test functions')
    parser.add_argument("--test_id",   type=int, default=0,            help="InputTest query")
    parser.add_argument("--data_id",   type=int, default=0,            help="InputData query")
    parser.add_argument("--data_dir_path",  type=str, default="./data",         help="dir path")
    parser.add_argument("--dir_path",  type=str, default="./",         help="dir path")
    parser.add_argument("--model_name",type=str, default="cnn1d_model", help="model name")
    parser.add_argument("--device",    type=str, default="cpu",        help="Device query")
    args = parser.parse_args()
    test_id = args.test_id
    data_id = args.data_id

    device = torch.device(args.device)
    model = Net(1,3).double()
    if test_id == 0:
        # Evaluation mode
        model.to(device)
        model.eval()
        smodel = torch.jit.script(model)
        script_model_name = "script_{}_{}.pt".format(args.model_name,args.device)
        save_path=os.path.join(args.dir_path, script_model_name)
        smodel.save(save_path)

    if test_id == 1:
        # Load model

        input = torch.randn(1, 1, 3)
        input = input.to(torch.device(args.device))

        model = model.to(torch.device(args.device))
        model.eval()

        out = model(x=input)
        print('OUT',out)

        smodel = torch.jit.script(model)
        script_model_name = "script_{}_{}.pt".format(args.model_name,args.device)
        save_path=os.path.join(args.dir_path, script_model_name)
        smodel.save(save_path)

