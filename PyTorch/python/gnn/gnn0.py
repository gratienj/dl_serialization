

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, Dropout, ReLU
from torch_geometric.nn import SplineConv, TopKPooling, global_mean_pool, DataParallel, BatchNorm, ChebConv, SAGEConv
from torch_geometric.nn import GCNConv

from typing import Optional, Union, Tuple, List
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor, LongTensor
from torch_sparse.tensor import SparseTensor
from torch_geometric.data import Data

torch.set_default_tensor_type('torch.DoubleTensor')

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels, 64).jittable()
        self.conv2 = GCNConv(64, out_channels).jittable()

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


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
    parser.add_argument("--dir_path",  type=str, default="./",         help="dir path")
    parser.add_argument("--model_name",type=str, default="gnn0_model",  help="model name")
    args = parser.parse_args()
    test_id = args.test_id
    data_id = args.data_id

    device = torch.device('cpu')
    model = Net(13,3).double()
    if test_id == 0:
        # Evaluation mode
        model.to(device)
        model.eval()
        smodel = torch.jit.script(model)
        script_model_name = "script_{}.pt".format(args.model_name)
        save_path=os.path.join(args.dir_path, script_model_name)
        smodel.save(save_path)

    if test_id == 1:
        # Load model
        model.to(device)
        model.eval()

        # load graphs and reshape it in torch format
        dataset = VOFDataSet(root='../../../DATA', num_file=3000, transform=T.Cartesian())
        test_data = dataset.test()
        print("NB SAMPLES",len(test_data))
        #data = test_data.get(data_id)

        data_loader = DataLoader(dataset=test_data, batch_size=1)
        i,data = next(enumerate(data_loader))
        print('DATA ',data)

        out = model(x=data.x,edge_index=data.edge_index)
        print('OUT',out)

        smodel = torch.jit.script(model)
        script_model_name = "script_{}.pt".format(args.model_name)
        save_path=os.path.join(args.dir_path, script_model_name)
        smodel.save(save_path)

    if test_id == 2:
        # Load model
        load_model_name = "{}.pt".format(args.model_name)
        load_path = os.path.join(args.dir_path, load_model_name)
        print("Load Model : ", load_path)
        model.load(load_path)
        model.to(device)
        model.eval()
