
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear, Sequential, Dropout, ReLU
from torch_geometric.nn import SplineConv, TopKPooling, global_mean_pool, DataParallel, BatchNorm, ChebConv, SAGEConv

from typing import Optional, Union, Tuple, List
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch import Tensor, LongTensor
from torch_sparse.tensor import SparseTensor
from torch_geometric.data import Data

import os
import argparse

torch.set_default_tensor_type('torch.DoubleTensor')


class GNNNet(nn.Module):
    def __init__(self):
        super(GNNNet, self).__init__()
        self.conv1 = SAGEConv(13, 128).jittable()
        self.conv2 = SAGEConv(128, 256).jittable()

        self.conv3 = SAGEConv(256, 256).jittable()
        self.conv4 = SAGEConv(256, 256).jittable()

        # self.Pool1 = TopKPooling(in_channels=128, ratio=0.8)

        self.conv5 = SAGEConv(256, 256).jittable()
        self.conv6 = SAGEConv(256, 256).jittable()

        # self.Pool2 = TopKPooling(in_channels=128, ratio=0.8)

        self.dropout1 = Dropout(p=0.5)
        self.BN1 = BatchNorm(256)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.BN2 = BatchNorm(128)
        self.fc3 = nn.Linear(128, 64)

        self.dropout_fc1 = Dropout(p=0.5)

        self.out = nn.Linear(in_features=64, out_features=3)


    #def forward(self, data ):
    #    batch,x,edge_index,pseudo = data.batch, data.x, data.edge_index, data.edge_attr
    #    return self.propagate(batch,x,edge_index,pseudo)

    #def forward(self, batch: Tensor, x: Tensor, edge_index: Tensor) -> Tensor:
    #    return self.propagate(batch,x,edge_index)

    def forward(self, batch: Tensor, x: Tensor, edge_index: Tensor, pseudo: Tensor) -> Tensor:
        return self.propagate(batch,x,edge_index,pseudo)


    #def propagate(self, batch: Tensor, x: Tensor, edge_index: Tensor) -> Tensor:
    def propagate(self, batch: Tensor, x: Tensor, edge_index: Tensor, pseudo: Tensor) -> Tensor:
        x = F.elu(self.conv1(x, edge_index, ))
        x = F.elu(self.conv2(x, edge_index, ))
        x = x + F.elu(self.conv3(x, edge_index, ))
        x = x + F.elu(self.conv4(x, edge_index, ))
        # x, edge_index, pseudo, batch, __, __ = self.Pool1(x, edge_index, pseudo, batch)

        x = x + F.elu(self.conv5(x, edge_index, ))
        x = x + F.elu(self.conv6(x, edge_index, ))
        # x, edge_index, pseudo, batch, __, __ = self.Pool2(x, edge_index, pseudo, batch)

        # x = self.BN1(x)
        # x = self.dropout1(x)

        x = x + F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.dropout_fc1(x)
        x = global_mean_pool(x, batch)
        x = self.BN2(x)
        x = F.elu(self.fc3(x))
        x = F.normalize(self.out(x))

        return x

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


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
    parser.add_argument("--test_id",   type=int, default=0,              help="Test query")
    parser.add_argument("--data_id",   type=int, default=0,              help="Data query")
    parser.add_argument("--dir_path",  type=str, default="./",           help="dir path")
    parser.add_argument("--model_name",type=str, default="gnn1_model",    help="model name")
    args = parser.parse_args()
    test_id = args.test_id
    data_id = args.data_id

    model = GNNNet().double()
    if test_id == 0:
        # Evaluation mode
        model.eval()
        smodel = torch.jit.script(model)
        script_model_name = "script_{}.pt".format(args.model_name)
        save_path=os.path.join(args.dir_path, script_model_name)
        smodel.save(save_path)

    if test_id == 1:
        # Load model
        load_model_name = "{}.pt".format(args.model_name)
        load_path       = os.path.join(args.dir_path, load_model_name)
        print("Load Model : ",load_path)
        model.load(load_path)

        print("Save script Model")
        model.eval()
        smodel = torch.jit.script(model)
        script_model_name = "script_{}.pt".format(args.model_name)
        save_path=os.path.join(args.dir_path, script_model_name)
        smodel.save(save_path)


    if test_id == 2:
        # Load model
        device = torch.device('cpu')
        model.to(device)
        model.eval()

        # load graphs and reshape it in torch format
        dataset = VOFDataSet(root='../../../DATA', num_file=3000, transform=T.Cartesian())
        test_data = dataset.test()
        print("NB SAMPLES",len(test_data))
        print("DATA ID : ",data_id)
        #data = test_data.get(data_id)
        #print('DATA ',data)

        data_loader = DataLoader(dataset=test_data, batch_size=1)
        for i,d in enumerate(data_loader):
            if i==data_id:
                data = d
        print("DATA:",data)
        #out = model(x=data.x,edge_index=data.edge_index )
        #out = model(batch=data.batch,x=data.x,edge_index=data.edge_index )
        out = model(batch=data.batch,x=data.x,edge_index=data.edge_index,pseudo=data.edge_attr )
        print('OUT',out)

        print("Save script Model")
        smodel = torch.jit.script(model)
        script_model_name = "script_{}.pt".format(args.model_name)
        save_path=os.path.join(args.dir_path, script_model_name)
        smodel.save(save_path)


    if test_id == 3:
        # Load model
        device = torch.device('cpu')
        load_model_name = "{}.pt".format(args.model_name)
        load_path       = os.path.join(args.dir_path, load_model_name)
        print("Load Model : ",load_path)
        model.load(load_path)
        model.to(device)
        model.eval()

        # load graphs and reshape it in torch format
        dataset = VOFDataSet(root='../../../DATA', num_file=3000, transform=T.Cartesian())
        test_data = dataset.test()
        print("NB SAMPLES",len(test_data))
        print("DATA ID : ",data_id)

        data_loader = DataLoader(dataset=test_data, batch_size=1)
        for i,d in enumerate(data_loader):
            if i==data_id:
                data = d
        print("DATA:",data)
        #print("BATCH:",data.batch)
        out = model(batch=data.batch,x=data.x,edge_index=data.edge_index,pseudo=data.edge_attr)
        print('OUT',out)

        print("Save script Model")
        smodel = torch.jit.script(model)
        script_model_name = "script_{}.pt".format(args.model_name)
        save_path=os.path.join(args.dir_path, script_model_name)
        smodel.save(save_path)