import torch
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
import numpy as np
import time
import sys
import os

from vof_data import VOFDataSet, TestDataLoader, TrainDataLoader
from gnn_model import GNNNet
from training_algo import TrainingAlgo

if __name__ == "__main__":

    data_path = '/work/gratienj/BigData/dl_serialization/data/gnn'
    dataset = VOFDataSet(root=data_path, num_file=5, train=True,transform=T.Cartesian())
    #loader_train = VOFDataSet(root=data_path, num_file=100, train=True,transform=T.Cartesian())
    #loader_val   = VOFDataSet(root=data_path, num_file=100, train=False,transform=T.Cartesian())
    loader_val   = TestDataLoader(dataset)
    loader_train = TrainDataLoader(dataset)
    loader_train = DataLoader(dataset=loader_train, batch_size=2)
    loader_val   = DataLoader(dataset=loader_val, batch_size=2)
    #loader_train = DataListLoader(dataset=loader_train, batch_size=2)
    #loader_val   = DataListLoader(dataset=loader_val, batch_size=2)


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # An instance of your model.
    model = GNNNet()
    model = GNNNet().double()
    model = model.to(device)
    #model = DataParallel(model)

    # Evaluation mode
    model.eval()

    print('EVAL MODEL ON DATA')

    '''
    for i, data in enumerate(loader_train):
        print("iter:",i)
        out_n = model(data.batch,data.x,data.edge_index, data.edge_attr)
    '''

    print('START TRAINING MODEL')
    algo = TrainingAlgo(0.001,10)
    algo.train(model,loader_train,loader_val,device)

    print('SAVE SCRIPTED MODEL')
    smodel = torch.jit.script(model)
    smodel.save("script_gnnnet_model.pt")
