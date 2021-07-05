import torch
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
import numpy as np
import time
import sys
import os

from dataset import VOFDataSet, TestDataLoader, TrainDataLoader
from gnn import GNNNet
from training_algo import TrainingAlgo

if __name__ == "__main__":

    data_path = '/work/gratienj/BigData/dl_serialization/data/gnn'
    dataset = VOFDataSet(root=data_path, num_file=5, train=True,transform=T.Cartesian())
    #loader_train = DataLoader(dataset=dataset.train(), batch_size=64)
    #loader_val   = DataLoader(dataset=dataset.test(), batch_size=64)
    loader_train = DataListLoader(dataset=dataset.train(), batch_size=2)
    loader_val   = DataListLoader(dataset=dataset.test(), batch_size=2)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    # An instance of your model.
    model = GNNNet()
    model = GNNNet().double()
    model = DataParallel(model)
    model = model.to(device)


    print('START TRAINING MODEL')
    algo = TrainingAlgo(0.001,500)
    algo.train(model,loader_train,loader_val,device)

    print('SAVE SCRIPTED MODEL')
    smodel = torch.jit.script(model)
    smodel.save("script_gnnnet_model.pt")
