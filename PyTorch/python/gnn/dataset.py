#######################################################
# Python script to generate pytorch_geometric dataset
# Author: Michele Alessandro Bucci
# mail: michele-alessandro.bucci@inria.fr
######################################################

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import pickle
import os
import os.path as osp

class TrainDataSet(InMemoryDataset):
    def __init__(self, vof):
        super(TrainDataSet, self).__init__(vof.root, vof.transform, vof.pre_transform)
        self.parent = vof
        path = self.parent.processed_paths[0]
        self.data, self.slices = torch.load(path)

class TestDataSet(InMemoryDataset):
    def __init__(self, vof):
        super(TestDataSet, self).__init__(vof.root, vof.transform, vof.pre_transform)
        self.parent = vof
        path = self.parent.processed_paths[1]
        self.data, self.slices = torch.load(path)

class VOFDataSet(InMemoryDataset):
    def __init__(self, root, num_file, ring=2, test_size=0.2, transform=None, pre_transform=None):
        self.num_file = num_file
        self.ring = ring
        self.test_size = test_size
        super(VOFDataSet, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        files = []
        for i in range(self.num_file):
            files.append('graph_'+str(i)+'.pickle') 
        return files

    @property
    def processed_file_names(self):
        return ['data_train.pt', 'data_test.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw/')

    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed/')

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        for i_file in range(self.num_file):
            print('Load file :' + self.raw_dir+self.raw_file_names[i_file])
            with open(self.raw_dir+self.raw_file_names[i_file],'rb') as handle:
                out= pickle.load(handle,encoding='latin1')
            for i in range(out['on_edge']):
                alpha= out[str(i)]['input'][str(self.ring)]['alpha']
                alpha= alpha.reshape(alpha.shape[0], -1)
                volume= out[str(i)]['input'][str(self.ring)]['volume']
                volume= volume.reshape(volume.shape[0], -1)

                vertices= out[str(i)]['input'][str(self.ring)]['vertices']

                edge_index = out[str(i)]['input'][str(self.ring)]['edges']
                pseudo     = out[str(i)]['input'][str(self.ring)]['position']
                normal     = out[str(i)]['label']['normal'].reshape(1,3)
                center     = out[str(i)]['label']['center'].reshape(1,3)

                Lx=vertices[0,:,0].max() - vertices[0,:,0].min()
                Ly=vertices[0,:,1].max() - vertices[0,:,1].min()
                Lz=vertices[0,:,2].max() - vertices[0,:,2].min()
                L = max([Lx, Ly, Lz])

                vertices[:,:,0] = (vertices[:,:,0] - pseudo[0,0])/L
                vertices[:,:,1] = (vertices[:,:,1] - pseudo[0,1])/L
                vertices[:,:,2] = (vertices[:,:,2] - pseudo[0,2])/L

                center[0,0] = (center[0,0] - pseudo[0,0])/L 
                center[0,1] = (center[0,1] - pseudo[0,1])/L
                center[0,2] = (center[0,2] - pseudo[0,2])/L

                pseudo[:,0] = (pseudo[:,0] - pseudo[0,0])/L
                pseudo[:,1] = (pseudo[:,1] - pseudo[0,1])/L
                pseudo[:,2] = (pseudo[:,2] - pseudo[0,2])/L

                vertices = vertices.reshape(-1, 4*3)
                normal   = normal/np.linalg.norm(normal)
                volume   = volume/(L**3) 
                x        = np.hstack((alpha, vertices))
                label    = normal

                x          = torch.tensor(x, dtype=torch.double)
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                pseudo     = torch.tensor(pseudo, dtype=torch.double)
                label      = torch.tensor(label, dtype=torch.double)

                data_list.append(Data(x=x, y=label, edge_index=edge_index, pos=pseudo))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_train, data_val=train_test_split(data_list, test_size=self.test_size)
        torch.save(self.collate(data_train), self.processed_paths[0])
        torch.save(self.collate(data_val), self.processed_paths[1])

    def test(self):
        return TestDataSet(self)

    def train(self):
        return TrainDataSet(self)


if __name__ == "__main__":
    # An instance of your model.

    from torch_geometric.data import DataLoader
    import torch_geometric.transforms as T
    import argparse
    import json

    parser = argparse.ArgumentParser(description='GNNNet test functions')
    parser.add_argument("--test_id",   type=int, default=0,    help="Input query")
    args = parser.parse_args()
    test_id = args.test_id

    device = torch.device('cpu')
    print('Running on : ', device)

    import os

    path = os.getcwd()

    print(path)

    # load graphs and reshape it in torch format
    dataset = VOFDataSet(root='../../../DATA', num_file=3000, transform=T.Cartesian())
    test_data = dataset.test()
    print("NB SAMPLES",len(test_data))
    print("TEST ID : ",test_id)
    ##data = test_data.get(test_id)

    data_loader = DataLoader(dataset=test_data, batch_size=1)
    for i,d in enumerate(data_loader):
        if i==test_id:
            data = d
    print("DATA",data)
    graph_dict = { 'x' : data.x.detach().cpu().numpy().tolist(),
                   'edge_index' : data.edge_index.detach().cpu().numpy().tolist(),
                   'y' : data.y.detach().cpu().numpy().tolist()
                 }
    if data.edge_attr is not  None :
        graph_dict['edge_attr'] = data.edge_attr.detach().cpu().numpy().tolist()
    with open("graph"+str(test_id)+".json", 'w') as outfile:
        json.dump(graph_dict, outfile)

    #loader_test    = DataLoader(dataset=dataset.test(), batch_size=1)
    #i, data = next(enumerate(loader_test))
    '''
    print("X",data.x)
    torch.save(data.x,"X.pt")
    print("E",data.edge_index)
    torch.save(data.edge_index,"E.pt")
    print("E_Attr",data.edge_attr)
    if data.edge_attr is not  None :
        torch.save(data.edge_attr, "EAttr.pt")
    print("Y",data.y)
    torch.save(data.y,"Y.pt")
    '''