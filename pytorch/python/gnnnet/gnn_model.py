import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Union, Tuple, List
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size

from torch import Tensor, LongTensor
from torch_sparse.tensor import SparseTensor

from torch_geometric.nn import SplineConv, TopKPooling, global_mean_pool, DataParallel, BatchNorm
from torch_geometric.data import Data


class GNNNet(nn.Module):

    propagate_type = { 'batch': LongTensor,
                       'x': Tensor,
                       'edge_index': SparseTensor,
                       'pseudo' : OptTensor}

    def __init__(self):
        super(GNNNet, self).__init__()
        self._degree = 1
        self.conv1 = SplineConv(13, 64, 3, 5, degree=self._degree).jittable()
        self.BN = BatchNorm(64)
        self.N_dropout = nn.Dropout(p=0.2)
        self.N_out = nn.Linear(in_features=64, out_features=3)


    __constants__ = ['_degree']

    @torch.jit.ignore
    def forward(self, data : Data ):
        batch,x,edge_index,pseudo = data.batch, data.x, data.edge_index, data.edge_attr
        print("BATCH",batch)
        return self.propagate(batch,x,edge_index,pseudo)

    #def forward(self, batch: LongTensor, x: Tensor, edge_index: Union[Tensor, SparseTensor], pseudo: Optional[Tensor] ) -> Tuple[Tensor,Tensor] :
    #def forward2(self, batch: Tensor, x: Tensor, edge_index: SparseTensor, pseudo: Tensor) -> Tuple[Tensor, Tensor]:
    def forward(self, batch: Tensor, x: Tensor, edge_index: Tensor, pseudo: Tensor) -> Tensor:
        return self.propagate(batch,x,edge_index,pseudo)

    def propagate(self, batch: Tensor, x: Tensor, edge_index: Tensor, pseudo: Tensor) -> Tensor:
        x = F.elu(self.conv1(x, edge_index, pseudo, None))
        x = global_mean_pool(x, batch)
        x = self.BN(x)
        ##### for normal estiation
        n = self.N_dropout(x)
        n = F.normalize(self.N_out(n.reshape(-1, 64)))
        return n

if __name__ == "__main__":

    # An instance of your model.
    model = GNNNet()

    # Evaluation mode
    model.eval()
    smodel = torch.jit.script(model)
    smodel.save("script_gnnnet_model.pt")