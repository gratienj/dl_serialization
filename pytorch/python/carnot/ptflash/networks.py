import torch
from torch import nn
import pytorch_lightning as pl

from typing import List


class ResidualBlock(nn.Module):
    """Residual block used by PTNet."""

    def __init__(self, dim, activation=nn.SiLU):
        super(ResidualBlock, self).__init__()
        self.act = activation()
        self.l1 = nn.Linear(dim, dim)
        self.l2 = nn.Linear(dim, dim)

    def forward(self, x):
        identity = x
        out = self.act(self.l1(x))
        out = self.l2(out)
        out += identity
        out = self.act(out)
        return out


class PTNet(nn.Module):
    """
    PTNet can be either an initializer to intialize the distribution coefficients or
        a classifier to predict the phases at equilibrium.

    Args:
        input_dim (int):
            The input dimension

        output_dim (int):
            The output dimension

        mean (torch.Tensor):
            The mean which is subtracted from the input.

        scale (torch.Tensor):
            The scale by which the input is divided to have unit variance

        units (List[int]):
            The number of neurons for each hidden layer

        activation (str, optional): default="SiLU"
            See https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
            for the namespalce of PyTorch's activation functions.

        concat (bool, optional): default=False
            If true, concatenate the input to the output of the last hidden layer.

        residual (bool, optional): default=False
            If true, use residual blocks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mean: torch.Tensor,
        scale: torch.Tensor,
        units: List[int],
        activation: str = "SiLU",
        concat: bool = False,
        residual: bool = False,
    ):
        super(PTNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)
        self.units = units
        self.act = getattr(nn, activation)
        self.concat = concat
        self.residual = residual
        self.build()

    def build(self):
        """Create sub-networks"""

        layers = []
        if self.residual:
            assert (len(self.units) % 2) == 1, "If resudual=True, layers must be odd."
            dim = self.units[0]
            layers.append(nn.Linear(self.input_dim, dim))
            layers.append(self.act())
            for _ in range((len(self.units) - 1) // 2):
                layers.append(ResidualBlock(dim, self.act))
        else:
            in_dim = self.input_dim
            for out_dim in self.units:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(self.act())
                in_dim = out_dim
            # initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                    nn.init.zeros_(m.bias)
        # chain all layers
        self.trunk = nn.Sequential(*layers)
        # output layer
        if self.concat:
            out_layer = nn.Linear(self.units[-1] + self.input_dim, self.output_dim)
        else:
            out_layer = nn.Linear(self.units[-1], self.output_dim)
        nn.init.xavier_uniform_(out_layer.weight)
        nn.init.zeros_(out_layer.bias)
        self.out_layer = out_layer

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.scale
        trunk = self.trunk(x)
        if self.concat:
            trunk = torch.cat((trunk, x), dim=1)
        return self.out_layer(trunk)


class TrainWrapper(pl.LightningModule):
    """A lightning module wrapping the training logic.

    Args:
        net:
            A network to be trained

        loss_func:
            Loss function

        metrics: torchmetrics.MetricCollection
            Metrics to monitor the network's performance during training, see the following link:
                https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metriccollection

        lr (float):
            The learning rate of Adam

        sche_params (dict):
            The parameters of CyclicLR scheduler, see the following link:
                https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html
    """

    def __init__(
        self,
        net,
        loss_func,
        metrics=None,
        lr=0.001,
        sche_args=None,
    ):
        super(TrainWrapper, self).__init__()
        self.net = net
        self.loss_func = loss_func
        self.metrics = metrics
        self.save_hyperparameters(ignore=["net", "loss_func", "metrics"])
        if metrics is not None:
            self.train_metrics = metrics.clone(prefix="train_")
            self.valid_metrics = metrics.clone(prefix="val_")
            self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_func(out, y)
        if self.metrics is not None:
            self.log_dict(self.train_metrics(out, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_func(out, y)
        self.log("valid_loss", loss, prog_bar=True)
        if self.metrics is not None:
            self.log_dict(self.valid_metrics(out, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = self.loss_func(out, y)
        self.log("test_loss", loss)
        if self.metrics is not None:
            self.log_dict(self.test_metrics(out, y))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        if self.hparams.sche_args:
            sche = dict(
                scheduler=torch.optim.lr_scheduler.CyclicLR(
                    opt, **self.hparams.sche_args
                ),
                interval="step",
                frequency=1,
            )
            return [opt], [sche]
        else:
            return opt
