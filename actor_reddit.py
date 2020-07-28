import ray

import os.path as osp
from typing import Union
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch_geometric.typing import OptPairTensor, Adj

ray.init()


@ray.remote
class SAGEConvActor(object):
    def __init__(self, in_channels, out_channels):
        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj):
        return self.conv(x, edge_index)


class SAGE(Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.first_layer = SAGEConvActor.remote(in_channels, hidden_channels)
        self.second_layer = SAGEConvActor.remote(hidden_channels, out_channels)

    def forward(self, x, adjs):
        x_target = x[:adjs[0][2][1]]
        x_ref = self.first_layer.forward.remote((x, x_target), adjs[0][0])
        x = ray.get(x_ref)

        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x_target = x[:adjs[1][2][1]]
        x_ref = self.second_layer.forward.remote((x, x_target), adjs[1][0])
        x = ray.get(x_ref)

        return x.log_softmax(dim=-1)


def train(model, optimizer, epoch, data, train_loader):
    model.train()

    x = data.x
    y = data.y.squeeze()

    total_loss = total_correct = 0

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    for batch_size, n_id, adjs in train_loader:
        # optimizer.zero_grad()
        out = model(x[n_id], adjs)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        # optimizer.step()

        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())
        pbar.update(batch_size)

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    pbar.close()

    return loss, approx_acc


def _run_trainer():
    path = osp.join(osp.dirname(osp.realpath(__file__)),
                    '..', 'data', 'Reddit')
    print("Load Dataset")
    dataset = Reddit(path)
    data = dataset[0]
    print("Load Train Sampler")
    train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                   sizes=[25, 10],
                                   batch_size=1024, shuffle=True,
                                   num_workers=12)

    print("Creating SAGE model")
    model = SAGE(dataset.num_features, 128, dataset.num_classes)

    #optimizer = Adam(model.parameters(), lr=0.01)

    print("Start training")
    for epoch in range(1, 11):
        loss, acc = train(model, None, epoch, data, train_loader)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')


_run_trainer()
