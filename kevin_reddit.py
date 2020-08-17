import os.path as osp
import os
import torch
import torch.nn.functional as F
import torch.distributed.rpc as rpc
import argparse
from tqdm import tqdm
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv
from torch.distributed.rpc import rpc_sync, RRef
from torch.distributed.optim.optimizer import DistributedOptimizer
import torch.distributed.autograd as dist_autograd


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.first_layer_rref = rpc.remote(
            "actor_1", SAGEConv, args=(in_channels, hidden_channels))
        self.second_layer_rref = rpc.remote(
            "actor_2", SAGEConv, args=(hidden_channels, out_channels))

    def forward(self, x, adjs):
        x_target = x[:adjs[0][2][1]]
        x = _remote_method(SAGEConv.__call__,
                           self.first_layer_rref, (x, x_target), adjs[0][0])

        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x_target = x[:adjs[1][2][1]]
        x = _remote_method(SAGEConv.__call__,
                           self.second_layer_rref, (x, x_target), adjs[1][0])

        return x.log_softmax(dim=-1)

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(_remote_method(
            _parameter_rrefs, self.first_layer_rref))
        remote_params.extend(_remote_method(
            _parameter_rrefs, self.second_layer_rref))
        return remote_params


def train(model, optimizer, epoch, data, train_loader):
    model.train()

    x = data.x
    y = data.y.squeeze()

    total_loss = total_correct = 0

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    for batch_size, n_id, adjs in train_loader:
        with dist_autograd.context() as context_id:
            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            dist_autograd.backward(context_id, [loss])
            optimizer.step(context_id)

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-
                                            1).eq(y[n_id[:batch_size]]).sum())
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
                                   num_workers=0)

    print("Creating SAGE model")
    model = SAGE(dataset.num_features, 256, dataset.num_classes)

    optimizer = DistributedOptimizer(
        torch.optim.Adam, model.parameter_rrefs(), lr=0.01)

    print("Start training")
    for epoch in range(1, 11):
        loss, acc = train(model, optimizer, epoch, data, train_loader)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')


def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    if rank == 0:
        print("Starting Trainer")
        print("Waiting...")
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        _run_trainer()
    else:
        print("Starting Worker")
        print("Waiting...")
        rpc.init_rpc(f"actor_{rank}", rank=rank, world_size=world_size)

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rank', type=int, default=0)
    args = parser.parse_args()
    run_worker(args.rank, 3)
