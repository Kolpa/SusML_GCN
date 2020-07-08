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


def _call_method(method, rref, *args):
    return method(rref.local_value(), *args)


def _remote_method(method, rref, args=[]):
    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args=args)


def _parameter_rrefs(module):
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(RRef(param))
    return param_rrefs


path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
dataset = Reddit(path)
data = dataset[0]

train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=[25, 10], batch_size=1024, shuffle=True,
                               num_workers=12)

subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=1024, shuffle=False,
                                  num_workers=12)


class SAGE(torch.nn.Module):
    def __init__(self, worker_name, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()

        self.local_layer = SAGEConv(in_channels, hidden_channels)
        self.remote_layer_rref = rpc.remote(
            worker_name, SAGEConv, args=(hidden_channels, out_channels))

        print("SAGE init done")

    def forward(self, x, adjs):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.

        x_target = x[:adjs[0][2][1]]
        x = self.local_layer((x, x_target), adjs[0][0])

        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x_target = x[:adjs[1][2][1]]
        x = _remote_method(
            SAGEConv.__call__,
            self.remote_layer_rref,
            args=((x, x_target), adjs[1][0])
        )

        return x.log_softmax(dim=-1)

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * 2)
        pbar.set_description('Evaluating')

        xs = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj
            x = x_all[n_id]
            x_target = x[:size[1]]
            x = self.local_layer((x, x_target), edge_index)
            x = F.relu(x)
            xs.append(x)

            pbar.update(batch_size)

        x_all = torch.cat(xs, dim=0)

        xs = []
        for batch_size, n_id, adj in subgraph_loader:
            edge_index, _, size = adj
            x = x_all[n_id]
            x_target = x[:size[1]]

            x = _remote_method(
                SAGEConv.__call__,
                self.remote_layer_rref,
                args=((x, x_target), edge_index)
            )

            xs.append(x)

            pbar.update(batch_size)

        x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(_parameter_rrefs(self.local_layer))
        remote_params.extend(_remote_method(
            _parameter_rrefs, self.remote_layer_rref))
        return remote_params


x = data.x
y = data.y.squeeze()


def train(model, optimizer, epoch):
    model.train()

    pbar = tqdm(total=int(data.train_mask.sum()))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        with dist_autograd.context() as context_id:
            adjs = [adj for adj in adjs]

            out = model(x[n_id], adjs)
            loss = F.nll_loss(out, y[n_id[:batch_size]])
            dist_autograd.backward(context_id, [loss])
            optimizer.step(context_id)

            total_loss += float(loss)
            total_correct += int(out.argmax(dim=-
                                            1).eq(y[n_id[:batch_size]]).sum())
            pbar.update(batch_size)

    pbar.close()

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / int(data.train_mask.sum())

    return loss, approx_acc


@torch.no_grad()
def test(model):
    model.eval()

    out = model.inference(x)

    y_true = y.unsqueeze(-1)
    y_pred = out.argmax(dim=-1, keepdim=True)

    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        results += [int(y_pred[mask].eq(y_true[mask]).sum()) / int(mask.sum())]

    return results


def _run_trainer():
    print("_run_trainer()")
    model = SAGE('ps', dataset.num_features, 256, dataset.num_classes)

    optimizer = DistributedOptimizer(
        torch.optim.Adam, model.parameter_rrefs(), lr=0.01)

    for epoch in range(1, 11):
        loss, acc = train(model, optimizer, epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test(model)
        print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')


def run_worker(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    if rank == 0:
        rpc.init_rpc("trainer", rank=rank, world_size=world_size)
        _run_trainer()
    else:
        rpc.init_rpc("ps", rank=rank, world_size=world_size)
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('rank', type=int, default=0)
    args = parser.parse_args()
    run_worker(args.rank, 2)
