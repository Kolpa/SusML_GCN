# Graph Convolutional Networks on Edge Device Clusters

## Getting Started

On your Raspberry Pi:

1. Install [poetry package manager](https://python-poetry.org/docs/#installation)
2. Run `poetry install` in the project root, to install dependencies
3. Using `poetry shell` will activate the venv
4. Run script using `python SCRIPT`

WARNING: Some dependencies might need shared libraries such as libssl

This has only been tested on linux

## Summary

We tried multiple versions of a pytorch geometric GCN impementation using the example [reddit dataset](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/reddit.py). Following is a short description of those versions.

### reddit.py

The default reddit dataset GCN implementation

### ddp_reddit.py

Using DistributedDataParallel from torch.nn.parallel this version is an implementation of data parallel model training. It is suitable for throughput/performance gain on multiple machines and less suitable for low spec hardware with little memory, since the whole model has to be hold by every node

### mpi_reddit.py
Using OpenMPI and Horovod this version distributes the model using horovod, which is a bit more memory efficient. Since the NeighbourSampler have to be hold by every node, this solution was still not efficient enough, because of the big sampler size in memory.

### actor_reddit.py
Ray is an actor framework built on Redis so nodes can communicate via message passing. We were not able to get backwards propagation running, because that has to be deeply integrated into pytorch. While calculating the forward propagation, the tensor graph has to be built and remembered. This is not trivial using actors and therefore was no solution suitable for us.

### gloo_rpc_reddit.py
The solution we chose at the end, as it gave us the best results regarding memory efficiency. Every layer and NeigboughrSampling could be run on different machines and therefore the pis did not run out of memory. Only problem is, that this solution requires the Gloo framework for communication, which is only compatible with 64bit systems. Fir that we installed 64bit ubuntu on our pi cluster and got it running.