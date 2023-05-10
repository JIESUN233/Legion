Legion-ATC23-Artifacts


# Legion
Legion is a system for large-scale GNN training.


## Hardware in Our Paper:
All platforms are bare-metal machines.
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | NVLinks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DGX-V100 | 96*Intel(R) Xeon(R) Platinum 8163 CPU @2.5GHZ | 2 | 1 | 384GB | PCIe 3.0x16, 4*PCIe switches, each connecting 2 GPUs | 8x16GB-V100 | NVLink Bridges, Kc = 2, Kg = 4 |
| Siton | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 1TB | PCIe 4.0x16, 2*PCIe switches, each connecting 4 GPUs | 8x40GB-VA00 | NVLink Bridges, Kc = 4, Kg = 2 |
| DGX-A100 | 128*Intel(R) Xeon(R) Platinum 8369B CPU @2.9GHZ | 2 | 1 | 1TB | PCIe 4.0x16, 4*PCIE switches, each connecting 2 GPUs | 8x80GB-A100 | NVSwitch, Kc = 1, Kg = 8 |

Kc means the number of groups in which GPUs connect each other. And Kg means the number of GPUs in each group.

## Hardware We Can Support Now:
The platforms above are currently unavailable. Alternatively, we offer a stable machine with fewer GPUs:
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | NVLinks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Siton2 | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 500GB | PCIe 4.0x16, 2*PCIe switches, each connecting 4 GPUs | 2x40GB-A100 | NVLink Bridges |

## Software: 
1. Nvidia Driver Version: 515.43.04(DGX-A100, Siton, Siton2), 470.82.01(V100)

2. CUDA 11.3(DGX-A100, Siton), CUDA 10.1(DGX-V100), CUDA 11.7(Siton2)

3. GCC/G++ 9.4.0+

4. Intel PCM

5. pytorch-cu113(DGX-A100, Siton), pytorch-cu101(DGX-V100), pytorch-cu117(Siton2)

6. dgl 0.9.1(DGX-A100, Siton, DGX-V100) dgl 1.0.1(Siton2)

7. MPI

## Dataset: 
| Datasets | PR | PA | CO | UKS | UKL | CL |
| --- | --- | --- | --- | --- | --- | --- |
| #Vertices | 2.4M | 111M | 65M | 133M | 0.79B | 1B |
| #Edges | 120M | 1.6B | 1.8B | 5.5B | 47.2B | 42.5B |
| Feature Size | 100 | 128 | 256 | 256 | 128 | 128 |
| Topology Storage | 640MB | 6.4GB | 7.2GB | 22GB | 189GB | 170GB |
| Feature Storage | 960MB | 56GB | 65GB | 136GB | 400GB | 512GB |

## How to run Experiment: Three steps.
There are several steps to train a GNN model in Legion. Before running Legion, please clone the source code:
```
$ git clone https://github.com/RC4ML/legion.git
```

### Graph Partitioning: XtraPulp
Prepare MPI in the machine and download XtraPulp
```
1. $ git clone https://github.com/luoxiaojian/xtrapulp.git
```
```
To make:

1.) Set MPICXX in Makefile to your c++ compiler, adjust CXXFLAGS if necessary
-OpenMP 3.1 support is required for parallel execution
-No other dependencies needed

2.) $ make 
-This will make xtrapulp executable and library

3.) $ make libxtrapulp
-This will just make libxtrapulp.a static library for use with xtrapulp.h
```


### Legion Compiling
```
1. $ cd legion-atc-artifacts/src/

2. $ make cuda && make main

3. $ source env.sh
```
### Running
```
1. $ cd legion-atc-artifacts/
```
Then set the meta data of specific datasets:
for example, /datasets_path/products/ 8000 2449029 123718280 100 196615 39323 2213091 400000000 100

After setting up meta data
```
2. $ ./src/legion #numberofGPUs #Kg
```
After Legion outputs "System is ready for serving", run the training backend.
```
3. $ cd pytorch-extension/
```
legion_graphsage.py, legion_gcn.py trains the GraphSAGE/GCN models, respectively.
```
4. $ python3 legion_graphsage.py
```


