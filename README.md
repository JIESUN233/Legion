


# Legion-ATC23-Artifacts
Legion is a system for large-scale GNN training. Legion use GPU to accelerate graph sampling, feature extraction and GNN training. And Legion use multi-GPU memory as unified cache to minimize PCIe traffic.


## Hardware in Our Paper:
All platforms are bare-metal machines.
Table 1
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | NVLinks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DGX-V100 | 96*Intel(R) Xeon(R) Platinum 8163 CPU @2.5GHZ | 2 | 1 | 384GB | PCIe 3.0x16, 4*PCIe switches, each connecting 2 GPUs | 8x16GB-V100 | NVLink Bridges, Kc = 2, Kg = 4 |
| Siton | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 1TB | PCIe 4.0x16, 2*PCIe switches, each connecting 4 GPUs | 8x40GB-A100 | NVLink Bridges, Kc = 4, Kg = 2 |
| DGX-A100 | 128*Intel(R) Xeon(R) Platinum 8369B CPU @2.9GHZ | 2 | 1 | 1TB | PCIe 4.0x16, 4*PCIE switches, each connecting 2 GPUs | 8x80GB-A100 | NVSwitch, Kc = 1, Kg = 8 |

Kc means the number of groups in which GPUs connect each other. And Kg means the number of GPUs in each group.

## Hardware We Can Support Now:
Unfortunately, the platforms above are currently unavailable. Alternatively, we offer a stable machine with fewer GPUs:
Table 2
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | NVLinks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Siton2 | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 500GB | PCIe 4.0x16, 2*PCIe switches, each connecting 4 GPUs | 2x40GB-A100 | NVLink Bridges, Kc = 1, Kg = 2 |

We will offer the way to access Siton2 in ATC artifacts submission. 

## Software: 
1. Nvidia Driver Version: 515.43.04(DGX-A100, Siton, Siton2), 470.82.01(V100)

2. CUDA 11.3(DGX-A100, Siton), CUDA 10.1(DGX-V100), **CUDA 11.7(Siton2)**

3. GCC/G++ 9.4.0+

4. OS: Ubuntu(other linux systems are ok)

5. Intel PCM(according to OS version)
```
wget https://download.opensuse.org/repositories/home:/opcm/xUbuntu_18.04/amd64/pcm_0-0+651.1_amd64.deb
```
6. pytorch-cu113(DGX-A100, Siton), pytorch-cu101(DGX-V100), **pytorch-cu117(Siton2)**
```
$ pip3 install torch-cu1xx
```
7. dgl 0.9.1(DGX-A100, Siton, DGX-V100) **dgl 1.0.1(Siton2)**
```
$ pip3 install dgl
```
8. MPI

## Dataset: 
Table 3
| Datasets | PR | PA | CO | UKS | UKL | CL |
| --- | --- | --- | --- | --- | --- | --- |
| #Vertices | 2.4M | 111M | 65M | 133M | 0.79B | 1B |
| #Edges | 120M | 1.6B | 1.8B | 5.5B | 47.2B | 42.5B |
| Feature Size | 100 | 128 | 256 | 256 | 128 | 128 |
| Topology Storage | 640MB | 6.4GB | 7.2GB | 22GB | 189GB | 170GB |
| Feature Storage | 960MB | 56GB | 65GB | 136GB | 400GB | 512GB |
| Class Number | 47 | 2 | 2 | 2 | 2 | 2 |

We store the pre-processed datasets in path of Siton2: /legion-dataset/. We also place the partitioning result for demos in Siton2 so that you needn't wait a lot of time for partitioning.

## Build Legion from Source.
```
$ git clone https://github.com/RC4ML/legion.git
```

### Prepare Graph Partitioning Tool: XtraPulp
Prepare MPI in the machine and download XtraPulp
```
1. $ git clone https://github.com/luoxiaojian/xtrapulp.git
```

To make:

1.) Set MPICXX in Makefile to your c++ compiler, adjust CXXFLAGS if necessary
-OpenMP 3.1 support is required for parallel execution
-No other dependencies needed

Then make xtrapulp executable and library
```
2. $cd xtrapulp/ && make 
```
This will just make libxtrapulp.a static library for use with xtrapulp.h
```
3. $ make libxtrapulp
```


### Legion Compiling
```
1. $ cd legion-atc-artifacts/src/

2. $ make cuda && make main

3. $ source env.sh
```

## Using Pre-installed Legion
```
1. $ cd legion-atc-artifacts/src/

2. $ source env.sh
```

## Run Legion
There are three steps to train a GNN model in Legion:
### Step 1. Open msr by root for PCM:
```
1. $ modprobe msr
```

### Step 2. Run Legion sampling server
Running the sampling server of Legion by root. In Siton2, we support two mode: NVLink, no NVLink.
User can modify these parameters:
#### Choose dataset
    argparser.add_argument('--dataset_path', type=str, default="/home/atc-artifacts-user/datasets")
    argparser.add_argument('--dataset', type=str, default="PR")
You can change "PR" into "PA", "CO", "UKS", "UKL", "CL".
#### Set sampling hyper-parameters
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--hops_num', type=int, default=2)
    argparser.add_argument('--nbrs_num', type=list, default=[25, 10])
    argparser.add_argument('--epoch', type=int, default=10)
#### Set GPU number, GPU meory limitation and whether to use NVLinks
    argparser.add_argument('--gpu_number', type=int, default=1)
    argparser.add_argument('--cache_memory', type=int, default=200000000)
    argparser.add_argument('--usenvlink', type=bool, default=True)
```
2. $ cd legion-atc-artifacts/ && python3 legion_server.py
```

### Step 3. Run Legion training backend
After Legion outputs "System is ready for serving", run the training backend by artifact-user.
"legion_graphsage.py" and "legion_gcn.py" trains the GraphSAGE/GCN models, respectively.
User can modify these parameters:
### Set dataset statistics
For specific numbers, please refer to Table 3(dataset).
```
    argparser.add_argument('--class_num', type=int, default=47)
    argparser.add_argument('--features_num', type=int, default=100)
```
### Set GNN hyper-parameters
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--hidden_dim', type=int, default=256)
    argparser.add_argument('--hops_num', type=int, default=2)
    argparser.add_argument('--nbrs_num', type=list, default=[25, 10])
    argparser.add_argument('--drop_rate', type=float, default=0.5)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    argparser.add_argument('--epoch', type=int, default=10)
Note that the train_batch_size, hops_num, nbrs_num, epoch should be the same as sampling hyper-parameters
```
3. $ cd pytorch-extension/ && python3 legion_graphsage.py
```





