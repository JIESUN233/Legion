# The latest repo is at https://github.com/RC4ML/Legion


# Legion-ATC23-Artifacts
Legion is a system for large-scale GNN training. Legion uses GPU to accelerate graph sampling, feature extraction and GNN training. And Legion utilizes multi-GPU memory as unified cache to minimize PCIe traffic. In this repo, we provide Legion's prototype and show how to run Legion. We provide two ways to build Legion: 1. building from source, 2. using pre-installed Legion. For artifacts evaluation, we recommend using the pre-installed Legion. Due to the machine limitation, we only show the functionality of Legion in the pre-installed environment.


## 1. Hardware 
### Hardware Used in Our Paper
All platforms are bare-metal machines.
Table 1
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | NVLinks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DGX-V100 | 96*Intel(R) Xeon(R) Platinum 8163 CPU @2.5GHZ | 2 | 1 | 384GB | PCIe 3.0x16, 4*PCIe switches, each connecting 2 GPUs | 8x16GB-V100 | NVLink Bridges, Kc = 2, Kg = 4 |
| Siton | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 1TB | PCIe 4.0x16, 2*PCIe switches, each connecting 4 GPUs | 8x40GB-A100 | NVLink Bridges, Kc = 4, Kg = 2 |
| DGX-A100 | 128*Intel(R) Xeon(R) Platinum 8369B CPU @2.9GHZ | 2 | 1 | 1TB | PCIe 4.0x16, 4*PCIE switches, each connecting 2 GPUs | 8x80GB-A100 | NVSwitch, Kc = 1, Kg = 8 |

Kc means the number of groups in which GPUs connect each other. And Kg means the number of GPUs in each group.

### Hardware We Can Support Now
Unfortunately, the platforms above are currently unavailable. Alternatively, we provide a stable machine with two GPUs:
Table 2
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | NVLinks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Siton2 | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 500GB | PCIe 4.0x16, 2*PCIe switches, one connecting 2 GPUs | 2x80GB-A100 | NVLink Bridges, Kc = 1, Kg = 2 |

We will provide the way to access Siton2 in ATC artifacts submission. 

## 2. Software 
Legion's software is light-weighted and portable. Here we list some tested environment.

1. Nvidia Driver Version: 515.43.04(DGX-A100, Siton, Siton2), 470.82.01(V100)

2. CUDA 11.3(DGX-A100, Siton), CUDA 10.1(DGX-V100), **CUDA 11.7(Siton2)**

3. GCC/G++ 9.4.0+(DGX-A100, Siton, DGX-V100), GCC/G++ 7.5.0+(Siton2)

4. OS: Ubuntu(other linux systems are ok)

5. Intel PCM(according to OS version)
```
$ wget https://download.opensuse.org/repositories/home:/opcm/xUbuntu_18.04/amd64/pcm_0-0+651.1_amd64.deb
```
6. pytorch-cu113(DGX-A100, Siton), pytorch-cu101(DGX-V100), **pytorch-cu117(Siton2)**, torchmetrics
```
$ pip3 install torch-cu1xx
```
7. dgl 0.9.1(DGX-A100, Siton, DGX-V100) **dgl 1.1.0(Siton2)**
```
$ pip3 install  dgl -f https://data.dgl.ai/wheels/cu1xx/repo.html
```
8. MPI

## 3. Datasets
Table 3
| Datasets | PR | PA | CO | UKS | UKL | CL |
| --- | --- | --- | --- | --- | --- | --- |
| #Vertices | 2.4M | 111M | 65M | 133M | 0.79B | 1B |
| #Edges | 120M | 1.6B | 1.8B | 5.5B | 47.2B | 42.5B |
| Feature Size | 100 | 128 | 256 | 256 | 128 | 128 |
| Topology Storage | 640MB | 6.4GB | 7.2GB | 22GB | 189GB | 170GB |
| Feature Storage | 960MB | 56GB | 65GB | 136GB | 400GB | 512GB |
| Class Number | 47 | 2 | 2 | 2 | 2 | 2 |

We store the pre-processed datasets in path of Siton2: /home/atc-artifacts-user/datasets. We also place the partitioning result for demos in Siton2 so that you needn't wait a lot of time for partitioning.

## 4. Use Pre-installed Legion
There are four steps to train a GNN model in Legion. In these steps, you need to change into root user of Siton2.
### Step 1. Add environment variables temporarily
```
1. $ cd /home/atc-artifacts-user/legion-atc-artifacts/src/ && source env.sh
```
### Step 2. Open msr by root for PCM
```
2. $ modprobe msr
```
After these two steps, you need prepare two sessions to run Legion's sampling server and training backend separately.
### Step 3. Run Legion sampling server
In Siton2, we can test Legion in two mode: NVLink, no NVLink.
User can modify these parameters:
#### Choose dataset
    argparser.add_argument('--dataset_path', type=str, default="/home/atc-artifacts-user/datasets")
    argparser.add_argument('--dataset', type=str, default="PR")
You can change "PR" into "PA", "CO", "UKS", "UKL", "CL".
#### Set sampling hyper-parameters
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--epoch', type=int, default=10)
#### Set GPU number, GPU meory limitation and whether to use NVLinks
    argparser.add_argument('--gpu_number', type=int, default=1)
    argparser.add_argument('--cache_memory', type=int, default=200000000) ## default is 200000000 Bytes
    argparser.add_argument('--usenvlink', type=int, default=1)## 1 means true, 0 means false.
#### Start server
```
3. $ cd /home/atc-artifacts-user/legion-atc-artifacts/ && python3 legion_server.py
```
#### Sampling server functionality
This figure shows that PCM is working.

![7164f5c512559008fda789051ee3846](https://github.com/JIESUN233/Legion/assets/109936863/a5c6dd95-02fb-48b5-9c53-7af8f2734346)

This figure shows the system outputs including dataset statistics, training statistics and cache management outputs.

![fe485222ae227d406bab1068eb1bed9](https://github.com/JIESUN233/Legion/assets/109936863/0c8476a8-81b7-4bbc-98a5-e926e9a80931)


### Step 4. Run Legion training backend
**After Legion outputs "System is ready for serving",** run the training backend by artifact-user.
"legion_graphsage.py" and "legion_gcn.py" trains the GraphSAGE/GCN models, respectively.
User can modify these parameters:
#### Set dataset statistics
For specific numbers, please refer to Table 3(dataset).
```
    argparser.add_argument('--class_num', type=int, default=47)
    argparser.add_argument('--features_num', type=int, default=100)
```
#### Set GNN hyper-parameters
    These are the default setting in Legion.
    argparser.add_argument('--train_batch_size', type=int, default=8000) 
    argparser.add_argument('--hidden_dim', type=int, default=256)
    argparser.add_argument('--drop_rate', type=float, default=0.5)
    argparser.add_argument('--learning_rate', type=float, default=0.003)
    argparser.add_argument('--epoch', type=int, default=10)
    argparser.add_argument('--gpu_num', type=int, default=1) 
    
Note that the train_batch_size, epoch, and gpu_num should be the same as sampling hyper-parameters
#### Start training backend
```
3. $ cd /home/atc-artifacts-user/legion-atc-artifacts/pytorch-extension/ && python3 legion_graphsage.py
```
#### Training backend functionality
When training backend successfully runs, system outputs information including epoch time, validation accuracy, and testing accuracy.

![30685d2d9a729ce84d52e8b72fcc1cb](https://github.com/JIESUN233/Legion/assets/109936863/dc93be15-6576-4dd5-ab70-477741b5df28)

![image](https://github.com/JIESUN233/Legion/assets/109936863/199863c0-2ca2-4cc3-8603-f15e6b4aa2b5)


If SEGMENT-FAULT occurs or you kill Legion's processes, please remove semaphores in /dev/shm, for example:
![14b24058fbcfe5bf0648f0d7082686a](https://github.com/JIESUN233/Legion/assets/109936863/c80f6453-6eda-4978-8655-3475cf045457)

### Step 5. Specific hyper-parameter settings to reproduce results in paper
To reproduce the results in paper, we need a 8-GPU machine and run the Python scripts, legion_server.py legion_graphsage.py for graphsage model. (legion_gcn.py for gcn model). The hyper-parameters in Python scripts are shown below.
#### Figure 8 DGX-V100, Legion hyper-parameters:
| Datasets | PR | PA | CO | UKS |
| --- | --- | --- | --- | --- | 
| train_batch_size | 8000 | 8000 | 8000 | 8000 |
| epoch | 10 | 10 | 10 | 10 | 
| gpu_number | 8 | 8 | 8 | 8 | 
| cache_memory | 13GB | 13GB | 11GB | 11GB |
| usenvlink | 1 | 1 | 1 | 1 | 
| class_num | 47 | 2 | 2 | 2 | 
| features_num | 100 | 128 | 256 | 256 | 
| hidden_dim | 256 | 256 | 256 | 256 |
| drop_rate | 0.5 | 0.5 | 0.5 | 0.5 | 
| learning_rate | 0.003 | 0.003 | 0.003 | 0.003 | 

#### Figure 8 DGX-V100, PaGraph hyper-parameters:
| Datasets | PR | PA | CO | UKS |
| --- | --- | --- | --- | --- |
| train_batch_size | 8000 | OOM | OOM | OOM |
| epoch | 10 | OOM | OOM | OOM |
| gpu number | 8 | OOM | OOM |  OOM |
| feature cache ratio | 100% | OOM | OOM |  OOM |
| CPU threads | 64 | OOM | OOM |  OOM |
| class_num | 47 | OOM | OOM | OOM |
| features_num | 100 | OOM | OOM | OOM |
| hidden_dim | 256 | OOM | OOM | OOM |
| drop_rate | 0.5 | OOM | OOM | OOM |
| learning_rate | 0.003 | OOM | OOM | OOM |

#### Figure 8 DGX-V100, GNNLab hyper-parameters:
| Datasets | PR | PA | CO | UKS |
| --- | --- | --- | --- | --- |
| train_batch_size | 8000 | 8000 | 8000 | OOM |
| epoch | 10 | 10 | 10 | OOM |
| sampling gpu number | 4 | 2 | 1 |  OOM |
| training gpu number | 4 | 6 | 7 |  OOM |
| feature cache ratio | 100% | 24% | 18% |  OOM |
| class_num | 47 | 2 | 2 | OOM |
| features_num | 100 | 128 | 256 | OOM |
| hidden_dim | 256 | 256 | 256 | OOM |
| drop_rate | 0.5 | 0.5 | 0.5 | OOM |
| learning_rate | 0.003 | 0.003 | 0.003 | OOM |

#### Figure 8 DGX-V100, DGL(UVA) hyper-parameters:
| Datasets | PR | PA | CO | UKS |
| --- | --- | --- | --- | --- | 
| train_batch_size | 8000 | 8000 | 8000 | 8000 |
| epoch | 10 | 10 | 10 | 10 | 
| gpu_number | 8 | 8 | 8 | 8 | 
| class_num | 47 | 2 | 2 | 2 | 
| features_num | 100 | 128 | 256 | 256 | 
| hidden_dim | 256 | 256 | 256 | 256 |
| drop_rate | 0.5 | 0.5 | 0.5 | 0.5 | 
| learning_rate | 0.003 | 0.003 | 0.003 | 0.003 | 

#### Figure 8 DGX-A100, Legion hyper-parameters:
| Datasets | PR | PA | CO | UKS | UKL | CL |
| --- | --- | --- | --- | --- |  --- | --- | 
| train_batch_size | 8000 | 8000 | 8000 | 8000 | 8000 | 8000 |
| epoch | 10 | 10 | 10 | 10 | 10 | 10 | 
| gpu_number | 8 | 8 | 8 | 8 |  8 | 8 | 
| cache_memory | 36GB | 36GB | 32GB | 32GB | 36GB | 36GB |
| usenvlink | 1 | 1 | 1 | 1 | 1 | 1 | 
| class_num | 47 | 2 | 2 | 2 | 2 | 2 | 
| features_num | 100 | 128 | 256 | 256 | 128 | 128 |
| hidden_dim | 256 | 256 | 256 | 256 | 256 | 256 |
| drop_rate | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 
| learning_rate | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 

#### Figure 8 DGX-A100, DGL(UVA) hyper-parameters:
| Datasets | PR | PA | CO | UKS | UKL | CL |
| --- | --- | --- | --- | --- |  --- | --- | 
| train_batch_size | 8000 | 8000 | 8000 | 8000 | 8000 | 8000 |
| epoch | 10 | 10 | 10 | 10 | 10 | 10 | 
| gpu_number | 8 | 8 | 8 | 8 |  8 | 8 | 
| class_num | 47 | 2 | 2 | 2 | 2 | 2 | 
| features_num | 100 | 128 | 256 | 256 | 128 | 128 |
| hidden_dim | 256 | 256 | 256 | 256 | 256 | 256 |
| drop_rate | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 
| learning_rate | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 

All systems will output the epoch time of each setting. Users need to use a external PCM tool to collect maximum PCIe traffic among different sockets.


## 5. Legion Code Structure
To help users understand Legion's implementation, we list the code structure in this part.
```
legion-atc-artifacts\
├─legion_server.py 
├─src\                                              ## codes of sampling server
└pytorch_extension\                                 ## codes of training backend

legion-atc-artifacts\src\
├─main.cpp                                          ## sampling server main
├─Server.cu, Server.h                               ## implementation of sampling server 
├─GPUGraphStore.cu, GPUGraphStore.cuh               ## initializing of graph storage
├─GPU_Memory_Graph_Storage.cu GPU_Graph_Storage.cuh ## graph topology storage
├─GPU_Memory_Node_Storage.cu GPU_Node_Storage.cuh   ## graph features storage
├─Operator.cu Operator.h                            ## graph operators in fine-grained pipeline
├─Kernels.cu Kernels.cuh                            ## CUDA implimentation of each operators
├─GPUCache.cu GPUCache.cuh                          ## unified cache management
├─GPUMemoryPool.cu GPUMemoryPool.cuh                ## internal buffers in system
├─CUDA_IPC_Service.cu CUDA_IPC_Service.h            ## inter process communication module for sampling server with training backend
├─Makefile
├─env.sh                                            ## setting enviromental variables
├─build/                                            ## pcm library
├─pcm/src/                                          ## pcm source code
├─include/                                          ## hashmap implementation
└Others

legion-atc-artifacts\pytorch_extension\
├─legion_graphsage.py                               ## training backend for graphsage model
├─legion_gcn.py                                     ## training backend for gcn model
├─setup.py                                          ## compiling the training backend
├─ipc_service.cpp ipc_service.h ipc_cuda_kernel.cu  ## inter process communication module for training backend with sampling server
└Others
```
## 6. Build Legion from Source
```
$ git clone https://github.com/JIESUN233/Legion.git
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
2. $ cd xtrapulp/ && make 
```
This will just make libxtrapulp.a static library for use with xtrapulp.h
```
3. $ make libxtrapulp
```

### Legion Compiling
#### Firstly, build Legion's sampling server
```
1. $ cd /home/atc-artifacts-user/legion-atc-artifacts/src/

2. $ make cuda && make main

```
#### Secondly, build Legion's training backend
```
3. $ cd /home/atc-artifacts-user/legion-atc-artifacts/pytorch_extension/
```
Change into root user and execute:
```
4. $ python3 setup.py install
```

### Run Legion
Similar to the way in using pre-installed Legion





