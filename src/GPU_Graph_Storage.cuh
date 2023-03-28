#ifndef GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_GRAPH_STORAGE_H_
#define GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_GRAPH_STORAGE_H_
#include <cstdint>
#include <string>
#include <vector>
#include "BuildInfo.h"

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
// #include <thrust/sort.h>
// #include <thrust/execution_policy.h>
#include "hashmap.h"
// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }
  

class GPUGraphStorage {
public: 
    virtual ~GPUGraphStorage() = default;
    //build
    virtual void Build(BuildInfo* info) = 0;
    virtual void GraphCache(std::vector<unsigned long long int*> &access_time, int32_t device_count) = 0;
    virtual void Finalize() = 0;
    //CSR
    virtual int32_t GetPartitionCount() const = 0;
	  virtual int64_t** GetCSRNodeIndex(int32_t dev_id) const = 0;
	  virtual int32_t** GetCSRNodeMatrix(int32_t dev_id) const = 0;
    virtual int64_t* GetCSRNodeIndexCPU() const = 0;
    virtual int32_t* GetCSRNodeMatrixCPU() const = 0;
    virtual int64_t Src_Size(int32_t part_id) const = 0;
    virtual int64_t Dst_Size(int32_t part_id) const = 0;
    virtual char* PartitionIndex(int32_t dev_id) const = 0;
    virtual int32_t* PartitionOffset(int32_t dev_id) const = 0;
    virtual void Find(int32_t* input_ids, char* partition_index, int32_t* partition_offset, int32_t batch_size, int32_t device_id, int32_t op_id, cudaStream_t strm_hdl) = 0;
};
extern "C" 
GPUGraphStorage* NewGPUMemoryGraphStorage();

#endif  // GRAPHLEARN_CORE_GRAPH_STORAGE_GPU_GRAPH_STORAGE_H_