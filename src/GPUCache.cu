#include "GPUCache.cuh"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <mutex>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
// #include <thrust/sort.h>
// #include <thrust/execution_policy.h>
#include <algorithm>
#include <functional>
#include <cstdlib>

#include <cstdint>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <thrust/sequence.h>
// #include <bcht.hpp>
// #include <cmd.hpp>
// #include <gpu_timer.hpp>
// #include <limits>
// #include <perf_report.hpp>
// #include <rkg.hpp>
// #include <type_traits>

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

#define SHMEMSIZE 10240

#define THRESHOLD1 1
#define THRESHOLD2 0

#define TOTAL_DEV_NUM 1
#define TOL 16
#define NCOUNT_1 4
#define NCOUNT_2 10
using pair_type = bght::pair<int32_t, int32_t>;

__global__ void feature_cache_hit(int32_t* cache_offset, int32_t batch_size, int32_t* global_count){
    __shared__ int32_t local_count[1];
    if(threadIdx.x == 0){
        local_count[0] = 0;
    }
    __syncthreads();
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (batch_size); thread_idx += blockDim.x * gridDim.x){
        int32_t offset = cache_offset[thread_idx];
        if(offset >= 0){
            atomicAdd(local_count, 1);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(global_count, local_count[0]);
    }
}


__global__ void Find_Kernel(
    int32_t* sampled_ids,
    int32_t* cache_offset,
    int32_t* node_counter,
    int32_t total_num_nodes,
    int32_t op_id,
    int32_t* cache_map)
{
    int32_t batch_size = 0;
	int32_t node_off = 0;
	if(op_id == 1){
		node_off = node_counter[3];
		batch_size = node_counter[4];
	}else if(op_id == 3){
		node_off = node_counter[5];
		batch_size = node_counter[6];
	}else if(op_id == 5){
		node_off = node_counter[7];
		batch_size = node_counter[8];
	}
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < batch_size; thread_idx += gridDim.x * blockDim.x){
        int32_t id = sampled_ids[node_off + thread_idx];
        // cache_offset[thread_idx] = -1;
        if(id < 0){
            cache_offset[thread_idx] = -1;
        }else{
            cache_offset[thread_idx] = cache_map[id%total_num_nodes];
        }
    }
}

__global__ void Init_Int64(
    int64_t* array,
    int32_t length,
    int32_t value)
{
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < length; thread_idx += gridDim.x * blockDim.x){
        array[thread_idx] = value;
    }
}

__global__ void Init_Int32(
    int32_t* array,
    int32_t length,
    int32_t value)
{
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < length; thread_idx += gridDim.x * blockDim.x){
        array[thread_idx] = value;
    }
}


__global__ void Init_Array_Seq(
    int32_t* array,
    int32_t length)
{
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < length; thread_idx += gridDim.x * blockDim.x){
        array[thread_idx] = thread_idx;
    }
}

__device__ int32_t SetHash(int32_t id, int32_t set_num, int32_t seed){
    return (id + seed) % set_num;
}

__global__ void CacheHitTimes(int32_t* sampled_node, int32_t* node_counter){
    __shared__ int32_t count[2];
    if(threadIdx.x == 0){
        count[0] = 0;
        count[1] = 0;
    }
    __syncthreads();
    int32_t num_nodes = node_counter[9];
    // int32_t num_edges = edge_counter[4];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_nodes; thread_idx += gridDim.x * blockDim.x){
        int32_t cid = sampled_node[thread_idx];
        if(cid >= 0){
            atomicAdd(count, 1);

            // int32_t cache_offset = cache_map[cid];
            // if(cache_offset >= 0){
            //     atomicAdd(count, 1);
            // }
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(node_counter + 10, count[0]);
    }
    // __syncthreads();
    // for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_edges; thread_idx += gridDim.x * blockDim.x){
    //     int32_t cid = sampled_edge[thread_idx];
    //     if(cid >= 0){
    //         int32_t cache_offset = partition_index[cid];
    //         if(cache_offset >= 0 && cache_offset < partition_count){
    //             atomicAdd(count + 1, 1);
    //         }
    //     }
    // }
    // __syncthreads();
    // if(threadIdx.x == 0){
    //     atomicAdd(edge_counter + 10, count[1]);
    // }
}


void mmap_cache_read(std::string &cache_file, std::vector<int32_t>& cache_map){
    int64_t t_idx = 0;
    int32_t fd = open(cache_file.c_str(), O_RDONLY);
    if(fd == -1){
        std::cout<<"cannout open file: "<<cache_file<<"\n";
    }
    // int64_t buf_len = lseek(fd, 0, SEEK_END);
    int64_t buf_len = int64_t(int64_t(cache_map.size()) * 4);
    const int32_t* buf = (int32_t *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int32_t* buf_end = buf + buf_len/sizeof(int32_t);
    int32_t temp;
    while(buf < buf_end){
        temp = *buf;
        cache_map[t_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}

__global__ void HotnessMeasure(int32_t* new_batch_ids, int32_t* node_counter, unsigned long long int* access_map){
    int32_t num_candidates = node_counter[9];
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < num_candidates; thread_idx += gridDim.x * blockDim.x){
        int32_t cid = new_batch_ids[thread_idx];
        if(cid >= 0){
            atomicAdd(access_map + cid, 1);
        }
    }
}

__global__ void InitPair(pair_type* pair, int32_t* cache_ids, int32_t* cache_offset, int32_t capacity){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity; thread_idx += gridDim.x * blockDim.x){
        pair[thread_idx].first = cache_ids[thread_idx];
        pair[thread_idx].second = cache_offset[thread_idx];
    }
}


class PreSCCacheController : public CacheController {
public:
    PreSCCacheController(int32_t train_step){
       train_step_ = train_step;
    }

    virtual ~PreSCCacheController(){}

    void Initialize(
        int32_t dev_id,
        int32_t capacity,
        int32_t sampled_num,
        int32_t total_num_nodes,
        int32_t batch_size) override
    {
        device_idx_ = dev_id;
        capacity_ = capacity;
        total_num_nodes_ = total_num_nodes;
        sampled_num_ = sampled_num;
        batch_size_ = batch_size;
        cudaSetDevice(dev_id);

        // dim3 block_num(80, 1);
        // dim3 thread_num(1024, 1);

        // cudaMalloc(&cache_map_, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));//may overflow when meet 1B nodes
        cudaMalloc(&cache_ids_, int64_t(int64_t(capacity_ * TOTAL_DEV_NUM) * sizeof(int32_t)));
        cudaMalloc(&cache_offset_, int64_t(int64_t(capacity_ * TOTAL_DEV_NUM) * sizeof(int32_t)));
        cudaCheckError();

        cudaMalloc(&access_time_, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        cudaMemset(access_time_, 0, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        cudaCheckError();

        cudaMalloc(&edge_access_time_, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        cudaMemset(edge_access_time_, 0, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        cudaCheckError();
        iter_ = 0;
        max_ids_ = 0;

        auto invalid_key = -1;
        auto invalid_value = -1;

        hash_map_ = new bght::bcht<int32_t, int32_t>(int64_t(capacity_ * TOTAL_DEV_NUM) * 2, invalid_key, invalid_value);
        cudaCheckError();
        // pos_map_ = new bght::bcht<int32_t, int32_t>(sampled_num * 2, invalid_key, invalid_value);
        // cudaCheckError();

        cudaMalloc(&pair_, int64_t(int64_t(capacity_ * TOTAL_DEV_NUM) * sizeof(pair_type)));
        cudaCheckError();


        // cudaMalloc(&sampled_offset_, sampled_num * sizeof(int32_t));
        // cudaCheckError();
        // Init_Array_Seq<<<block_num, thread_num>>>(sampled_offset_, sampled_num);
        // cudaMalloc(&graph_pair_, int64_t(int64_t(sampled_num) * sizeof(pair_type)));
        // cudaCheckError();

        cudaMalloc(&d_global_count_, 4);
        h_global_count_ = (int32_t*)malloc(4);
        find_iter_ = 0;
        h_cache_hit_ = 0;
    }

    void Finalize() override {
        // pos_map_->clear();
    }

    void MakePlan(
                    int32_t* sampled_ids,
                    int32_t* agg_src_id,
                    int32_t* agg_dst_id,
                    int32_t* agg_src_off,
                    int32_t* agg_dst_off,
                    int32_t* node_counter,
                    int32_t* edge_counter,
                    bool is_presc,
                    void* stream) override
    {
        dim3 block_num(48, 1);
        dim3 thread_num(1024, 1);

        // cudaMemcpy(h_edge_counter, edge_counter, 64, cudaMemcpyDeviceToHost);
        // InitPair<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(graph_pair_, sampled_ids, sampled_offset_, h_node_counter[9]);
        // pos_map_->insert(graph_pair_, (graph_pair_ + h_node_counter[9]), static_cast<cudaStream_t>(stream));
        // cudaCheckError();
        // pos_map_->find(agg_src_id, agg_src_id + (h_edge_counter[4]), agg_src_off, static_cast<cudaStream_t>(stream));
        // pos_map_->find(agg_dst_id, agg_dst_id + (h_edge_counter[4]), agg_dst_off, static_cast<cudaStream_t>(stream));

        if(is_presc){
            int32_t* h_node_counter = (int32_t*)malloc(16*sizeof(int32_t));
            // int32_t* h_edge_counter = (int32_t*)malloc(16*sizeof(int32_t));
            cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);
            HotnessMeasure<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(sampled_ids, node_counter, access_time_);

            if(h_node_counter[9] > max_ids_){
                max_ids_ = h_node_counter[9];
            }
            if(iter_ == (train_step_ - 1)){
                iter_ = 0;
            }
            free(h_node_counter);
        }else{

        }
        
        iter_++;
    }

    /*num candidates = sampled num*/
    void Update(int cache_expand) override
    {
        cudaSetDevice(device_idx_);
        dim3 block_num(80, 1);
        dim3 thread_num(1024, 1);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        InitPair<<<block_num, thread_num>>>(pair_, cache_ids_, cache_offset_, capacity_ * cache_expand);
        cudaCheckError();
        bool success = hash_map_->insert(pair_, (pair_ + capacity_ * cache_expand), stream);
        cudaCheckError();
        if(success){
            std::cout<<"Feature Cache Successfully Initialized\n";
        }
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaFree(pair_);
        cudaCheckError();
        cudaFree(cache_ids_);
        cudaCheckError();
        cudaFree(cache_offset_);
        cudaCheckError();
    }

    void AccessCount(
        int32_t* d_key,
        int32_t num_keys,
        void* stream) override
    {}

    unsigned long long int* GetAccessedMap(){
        return access_time_;
    }

    unsigned long long int* GetEdgeAccessedMap(){
        return edge_access_time_;
    }

    int32_t* GetCacheId() override {
        return cache_ids_;
    }

    int32_t* GetCacheOffset() override {
        return cache_offset_;
    }

    int32_t* FutureBatch() override
    {
        return nullptr;
    }

    int32_t Capacity() override
    {
        return capacity_;
    }

    int32_t* AllCachedIds() override
    {
        return nullptr;
    }

    int32_t* RecentMark() override {
        return nullptr;
    }

    void Find(
        int32_t* sampled_ids,
        int32_t* cache_offset,
        int32_t* node_counter,
        int32_t op_id,
        void* stream) override
    {
        // dim3 block_num(40, 1);
        // dim3 thread_num(1024, 1);
        // Find_Kernel<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(sampled_ids, cache_offset, node_counter, total_num_nodes_, op_id, cache_map_);
        // cudaCheckError();
        int32_t* h_node_counter = (int32_t*)malloc(64);
        cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);

        int32_t batch_size = 0;
        int32_t node_off = 0;
        if(op_id == 1){
            node_off = h_node_counter[3];
            batch_size = h_node_counter[4];
        }else if(op_id == 3){
            node_off = h_node_counter[5];
            batch_size = h_node_counter[6];
        }else if(op_id == 5){
            node_off = h_node_counter[7];
            batch_size = h_node_counter[8];
        }
        if(batch_size == 0){
            std::cout<<"invalid batchsize for feature extraction\n";
            return;
        }
        hash_map_->find(sampled_ids + node_off, sampled_ids + (node_off + batch_size), cache_offset, static_cast<cudaStream_t>(stream));
        if(find_iter_ % 500 == 0){
            cudaMemsetAsync(d_global_count_, 0, 4, static_cast<cudaStream_t>(stream));
            dim3 block_num(48, 1);
            dim3 thread_num(1024, 1);
            feature_cache_hit<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(cache_offset, batch_size, d_global_count_);
            cudaMemcpy(h_global_count_, d_global_count_, 4, cudaMemcpyDeviceToHost);
            h_cache_hit_ += h_global_count_[0];
            if(op_id == 5){
                std::cout<<device_idx_<<" Feature Cache Hit: "<<h_cache_hit_<<" "<<(h_cache_hit_ * 1.0 / h_node_counter[9])<<"\n";    
                h_cache_hit_ = 0;
            }
        }
        if(op_id == 5){
            // std::cout<<device_idx_<<" Feature Cache Hit: "<<h_cache_hit_<<" "<<(h_cache_hit_ * 1.0 / h_node_counter[9])<<"\n";    
            // h_cache_hit_ = 0;
            find_iter_++;
            // std::cout<<"find_iter "<<find_iter_<<"\n";
        }
    }

    int32_t MaxIdNum() override
    {
        return max_ids_;
    }

private:
    int32_t device_idx_;
    int32_t capacity_;
    int32_t total_num_nodes_;
    int32_t set_num_;
    int32_t way_num_;
    int32_t sampled_num_;
    int32_t k_batch_;
    int32_t batch_size_;

    int32_t* cache_map_;
    unsigned long long int* access_time_;
    unsigned long long int* edge_access_time_;
    int32_t train_step_;
    int32_t iter_;

    int32_t max_ids_;//for allocating feature buffer

    bght::bcht<int32_t, int32_t>* hash_map_;

    bght::bcht<int32_t, int32_t>* pos_map_;

    int32_t* cache_ids_;
    int32_t* cache_offset_;
    int32_t* sampled_offset_;
    pair_type* pair_;
    pair_type* graph_pair_;

    int32_t* d_global_count_;
    int32_t* h_global_count_;
    int32_t  h_cache_hit_;
    int32_t  find_iter_;
};

CacheController* NewPreSCCacheController(int32_t train_step)
{
    return new PreSCCacheController(train_step);
}

__global__ void init_feature_cache(float** pptr, float* ptr, int dev_id){
    pptr[dev_id] = ptr;
}

void GPUCache::Initialize(
    std::vector<int> device,
    int32_t capacity,
    int32_t int_attr_len,
    int32_t float_attr_len,
    int32_t K_batch,
    int32_t way_num,
    int32_t train_step)
{
    //allocate essential buffer
    dev_ids_.resize(TOTAL_DEV_NUM);
    cache_controller_.resize(TOTAL_DEV_NUM);

    k_batch_ = K_batch;
    way_num_ = way_num;

    for(int32_t i = 0; i < TOTAL_DEV_NUM; i++){
        dev_ids_[i] = false;
    }

    if(int_attr_len > 0){
        int_feature_cache_.resize(TOTAL_DEV_NUM);
    }
    if(float_attr_len > 0){
        float_feature_cache_.resize(TOTAL_DEV_NUM);
    }
    cudaCheckError();

    // d_float_feature_cache_ptr_.resize(TOTAL_DEV_NUM);

    // for(int32_t i = 0; i < device.size(); i++){
    //     int32_t dev_id = device[i];
    //     cudaSetDevice(dev_id);
    //     float** new_ptr;
    //     cudaMalloc(&new_ptr, device.size() * sizeof(float*));
    //     d_float_feature_cache_ptr_[dev_id] = new_ptr;
    // }
    for(int32_t i = 0; i < device.size(); i++){
        int32_t dev_id = device[i];
        dev_ids_[dev_id] = true;

        CacheController* cctl = NewPreSCCacheController(train_step);
        cache_controller_[dev_id] = cctl;

        // if(float_attr_len > 0){
        //     cudaSetDevice(dev_id);
        //     float* new_float_feature_cache;
        //     cudaMalloc(&new_float_feature_cache, int64_t(int64_t(int64_t(capacity) * float_attr_len) * sizeof(float)));
        //     float_feature_cache_[dev_id] = new_float_feature_cache;
        //     init_feature_cache<<<1,1>>>(d_float_feature_cache_ptr_[0], new_float_feature_cache, dev_id);
        //     cudaCheckError();
        // }
    }
    // for(int32_t i = 0; i < device.size(); i++){
    //     int32_t dev_id = device[i];
    //     cudaSetDevice(dev_id);
    //     cudaMemcpy(d_float_feature_cache_ptr_[dev_id], d_float_feature_cache_ptr_[0], device.size() * sizeof(float**), cudaMemcpyDeviceToDevice);
    //     cudaCheckError();
    // }
    std::cout<<"Cache Space Initialize\n";
    capacity_ = capacity;
    int_attr_len_ = int_attr_len;
    float_attr_len_ = float_attr_len;
    is_presc_ = true;
}

void GPUCache::InitializeCacheController(
    int32_t dev_id,
    int32_t capacity,
    int32_t sampled_num,
    int32_t total_num_nodes,
    int32_t batch_size)
{
    if(dev_ids_[dev_id] == true){
        cache_controller_[dev_id]->Initialize(dev_id, capacity, sampled_num, total_num_nodes, batch_size);
    }else{
        std::cout<<"invalid device for cache\n";
    }
}

void GPUCache::Finalize(int32_t dev_id){
    if(dev_ids_[dev_id] == true){
        cudaSetDevice(dev_id);
        cache_controller_[dev_id]->Finalize();
        // cudaFree(float_feature_cache_[i]);
    }
    // for(int32_t i = 0; i < TOTAL_DEV_NUM; i++){
    //     if(dev_ids_[i] == true){
    //         cudaSetDevice(i);
    //         cache_controller_[i]->Finalize();
    //         // cudaFree(float_feature_cache_[i]);
    //     }
    // }
}

int32_t GPUCache::Capacity(){
    return capacity_;
}

void GPUCache::Find(
    int32_t* sampled_ids,
    int32_t* cache_offset,
    int32_t* node_counter,
    int32_t op_id,
    void* stream,
    int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        cache_controller_[dev_id]->Find(sampled_ids, cache_offset, node_counter, op_id, stream);
    }else{
        std::cout<<"invalid device for cache\n";
    }
}

void GPUCache::MakePlan(
    int32_t* sampled_ids,
    int32_t* agg_src_id,
    int32_t* agg_dst_id,
    int32_t* agg_src_off,
    int32_t* agg_dst_off,
    int32_t* node_counter,
    int32_t* edge_counter,
    void* stream,
    int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        cache_controller_[dev_id]->MakePlan(sampled_ids, agg_src_id, agg_dst_id, agg_src_off, agg_dst_off, node_counter, edge_counter, is_presc_, stream);
    }else{
        std::cout<<"invalid device for cache\n";
    }
}

void GPUCache::Update(
    int32_t* candidates_ids,
    float* candidates_float_feature,
    float* cache_float_feature,
    int32_t float_attr_len,
    void* stream,
    int32_t dev_id)
{
    // if(dev_ids_[dev_id] == true){
    //     cache_controller_[dev_id]->Update();
    // }else{
    //     std::cout<<"invalid device for cache\n";
    // }
}

void GPUCache::AccessCount(
    int32_t* d_key,
    int32_t num_keys,
    void* stream,
    int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        cache_controller_[dev_id]->AccessCount(d_key, num_keys, stream);
    }else{
        std::cout<<"invalid device for cache\n";
    }
}

__global__ void aggregate_access(unsigned long long int* agg_access_time, unsigned long long int* new_access_time, int32_t total_num_nodes){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (total_num_nodes); thread_idx += blockDim.x * gridDim.x){
        agg_access_time[thread_idx] += new_access_time[thread_idx];
    }
}

__global__ void init_cache_order(int32_t* cache_order, int32_t total_num_nodes){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (total_num_nodes); thread_idx += blockDim.x * gridDim.x){
        cache_order[thread_idx] = thread_idx;
    }
}

void GPUCache::Coordinate(int cache_agg_mode, GPUNodeStorage* noder, GPUGraphStorage* graph, std::vector<uint64_t>& counters, int train_step){
    std::cout<<"Start initialize cache\n";
    std::vector<unsigned long long int*> access_time;
    std::vector<unsigned long long int*> edge_access_time;
    std::vector<int32_t*> d_cache_ids;
    std::vector<int32_t*> d_cache_offset;
    for(int32_t i = 0; i < TOTAL_DEV_NUM; i++){
        if(dev_ids_[i] == true){
            access_time.push_back(cache_controller_[i]->GetAccessedMap());
            edge_access_time.push_back(cache_controller_[i]->GetEdgeAccessedMap());
            d_cache_ids.push_back(cache_controller_[i]->GetCacheId());
            d_cache_offset.push_back(cache_controller_[i]->GetCacheOffset());
        }
    }

    dim3 block_num(80,1);
    dim3 thread_num(1024,1);
    int32_t total_num_nodes = noder->TotalNodeNum();
    float* cpu_float_attrs = noder->GetAllFloatAttr();
    int32_t float_attr_len = noder->GetFloatAttrLen();
    int64_t* csr_index = graph->GetCSRNodeIndexCPU();


    // if(cache_agg_mode == 1){
    //     for(int32_t i = 0; i < 4; i++){//4 clique
    //         cudaSetDevice(i*2);
    //         int max_payload_size = 64;

    //         int64_t memory_step = 800000000;//100MB
    //         uint64_t total_trans_of_topo = counters[i/2] / 2;
    //         uint64_t total_trans_of_feat = int64_t((int64_t(int64_t(cache_controller_[i]->MaxIdNum()) * train_step) * float_attr_len) * sizeof(float)) / max_payload_size;

    //         std::cout<<"Total topo trans "<<total_trans_of_topo<<"\n";
    //         std::cout<<"Total feat trans "<<total_trans_of_feat<<"\n";
    //         int32_t* node_cache_order;
    //         cudaMalloc(&node_cache_order, total_num_nodes * sizeof(int32_t));
    //         cudaCheckError();
    //         init_cache_order<<<block_num, thread_num>>>(node_cache_order, total_num_nodes);
    //         cudaCheckError();
    //         thrust::sort_by_key(thrust::device, access_time[i], access_time[i] + total_num_nodes, node_cache_order, thrust::greater<int32_t>());
    //         cudaCheckError();
    //         // int32_t* h_node_cache_order = (int32_t*)malloc(total_num_nodes * sizeof(int32_t));
    //         // cudaMemcpy(h_node_cache_order, node_cache_order, total_num_nodes * sizeof(int32_t), cudaMemcpyDeviceToHost);
    //         // cudaCheckError();
    //         cudaFree(node_cache_order);

    //         int32_t* edge_cache_order;
    //         cudaMalloc(&edge_cache_order, total_num_nodes * sizeof(int32_t));
    //         cudaCheckError();
    //         init_cache_order<<<block_num, thread_num>>>(edge_cache_order, total_num_nodes);
    //         cudaCheckError();
    //         thrust::sort_by_key(thrust::device, edge_access_time[i], edge_access_time[i] + total_num_nodes, edge_cache_order, thrust::greater<int32_t>());
    //         cudaCheckError();
    //         int32_t* h_edge_cache_order = (int32_t*)malloc(total_num_nodes * sizeof(int32_t));
    //         cudaMemcpy(h_edge_cache_order, edge_cache_order, total_num_nodes * sizeof(int32_t), cudaMemcpyDeviceToHost);
    //         cudaCheckError();
    //         cudaFree(edge_cache_order);
            

    //         int32_t* d_node_prefix;
    //         int32_t* d_edge_prefix;
    //         cudaMalloc(&d_node_prefix, int64_t(total_num_nodes)*sizeof(int32_t));
    //         cudaMalloc(&d_edge_prefix, int64_t(total_num_nodes)*sizeof(int32_t));
    //         thrust::inclusive_scan(thrust::device, access_time[i], access_time[i] + total_num_nodes, d_node_prefix);
    //         thrust::inclusive_scan(thrust::device, edge_access_time[i], edge_access_time[i] + total_num_nodes, d_edge_prefix);
    //         int32_t* h_node_prefix = (int32_t*)malloc(int64_t(total_num_nodes)*sizeof(int32_t));
    //         int32_t* h_edge_prefix = (int32_t*)malloc(int64_t(total_num_nodes)*sizeof(int32_t));
    //         cudaMemcpy(h_node_prefix, d_node_prefix, int64_t(total_num_nodes)*sizeof(int32_t), cudaMemcpyDeviceToHost);
    //         cudaMemcpy(h_edge_prefix, d_edge_prefix, int64_t(total_num_nodes)*sizeof(int32_t), cudaMemcpyDeviceToHost);
    //         std::cout<<"total node hotness "<<h_node_prefix[total_num_nodes - 1]<<"\n";
    //         std::cout<<"total edge hotness "<<h_edge_prefix[total_num_nodes - 1]<<"\n";
    //         int64_t current_mem = 0;
    //         int64_t total_mem = 10000000000;//10GB
    //         int64_t steps = (total_mem  - 1) / memory_step + 1;
    //         int64_t current_steps = 0;
    //         std::cout<<"mem steps "<<steps<<"\n";
    //         int32_t node_num_topo = 0;
    //         int64_t current_edge_mem = 0;
    //         std::vector<float> trans_of_topo((steps + 1), 0);
    //         std::vector<float> trans_of_feat((steps + 1), 0);
    //         for( ;current_mem < total_mem ; current_mem += memory_step){
    //             int32_t node_num_feat = current_steps * (memory_step / (float_attr_len * sizeof(float)));
    //             // std::cout<<"current step "<<current_steps<<" "<<node_num_feat<<" "<<node_num_topo<<"\n";
    //             while(1){
    //                 if(current_edge_mem < (current_mem + memory_step) && (node_num_topo < total_num_nodes)){
    //                     int32_t cache_id = h_edge_cache_order[node_num_topo];
    //                     current_edge_mem += (sizeof(int64_t) + sizeof(int32_t) * (csr_index[cache_id + 1] - csr_index[cache_id]));
    //                     node_num_topo++;
    //                 }else{
    //                     break;
    //                 }
    //             }
    //             if(node_num_topo < total_num_nodes){
    //                 trans_of_topo[current_steps] = total_trans_of_topo * 1.0 / h_edge_prefix[total_num_nodes - 1] * h_edge_prefix[node_num_topo - 1];
    //             }
    //             if(node_num_feat < total_num_nodes){
    //                 trans_of_feat[current_steps] = total_trans_of_feat * 1.0 / h_node_prefix[total_num_nodes - 1] * h_node_prefix[node_num_feat - 1];
    //             }
    //             current_steps++;
    //         }
    //         for(int i = 0; i < steps; i++){
    //             std::cout<<"trans "<<trans_of_topo[i]<<" "<<trans_of_feat[i]<<"\n";
    //         }
    //     }
    // }else if(cache_agg_mode == 3){
    //     for(int32_t i = 0; i < 1; i++){//1 clique

    //         cudaSetDevice(i);
    //         int max_payload_size = 64;

    //         int64_t memory_step = 2560000000;//400MB
    //         uint64_t total_trans_of_topo = counters[0] + counters[1]; 
    //         uint64_t total_trans_of_feat = 0;
    //         for(int32_t j = 0; j < 8; j++){
    //             total_trans_of_feat += (int64_t((int64_t(int64_t(cache_controller_[j]->MaxIdNum()) * train_step) * float_attr_len) * sizeof(float)) / max_payload_size);
    //         }
    //         std::cout<<"Total topo trans "<<total_trans_of_topo<<"\n";
    //         std::cout<<"Total feat trans "<<total_trans_of_feat<<"\n";
    //         int32_t* node_cache_order;
    //         cudaMalloc(&node_cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
    //         cudaCheckError();
    //         init_cache_order<<<block_num, thread_num>>>(node_cache_order, total_num_nodes);
    //         cudaCheckError();
    //         uint64_t* node_agg_access_time;
    //         cudaMalloc(&node_agg_access_time, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));
    //         cudaMemset(node_agg_access_time, 0, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));
    //         uint64_t* edge_agg_access_time;
    //         cudaMalloc(&edge_agg_access_time, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));
    //         cudaMemset(edge_agg_access_time, 0, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));

    //         for(int32_t j = 0; j < 8; j++){
    //             aggregate_access<<<block_num, thread_num>>>((unsigned long long*)node_agg_access_time, access_time[j], total_num_nodes);
    //             aggregate_access<<<block_num, thread_num>>>((unsigned long long*)edge_agg_access_time, edge_access_time[j], total_num_nodes);
    //         }
    //         thrust::sort_by_key(thrust::device, node_agg_access_time, node_agg_access_time + total_num_nodes, node_cache_order, thrust::greater<uint64_t>());
    //         cudaCheckError();
    //         // int32_t* h_node_cache_order = (int32_t*)malloc(total_num_nodes * sizeof(int32_t));
    //         // cudaMemcpy(h_node_cache_order, node_cache_order, total_num_nodes * sizeof(int32_t), cudaMemcpyDeviceToHost);
    //         // cudaCheckError();
    //         cudaFree(node_cache_order);

    //         int32_t* edge_cache_order;
    //         cudaMalloc(&edge_cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
    //         cudaCheckError();
    //         init_cache_order<<<block_num, thread_num>>>(edge_cache_order, total_num_nodes);
    //         cudaCheckError();
    //         thrust::sort_by_key(thrust::device, edge_agg_access_time, edge_agg_access_time + total_num_nodes, edge_cache_order, thrust::greater<uint64_t>());
    //         cudaCheckError();
    //         int32_t* h_edge_cache_order = (int32_t*)malloc(int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
    //         cudaMemcpy(h_edge_cache_order, edge_cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)), cudaMemcpyDeviceToHost);
    //         cudaCheckError();
    //         cudaFree(edge_cache_order);
            
            

    //         uint64_t* d_node_prefix;
    //         uint64_t* d_edge_prefix;
    //         cudaMalloc(&d_node_prefix, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));
    //         cudaMalloc(&d_edge_prefix, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));
    //         thrust::inclusive_scan(thrust::device, node_agg_access_time, node_agg_access_time + total_num_nodes, d_node_prefix);
    //         thrust::inclusive_scan(thrust::device, edge_agg_access_time, edge_agg_access_time + total_num_nodes, d_edge_prefix);
    //         uint64_t* h_node_prefix = (uint64_t*)malloc(int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));
    //         uint64_t* h_edge_prefix = (uint64_t*)malloc(int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));
    //         cudaMemcpy(h_node_prefix, d_node_prefix, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)), cudaMemcpyDeviceToHost);
    //         cudaMemcpy(h_edge_prefix, d_edge_prefix, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)), cudaMemcpyDeviceToHost);
    //         std::cout<<"total node hotness "<<h_node_prefix[total_num_nodes - 1]<<" "<<h_node_prefix[0]<<" "<<h_node_prefix[1]<<"\n";
    //         std::cout<<"total edge hotness "<<h_edge_prefix[total_num_nodes - 1]<<"\n";
    //         int64_t current_mem = 0;
    //         int64_t total_mem = 256000000000;//10GB
    //         int64_t steps = (total_mem  - 1) / memory_step + 1;
    //         int64_t current_steps = 0;
    //         std::cout<<"mem steps "<<steps<<"\n";
    //         int32_t node_num_topo = 0;
    //         int64_t current_edge_mem = 0;
    //         std::vector<float> trans_of_topo((steps + 1), 0);
    //         std::vector<float> trans_of_feat((steps + 1), 0);
    //         for( ;current_mem < total_mem ; current_mem += memory_step){
    //             int32_t node_num_feat = (current_steps + 1) * (memory_step / (float_attr_len * sizeof(float)));
    //             // std::cout<<"current step "<<current_steps<<" "<<node_num_feat<<" "<<node_num_topo<<"\n";
    //             while(1){
    //                 if(current_edge_mem < (current_mem + memory_step) && (node_num_topo < total_num_nodes)){
    //                     int32_t cache_id = h_edge_cache_order[node_num_topo];
    //                     current_edge_mem += (sizeof(int64_t) + sizeof(int32_t) * (csr_index[cache_id + 1] - csr_index[cache_id]));
    //                     node_num_topo++;
    //                 }else{
    //                     break;
    //                 }
    //             }
    //             if(node_num_topo < total_num_nodes){
    //                 trans_of_topo[current_steps] = total_trans_of_topo * 1.0 / h_edge_prefix[total_num_nodes - 1] * h_edge_prefix[node_num_topo - 1];
    //             }
    //             if(node_num_feat < total_num_nodes){
    //                 trans_of_feat[current_steps] = total_trans_of_feat * 1.0 / h_node_prefix[total_num_nodes - 1] * h_node_prefix[node_num_feat - 1];
    //             }
    //             current_steps++;
    //         }
    //         // std::cout<<"feat trans "<<total_trans_of_feat * 1.0 / h_node_prefix[total_num_nodes - 1] * h_node_prefix[0]<<"\n";
    //         for(int sidx = 0; sidx < steps; sidx++){
    //             std::cout<<"trans "<<trans_of_topo[sidx]<<" "<<trans_of_feat[sidx]<<"\n";
    //         }
    //     }
    // }



    if(cache_agg_mode == 0){//individual mode
        for(int didx = 0; didx < TOTAL_DEV_NUM; didx++){
            cudaSetDevice(didx);
            int32_t* cache_order;
            cudaMalloc(&cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
            cudaCheckError();

            cudaCheckError();
            init_cache_order<<<block_num, thread_num>>>(cache_order, total_num_nodes);
            // thrust::sort_by_key(thrust::device, access_time[didx], access_time[didx] + total_num_nodes, cache_order, thrust::greater<unsigned long long int>());
            cudaCheckError();
            cudaFree(access_time[didx]);
            int32_t* h_cache_order = (int32_t*)malloc(int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
            cudaMemcpy(h_cache_order, cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)), cudaMemcpyDeviceToHost);
            cudaCheckError();
            cudaFree(cache_order);

            std::vector<int32_t> h_cache_ids(capacity_, -1);
            std::vector<int32_t> h_cache_offset(capacity_, -1);
    
            for(int32_t idx = 0; idx < capacity_; idx ++){
                int32_t cached_id = h_cache_order[idx];
                h_cache_ids[idx] = cached_id;
                h_cache_offset[idx] = (didx) * capacity_ + idx;
            }
            cudaMemcpy(d_cache_ids[didx], &h_cache_ids[0], int64_t(int64_t(capacity_) * sizeof(int32_t)), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cache_offset[didx], &h_cache_offset[0], int64_t(int64_t(capacity_) * sizeof(int32_t)), cudaMemcpyHostToDevice);
    
            cudaCheckError();
        }
        
    }else if(cache_agg_mode == 1){
        for(int32_t i = 0; i < 4; i++){//4 nvlink clique
            cudaSetDevice(i*2);
            int32_t* cache_order;
            cudaMalloc(&cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
            cudaCheckError();

            cudaCheckError();
            unsigned long long int* agg_access_time;
            cudaMalloc(&agg_access_time, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
            cudaMemset(agg_access_time, 0, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
            cudaCheckError();

            std::vector<unsigned long long int*> h_access_time;
            h_access_time.resize(2);
            for(int32_t j = 0; j < 2; j++){
                h_access_time[j] = (unsigned long long int*)malloc(int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
                aggregate_access<<<block_num, thread_num>>>(agg_access_time, access_time[i*2+j], total_num_nodes);
                cudaCheckError();

                cudaMemcpy(h_access_time[j], access_time[i*2+j], int64_t(int64_t(total_num_nodes)) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
                cudaCheckError();

            }

            for(int32_t j = 0; j < 2; j++){
                cudaSetDevice(j);
                cudaFree(access_time[i*2+j]);
                cudaCheckError();
            }
            cudaSetDevice(i*2);
            cudaCheckError();
            init_cache_order<<<block_num, thread_num>>>(cache_order, total_num_nodes);
            // thrust::sort_by_key(thrust::device, agg_access_time, agg_access_time + total_num_nodes, cache_order, thrust::greater<unsigned long long int>());
            cudaCheckError();
            int32_t* h_cache_order = (int32_t*)malloc(int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
            cudaMemcpy(h_cache_order, cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)), cudaMemcpyDeviceToHost);
            cudaCheckError();
            cudaSetDevice(i*2);
            cudaFree(cache_order);
            cudaFree(agg_access_time);
            std::vector<int32_t> h_cache_ids(capacity_ * 2, -1);
            std::vector<int32_t> h_cache_offset(capacity_ * 2, -1);
            std::vector<int32_t> h_size(2, 0);
            std::vector<int32_t> h_order(2, -1);

            for(int32_t idx = 0; idx < capacity_ * 2; idx ++){
                int32_t cached_id = h_cache_order[idx];
                // for(int32_t oidx = 0; oidx < 2; oidx++){
                //     unsigned long long int acc_max = 0;
                //     for(int32_t didx = 0; didx < 2; didx ++){
                //         unsigned long long int acc_temp = h_access_time[didx][cached_id];
                //         if(acc_temp >= acc_max){
                //             acc_max = acc_temp;
                //             h_order[oidx] = didx;
                //         }
                //     }
                //     h_access_time[h_order[oidx]][cached_id] = 0;
                // }

                if(h_access_time[0][cached_id] > h_access_time[1][cached_id]){
                    h_order[0] = 0;
                    h_order[1] = 1;
                }else{
                    h_order[1] = 0;
                    h_order[0] = 1;
                }
                for(int32_t oidx = 0; oidx < 2; oidx++){
                    int32_t order_id = h_order[oidx];
                    if(h_size[order_id] < capacity_){
                        h_cache_ids[idx] = cached_id;
                        h_cache_offset[idx] = (i * 2 + order_id) * capacity_ + h_size[order_id];
                        h_size[h_order[oidx]] += 1;
                        break;
                    }
                }
            }
            for(int32_t j = 0; j < 2; j++){
                std::cout<<"cache node num "<<h_size[j]<<"\n";
            } 
            for(int32_t j = 0; j < 2; j++){
                cudaMemcpy(d_cache_ids[i * 2 + j], &h_cache_ids[0], int64_t(int64_t(capacity_ * 2) * sizeof(int32_t)), cudaMemcpyHostToDevice);
                cudaMemcpy(d_cache_offset[i * 2 + j], &h_cache_offset[0], int64_t(int64_t(capacity_ * 2) * sizeof(int32_t)), cudaMemcpyHostToDevice);
                cudaCheckError();
            }

            cudaCheckError();
        }
        cudaCheckError();

    }else if(cache_agg_mode == 2){
        for(int32_t i = 0; i < 2; i++){//2 nvlink clique
            cudaSetDevice(i*4);
            int32_t* cache_order;
            cudaMalloc(&cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
            cudaCheckError();

            cudaCheckError();
            unsigned long long int* agg_access_time;
            cudaMalloc(&agg_access_time, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
            cudaMemset(agg_access_time, 0, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
            cudaCheckError();

            std::vector<unsigned long long int*> h_access_time;
            h_access_time.resize(4);
            for(int32_t j = 0; j < 4; j++){
                h_access_time[j] = (unsigned long long int*)malloc(int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
                aggregate_access<<<block_num, thread_num>>>(agg_access_time, access_time[i*4+j], total_num_nodes);
                cudaCheckError();

                cudaMemcpy(h_access_time[j], access_time[i*4+j], int64_t(int64_t(total_num_nodes)) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
                cudaCheckError();

            }

            for(int32_t j = 0; j < 4; j++){
                cudaSetDevice(j);
                cudaFree(access_time[i*4+j]);
                cudaCheckError();
            }
            cudaSetDevice(i*4);
            cudaCheckError();
            init_cache_order<<<block_num, thread_num>>>(cache_order, total_num_nodes);
            // thrust::sort_by_key(thrust::device, agg_access_time, agg_access_time + total_num_nodes, cache_order, thrust::greater<unsigned long long int>());
            cudaCheckError();
            int32_t* h_cache_order = (int32_t*)malloc(int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
            cudaMemcpy(h_cache_order, cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)), cudaMemcpyDeviceToHost);
            cudaCheckError();
            cudaSetDevice(i*4);
            cudaFree(cache_order);
            cudaFree(agg_access_time);
            std::vector<int32_t> h_cache_ids(capacity_ * 4, -1);
            std::vector<int32_t> h_cache_offset(capacity_ * 4, -1);
            std::vector<int32_t> h_size(4, 0);
            std::vector<int32_t> h_order(4, -1);

            for(int32_t idx = 0; idx < capacity_ * 4; idx ++){
                int32_t cached_id = h_cache_order[idx];
                // for(int32_t oidx = 0; oidx < 2; oidx++){
                //     unsigned long long int acc_max = 0;
                //     for(int32_t didx = 0; didx < 2; didx ++){
                //         unsigned long long int acc_temp = h_access_time[didx][cached_id];
                //         if(acc_temp >= acc_max){
                //             acc_max = acc_temp;
                //             h_order[oidx] = didx;
                //         }
                //     }
                //     h_access_time[h_order[oidx]][cached_id] = 0;
                // }

                for(int32_t oidx = 0; oidx < 4; oidx++){
                    unsigned long long int acc_max = 0;
                    for(int32_t didx = 0; didx < 4; didx ++){
                        unsigned long long int acc_temp = h_access_time[didx][cached_id];
                        if(acc_temp > acc_max){
                            acc_max = acc_temp;
                            h_order[oidx] = didx;
                        }
                    }
                    h_access_time[h_order[oidx]][cached_id] = 0;
                }

                for(int32_t oidx = 0; oidx < 4; oidx++){
                    int32_t order_id = h_order[oidx];
                    if(h_size[order_id] < capacity_){
                        h_cache_ids[idx] = cached_id;
                        h_cache_offset[idx] = (i * 4 + order_id) * capacity_ + h_size[order_id];
                        h_size[h_order[oidx]] += 1;
                        break;
                    }
                }
            }
            for(int32_t j = 0; j < 4; j++){
                std::cout<<"cache node num "<<h_size[j]<<"\n";
            } 
            for(int32_t j = 0; j < 4; j++){
                cudaMemcpy(d_cache_ids[i * 4 + j], &h_cache_ids[0], int64_t(int64_t(capacity_ * 4) * sizeof(int32_t)), cudaMemcpyHostToDevice);
                cudaMemcpy(d_cache_offset[i * 4 + j], &h_cache_offset[0], int64_t(int64_t(capacity_ * 4) * sizeof(int32_t)), cudaMemcpyHostToDevice);
                cudaCheckError();
            }

            cudaCheckError();
        }
        cudaCheckError();

    }else if(cache_agg_mode == 3){//nvswitch
            cudaSetDevice(0);
            int32_t* cache_order;
            cudaMalloc(&cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
            cudaCheckError();

            cudaCheckError();
            unsigned long long int* agg_access_time;
            cudaMalloc(&agg_access_time, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
            cudaMemset(agg_access_time, 0, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
            cudaCheckError();

            std::vector<unsigned long long int*> h_access_time;
            h_access_time.resize(TOTAL_DEV_NUM);
            for(int32_t j = 0; j < TOTAL_DEV_NUM; j++){
                h_access_time[j] = (unsigned long long int*)malloc(int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
                aggregate_access<<<block_num, thread_num>>>(agg_access_time, access_time[j], total_num_nodes);
                cudaCheckError();

                cudaMemcpy(h_access_time[j], access_time[j], int64_t(int64_t(total_num_nodes)) * sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
                cudaCheckError();

            }

            for(int32_t j = 0; j < TOTAL_DEV_NUM; j++){
                cudaSetDevice(j);
                cudaFree(access_time[j]);
                cudaCheckError();
            }
            cudaSetDevice(0);
            cudaCheckError();
            init_cache_order<<<block_num, thread_num>>>(cache_order, total_num_nodes);
            // thrust::sort_by_key(thrust::device, agg_access_time, agg_access_time + total_num_nodes, cache_order, thrust::greater<unsigned long long int>());
            cudaCheckError();
            int32_t* h_cache_order = (int32_t*)malloc(int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
            cudaMemcpy(h_cache_order, cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)), cudaMemcpyDeviceToHost);
            cudaCheckError();
            cudaSetDevice(0);
            cudaFree(cache_order);
            cudaFree(agg_access_time);
            std::vector<int32_t> h_cache_ids(capacity_ * TOTAL_DEV_NUM, -1);
            std::vector<int32_t> h_cache_offset(capacity_ * TOTAL_DEV_NUM, -1);
            std::vector<int32_t> h_size(TOTAL_DEV_NUM, 0);
            std::vector<int32_t> h_order(TOTAL_DEV_NUM, -1);

            for(int32_t idx = 0; idx < capacity_ * TOTAL_DEV_NUM; idx ++){
                int32_t cached_id = h_cache_order[idx];

                for(int32_t oidx = 0; oidx < TOTAL_DEV_NUM; oidx++){
                    unsigned long long int acc_max = 0;
                    for(int32_t didx = 0; didx < TOTAL_DEV_NUM; didx ++){
                        unsigned long long int acc_temp = h_access_time[didx][cached_id];
                        if(acc_temp > acc_max){
                            acc_max = acc_temp;
                            h_order[oidx] = didx;
                        }
                    }
                    h_access_time[h_order[oidx]][cached_id] = 0;
                }

                // if(h_access_time[0][cached_id] > h_access_time[1][cached_id]){
                //     h_order[0] = 0;
                //     h_order[1] = 1;
                // }else{
                //     h_order[1] = 0;
                //     h_order[0] = 1;
                // }
                for(int32_t oidx = 0; oidx < TOTAL_DEV_NUM; oidx++){
                    int32_t order_id = h_order[oidx];
                    if(h_size[order_id] < capacity_){
                        h_cache_ids[idx] = cached_id;
                        h_cache_offset[idx] = (order_id) * capacity_ + h_size[order_id];
                        h_size[h_order[oidx]] += 1;
                        break;
                    }
                }
            }
            for(int32_t j = 0; j < TOTAL_DEV_NUM; j++){
                std::cout<<"cache node num "<<h_size[j]<<"\n";
            } 
            for(int32_t j = 0; j < TOTAL_DEV_NUM; j++){
                cudaMemcpy(d_cache_ids[j], &h_cache_ids[0], int64_t(int64_t(capacity_ * TOTAL_DEV_NUM) * sizeof(int32_t)), cudaMemcpyHostToDevice);
                cudaMemcpy(d_cache_offset[j], &h_cache_offset[0], int64_t(int64_t(capacity_ * TOTAL_DEV_NUM) * sizeof(int32_t)), cudaMemcpyHostToDevice);
                cudaCheckError();
            }

            cudaCheckError();
    }

    for(int32_t i = 0; i < TOTAL_DEV_NUM; i++){
        int cache_expand;
        if(cache_agg_mode == 0){
            cache_expand = 1;
        }else if(cache_agg_mode == 1){
            cache_expand = 2;
        }else if(cache_agg_mode == 2){
            cache_expand = 4;
        }else if(cache_agg_mode == 3){
            cache_expand = 8;
        }
        cache_controller_[i]->Update(cache_expand);
    }

    graph->GraphCache(edge_access_time, TOTAL_DEV_NUM);
    std::cout<<"finish load topo cache\n";
    is_presc_ = false;

    d_float_feature_cache_ptr_.resize(TOTAL_DEV_NUM);

    for(int32_t i = 0; i < TOTAL_DEV_NUM; i++){
        cudaSetDevice(i);
        float** new_ptr;
        cudaMalloc(&new_ptr, TOTAL_DEV_NUM * sizeof(float*));
        d_float_feature_cache_ptr_[i] = new_ptr;
    }

    for(int32_t i = 0; i < TOTAL_DEV_NUM; i++){
        int32_t dev_id = i;//device[i];
        dev_ids_[dev_id] = true;

        if(float_attr_len_ > 0){
            cudaSetDevice(dev_id);
            float* new_float_feature_cache;
            cudaMalloc(&new_float_feature_cache, int64_t(int64_t(int64_t(capacity_) * float_attr_len_) * sizeof(float)));
            float_feature_cache_[dev_id] = new_float_feature_cache;
            init_feature_cache<<<1,1>>>(d_float_feature_cache_ptr_[0], new_float_feature_cache, dev_id);
            cudaCheckError();
        }
    }
    for(int32_t i = 0; i < TOTAL_DEV_NUM; i++){
        int32_t dev_id = i;
        cudaSetDevice(dev_id);
        cudaMemcpy(d_float_feature_cache_ptr_[dev_id], d_float_feature_cache_ptr_[0], TOTAL_DEV_NUM * sizeof(float**), cudaMemcpyDeviceToDevice);
        cudaCheckError();
    }
    cudaDeviceSynchronize();
    std::cout<<"Finish load presc feature cache\n";

}

int32_t* GPUCache::FutureBatch(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return cache_controller_[dev_id]->FutureBatch();
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

int32_t* GPUCache::AllCachedIds(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return cache_controller_[dev_id]->AllCachedIds();
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

int32_t* GPUCache::RecentMark(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return cache_controller_[dev_id]->RecentMark();
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

float* GPUCache::Float_Feature_Cache(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return float_feature_cache_[dev_id];
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

float** GPUCache::Global_Float_Feature_Cache(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return d_float_feature_cache_ptr_[dev_id];
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

int64_t* GPUCache::Int_Feature_Cache(int32_t dev_id)
{
    if(dev_ids_[dev_id] == true){
        return int_feature_cache_[dev_id];
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}

int32_t GPUCache::K_Batch()
{
    return k_batch_;
}

int32_t GPUCache::MaxIdNum(int32_t dev_id){
    if(dev_ids_[dev_id] == true){
        return cache_controller_[dev_id]->MaxIdNum();
    }else{
        std::cout<<"invalid device for cache\n";
        return 0;
    }
}

unsigned long long int* GPUCache::GetEdgeAccessedMap(int32_t dev_id){
    if(dev_ids_[dev_id] == true){
        return cache_controller_[dev_id]->GetEdgeAccessedMap();
    }else{
        std::cout<<"invalid device for cache\n";
        return nullptr;
    }
}
