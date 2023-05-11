#include "GPU_Graph_Storage.cuh"
#include <iostream>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__global__ void assign_memory(int32_t** int32_pptr, int32_t* int32_ptr, int64_t** int64_pptr, int64_t* int64_ptr, int32_t device_id){
    int32_pptr[device_id] = int32_ptr;
    int64_pptr[device_id] = int64_ptr;
}


__global__ void GetNeighborCount(int32_t* QT, int32_t Kg, int32_t Ki, int32_t capacity, int64_t* csr_node_index_cpu, int64_t* neighbor_count){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity; thread_idx += gridDim.x * blockDim.x){
        int32_t cache_id = QT[thread_idx * Kg + Ki];
        int64_t count = csr_node_index_cpu[cache_id + 1] - csr_node_index_cpu[cache_id];
        neighbor_count[thread_idx] = count;
    }
}

__global__ void TopoFillUp(int32_t* QT, int32_t Kg, int32_t Ki, int32_t capacity, 
                            int64_t* csr_node_index_cpu, int32_t* csr_dst_node_ids_cpu, 
                             int64_t* d_csr_node_index, int32_t* d_csr_dst_node_ids){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity; thread_idx += gridDim.x * blockDim.x){
        int32_t cache_id = QT[thread_idx * Kg + Ki];
        int64_t count = csr_node_index_cpu[cache_id + 1] - csr_node_index_cpu[cache_id];
        for(int i = 0; i < count; i++){
            int32_t neighbor_id = csr_dst_node_ids_cpu[csr_node_index_cpu[cache_id] + i];
            int64_t start_off = d_csr_node_index[thread_idx];
            d_csr_dst_node_ids[start_off + i] = neighbor_id;
        }
    }
}

/*in this version, partition id = shard id = device id*/
class GPUMemoryGraphStorage : public GPUGraphStorage {
public:
    GPUMemoryGraphStorage() {
    }

    virtual ~GPUMemoryGraphStorage() {
    }

    void Build(BuildInfo* info) override {
        int32_t partition_count = info->partition_count;
        partition_count_ = partition_count;
        node_num_ = info->total_num_nodes;
        edge_num_ = info->total_edge_num;
        cache_edge_num_ = info->cache_edge_num;

        // shard count == partition count now
        csr_node_index_.resize(partition_count_);
        csr_dst_node_ids_.resize(partition_count_);
        partition_index_.resize(partition_count_);
        partition_offset_.resize(partition_count_);

        d_global_count_.resize(partition_count);
        h_global_count_.resize(partition_count);
        h_cache_hit_.resize(partition_count);
        find_iter_.resize(partition_count);
        h_batch_size_.resize(partition_count);

        for(int32_t i = 0; i < partition_count; i++){
            cudaSetDevice(i);
            cudaMalloc(&csr_node_index_[i], (partition_count + 1) * sizeof(int64_t*));
            cudaMalloc(&csr_dst_node_ids_[i], (partition_count + 1) * sizeof(int32_t*));
            cudaMalloc(&d_global_count_[i], 4);
            h_global_count_[i] = (int32_t*)malloc(4);
            h_cache_hit_[i] = 0;
            find_iter_[i] = 0;
            h_batch_size_[i] = 0;
        }

        src_size_.resize(partition_count);
        dst_size_.resize(partition_count);
        cudaCheckError();

        cudaSetDevice(0);

        int64_t* pin_csr_node_index;
        int32_t* pin_csr_dst_node_ids;

        h_csr_node_index_ = info->csr_node_index;
        h_csr_dst_node_ids_ = info->csr_dst_node_ids;
        
        cudaHostGetDevicePointer(&pin_csr_node_index, h_csr_node_index_, 0);
        cudaHostGetDevicePointer(&pin_csr_dst_node_ids, h_csr_dst_node_ids_, 0);
        assign_memory<<<1,1>>>(csr_dst_node_ids_[0], pin_csr_dst_node_ids, csr_node_index_[0], pin_csr_node_index, partition_count);
        cudaCheckError();

        csr_node_index_cpu_ = pin_csr_node_index;
        csr_dst_node_ids_cpu_ = pin_csr_dst_node_ids;
        
    }
    

    void GraphCache(int32_t* QT, int32_t Ki, int32_t Kg, int32_t capacity){
        cudaMemcpy(csr_node_index_[Ki * Kg], csr_node_index_[0], (partition_count_ + 1) * sizeof(int64_t*), cudaMemcpyDeviceToDevice);
        cudaCheckError();
        cudaMemcpy(csr_dst_node_ids_[Ki * Kg], csr_dst_node_ids_[0], (partition_count_ + 1) * sizeof(int32_t*), cudaMemcpyDeviceToDevice);
        cudaCheckError();
        for(int32_t i = 0; i < Kg; i++){
            cudaSetDevice(Ki * Kg + i);
            int64_t* neighbor_count;
            cudaMalloc(&neighbor_count, capacity * sizeof(int64_t));
            GetNeighborCount<<<80, 1024>>>(QT, Kg, i, capacity, csr_node_index_cpu_, neighbor_count);

            int64_t* d_csr_node_index;
            cudaMalloc(&d_csr_node_index, (int64_t(capacity + 1)*sizeof(int64_t)));
            cudaMemset(d_csr_node_index, 0, (int64_t(capacity + 1)*sizeof(int64_t)));
            thrust::inclusive_scan(thrust::device, neighbor_count, neighbor_count + capacity, d_csr_node_index + 1);
            cudaCheckError();
            int64_t* h_csr_node_index = (int64_t*)malloc((capacity + 1) * sizeof(int64_t));
            cudaMemcpy(h_csr_node_index, d_csr_node_index, (capacity + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);
            
            int32_t* d_csr_dst_node_ids;
            cudaMalloc(&d_csr_dst_node_ids, int64_t(int64_t(h_csr_node_index[capacity]) * sizeof(int32_t)));

            TopoFillUp<<<80, 1024>>>(QT, Kg, i, capacity, csr_node_index_cpu_, csr_dst_node_ids_cpu_, d_csr_node_index, d_csr_dst_node_ids);
            cudaCheckError();
    
            assign_memory<<<1,1>>>(csr_dst_node_ids_[Ki * Kg], d_csr_dst_node_ids, csr_node_index_[Ki * Kg], d_csr_node_index, Ki * Kg + i);
            cudaCheckError();
            cudaFree(neighbor_count);
        }
        for(int32_t i = 1; i < Kg; i++){
            cudaMemcpy(csr_node_index_[Ki * Kg + i], csr_node_index_[Ki * Kg], (partition_count_ + 1) * sizeof(int64_t*), cudaMemcpyDeviceToDevice);
            cudaCheckError();
            cudaMemcpy(csr_dst_node_ids_[Ki * Kg + i], csr_dst_node_ids_[Ki * Kg], (partition_count_ + 1) * sizeof(int32_t*), cudaMemcpyDeviceToDevice);
            cudaCheckError();
        }
    }

    void Finalize() override {
        cudaFreeHost(csr_node_index_cpu_);
        cudaFreeHost(csr_dst_node_ids_cpu_);
        // for(int32_t i = 0; i < partition_count_; i++){
        //     cudaFree(partition_index_[i]);
        //     cudaFree(partition_offset_[i]);
        // }
    }

    //CSR
    int32_t GetPartitionCount() const override {
        return partition_count_;
    }
	int64_t** GetCSRNodeIndex(int32_t dev_id) const override {
		return csr_node_index_[dev_id];
	}
	int32_t** GetCSRNodeMatrix(int32_t dev_id) const override {
        return csr_dst_node_ids_[dev_id];
    }
    
    int64_t* GetCSRNodeIndexCPU() const override {
        return csr_node_index_cpu_;
    }

    int32_t* GetCSRNodeMatrixCPU() const override {
        return csr_dst_node_ids_cpu_;
    }

    int64_t Src_Size(int32_t part_id) const override {
        return src_size_[part_id];
    }
    int64_t Dst_Size(int32_t part_id) const override {
        return dst_size_[part_id];
    }
    char* PartitionIndex(int32_t dev_id) const override {
        return partition_index_[dev_id];
    }
    int32_t* PartitionOffset(int32_t dev_id) const override {
        return partition_offset_[dev_id];
    }

private:
    std::vector<int64_t> src_size_;	
	std::vector<int64_t> dst_size_;

    int32_t node_num_;
    int64_t edge_num_;
    int64_t cache_edge_num_;

	//CSR graph, every partition has a ptr copy
    int32_t partition_count_;
	std::vector<int64_t**> csr_node_index_;
	std::vector<int32_t**> csr_dst_node_ids_;	
    int64_t* csr_node_index_cpu_;
    int32_t* csr_dst_node_ids_cpu_;

    int64_t* h_csr_node_index_;
    int32_t* h_csr_dst_node_ids_;

    std::vector<char*> partition_index_;
    std::vector<int32_t*> partition_offset_;

    std::vector<int32_t*> h_global_count_;
    std::vector<int32_t*> d_global_count_;


    std::vector<int32_t> find_iter_;
    std::vector<int32_t> h_cache_hit_;
    std::vector<int32_t> h_batch_size_;
};

extern "C" 
GPUGraphStorage* NewGPUMemoryGraphStorage(){
    GPUMemoryGraphStorage* ret = new GPUMemoryGraphStorage();
    return ret;
}
