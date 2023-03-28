#include "GPU_Graph_Storage.cuh"
#include <iostream>

#define DEVCOUNT 1
using index_pair_type = bght::pair<int32_t, char>;
using offset_pair_type = bght::pair<int32_t, int32_t>;

__global__ void InitIndexPair(index_pair_type* pair, int32_t* cache_ids, char* cache_index, int32_t capacity){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity; thread_idx += gridDim.x * blockDim.x){
        pair[thread_idx].first = cache_ids[thread_idx];
        pair[thread_idx].second = cache_index[thread_idx];
    }
}

__global__ void InitOffsetPair(offset_pair_type* pair, int32_t* cache_ids, int32_t* cache_offset, int32_t capacity){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity; thread_idx += gridDim.x * blockDim.x){
        pair[thread_idx].first = cache_ids[thread_idx];
        pair[thread_idx].second = cache_offset[thread_idx];
    }
}


__global__ void assign_memory(int32_t** int32_pptr, int32_t* int32_ptr, int64_t** int64_pptr, int64_t* int64_ptr, int32_t device_id){
    int32_pptr[device_id] = int32_ptr;
    int64_pptr[device_id] = int64_ptr;
}

__global__ void agg_acc(unsigned long long int* agg_access_time, unsigned long long int* new_access_time, int32_t total_num_nodes){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (total_num_nodes); thread_idx += blockDim.x * gridDim.x){
        agg_access_time[thread_idx] += new_access_time[thread_idx];
    }
}

__global__ void init_co(int32_t* cache_order, int32_t total_num_nodes){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (total_num_nodes); thread_idx += blockDim.x * gridDim.x){
        cache_order[thread_idx] = thread_idx;
    }
}

__global__ void cache_hit(char* partition_index, int32_t batch_size, int32_t* global_count){
    __shared__ int32_t local_count[1];
    if(threadIdx.x == 0){
        local_count[0] = 0;
    }
    __syncthreads();
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (batch_size); thread_idx += blockDim.x * gridDim.x){
        int32_t offset = partition_index[thread_idx];
        if(offset >= 0){
            atomicAdd(local_count, 1);
        }
    }
    __syncthreads();
    if(threadIdx.x == 0){
        atomicAdd(global_count, local_count[0]);
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

        index_map_.resize(partition_count);
        offset_map_.resize(partition_count);

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
    

    void GraphCache(std::vector<unsigned long long int*> &access_time, int32_t device_count){
        
        dim3 block_num(80,1);
        dim3 thread_num(1024,1);
        cudaSetDevice(0);
        int32_t* cache_order;
        cudaMalloc(&cache_order, int64_t(int64_t(node_num_) * sizeof(int32_t)));
        cudaCheckError();
        unsigned long long int* agg_access_time;
        cudaMalloc(&agg_access_time, int64_t(int64_t(node_num_) * sizeof(unsigned long long int)));
        cudaMemset(agg_access_time, 0, int64_t(int64_t(node_num_) * sizeof(unsigned long long int)));
        std::vector<unsigned long long int*> h_access_time;
        h_access_time.resize(DEVCOUNT);
        for(int32_t j = 0; j < DEVCOUNT; j++){
            h_access_time[j] = (unsigned long long int*)malloc(int64_t(int64_t(node_num_) * sizeof(unsigned long long int)));
            agg_acc<<<block_num, thread_num>>>(agg_access_time, access_time[j], node_num_);
            cudaMemcpy(h_access_time[j], access_time[j], int64_t(int64_t(node_num_) * sizeof(unsigned long long int)), cudaMemcpyDeviceToHost);
        }

        for(int32_t j = 0; j < DEVCOUNT; j++){
                cudaSetDevice(j);
                // cudaFree(access_time[i]);
                // cudaCheckError();
                cudaFree(access_time[j]);
                cudaCheckError();
        }
        cudaSetDevice(0);

        cudaCheckError();
        init_co<<<block_num, thread_num>>>(cache_order, node_num_);
        // thrust::sort_by_key(thrust::device, agg_access_time, agg_access_time + node_num_, cache_order, thrust::greater<unsigned long long int>());
        cudaCheckError();
        int32_t* h_cache_order = (int32_t*)malloc(int64_t(int64_t(node_num_) * sizeof(int32_t)));
        cudaMemcpy(h_cache_order, cache_order, int64_t(int64_t(node_num_) * sizeof(int32_t)), cudaMemcpyDeviceToHost);
        cudaFree(cache_order);
        cudaFree(agg_access_time);

        std::vector<int32_t> h_partition_offset(node_num_, -1);
        std::vector<char>    h_partition_index(node_num_, -1);
        std::vector<int32_t> h_node_ids(node_num_, -1);
        std::vector<int64_t> h_e_size(DEVCOUNT, 0);
        std::vector<int32_t> h_n_size(DEVCOUNT, 0);
        std::vector<int32_t> h_order(DEVCOUNT, -1);
        int64_t total_cache_edge = 0;
        int64_t total_cache_node = 0;

        std::vector<std::vector<int64_t>> csr_node_index;
        std::vector<std::vector<int32_t>> csr_dst_node_ids;
        csr_node_index.resize(DEVCOUNT);
        csr_dst_node_ids.resize(DEVCOUNT);
        int32_t idx = 0;
        // for( ; idx < node_num_; idx ++){
        //     if(total_cache_edge >= cache_edge_num_ * DEVCOUNT){
        //         break;
        //     }
        //     int32_t cached_id = h_cache_order[idx];

        //     for(int32_t oidx = 0; oidx < DEVCOUNT; oidx++){
        //         unsigned long long int acc_max = 0;
        //         for(int32_t didx = 0; didx < DEVCOUNT; didx ++){
        //             unsigned long long int acc_temp = h_access_time[didx][cached_id];
        //             if(acc_temp > acc_max){
        //                 acc_max = acc_temp;
        //                 h_order[oidx] = didx;
        //             }
        //         }
        //         h_access_time[h_order[oidx]][cached_id] = 0;
        //     }

        //     // std::cout<<"\n";
        //     for(int32_t oidx = 0; oidx < DEVCOUNT; oidx++){
        //         if(h_e_size[h_order[oidx]] < cache_edge_num_){
        //             h_node_ids[idx] = cached_id;
        //             h_partition_index[idx] = h_order[oidx];
        //             h_partition_offset[idx] = h_n_size[h_order[oidx]]; 
        //             int32_t neighbor_count = h_csr_node_index_[cached_id + 1] - h_csr_node_index_[cached_id];

        //             csr_node_index[h_order[oidx]].push_back(h_e_size[h_order[oidx]]);
        //             for(int32_t nid = 0; nid < neighbor_count; nid++){
        //                 csr_dst_node_ids[h_order[oidx]].push_back(h_csr_dst_node_ids_[h_csr_node_index_[cached_id] + nid]);
        //             }

        //             h_e_size[h_order[oidx]] += neighbor_count;
        //             h_n_size[h_order[oidx]] += 1;
        //             total_cache_edge += neighbor_count;                        
        //             total_cache_node += 1;
        //             break;
        //         }
        //     }
        // }

        for( ; idx < node_num_; idx ++){
            if(total_cache_edge >= cache_edge_num_ * DEVCOUNT){
                break;
            }
            int32_t cached_id = h_cache_order[idx];
            h_node_ids[idx] = cached_id;
            h_partition_index[idx] = idx % DEVCOUNT;
            h_partition_offset[idx] = h_n_size[idx % DEVCOUNT]; 
            int32_t neighbor_count = h_csr_node_index_[cached_id + 1] - h_csr_node_index_[cached_id];

            csr_node_index[idx % DEVCOUNT].push_back(h_e_size[idx % DEVCOUNT]);
            for(int32_t nid = 0; nid < neighbor_count; nid++){
                csr_dst_node_ids[idx % DEVCOUNT].push_back(h_csr_dst_node_ids_[h_csr_node_index_[cached_id] + nid]);
            }

            h_e_size[idx % DEVCOUNT] += neighbor_count;
            h_n_size[idx % DEVCOUNT] += 1;
            total_cache_edge += neighbor_count;                        
            total_cache_node += 1;
        }

        std::cout<<"total cache nodes "<<total_cache_node<<" "<<idx<<"\n";

        for(int32_t didx = 0; didx < DEVCOUNT; didx ++){
            csr_node_index[didx].push_back(h_e_size[didx]);
            std::cout<<csr_node_index[didx].size()<<" "<<csr_dst_node_ids[didx].size()<<" "<<h_e_size[didx]<<"\n";
        }

        for(int32_t j = 0; j < DEVCOUNT; j++){
            // std::cout<<"Initialize Topo Cache 1\n";
            cudaSetDevice(j);
            index_map_[j] = new bght::bcht<int32_t, char>((total_cache_node * 2), -1, -1);
            cudaCheckError();

            offset_map_[j] = new bght::bcht<int32_t, int32_t>((total_cache_node * 2), -1, -1);
            // std::cout<<"Initialize Topo Cache 2\n";

            index_pair_type* index_pair;
            offset_pair_type* offset_pair;
            cudaMalloc(&index_pair, int64_t(int64_t(total_cache_node) * sizeof(index_pair_type)));
            cudaCheckError();
            cudaMalloc(&offset_pair, int64_t(int64_t(total_cache_node) * sizeof(offset_pair_type)));
            cudaCheckError();
            // std::cout<<"Initialize Topo Cache 3\n";

            int32_t* d_cache_ids;
            cudaMalloc(&d_cache_ids, int64_t(int64_t(total_cache_node) * sizeof(int32_t)));
            cudaMemcpy(d_cache_ids, &h_node_ids[0], int64_t(int64_t(total_cache_node) * sizeof(int32_t)), cudaMemcpyHostToDevice);
            char* d_cache_index;
            cudaMalloc(&d_cache_index, int64_t(int64_t(total_cache_node) * sizeof(char)));
            cudaMemcpy(d_cache_index, &h_partition_index[0], int64_t(int64_t(total_cache_node) * sizeof(char)), cudaMemcpyHostToDevice);

            int32_t* d_cache_offset;
            cudaMalloc(&d_cache_offset, int64_t(int64_t(total_cache_node) * sizeof(int32_t)));
            cudaMemcpy(d_cache_offset, &h_partition_offset[0], int64_t(int64_t(total_cache_node) * sizeof(int32_t)), cudaMemcpyHostToDevice);
            // std::cout<<"Initialize Topo Cache 4\n";

            InitIndexPair<<<block_num, thread_num>>>(index_pair, d_cache_ids, d_cache_index, total_cache_node);
            InitOffsetPair<<<block_num, thread_num>>>(offset_pair, d_cache_ids, d_cache_offset, total_cache_node);
            cudaStream_t stream;
            cudaStreamCreate(&stream);
            cudaCheckError();
            // std::cout<<"Initialize Topo Cache 5\n";

            index_map_[j]->insert(index_pair, (index_pair + total_cache_node), stream);
            cudaCheckError();
            // std::cout<<"Initialize Topo Cache 5.1\n";

            offset_map_[j]->insert(offset_pair, (offset_pair + total_cache_node), stream);
            // std::cout<<"Initialize Topo Cache 5.2\n";

            cudaCheckError();
            cudaDeviceSynchronize();
            cudaFree(index_pair);
            cudaFree(offset_pair);
            cudaFree(d_cache_ids);
            cudaFree(d_cache_index);
            cudaFree(d_cache_offset);
            // std::cout<<"Initialize Topo Cache 6\n";

            int64_t* d_csr_node_index;
            cudaMalloc(&d_csr_node_index, (int64_t(h_n_size[j] + 1))*sizeof(int64_t));
            cudaCheckError();
            cudaMemcpy(d_csr_node_index, &csr_node_index[j][0], (int64_t(h_n_size[j] + 1))*sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaCheckError();
            int32_t* d_csr_dst_node_ids;
            cudaMalloc(&d_csr_dst_node_ids, int64_t(int64_t(h_e_size[j]) * sizeof(int32_t)));
            cudaCheckError();
            cudaMemcpy(d_csr_dst_node_ids, &csr_dst_node_ids[j][0], int64_t(int64_t((h_e_size[j]) * sizeof(int32_t))), cudaMemcpyHostToDevice);
            cudaCheckError();
            // std::cout<<"Initialize Topo Cache 7\n";

            assign_memory<<<1,1>>>(csr_dst_node_ids_[0], d_csr_dst_node_ids, csr_node_index_[0], d_csr_node_index, j);
            cudaCheckError();
        }
        // std::cout<<"Initialize Topo Cache 8\n";

        for(int32_t didx = 0; didx < DEVCOUNT; didx ++){
            csr_node_index[didx].clear();
            csr_dst_node_ids[didx].clear();
            free(h_access_time[didx]);
        }
        free(h_cache_order);
        h_partition_offset.clear();
        h_partition_index.clear();
        // std::cout<<"Initialize Topo Cache 9\n";

        for(int32_t j = 1; j < partition_count_; j++){
            cudaMemcpy(csr_node_index_[j], csr_node_index_[0], (partition_count_ + 1) * sizeof(int64_t*), cudaMemcpyDeviceToDevice);
            cudaCheckError();
            cudaMemcpy(csr_dst_node_ids_[j], csr_dst_node_ids_[0], (partition_count_ + 1) * sizeof(int32_t*), cudaMemcpyDeviceToDevice);
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

    void Find(int32_t* input_ids, char* partition_index, int32_t* partition_offset, int32_t batch_size, int32_t device_id, int32_t op_id, cudaStream_t strm_hdl) override {
        index_map_[device_id]->find(input_ids, input_ids + batch_size, partition_index, strm_hdl);
        offset_map_[device_id]->find(input_ids, input_ids + batch_size, partition_offset, strm_hdl);
        
        if(find_iter_[device_id] % 500 == 0){
            cudaMemsetAsync(d_global_count_[device_id], 0, 4, strm_hdl);
            dim3 block_num(48, 1);
            dim3 thread_num(1024, 1);
            cache_hit<<<block_num, thread_num, 0, strm_hdl>>>(partition_index, batch_size, d_global_count_[device_id]);
            cudaMemcpy(h_global_count_[device_id], d_global_count_[device_id], 4, cudaMemcpyDeviceToHost);
            h_cache_hit_[device_id] += ((h_global_count_[device_id])[0]);
            h_batch_size_[device_id] += batch_size;
            if(op_id == 4){
                std::cout<<device_id<<" Topo Cache Hit: "<<h_cache_hit_[device_id]<<" "<<(h_cache_hit_[device_id] * 1.0 / h_batch_size_[device_id])<<"\n";    
                h_cache_hit_[device_id] = 0;
                h_batch_size_[device_id] = 0;
            }
        }
        if(op_id == 4){
            find_iter_[device_id] += 1;
        }
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

    std::vector<bght::bcht<int32_t, char>*> index_map_;
    std::vector<bght::bcht<int32_t, int32_t>*> offset_map_;

    std::vector<int32_t> find_iter_;
    std::vector<int32_t> h_cache_hit_;
    std::vector<int32_t> h_batch_size_;
};

extern "C" 
GPUGraphStorage* NewGPUMemoryGraphStorage(){
    GPUMemoryGraphStorage* ret = new GPUMemoryGraphStorage();
    return ret;
}
