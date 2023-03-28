#include "GPU_Node_Storage.cuh"
#include <iostream>

#include <cuda.h>
#include <nvm_ctrl.h>
#include <nvm_types.h>
#include <nvm_queue.h>
#include <nvm_util.h>
#include <nvm_admin.h>
#include <nvm_error.h>
#include <nvm_cmd.h>
#include <string>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <ctrl.h>
#include <buffer.h>

#include <event.h>
#include <queue.h>
#include <nvm_parallel_queue.h>
#include <nvm_io.h>
#include <page_cache.h>
#include <util.h>
#include <iostream>
#include <fstream>
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

//uint32_t n_ctrls = 1;
const char* const sam_ctrls_paths[] = {"/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm4", "/dev/libnvm9", "/dev/libnvm2", "/dev/libnvm3", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7", "/dev/libnvm8", "/dev/libnvm10", "/dev/libnvm11"};
const char* const intel_ctrls_paths[] = {"/dev/libinvm0", "/dev/libinvm1", "/dev/libinvm4", "/dev/libinvm9", "/dev/libinvm2", "/dev/libinvm3", "/dev/libinvm5", "/dev/libinvm6", "/dev/libinvm7", "/dev/libinvm8"};


__global__ void zero_copy_with_bam(
	array_d_t<float>* bam_float_attrs, float** cache_float_attrs, int32_t float_attr_len,
	int32_t* sampled_ids, int32_t* cache_index, int32_t cache_capacity,
	int32_t* node_counter, float* dst_float_buffer,
	int32_t total_num_nodes,
	int32_t dev_id,
	int32_t op_id)
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
	int32_t gidx;//global cache index
	int32_t fidx;//local cache index
	int32_t didx;//device index
	int32_t foffset;
	if(float_attr_len > 0){
		for(int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < (int64_t(batch_size) * float_attr_len); thread_idx += blockDim.x * gridDim.x){
			gidx = -1;//(cache_index[thread_idx / float_attr_len]);
			didx = gidx / cache_capacity;
			fidx = gidx % cache_capacity;
			foffset = thread_idx % float_attr_len;
			if(gidx < 0){/*cache miss*/
				fidx = sampled_ids[node_off + (thread_idx / float_attr_len)];
				if(fidx >= 0){
					dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_attr_len)) + thread_idx)] = (*bam_float_attrs)[int64_t(int64_t(int64_t(fidx%total_num_nodes) * float_attr_len) + foffset)];
				}
			}else{/*cache hit, find global position*/
				dst_float_buffer[int64_t(int64_t((int64_t(node_off) * float_attr_len)) + thread_idx)] = cache_float_attrs[didx][int64_t(int64_t(int64_t(fidx) * float_attr_len) + foffset)];
			}
		}
	}
}


__global__ void zero_copy_with_bam_para(
	array_d_t<float>* bam_float_attrs, int32_t float_attr_len,
	int32_t* sampled_ids, float* dst_float_buffer,
	int32_t batch_size, int32_t node_off,
	int32_t total_num_nodes,
	int32_t op_id)
{
	int32_t fidx;
	int32_t foffset;
    int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(thread_idx < (int64_t(batch_size) * float_attr_len)){
        foffset = thread_idx % float_attr_len;
        fidx = sampled_ids[node_off + (thread_idx / float_attr_len)];
        int64_t dst_offset = ((int64_t(node_off) * float_attr_len)) + thread_idx;
        int64_t src_offset = (int64_t(fidx%total_num_nodes) * float_attr_len) + foffset;
        if(fidx >= 0){
            dst_float_buffer[dst_offset] = (*bam_float_attrs)[src_offset];
        }
    }
}

__global__ void zero_copy_with_bam_para2(
	array_d_t<float>* bam_float_attrs, int32_t float_attr_len,
	int32_t* sampled_ids, float* dst_float_buffer,
	int32_t batch_size, int32_t node_off,
	int32_t total_num_nodes,
	int32_t op_id)
{
	int32_t fidx;
    int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(thread_idx < (int64_t(batch_size))){
        fidx = sampled_ids[node_off + (thread_idx)];
        if(fidx >= 0){
            int64_t dst_offset = ((int64_t(node_off) * float_attr_len)) + thread_idx * float_attr_len;
            int64_t src_offset = (int64_t(fidx%total_num_nodes) * float_attr_len);
            dst_float_buffer[dst_offset] = (*bam_float_attrs)[src_offset];
            // for(int32_t foffset = 0; foffset < float_attr_len; foffset++){
            //     dst_float_buffer[dst_offset + foffset] = (*bam_float_attrs)[src_offset + foffset];
            // }
        }
    }
}

__global__ void zero_copy_with_bam_request(
	array_d_t<float>* bam_float_attrs, int32_t float_attr_len,
	int32_t* sampled_ids, float* d_req_count,
	int32_t batch_size, int32_t node_off,
	int32_t total_num_nodes,
	int32_t op_id)
{
	int32_t fidx;
    int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(thread_idx < (int64_t(batch_size))){
        fidx = sampled_ids[node_off + (thread_idx)];
        if(fidx >= 0){
            int64_t dst_offset = ((int64_t(node_off) * float_attr_len)) + thread_idx * float_attr_len;
            int64_t src_offset = (int64_t(fidx%total_num_nodes) * float_attr_len);
            d_req_count[0] += (*bam_float_attrs)[src_offset];
            // for(int32_t foffset = 0; foffset < float_attr_len; foffset++){
            //     dst_float_buffer[dst_offset + foffset] = (*bam_float_attrs)[src_offset + foffset];
            // }
        }
    }
}

__global__ void zero_copy_with_bam_recieve(
	array_d_t<float>* bam_float_attrs, int32_t float_attr_len,
	int32_t* sampled_ids, float* dst_float_buffer,
	int32_t batch_size, int32_t node_off,
	int32_t total_num_nodes,
	int32_t op_id)
{
	int32_t fidx;
    int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(thread_idx < (int64_t(batch_size) * float_attr_len)){
        fidx = sampled_ids[node_off + (thread_idx / float_attr_len)];
        if(fidx >= 0){
            int64_t dst_offset = ((int64_t(node_off) * float_attr_len)) + thread_idx;
            int64_t src_offset = (int64_t(fidx%total_num_nodes) * float_attr_len);
            int32_t foffset = thread_idx % float_attr_len;
            dst_float_buffer[dst_offset] = (*bam_float_attrs)[src_offset + foffset];
            // for(; foffset < float_attr_len; foffset++){
            //     dst_float_buffer[dst_offset + foffset] = (*bam_float_attrs)[src_offset + foffset];
            // }
        }
    }
}

__global__ void zero_copy_with_bam_warp(
	array_d_t<float>* bam_float_attrs, int32_t float_attr_len,
	int32_t* sampled_ids, float* dst_float_buffer,
	int32_t batch_size, int32_t node_off,
	int32_t total_num_nodes,
	int32_t op_id)
{
	int32_t fidx;
    int64_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(thread_idx < (int64_t(batch_size) * 32)){
        fidx = sampled_ids[node_off + (thread_idx/32)];
        if(fidx >= 0){
            int64_t dst_offset = ((int64_t(node_off) * float_attr_len)) + (thread_idx / 32) * float_attr_len;
            int64_t src_offset = (int64_t(fidx%total_num_nodes) * float_attr_len);
            // dst_float_buffer[dst_offset] = (*bam_float_attrs)[src_offset];
            for(int32_t foffset = threadIdx.x & 31; foffset < float_attr_len; foffset+=32){
                dst_float_buffer[dst_offset + foffset] = (*bam_float_attrs)[src_offset + foffset];
            }
        }
    }
}

class GPUMemoryNodeStorage : public GPUNodeStorage{
public: 
    GPUMemoryNodeStorage(){
    }

    virtual ~GPUMemoryNodeStorage(){};

    void Build(BuildInfo* info) override {
        int32_t partition_count = info->partition_count;
        total_num_nodes_ = info->total_num_nodes;
        int_attr_len_ = info->int_attr_len;
        float_attr_len_ = info->float_attr_len;
        int64_t* host_int_attrs = info->host_int_attrs;
        float* host_float_attrs = info->host_float_attrs;

        if(int_attr_len_ > 0){
            cudaHostGetDevicePointer(&int_attrs_, host_int_attrs, 0);
        }
        if(float_attr_len_ > 0){
            cudaHostGetDevicePointer(&float_attrs_, host_float_attrs, 0);
        }
        cudaCheckError();

        cudaSetDevice(0);
        std::vector<Controller*> ctrls(info->n_ctrls);
        for (size_t i = 0 ; i < info->n_ctrls; i++){
            ctrls[i] = new Controller(info->ssdtype == 0 ? sam_ctrls_paths[i] : intel_ctrls_paths[i], info->nvmNamespace, info->cudaDevice, info->queueDepth, info->numQueues);
        }

        // uint64_t b_size = info->blkSize;//64;
        // uint64_t g_size = (info->numThreads + b_size - 1)/b_size;//80*16;
        // uint64_t n_threads = b_size * g_size;
        uint64_t page_size = info->pageSize;
        uint64_t n_pages = info->numPages;
        uint64_t total_cache_size = (page_size * n_pages);

        h_pc_ = new page_cache_t(page_size, n_pages, info->cudaDevice, ctrls[0][0], (uint64_t) 64, ctrls);
        std::cout << "finished creating bam cache\n";

        page_cache_t* d_pc = (page_cache_t*) (h_pc_->d_pc_ptr);
        uint64_t n_elems = info->numElems;
        std::cout<<"input page size "<<page_size<<"\n";
        std::cout<<"input num elem "<<n_elems<<"\n";
        std::cout<<"input num pages "<<n_pages<<"\n";
        uint64_t t_size = n_elems * sizeof(float);

        h_range_ = new range_t<float>((uint64_t)0, (uint64_t)n_elems, (uint64_t)0, (uint64_t)(t_size/page_size), (uint64_t)0, (uint64_t)page_size, h_pc_, info->cudaDevice);
        range_t<float>* d_range = (range_t<float>*) h_range_->d_range_ptr;

        std::vector<range_t<float>*> vr(1);
        vr[0] = h_range_;

        bam_float_attrs_ = new array_t<float>(n_elems, 0, vr, info->cudaDevice);

        std::cout << "finished creating range\n";


        training_set_num_.resize(partition_count);
        training_set_ids_.resize(partition_count);
        training_labels_.resize(partition_count);

        validation_set_num_.resize(partition_count);
        validation_set_ids_.resize(partition_count);
        validation_labels_.resize(partition_count);

        testing_set_num_.resize(partition_count);
        testing_set_ids_.resize(partition_count);
        testing_labels_.resize(partition_count);

        partition_count_ = partition_count;

        for(int32_t i = 0; i < info->shard_to_partition.size(); i++){
            int32_t part_id = info->shard_to_partition[i];
            int32_t device_id = info->shard_to_device[i];
            /*part id = 0, 1, 2...*/

            training_set_num_[part_id] = info->training_set_num[part_id];
            std::cout<<"Training set count "<<training_set_num_[part_id]<<" "<<info->training_set_num[part_id]<<"\n";

            validation_set_num_[part_id] = info->validation_set_num[part_id];
            testing_set_num_[part_id] = info->testing_set_num[part_id];

            cudaSetDevice(device_id);
            cudaCheckError();

            std::cout<<"Training set on device "<<part_id<<" "<<training_set_num_[part_id]<<"\n";
            // std::cout<<"Testing set on device "<<part_id<<" "<<testing_set_num_[part_id]<<"\n";

            int32_t* train_ids;
            cudaMalloc(&train_ids, training_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(train_ids, info->training_set_ids[part_id].data(), training_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            training_set_ids_[part_id] = train_ids;
            cudaCheckError();

            int32_t* valid_ids;
            cudaMalloc(&valid_ids, validation_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(valid_ids, info->validation_set_ids[part_id].data(), validation_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            validation_set_ids_[part_id] = valid_ids;
            cudaCheckError();

            int32_t* test_ids;
            cudaMalloc(&test_ids, testing_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(test_ids, info->testing_set_ids[part_id].data(), testing_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            testing_set_ids_[part_id] = test_ids;
            cudaCheckError();

            int32_t* train_labels;
            cudaMalloc(&train_labels, training_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(train_labels, info->training_labels[part_id].data(), training_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            training_labels_[part_id] = train_labels;
            cudaCheckError();

            int32_t* valid_labels;
            cudaMalloc(&valid_labels, validation_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(valid_labels, info->validation_labels[part_id].data(), validation_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            validation_labels_[part_id] = valid_labels;
            cudaCheckError();

            int32_t* test_labels;
            cudaMalloc(&test_labels, testing_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(test_labels, info->testing_labels[part_id].data(), testing_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            testing_labels_[part_id] = test_labels;
            cudaCheckError();

        }

        cudaMalloc(&d_req_count_, sizeof(float));
        cudaMemset(d_req_count_, 0, sizeof(float));
        cudaCheckError();

    };

    void Finalize() override {
        cudaFreeHost(float_attrs_);
        for(int32_t i = 0; i < partition_count_; i++){
            cudaSetDevice(i);
            cudaFree(training_set_ids_[i]);
            cudaFree(validation_set_ids_[i]);
            cudaFree(testing_set_ids_[i]);
            cudaFree(training_labels_[i]);
            cudaFree(validation_labels_[i]);
            cudaFree(testing_labels_[i]);
        }
    }

    int32_t* GetTrainingSetIds(int32_t part_id) const override {
        return training_set_ids_[part_id];
    }
    int32_t* GetValidationSetIds(int32_t part_id) const override {
        return validation_set_ids_[part_id];
    }
    int32_t* GetTestingSetIds(int32_t part_id) const override {
        return testing_set_ids_[part_id];
    }

	int32_t* GetTrainingLabels(int32_t part_id) const override {
        return training_labels_[part_id];
    };
    int32_t* GetValidationLabels(int32_t part_id) const override {
        return validation_labels_[part_id];
    }
    int32_t* GetTestingLabels(int32_t part_id) const override {
        return testing_labels_[part_id];
    }

    int32_t TrainingSetSize(int32_t part_id) const override {
        return training_set_num_[part_id];
    }
    int32_t ValidationSetSize(int32_t part_id) const override {
        return validation_set_num_[part_id];
    }
    int32_t TestingSetSize(int32_t part_id) const override {
        return testing_set_num_[part_id];
    }

    int32_t TotalNodeNum() const override {
        return total_num_nodes_;
    }
	int64_t* GetAllIntAttr() const override {
        return int_attrs_;
    }
    int32_t GetIntAttrLen() const override {
        return int_attr_len_;
    }
    float* GetAllFloatAttr() const override {
        return float_attrs_;
    }
    int32_t GetFloatAttrLen() const override {
        return float_attr_len_;
    }

    void Print(BuildInfo* info) override {
    }

    void GetBamFloatAttr(float** cache_float_attrs, int32_t float_attr_len,
                        int32_t* sampled_ids, int32_t* cache_index, int32_t cache_capacity,
                        int32_t* node_counter, float* dst_float_buffer,
                        int32_t total_num_nodes,
                        int32_t dev_id,
                        int32_t op_id, cudaStream_t strm_hdl) override {
        // dim3 block_num(1024, 1);
        // dim3 thread_num(512, 1);

        // // bam_float_attrs_ = (array_t<uint64_t>*)addr_;
        // // printf("in kernel %p\n", bam_float_attrs_);
        // // cudaCheckError();

        // // sequential_access_kernel<<<64, 1024>>>(bam_float_attrs_->d_array_ptr, 4096, d_req_count_, 1);
        // // cudaCheckError();

        // zero_copy_with_bam<<<block_num, thread_num, 0, (strm_hdl)>>>(
		// 	bam_float_attrs_->d_array_ptr, cache_float_attrs, float_attr_len,
		// 	sampled_ids, cache_index, cache_capacity,
		// 	node_counter, dst_float_buffer,
		// 	total_num_nodes,
		// 	dev_id, op_id
		// );
        // cudaCheckError();
        

        int* h_node_counter = (int*)malloc(64);
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
        // dim3 block_num((batch_size * float_attr_len - 1) / 1024 + 1, 1);
        // dim3 thread_num(1024, 1);

        // zero_copy_with_bam_para<<<block_num, thread_num, 0, (strm_hdl)>>>(
		// 	bam_float_attrs_->d_array_ptr, float_attr_len,
		// 	sampled_ids, dst_float_buffer,
		// 	batch_size, node_off, 
		// 	total_num_nodes,
		// 	op_id
		// );


        dim3 block_num((batch_size - 1) / 1024 + 1, 1);
        dim3 thread_num(1024, 1);

        zero_copy_with_bam_request<<<block_num, thread_num, 0, (strm_hdl)>>>(
			bam_float_attrs_->d_array_ptr, float_attr_len,
			sampled_ids, d_req_count_,
			batch_size, node_off, 
			total_num_nodes,
			op_id
		);
        dim3 block_num_2((batch_size * float_attr_len - 1) / 1024 + 1, 1);
        zero_copy_with_bam_recieve<<<block_num_2, thread_num, 0, (strm_hdl)>>>(
			bam_float_attrs_->d_array_ptr, float_attr_len,
			sampled_ids, dst_float_buffer,
			batch_size, node_off, 
			total_num_nodes,
			op_id
		);

        
        // dim3 block_num((batch_size * 32 - 1) / 1024 + 1, 1);
        // dim3 thread_num(1024, 1);

        // zero_copy_with_bam_warp<<<block_num, thread_num, 0, (strm_hdl)>>>(
		// 	bam_float_attrs_->d_array_ptr, float_attr_len,
		// 	sampled_ids, dst_float_buffer,
		// 	batch_size, node_off, 
		// 	total_num_nodes,
		// 	op_id
		// );
    }



private:
    std::vector<int> training_set_num_;
    std::vector<int> validation_set_num_;
    std::vector<int> testing_set_num_;

    std::vector<int32_t*> training_set_ids_;
    std::vector<int32_t*> validation_set_ids_;
    std::vector<int32_t*> testing_set_ids_;

    std::vector<int32_t*> training_labels_;
    std::vector<int32_t*> validation_labels_;
    std::vector<int32_t*> testing_labels_;

    array_t<float>* bam_float_attrs_;
    // void* bam_float_attrs_;
    range_t<float>* h_range_;

    int32_t partition_count_;
    int32_t total_num_nodes_;
    int64_t* int_attrs_;
    int32_t int_attr_len_;
    float* float_attrs_;
    int32_t float_attr_len_;
    page_cache_t* h_pc_;

    float* d_req_count_;


    friend GPUNodeStorage* NewGPUMemoryNodeStorage();
};

extern "C" 
GPUNodeStorage* NewGPUMemoryNodeStorage(){
    GPUMemoryNodeStorage* ret = new GPUMemoryNodeStorage();
    return ret;
}
