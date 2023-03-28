#ifndef __NVM_DMA_H__
#define __NVM_DMA_H__
// #ifndef __CUDACC__
// #define __device__
// #define __host__
// #endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __DIS_CLUSTER__
#include <sisci_types.h>
#endif



/*
 * Create DMA mapping descriptor from physical/bus addresses.
 *
 * Create a DMA mapping descriptor, describing a region of memory that is
 * accessible for the NVM controller. The caller must supply physical/bus  
 * addresses of physical memory pages, page size and total number of pages.
 * As the host's page size may differ from the controller's page size (MPS),
 * this function will calculate the necessary offsets into the actual memory
 * pages.
 *
 * While virtual memory is assumed to be continuous, the physical pages do not
 * need to be contiguous. Physical/bus addresses must be aligned to the 
 * controller's page size.
 *
 * Note: vaddr can be NULL.
 */
int nvm_dma_map(nvm_dma_t** map,                // Mapping descriptor reference
                const nvm_ctrl_t* ctrl,         // NVM controller reference
                void* vaddr,                    // Pointer to userspace memory (can be NULL if not required)
                size_t page_size,               // Physical page size
                size_t n_pages,                 // Number of pages to map
                const uint64_t* page_addrs);    // List of physical/bus addresses to the pages



/*
 * Create DMA mapping descriptor using offsets from a previously 
 * created DMA descriptor.
 */
int nvm_dma_remap(nvm_dma_t** new_map, const nvm_dma_t* other_map);



/*
 * Remove DMA mapping descriptor.
 *
 * Unmap DMA mappings (if necessary) and remove the descriptor.
 * This function destroys the descriptor.
 */
void nvm_dma_unmap(nvm_dma_t* map);



/*
 * Create DMA mapping descriptor from virtual address using the kernel module.
 * This function is similar to nvm_dma_map, except the user is not required
 * to pass physical/bus addresses. 
 *
 * Note: vaddr can not be NULL, and must be aligned to system page size.
 */
int nvm_dma_map_host(nvm_dma_t** map, const nvm_ctrl_t* ctrl, void* vaddr, size_t size);



//#if ( defined( __CUDA__ ) || defined( __CUDACC__ ) )

/*
 * Create DMA mapping descriptor from CUDA device pointer using the kernel
 * module. This function is similar to nvm_dma_map_host, except the memory
 * pointer must be a valid CUDA device pointer (see manual for 
 * cudaGetPointerAttributes).
 *
 * The controller handle must have been created using the kernel module.
 *
 * Note: vaddr can not be NULL, and must be aligned to GPU page size.
 */
int nvm_dma_map_device(nvm_dma_t** map, const nvm_ctrl_t* ctrl, void* devptr, size_t size);

//#endif /* __CUDA__ */



#if defined( __DIS_CLUSTER__ )

/*
 * Create DMA mapping descriptor from local SISCI segment.
 *
 * Create DMA mapping descriptor from a local segment handler, and 
 * reverse-map the segment making it accessible from the controller.
 * As segment memory is always continuous and page-aligned, it is not
 * necessary to calculate physical memory addresses. However, the user
 * should ensure that the mapping size is aligned to a controller
 * page-size (MPS).
 * 
 * The controller handle must have been created using SmartIO, and
 * the segment must already be prepared on the local adapter.
 */
int nvm_dis_dma_map_local(nvm_dma_t** map,              // Mapping descriptor reference
                          const nvm_ctrl_t* ctrl,       // NVM controller handle
                          uint32_t dis_adapter,         // Local DIS adapter segment is prepared on
                          sci_local_segment_t segment,  // Local segment descriptor
                          bool map_vaddr);              // Should function also map segment into local space

#endif /* __DIS_CLUSTER__ */



#if defined( __DIS_CLUSTER__ )

/*
 * Create DMA mapping descriptor from remote SISCI segment.
 *
 * Create DMA mapping descriptor from a remote segment handler, and 
 * reverse-map the segment making it accessible from the controller.
 * This function is similar to nvm_dis_dma_map_local.
 *
 * The remote segment must already be connected.
 *
 * Note: You should generally prefer write combining, except
 *       for mapped device registers that require fine-grained writes.
 */
int nvm_dis_dma_map_remote(nvm_dma_t** map,             // Mapping descriptor reference
                           const nvm_ctrl_t* ctrl,      // NVM controller handle
                           sci_remote_segment_t segment,// Remote segment descriptor
                           bool map_vaddr,              // Should function also map segment into local space
                           bool map_wc);                // Should function map with write combining

#endif /* __DIS_CLUSTER__ */



#if ( !defined( __CUDA__ ) && !defined( __CUDACC__ ) ) && ( defined (__unix__) )
/* 
 * Short-hand function for allocating a page aligned buffer and mapping it 
 * for the controller.
 *
 * Note: this function will not work if you are using the CUDA API
 */
int nvm_dma_create(nvm_dma_t** map, const nvm_ctrl_t* ctrl, size_t size);
#endif



#if defined( __DIS_CLUSTER__ )
/*
 * Create device memory segment and map it for the controller.
 * Short-hand function for creating a device memory segment.
 * If mem_hints is 0, the API will create a local segment instead.
 */
int nvm_dis_dma_create(nvm_dma_t** map, const nvm_ctrl_t* ctrl, size_t size, unsigned int mem_hints);

#endif /* __DIS_CLUSTER__ */



#if defined ( __DIS_CLUSTER__ )

/*
 * Note: This function requires the IOMMU to be enabled.
 */
int nvm_dis_dma_map_host(nvm_dma_t** map, const nvm_ctrl_t* ctrl, void* vaddr, size_t size);

#endif


#if ( ( defined( __CUDA__ ) || defined( __CUDACC__ ) ) && defined( __DIS_CLUSTER__ ) )

int nvm_dis_dma_map_device(nvm_dma_t** map, const nvm_ctrl_t* ctrl, void* devptr, size_t size);

#endif /* __DIS_CLUSTER__ && __CUDA__ */




#endif /* __NVM_DMA_H__ */
