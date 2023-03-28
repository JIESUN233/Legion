#ifndef __NVM_UTIL_H__
#define __NVM_UTIL_H__
//#ifndef __CUDACC__
//#define __device__
//#define __host__
//#endif

#include <nvm_types.h>
#include <stddef.h>
#include <stdint.h>
#include <cuda.h>

#ifdef __DIS_CLUSTER__
#include <sisci_types.h>
#include <sisci_api.h>
#endif



/* Convenience macro for creating a bit mask */
#define _NVM_MASK(num_bits) \
    ((1ULL << (num_bits)) - 1)

#define _NVM_MASK_PART(hi, lo) \
    (_NVM_MASK((hi) + 1) - _NVM_MASK(lo))


#if defined( __NO_COHERENCE__ ) && defined( __DIS_CLUSTER__ )
#ifdef __CUDACC__
__host__ __device__
#endif
static inline
void _nvm_cache_flush(void* ptr, size_t size)
{
#ifndef __CUDA_ARCH__ 
    sci_error_t err;
    SCICacheSync(NULL, ptr, size, SCI_FLAG_CACHE_FLUSH, &err);
#endif
}

#define nvm_cache_flush(ptr, size) _nvm_cache_flush(ptr, size)
#else
#define nvm_cache_flush(ptr, size)
#endif



#if defined( __NO_COHERENCE__ ) && defined( __DIS_CLUSTER__ )
#ifdef __CUDACC__
__host__ __device__
#endif
static inline
void _nvm_cache_invalidate(void* ptr, size_t size)
{
#ifndef __CUDA_ARCH__ 
    sci_error_t err;
    SCICacheSync(NULL, ptr, size, SCI_FLAG_CACHE_FLUSH | SCI_FLAG_CACHE_INVALIDATE, &err);
#endif
}

#define nvm_cache_invalidate(ptr, size) _nvm_cache_invalidate(ptr, size)
#else
#define nvm_cache_invalidate(ptr, size)
#endif



#if defined( __DIS_CLUSTER__ )
#ifdef __CUDACC__
__host__ __device__
#endif
static inline
void _nvm_wcb_flush()
{
#ifndef __CUDA_ARCH__
    SCIFlush(NULL, 0);
#endif
}

#define nvm_wcb_flush() _nvm_wcb_flush()
#else
#define nvm_wcb_flush()
#endif



/* Extract specific bits */
#define _RB(v, hi, lo)      \
    ( ( (v) & _NVM_MASK_PART((hi), (lo)) ) >> (lo) )


/* Set specifics bits */
#define _WB(v, hi, lo)      \
    ( ( (v) << (lo) ) & _NVM_MASK_PART((hi), (lo)) )


/* Offset to a register */
#define _REG(p, offs, bits) \
    ((volatile uint##bits##_t *) (((volatile unsigned char*) ((volatile void*) (p))) + (offs)))



/*
 * Calculate block number from page number.
 */
#define NVM_PAGE_TO_BLOCK(page_size, block_size, pageno)    \
    (((page_size) * (pageno)) / (block_size))
    


/*
 * Calculate page number from block number.
 */
#define NVM_BLOCK_TO_PAGE(page_size, block_size, blockno)   \
    (((block_size) * (blockno)) / (page_size))


/*
 * Create mask to clear away address offset.
 */
#define NVM_PAGE_MASK(page_size)                    \
    ~((page_size) - 1)


/*
 * Round address down to nearest page alignment.
 */
#define NVM_ADDR_MASK(addr, page_size)              \
    (((uint64_t) (addr)) & NVM_PAGE_MASK((page_size)))



/*
 * Align size to page boundary.
 */
#define NVM_PAGE_ALIGN(size, page_size)             \
    (((size) + (page_size) - 1) & NVM_PAGE_MASK((page_size)))


/*
 * Calculate page-aligned offset into address.
 */
#define NVM_ADDR_OFFSET(addr, page_size, pageno)    \
    (((uint64_t) (addr)) + ((page_size) * (pageno)))


/*
 * Calculate page-aligned offset into pointer.
 */
#define NVM_PTR_OFFSET(ptr, page_size, pageno)      \
    ((void*) (((unsigned char*) (ptr)) + ((page_size) * (pageno))))


/*
 * Align size to controller pages.
 */
#define NVM_CTRL_ALIGN(ctrl_ptr, size)              \
    NVM_PAGE_ALIGN((size), (ctrl_ptr)->page_size)


/*
 * Convert size to number of controller pages.
 */
#define NVM_CTRL_PAGES(ctrl_ptr, size)              \
    (NVM_CTRL_ALIGN((ctrl_ptr), (size)) / (ctrl_ptr)->page_size)


/*
 * Align size to page size.
 */
#define NVM_DMA_ALIGN(dma_ptr, size)                \
    NVM_PAGE_ALIGN((size), (dma_ptr)->page_size)


/*
 * Calculate controller page-aligned offset into DMA handle pointer.
 */
#define NVM_DMA_OFFSET(dma_ptr, pageno)             \
    NVM_PTR_OFFSET((dma_ptr)->vaddr, (dma_ptr)->page_size, (pageno))


/*
 * Calculate number of pages needed for a 
 * submission queue (SQ) with a given size.
 */
#define NVM_SQ_PAGES(ctrl_ptr, qs) \
    ((((uint16_t) ((qs) - 1))* sizeof(nvm_cmd_t)) / (ctrl_ptr)->page_size + 1)


/*
 * Calculate number of pages needed for a 
 * completion queue (CQ) with a given size.
 */
#define NVM_CQ_PAGES(ctrl_ptr, qs) \
    ((((uint16_t) ((qs) - 1)) * sizeof(nvm_cpl_t)) / (ctrl_ptr)->page_size + 1)


/*
 * Number of submission queue entries aligned to a page size.
 */
#define NVM_SQ_SIZE(ctrl_ptr, num_pages)   \
    ((ctrl_ptr)->page_size / sizeof(nvm_cmd_t))


/*
 * Number of completion queue entries aligned to a page size.
 */
#define NVM_CQ_SIZE(ctrl_ptr, num_pages)   \
    ((ctrl_ptr)->page_size / sizeof(nvm_cpl_t))



/* Standard fields in a command */
#define NVM_CMD_CID(p)              _REG(p, 2, 16)
#define NVM_CMD_NSID(p)             _REG(p, 1, 32)


/* Standard fields in a completion */
#define NVM_CPL_CID(p)              _REG(p, 12, 16)
#define NVM_CPL_SQHD(p)             _REG(p,  8, 16)
#define NVM_CPL_SQID(p)             _REG(p, 10, 16)
#define NVM_CPL_SF(p)               _REG(p, 14, 16)
#define NVM_CPL_STATUS(p)           NVM_CPL_SF(p)


/* Convenience macro for creating a default CID based on submission queue */
#define NVM_DEFAULT_CID(sq)         ((uint16_t) ((sq)->tail + (!(sq)->phase) * (sq)->qs))


#ifdef __cplusplus
extern "C" {
#endif
/*
 * Get controller associated with admin queue-pair reference.
 */
const nvm_ctrl_t* nvm_ctrl_from_aq_ref(nvm_aq_ref ref);
#ifdef __cplusplus
}
#endif



#ifdef __cplusplus
extern "C" {
#endif
/*
 * Get controller associated with DMA window
 */
const nvm_ctrl_t* nvm_ctrl_from_dma(const nvm_dma_t* dma);
#ifdef __cplusplus
}
#endif



#if defined( __DIS_CLUSTER__ )
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Get cluster node identifier from map.
 */
uint32_t nvm_dis_node_from_dma(const nvm_dma_t* dma);


/*
 * Get cluster node identifier from controller.
 */
uint32_t nvm_dis_node_from_ctrl(const nvm_ctrl_t* ctrl);

#ifdef __cplusplus
}
#endif
#endif

__forceinline__ __device__ uint32_t lane_id()
{
    uint32_t ret;
    asm volatile ("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ unsigned warp_id()
{
    // this is not equal to threadIdx.x / 32
    unsigned ret;
    asm volatile ("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

__forceinline__ __device__ uint32_t get_smid() {
     uint32_t ret;
     asm  ("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

#endif /* __NVM_UTIL_H__ */
