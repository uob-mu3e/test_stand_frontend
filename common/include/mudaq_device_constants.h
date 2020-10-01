/**
 * constants for the mudaq device, i.e. register address, bit patterns, ...
 *
 * @author      Moritz Kiehn <kiehn@physi.uni-heidelberg.de>
 * @author      Lennart Huth <huth@physi.uni-heidelberg.de>
 * @author      Heiko Augustin <augustin@physi.uni-heidelberg.de>
 * @author      Niklaus Berger <niberger@uni-mainz.de>
 * @date        2013-11-14
 * @updated     2017-02-28
 */





// register banks
#define MUDAQ_REGS_RW_INDEX     0
#define MUDAQ_REGS_RW_LEN       256
#define MUDAQ_REGS_RO_INDEX     1
#define MUDAQ_REGS_RO_LEN       256
// device memory
#define MUDAQ_MEM_RW_INDEX      2
#define MUDAQ_MEM_RW_ORDER      16  // to be changed if writeable memory size differs!
#define MUDAQ_MEM_RW_LEN        (1 << MUDAQ_MEM_RW_ORDER)
#define MUDAQ_MEM_RW_MASK       (MUDAQ_MEM_RO_LEN - 1)

#define MUDAQ_MEM_RO_INDEX      3
#define MUDAQ_MEM_RO_ORDER      16
#define MUDAQ_MEM_RO_LEN        (1 << MUDAQ_MEM_RO_ORDER)
#define MUDAQ_MEM_RO_MASK       (MUDAQ_MEM_RO_LEN - 1)
// dma buffers
#define MUDAQ_DMABUF_CTRL_INDEX 4
#define MUDAQ_DMABUF_CTRL_WORDS 4 // in words
// data buffer is 512MB ( = 128k uint32 words)
// currently the maximum is 4GB, if larger buffer is needed, the variable on the FPGA looping through the ring buffer
// needs to have more bits
#define MUDAQ_DMABUF_DATA_ORDER         25 // 29, 25 for 32 MB
#define MUDAQ_DMABUF_DATA_LEN           (1 << MUDAQ_DMABUF_DATA_ORDER)  // in bytes
#define MUDAQ_DMABUF_DATA_MASK          (MUDAQ_DMABUF_DATA_LEN - 1)
#define MUDAQ_DMABUF_DATA_ORDER_WORDS   (MUDAQ_DMABUF_DATA_ORDER - 2)
#define MUDAQ_DMABUF_DATA_WORDS         (1 << MUDAQ_DMABUF_DATA_ORDER_WORDS) // in words
#define MUDAQ_DMABUF_DATA_WORDS_MASK    (MUDAQ_DMABUF_DATA_WORDS - 1)

// fpga uses 8kb blocks internally. again given in uint32_t elements
// Actually it's 0xFFFF 32 bit words 
#define MUDAQ_BLOCK_ORDER       9
#define MUDAQ_BLOCK_LEN         (1 << MUDAQ_BLOCK_ORDER)
#define MUDAQ_BLOCK_MASK        (MUDAQ_BLOCK_LEN - 1)


/* smallest data unit is u32. all addresses / sizes must be u32 aligned */
/* 128 bit control buffer  */
#define MUDAQ_BUFFER_CTRL_SIZE (4 * 4)

#define PAGES_PER_INTERRUPT           64
#define N_BUFFERS                     (MUDAQ_DMABUF_DATA_LEN / ( PAGES_PER_INTERRUPT * PAGE_SIZE ))
#define N_PAGES                       (MUDAQ_DMABUF_DATA_LEN / PAGE_SIZE)
