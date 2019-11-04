/**
 * a driver for the mu3e pcie readout board
 *
 * @author  Moritz Kiehn <kiehn@physi.uni-heidelberg.de>
 * @date    2013-10-28
 *
 * modified to work with streaming DMA and not using the uio functions anymore by
 * @auther Dorothea vom Bruch <vombruch@physi.uni-heidelberg.de>
 * @date   2015-08-05
 * Some code is based on uio driver (drivers/uio/uio.c), partially exactly the same code
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/version.h>

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>

#include <linux/pci.h>

#include <linux/atomic.h>
#include <linux/err.h>
#include <linux/errno.h>
#include <linux/idr.h>
#include <linux/kfifo.h>
#include <linux/mm.h>
#include <linux/mutex.h>
#include <linux/string.h>
#include <linux/types.h>
#include <linux/interrupt.h>
#include <linux/wait.h>
#include <linux/sched.h>
#if LINUX_VERSION_CODE > KERNEL_VERSION(4,10,17)
#include <linux/sched/signal.h>
#endif
#include <linux/pagemap.h>

#include <asm/uaccess.h>

#include "mudaq.h"
#include "../include/mudaq_device_constants.h"
#include "../include/mudaq_registers.h"

#define ERROR(fmt, args...) printk(KERN_ERR   "mudaq: " fmt, ## args)
#define INFO(fmt, args...)  printk(KERN_INFO  "mudaq: " fmt, ## args)
#define DEBUG(fmt, args...) printk(KERN_DEBUG "mudaq: " fmt, ## args)

/* macros were removed in 3.9; allow compilation on newer kernels */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,9,0)
#define __devinit
#define __devexit
#define __devexit_p(arg) arg
#endif

/**
 * module-wide global variables
 */

static struct class *   mudaq_class = NULL;
static int              major = 0;
static DEFINE_IDR(      minor_idr);
static DEFINE_MUTEX(    minor_lock);

static const struct pci_device_id   PCI_DEVICE_IDS[] = {
    {PCI_DEVICE(0x1172, 0x0004),},
    {0,} };

/**
 * helper functions
 */
int rest( int int1, int int2, int wrap ) {
  int result;
  if ( ( int1 + int2 ) < wrap ) {
    result =  int1 + int2;
  }
  else if ( ( int1 + int2 ) >= wrap ) {
    result = ( int1 + int2 ) % wrap;
  }
  return result;
}

int wrap_ring( int int1, int int2, int wrap, int divisor ) {
  int result = 0;
  if (  ( int1 - int2 ) > 0 ) {
    result =  ( int1 - int2 ) / divisor;
  }
  else if ( ( int1 - int2 ) < 0 ) {
    result = wrap + ( int1 - int2 ) / divisor;
  }
  else if ( ( int1 - int2 ) == 0 ) {
    result = 0;
  }
  return result;
}

int is_page_aligned( void * pointer ) {
  DEBUG("diff to page: %lu", ( (uintptr_t)(const void *)(pointer) ) % PAGE_SIZE );
  return !( ( (uintptr_t)(const void *)(pointer) ) % PAGE_SIZE  == 0 );
}

void * align_page( void * pointer ) {
  void * aligned_pointer = (void *)( pointer + PAGE_SIZE - ((uintptr_t)(const void *)(pointer) ) % PAGE_SIZE );
  return aligned_pointer;
}

/**
 * mudaq structures and related functions
 */

struct mudaq {
  struct device * dev;
  struct pci_dev * pci_dev;
  struct cdev char_dev;
  u32 to_user[2];
  long irq;
  atomic_t event;
  wait_queue_head_t wait;
  struct mesg msg;
  struct mudaq_mem * mem;
  struct mudaq_dma * dma;
};

/**
   First four entries of internal_addr, phys_size and phys_addr are for
   the four PCIe BAR regions
   fifth entry is for the DMA control buffer
 */
struct mudaq_mem {
  __iomem u32 * internal_addr[5];
  u32 phys_size[5];
  u32 phys_addr[5];
  dma_addr_t bus_addr_ctrl;
};

struct mudaq_dma {
  bool flag;
  struct page **pages;
  dma_addr_t bus_addrs[N_PAGES];
  int n_pages[N_PAGES];
  int npages;
  struct sg_table * sgt;
};

/** allocate a new mudaq struct and initialize its state */
static int mudaq_alloc(struct mudaq ** mu)
{
    int retval;
    struct mudaq * tmp;
    struct mudaq_mem * tmp_mem;
    struct mudaq_dma * tmp_dma;

    /* allocate memory for the device structure */
    tmp = kzalloc(sizeof(struct mudaq), GFP_KERNEL);
    if (tmp == NULL) {
        ERROR("could not allocate memory for 'struct mudaq'\n");
        retval = -ENOMEM;
        goto fail;
    }

    tmp_mem = kzalloc(sizeof(struct mudaq_mem), GFP_KERNEL);
    if (tmp_mem == NULL) {
        ERROR("could not allocate memory for 'struct mudaq_mem'\n");
        retval = -ENOMEM;
        goto fail_mem;
    }
    tmp->mem = tmp_mem;

    tmp_dma = kzalloc(sizeof(struct mudaq_dma), GFP_KERNEL);
    if (tmp_dma == NULL) {
        ERROR("could not allocate memory for 'struct mudaq_dma'\n");
        retval = -ENOMEM;
        goto fail_dma;
    }
    tmp->dma = tmp_dma;

    *mu = tmp;
    return 0;

 fail_dma:
    kfree(tmp_mem);
    tmp_dma = NULL;
 fail_mem:
    kfree(tmp);
    tmp_mem = NULL;
 fail:
    *mu = NULL;
    return retval;
}

/** free the given mudaq struct and all associated memory */
static void mudaq_free(struct mudaq * mu)
{
  kfree( mu->dma );
  kfree( mu->mem );
  kfree( mu );
}


/**
 * minor number handling
 */

/** aquire a new minor number and associate it with the given data */
int minor_aquire(void * data)
{
    int retval;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,9,0)
    mutex_lock(&minor_lock);
    retval = idr_alloc(&minor_idr, data, 0, MAX_NUM_DEVICES, GFP_KERNEL);
    mutex_unlock(&minor_lock);
#else
    do {
        if (idr_pre_get(&minor_idr, GFP_KERNEL) == 0) {
            ERROR("no memory for idr_pre_get\n");
            return -ENOMEM;
        }

        mutex_lock(&minor_lock);
        retval = idr_get_new_above(&minor_idr, data, 0, &minor);
        mutex_unlock(&minor_lock);

    } while (retval == -EAGAIN );
    retval = minor;
#endif

    if (retval < 0)
        ERROR("could not allocate a minor number\n");

    return retval;
}

/** find the data associated with the given minor number */
void * minor_find_data(int minor)
{
    void * data;

        mutex_lock(&minor_lock);
        data = idr_find(&minor_idr, minor);
        mutex_unlock(&minor_lock);

    return data;
}

/** aquire a new minor number and associate it with the given device */
void minor_release(int minor)
{
        mutex_lock(&minor_lock);
        idr_remove(&minor_idr, minor);
        mutex_unlock(&minor_lock);
}

static int mudaq_interrupt_control(struct mudaq * info, s32 irq_on)
{
    /* no need to do anything. interupts are activated from userspace */
    DEBUG("Called interrupt control w/ %#x", irq_on);
    return 0;
}

static irqreturn_t mudaq_interrupt_handler(int irq, struct mudaq * mu)
{
    /* no need to do anything. just acknowledge that something happened. */
    DEBUG("Received interrupt");
    return IRQ_HANDLED;
}

/* Trigger an interrupt event */
void mudaq_event_notify( struct mudaq * mu )
{
  atomic_inc( &mu->event ); // interrupt counter
  wake_up_interruptible( &mu->wait ); // wake up read function
}

/* Hardware interrupt handler */
static irqreturn_t mudaq_interrupt( int irq, void *dev_id )
{
  struct mudaq * mu = (struct mudaq *)dev_id;
  irqreturn_t ret = mudaq_interrupt_handler( irq, mu );

  if ( ret == IRQ_HANDLED )
    mudaq_event_notify( mu );

  return ret;
}

/* Access registers  */
inline __iomem u32 * mudaq_register_rw( struct mudaq * mu, unsigned index )
{
  __iomem u32 * base = mu->mem->internal_addr[0];
  return base + index;
}

inline __iomem u32 * mudaq_register_ro( struct mudaq * mu, unsigned index )
{
  __iomem u32 * base = mu->mem->internal_addr[1];
  return base + index;
}

static void mudaq_deactivate( struct mudaq * mu ) {

  u32 test;

  iowrite32(0x0, mudaq_register_rw( mu, DATAGENERATOR_REGISTER_W));
  test = ioread32(      mudaq_register_rw( mu, DATAGENERATOR_REGISTER_W));
  if ( test != 0 )
    ERROR("read back %u instead of 0", test);
  iowrite32(0x0, mudaq_register_rw( mu, DMA_REGISTER_W));
  test = ioread32(      mudaq_register_rw( mu, DMA_REGISTER_W));
  if ( test != 0 )
    ERROR("read back %u instead of 0", test);
}

/* Copy dma bus addresses to device registers */
static void mudaq_set_dma_ctrl_addr( struct mudaq * mu,
                                     dma_addr_t ctrl_handle)
{
  dma_addr_t test;

  iowrite32((ctrl_handle & 0xFFFFFFFF), mudaq_register_rw(mu, DMA_CTRL_ADDR_LOW_REGISTER_W));
  iowrite32((ctrl_handle >> 32),        mudaq_register_rw(mu, DMA_CTRL_ADDR_HI_REGISTER_W));

  test = (dma_addr_t)ioread32(mudaq_register_rw(mu, DMA_CTRL_ADDR_LOW_REGISTER_W)) |
    (dma_addr_t)ioread32(mudaq_register_rw(mu,DMA_CTRL_ADDR_HI_REGISTER_W )) << 32;
  if ( (size_t)ctrl_handle != (size_t)test )
    ERROR("dma control buffer. wrote %#zx read %#zx", (size_t)ctrl_handle, (size_t)test);
}

static void mudaq_read_dma_data_addr( struct mudaq * mu,
                                      u32 mem_location)
{
  dma_addr_t test;
  u32 n_pages;
  u32 oldreg;

  oldreg = (u32)ioread32(mudaq_register_rw(mu, DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W) );
  iowrite32( SET_DMA_RAM_LOCATION_RANGE(oldreg, mem_location), mudaq_register_rw(mu, DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W) );
  test = (dma_addr_t)ioread32(mudaq_register_ro(mu, DMA_DATA_ADDR_LOW_REGISTER_R)) |
    (dma_addr_t)ioread32(mudaq_register_ro(mu, DMA_DATA_ADDR_HI_REGISTER_R)) << 32;
  //DEBUG("At %u: address = %lx", mem_location, (long unsigned)test);
  if ( test != mu->dma->bus_addrs[mem_location] )
    ERROR("Bus address is %lx, should be %lx", (long unsigned)test, (long unsigned)mu->dma->bus_addrs[mem_location]);

  n_pages = (u32)ioread32(mudaq_register_ro(mu, DMA_NUM_PAGES_REGISTER_R) );
  //DEBUG("# of pages per address: %u", n_pages);
  if ( n_pages != mu->dma->n_pages[mem_location] )
    ERROR("# of pages is %x, should be %x", n_pages, mu->dma->n_pages[mem_location]);
}

static void mudaq_set_dma_data_addr( struct mudaq * mu,
                                     dma_addr_t data_handle,
                                     u32 mem_location,
                                     u32 n_pages )
{
  dma_addr_t test;
  u32 test_n_pages;
  u32 regcontent;
  regcontent  = SET_DMA_NUM_PAGES_RANGE(0x0, n_pages);
  regcontent  = SET_DMA_RAM_LOCATION_RANGE(regcontent, mem_location);

  iowrite32( regcontent, mudaq_register_rw(mu, DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W) );
  iowrite32((data_handle >> 32),        mudaq_register_rw(mu, DMA_DATA_ADDR_HI_REGISTER_W));
  iowrite32((data_handle & 0xFFFFFFFF), mudaq_register_rw(mu, DMA_DATA_ADDR_LOW_REGISTER_W));

  test = (dma_addr_t)ioread32(mudaq_register_ro(mu, DMA_DATA_ADDR_LOW_REGISTER_R)) |
    (dma_addr_t)ioread32(mudaq_register_ro(mu, DMA_DATA_ADDR_HI_REGISTER_R)) << 32;
  if ( (size_t)data_handle != (size_t)test )
    ERROR("dma data buffer. wrote %#zx read %#zx", (size_t)data_handle, (size_t)test);

  test_n_pages = (u32)ioread32(mudaq_register_ro(mu, DMA_NUM_PAGES_REGISTER_R) );
  if ( test_n_pages != n_pages ) {
    ERROR("number of pages. wrote %u read %u", n_pages, test_n_pages);
  }
}

static void mudaq_set_dma_n_buffers( struct mudaq * mu,
                                     u32 n_buffers)
{
  u32 test;
  iowrite32( n_buffers, mudaq_register_rw(mu, DMA_NUM_ADDRESSES_REGISTER_W) );
  test = (u32)ioread32( mudaq_register_rw(mu,DMA_NUM_ADDRESSES_REGISTER_W ) );
  if ( test != n_buffers )
    ERROR("number of buffers. wrote %u, read %u", n_buffers, test);
}

/**
 * mudaq device file operatations
 */

ssize_t mudaq_fops_read(struct file * filp, char __user * buf, size_t count, loff_t * f_pos)
{
    struct mudaq * mu = (struct mudaq *)filp->private_data;
    DECLARE_WAITQUEUE( wait, current );
    size_t retval;
    u32 new_event_count;
    u32 * dma_buf_ctrl;
    int n_interrupts = 0;

    if ( !mu->irq ) return -EIO;
    if ( count != sizeof( mu->to_user ) ) return -EINVAL;

    add_wait_queue( &mu->wait, &wait ); // add read process to wait queue

    do {
      set_current_state( TASK_INTERRUPTIBLE ); // mark process as being asleep but interruptible

      new_event_count = atomic_read( &mu->event );
      if ( new_event_count != mu->to_user[0] ) {       // interrupt occured
        mu->to_user[0] = new_event_count;              // pass interrupt number
        dma_buf_ctrl = mu->mem->internal_addr[4];
        /* How many DMA blocks were transfered (in units of interrupts (64 blocks) )?
         * Offset in ring buffer comes in bytes from FPGA, transform to uint32_t words here
         */
        DEBUG("interrupt number: %d, address: %x%x", new_event_count, (int)dma_buf_ctrl[1], (int)dma_buf_ctrl[0]);
        n_interrupts = wrap_ring( (int)dma_buf_ctrl[3] >> 2, (int)mu->to_user[1], N_BUFFERS, PAGES_PER_INTERRUPT * PAGE_SIZE / 4 );
        if ( n_interrupts != 1 ) {
          DEBUG("ctrl buffer: %x, previous: %x", (int)dma_buf_ctrl[3] >> 2, (int)mu->to_user[1] );
          INFO("Missed %d interrupt(s)", n_interrupts);
        }

        mu->to_user[1] = (u32)dma_buf_ctrl[3] >> 2;  // position in ring buffer last written to + 1 (in words)
        if ( copy_to_user(buf, &(mu->to_user), sizeof(mu->to_user) ) )
          retval = -EFAULT;
        else {
          mu->to_user[0] = new_event_count;             // save event_count in case copying to user space fails
          retval = count;
        }
        break;
      }

      if ( filp->f_flags & O_NONBLOCK) { // check if user requested non-blocking I/O
        retval = -EAGAIN;
        break;
      }
      if ( signal_pending( current ) ) { // restart system when receiving interrupt
        retval = -ERESTARTSYS;
        break;
      }
      schedule(); // check whether interrupt occured while going through above operations

    } while (1);

    __set_current_state( TASK_RUNNING ); // process is able to run
    remove_wait_queue( &mu->wait, &wait ); // remove process from wait queue so that it can only be awakened once

    return retval;

}

ssize_t mudaq_fops_write(struct file * filp, const char __user * buf, size_t count, loff_t * f_pos)
{
    struct mudaq * mu = (struct mudaq *)filp->private_data;
    s32 irq_on;
    ssize_t retval;

    if ( !mu->irq ) return -EIO;
    if ( count != sizeof(s32) ) return -EINVAL;

    if ( copy_from_user( &irq_on, buf, count ) != 0 ) return -EFAULT;

    retval = mudaq_interrupt_control( mu, irq_on );

    return sizeof(s32);
}

int mudaq_fops_mmap(struct file * filp, struct vm_area_struct * vma)
{
    struct mudaq * mu = (struct mudaq *)filp->private_data;
    int index = (int)vma->vm_pgoff;
    unsigned long requested_pages = 0, actual_pages;
    int rv = 0;

    DEBUG( "Mapping for index %d", index );

    /* use the requested page offset as an index to select the pci region that
       should be mapped. readout board has 4 accessible memory regions
       registers rw, registers ro, memory rw and memory ro
       in addition, we have one dma ctrl buffer */
    if ( (index < 0) || (index > 4) ) {
      DEBUG("invalid mmap memory index %d", index);
      return -EINVAL;
    }

    /* only allow mapping of the whole selected memory.
       WARNING actual size must not be page-aligned
       But minimum virtual address space of vma
       is one page */
    actual_pages  = mu->mem->phys_size[index] / PAGE_SIZE;
    actual_pages += (( mu->mem->phys_size[index] % PAGE_SIZE) == 0) ? 0 : 1;
    requested_pages = (vma->vm_end - vma->vm_start) >> PAGE_SHIFT;

    if ( requested_pages != actual_pages ) {
      ERROR("invalid mmap pages requested. requested %lu actual %lu",
            requested_pages, actual_pages);
      rv = -EINVAL;
      goto out;
    }
    if ( mu->mem->phys_addr[index] + mu->mem->phys_size[index] < mu->mem->phys_addr[index] ) {
          ERROR("invalid memory settings. phys_addr %d size %d",
                mu->mem->phys_addr[index], mu->mem->phys_size[index]);
          rv = -EINVAL;
          goto out;
    }

    /* only the read/write registers and the read/write memory need to be
       writable. everything else can be forced to be read-only */
    vma->vm_flags |= VM_READ;
    vma->vm_flags &= ~VM_WRITE;
    vma->vm_flags &= ~VM_EXEC;
    /* dma_mmap_* and vm_iomap_memory use pgoff as additional offset inside
       the buffer, but we always want to map the whole area. */
    vma->vm_pgoff = 0;

    switch ( index ) {
    case 0: // rw registers
      vma->vm_flags |= VM_WRITE;
    case 1: // ro registers
    case 2: // rw memory
      vma->vm_flags |= VM_WRITE;
      vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
      return io_remap_pfn_range(
                                vma,
                                vma->vm_start,
                                mu->mem->phys_addr[index] >> PAGE_SHIFT,
                                vma->vm_end - vma->vm_start,
                                vma->vm_page_prot);
    case 3: // ro memory
      vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
      return io_remap_pfn_range(
                              vma,
                              vma->vm_start,
                              mu->mem->phys_addr[index] >> PAGE_SHIFT,
                              vma->vm_end - vma->vm_start,
                              vma->vm_page_prot);
    case 4: // dma control buffer
      return dma_mmap_coherent( mu->dev,
                                vma,
                                mu->mem->internal_addr[index],
                                mu->mem->bus_addr_ctrl,
                                mu->mem->phys_size[index]);
    }

    /* this should NEVER happen */
    ERROR("something went horribly wrong. run for your lives");
    rv = -EINVAL;

    out:
    return rv;
}

static int mudaq_setup_mmio( struct pci_dev * pdev, struct mudaq * mu ) {

  int i, j;
  const int bars[] = {0, 1, 2, 3};
  const char * names[] = {"registers_rw", "registers_ro", "memory_rw", "memory_ro"};
  int rv = 0;

  /* request access to pci BARs */
  rv = pci_request_regions(pdev, DRIVER_NAME);
  if (rv < 0) {
    ERROR("could not request memory regions\n");
    goto out;
  }

  /* Map PCI bar regions to __iomem addresses (addressable from kernel)  */
  for ( i = 0; i < 4; i++ ) {
    mu->mem->phys_addr[i]        = pci_resource_start( pdev, bars[i] );
    mu->mem->phys_size[i]        = pci_resource_len( pdev, bars[i] );
    mu->mem->internal_addr[i]    = (__iomem u32 *)pci_iomap( pdev, bars[i], mu->mem->phys_size[i] );
    if ( mu->mem->internal_addr[i] == NULL ) {
      ERROR("pci_iomap failed for '%s'", names[i]);
      rv = -ENODEV;
      goto fail_unmap;
    }
    DEBUG("Bar %d: at %x with size %d at internal address %lx", bars[i], mu->mem->phys_addr[i], mu->mem->phys_size[i], (long unsigned)  mu->mem->internal_addr[i]);
  }
  return 0;

 fail_unmap:
  for (j = i; j > 0; --j) pci_iounmap( pdev, mu->mem->internal_addr[i] );
  pci_release_regions(pdev);
 out:
  return rv;
}

static void mudaq_clear_mmio( struct pci_dev * dev, struct mudaq * mu ) {

  pci_iounmap( dev, mu->mem->internal_addr[0] );
  pci_iounmap( dev, mu->mem->internal_addr[1] );
  pci_iounmap( dev, mu->mem->internal_addr[2] );
  pci_iounmap( dev, mu->mem->internal_addr[3] );
  pci_release_regions(dev);

}

/** setup coherent dma
 *
 * set dma masks and create one DMA buffer for control
 *
 * WARNING
 * assumes that the read/write registers are already mapped
 */
static int mudaq_setup_dma( struct pci_dev * pdev, struct mudaq * mu )
{
  int rv;
  dma_addr_t ctrl_addr;
  void *     ctrl_internal;
  size_t     ctrl_size = MUDAQ_BUFFER_CTRL_SIZE;
  const char * name = "dma_ctrl";

  if ((rv = pci_set_dma_mask( pdev, DMA_BIT_MASK(64)) ) < 0) return rv;
  if ((rv = pci_set_consistent_dma_mask( pdev, DMA_BIT_MASK(64)) ) < 0) return rv;

  #if LINUX_VERSION_CODE >= KERNEL_VERSION(5,0,0)
  // https://lkml.org/lkml/2019/1/8/391 --> kernel > 5.0
  ctrl_internal = dma_alloc_coherent(&pdev->dev, ctrl_size, &ctrl_addr, GFP_KERNEL);
  #else
  ctrl_internal = dma_zalloc_coherent(&pdev->dev, ctrl_size, &ctrl_addr, GFP_KERNEL);
  #endif
  if (ctrl_internal == NULL) {
    ERROR("could not allocate dma control buffer");
    rv = -ENOMEM;
    goto out;
  }

  mu->mem->bus_addr_ctrl    = ctrl_addr;
  mu->mem->phys_size[4]     = ctrl_size;
  mu->mem->internal_addr[4] = ctrl_internal;

  //DEBUG("Addr: %d , Size: %d, Internal Add %d",int(ctrl_addr) int(ctrl_size) int(ctrl_internal));
  DEBUG( "setup dma for '%s' dma_addr %#zx size %d bytes",
         name, (size_t)mu->mem->bus_addr_ctrl, mu->mem->phys_size[4] );

  /* deactivate any readout activity to always start in a consistent state
     before setting up the dma addresses */
  mudaq_deactivate( mu );
  mudaq_set_dma_ctrl_addr( mu, ctrl_addr );
  return 0;

 out:
  return rv;
}

static void mudaq_clear_dma( struct pci_dev * pdev, struct mudaq * mu )
{

  /* deactivate any readout activity before removing the dma memory */
  mudaq_deactivate( mu );
  mudaq_set_dma_ctrl_addr( mu, 0x0 );

  dma_free_coherent( &pdev->dev, mu->mem->phys_size[4], mu->mem->internal_addr[4], mu->mem->bus_addr_ctrl );
}

static int mudaq_setup_msi( struct mudaq * mu ) {

  int rv;
  pci_set_master( mu->pci_dev );

  /* enable message-write-invalidate transactions for cache coherency. */
  if ((rv = pci_set_mwi( mu->pci_dev )) < 0) return rv;

  /* use message-based interrupts */
  if ((rv = pci_enable_msi( mu->pci_dev )) < 0) goto out_clear_mwi;
  mu->irq = mu->pci_dev->irq;

  /* request irq */
  init_waitqueue_head( &mu->wait );
  atomic_set( &mu->event, 0 );
  rv = devm_request_irq( mu->dev, mu->irq, mudaq_interrupt, 0, DRIVER_NAME, mu );
  if ( rv ) {
    DEBUG("Failed to initialize irq, error: %d", rv);
    goto out_clear_msi;
  }

  return 0;

 out_clear_msi:
  pci_disable_msi( mu->pci_dev );
 out_clear_mwi:
  pci_clear_mwi( mu->pci_dev );
  return rv;
}

static void mudaq_clear_msi( struct mudaq * mu ) {

  /* release irq */
  devm_free_irq( mu->dev, mu->irq, mu );
  DEBUG("Freed irq");

  /* disable MSI */
  pci_disable_msi( mu->pci_dev );
  pci_clear_mwi( mu->pci_dev );

}

static void mudaq_free_dma( struct mudaq * mu )
{
  int i_list, i_page;
  struct scatterlist * sg;

  mudaq_deactivate( mu );

  for_each_sg(  mu->dma->sgt->sgl, sg, mu->dma->sgt->nents, i_list ) {
    dma_unmap_sg( mu->dev, sg, 1, DMA_FROM_DEVICE );
  }

  for ( i_page = 0; i_page < mu->dma->npages; i_page++) {
    if ( PageReserved( mu->dma->pages[i_page] ) )
      INFO("Page %d is in reserved space", i_page );
    if ( !PageReserved( mu->dma->pages[i_page] ) )
      SetPageDirty( mu->dma->pages[i_page] );
    //page_cache_release( mu->dma->pages[i_page] );
    put_page(mu->dma->pages[i_page]);
  }

  kfree( mu->dma->sgt );
  kfree( mu->dma->pages );
  INFO("Freed pinned DMA buffer in kernel space");

  mu->dma->flag = false;

}

/* release function is called when a process closes the device file */
   static int mudaq_fops_release(struct inode * inode, struct file * filp)
{
  struct mudaq * mu  = (struct mudaq *)filp->private_data;
  if ( mu->dma->flag == true)
    mudaq_free_dma( mu );

  return 0;
}

static int mudaq_fops_open(struct inode * inode, struct file * filp)
{
  struct mudaq * mu = minor_find_data(iminor(inode));
  if (mu == NULL) return -ENODEV;
  filp->private_data = mu;

  return 0;

}

long mudaq_fops_ioctl( struct file * filp,
                      unsigned int cmd,	/* magic and sequential number for ioctl */
                      unsigned long ioctl_param)
{
  int retval = 0;
  struct mudaq * mu = (struct mudaq *)filp->private_data;
  int err = 0;
  int i_page = 0, i_list = 0, n_mapped = 0, count = 0;
  struct page **pages_tmp;
  void * aligned_pointer;
  struct scatterlist * sg;
  struct sg_table * sgt_tmp;
  u32 new_event_count;

  mu->dma->flag = false; // in case something goes wrong

  /*
   * Extract type and number bitfields from IOCTL cmd
   * Return error in case of wrong cmd's
   * type = magic, number = sequential number
   */

  if ( _IOC_TYPE(cmd) != MAGIC_NR ) {
    retval = -ENOTTY;
    goto fail;
  }
  if ( _IOC_NR(cmd)   > IOC_MAXNR ) {
    retval = -ENOTTY;
    goto fail;
  }

  /*
   * Verify that address of parameter does not point to kernel space memory
   * access_ok checks for this
   *    returns 1 for access ok
   *            0 for acces not ok
   * VERIFY_READ  = read user space memory
   * VERIFY_WRITE = write to user space memory
   * _IOC_READ    = read kernel space memory
   * _IOC_WRITE   = write data to kernel space memory
   * https://github.com/zfsonlinux/zfs/issues/8261 --> kernel > 5.0
   */

  if ( _IOC_DIR(cmd) & _IOC_READ )
    #if LINUX_VERSION_CODE >= KERNEL_VERSION(5,0,0)
    err = !access_ok((void __user *)ioctl_param, _IOC_SIZE(cmd));
    #else
    err = !access_ok(VERIFY_WRITE, (void __user *)ioctl_param, _IOC_SIZE(cmd));
    #endif
  else if ( _IOC_DIR(cmd) & _IOC_WRITE)
    #if LINUX_VERSION_CODE >= KERNEL_VERSION(5,0,0)
    err = !access_ok((void __user *)ioctl_param, _IOC_SIZE(cmd));
    #else
    err = !access_ok(VERIFY_READ, (void __user *)ioctl_param, _IOC_SIZE(cmd));
    #endif
  if (err) {
    retval = -EFAULT;
    goto fail;
  }

  /*
   * Switch according to the ioctl called
   */
  switch (cmd) {
  case REQUEST_INTERRUPT_COUNTER: /* Send current interrupt counter to user space */
    new_event_count = atomic_read( &mu->event );
    retval = copy_to_user((char __user *) ioctl_param, &new_event_count,  sizeof(new_event_count));
    if (retval > 0 ) {
      ERROR("copy_to_user failed with error %d \n", retval);
      goto fail;
    }
  case MAP_DMA: /* Receive a pointer to a virtual address in user space */
    retval = copy_from_user(&(mu->msg), (char __user *) ioctl_param, sizeof(mu->msg));
    if (retval > 0 ) {
      ERROR("copy_from_user failed with error %d \n", retval);
      goto fail;
    }
    INFO( "Recieved following virtual address: 0x%0llx \n", (u64)(mu->msg).address);
    INFO( "Size of allocated memory: %zu bytes\n", mu->msg.size);

    /* allocte memory for page pointers */
    if (( pages_tmp = kzalloc(sizeof(long unsigned) * N_PAGES, GFP_KERNEL)) == NULL) {
      DEBUG("Error allocating memory for page table");
      retval = -ENOMEM;
      goto fail;
    }
    mu->dma->pages = pages_tmp;

    /* allocate memory for scatter gather table containing scatter gather lists chained to each other */
    if (( sgt_tmp = kzalloc(sizeof(struct sg_table), GFP_KERNEL)) == NULL) {
      DEBUG("Error allocating memory for scatter gather table");
      retval = -ENOMEM;
      goto free_table;
    }
    mu->dma->sgt = sgt_tmp;

    DEBUG("Allocated memory");

    /* check page alignment of virtual address
     * (should not be necessary since pinned memory SHOULD be page aligned)
     * required for DMA to malloced memory
     */
    retval = is_page_aligned( (void *)(mu->msg).address );
    if ( retval != 0 ) {
      ERROR("Memory buffer is not page aligned");
      aligned_pointer = align_page( (void *)(mu->msg).address );
      retval = is_page_aligned( aligned_pointer );
      if ( retval != 0 )
        goto free_sgt;
      (mu->msg).address = aligned_pointer;
    }

    /* get pages in kernel space from user space address */
    down_read( &current->mm->mmap_sem ); //lock the memory management structure for our process

    // check kernel version as there was a change from 4.4.92.xx (??) on
    INFO("Found Kernel %d\n",LINUX_VERSION_CODE);
#if LINUX_VERSION_CODE <= KERNEL_VERSION(4,4,104)
    retval = get_user_pages(
                       current,
                       current->mm,
                       (unsigned long)(mu->msg).address,
                       N_PAGES,
                       1,  // write
                       0,  // do not force overriding of permissions
                       mu->dma->pages,
                       NULL
                       );
#elif LINUX_VERSION_CODE <= KERNEL_VERSION(4,4,104)
    retval = get_user_pages(
                       current,
                       current->mm,
                       (unsigned long)(mu->msg).address,
                       N_PAGES,
                       1,  // write
                       0,  // do not force overriding of permissions
                       mu->dma->pages,
                       NULL
                       );
#elif LINUX_VERSION_CODE <= KERNEL_VERSION(4,8,17)
    retval = get_user_pages_remote(
                     current,
                     current->mm,
                     (unsigned long)(mu->msg).address,
                     N_PAGES,
                     1,  // write
                     0,  // do not force overriding of permissions
                     mu->dma->pages,
                     NULL
                     );
#elif LINUX_VERSION_CODE <= KERNEL_VERSION(4,9,112)
        retval = get_user_pages_remote(
                     current,
                     current->mm,
                     (unsigned long)(mu->msg).address,
                     N_PAGES,
                     FOLL_WRITE, // write
                     mu->dma->pages,
                     NULL
                     );
#else
        retval = get_user_pages_remote(
                     current,
                     current->mm,
                     (unsigned long)(mu->msg).address,
                     N_PAGES,
                     FOLL_WRITE, // write
                     mu->dma->pages,
                     NULL, NULL
                     );
#endif

    up_read( &current->mm->mmap_sem );  // unlock
    if ( retval < 1 ) {
      ERROR( "Error %d while getting user pages \n", retval );
      mu->dma->npages = 0;
      goto free_sgt;
    }
    else {
      DEBUG("Number of pages received: %d", retval );
      mu->dma->npages = retval;
    }

    if ((retval = pci_set_dma_mask( mu->pci_dev, DMA_BIT_MASK(64)) ) < 0) goto release;
    if ((retval = pci_set_consistent_dma_mask( mu->pci_dev, DMA_BIT_MASK(64)) ) < 0) goto release;

    /* allocate scatter gather table */
    retval = sg_alloc_table_from_pages( mu->dma->sgt, mu->dma->pages, mu->dma->npages, 0, PAGE_SIZE * mu->dma->npages, GFP_KERNEL );
    if ( retval < 0 ) {
      ERROR("Could not set scatter gather table");
      goto release;
    }

    /* FPGA can only take 4096 addresses
     * => check for this limit
     */
    if ( mu->dma->sgt->nents > 4096 ) {
      ERROR("Number of DMA addresses: %d > 4096! Too much for FPGA", mu->dma->sgt->nents );
      retval = -ENOMEM;
      goto release;
    }

    mudaq_deactivate( mu ); // deactivate readout before setting dma addresses

    /** Map scatter gather lists to DMA addresses
     * calculate number of pages per list
     * pass address and number of pages to FPGA
     */
    for_each_sg(  mu->dma->sgt->sgl, sg, mu->dma->sgt->nents, i_list ) {
      count = dma_map_sg( &mu->pci_dev->dev, sg, 1, DMA_FROM_DEVICE );
      if ( count == 0 ) {
        ERROR("Could not map list %d", i_list);
        retval   = -EFAULT;
        n_mapped = i_list;
        goto unmap;
      }
      if ( dma_mapping_error( &mu->pci_dev->dev, sg_dma_address(sg) ) ) {
        ERROR("Error during mapping");
        retval   = -EADDRNOTAVAIL;
        n_mapped = i_list;
        goto unmap;
      }
      DEBUG("At %d: address %lx, length in pages: %lx", i_list, (long unsigned)sg_dma_address(sg), sg->length / PAGE_SIZE );
      if ( sg->length > MUDAQ_DMABUF_DATA_LEN )
        {
          ERROR("Length of scatter gather list larger than ring buffer");
          retval   = -EFAULT;
          n_mapped = i_list;
          goto unmap;
        }
      mudaq_set_dma_data_addr( mu, sg_dma_address(sg), i_list, sg->length / PAGE_SIZE );
      mu->dma->bus_addrs[i_list] = sg_dma_address(sg);
      mu->dma->n_pages[i_list]   = sg->length / PAGE_SIZE;
    }
    mudaq_set_dma_n_buffers( mu, mu->dma->sgt->nents );
    DEBUG("Found %d discontinuous pieces in memory buffer", mu->dma->sgt->nents );

    for ( i_list = 0; i_list < mu->dma->sgt->nents; i_list++) {
      mudaq_read_dma_data_addr( mu, i_list );
    }

    INFO("Setup mapping for pinned DMA data buffer");

    mu->to_user[1] = 0;      // reset offset in ring buffer
    mu->dma->flag   = true;  // flag to release pages and free memory when removing device

    break;
  }
  return 0;

 unmap:
  for_each_sg(  mu->dma->sgt->sgl, sg, n_mapped, i_list ) {
    dma_unmap_sg( mu->dev, sg, 1, DMA_FROM_DEVICE );
  }
 release:
  for ( i_page = 0; i_page < mu->dma->npages; i_page++) {
    if ( !PageReserved( mu->dma->pages[i_page] ) )
      SetPageDirty( mu->dma->pages[i_page] );
    //page_cache_release( mu->dma->pages[i_page] );
    put_page(mu->dma->pages[i_page]);
  }
 free_sgt:
  kfree( mu->dma->sgt );
 free_table:
  kfree( mu->dma->pages );
 fail:
  return retval;
}

static const struct file_operations mudaq_fops = {
    .owner          = THIS_MODULE,
    .read           = mudaq_fops_read,
    .write          = mudaq_fops_write,
    .mmap           = mudaq_fops_mmap,
    .open           = mudaq_fops_open,
    .unlocked_ioctl = mudaq_fops_ioctl,
    .release        = mudaq_fops_release,
};


/**
 * register / unregister mudaq device with the kernel
 */

/** register the mudaq device. device is live after succesful call. */
static int mudaq_register(struct mudaq * mu)
{
    int retval;
    dev_t devno;

    retval = minor_aquire(mu);
    if (retval < 0) goto fail_minor;

    devno = MKDEV(major, retval);
    DEBUG("allocated device(%d, %d)\n", MAJOR(devno), MINOR(devno));

    /* register the char device */
    cdev_init(&mu->char_dev, &mudaq_fops);
    mu->char_dev.owner = THIS_MODULE;
    retval = cdev_add(&mu->char_dev, devno, 1);
    if (retval < 0) {
        ERROR("could not add cdev(%d, %d)\n", MAJOR(devno), MINOR(devno));
        goto fail_cdev;
    }
    DEBUG("added cdev(%d, %d)\n", MAJOR(devno), MINOR(devno));

    /* register the sysfs device / kernel object */
    mu->dev = device_create(mudaq_class, &mu->pci_dev->dev, devno, mu,
                            DEVICE_NAME_TEMPLATE, MINOR(devno));
    if (IS_ERR(mu->dev)) {
        ERROR("could not create sys device\n");
        retval = PTR_ERR(mu->dev);
        goto fail_device;
    }

    INFO("registered '%s' cdev(%d, %d)\n", dev_name(mu->dev),
         MAJOR(devno), MINOR(devno));

    return 0;

 fail_device:
    cdev_del( &mu->char_dev );
 fail_cdev:
    minor_release( MINOR(devno) );
 fail_minor:
    return retval;
}

/** unregister the mudaq device */
static void mudaq_unregister(struct mudaq * mu)
{
    INFO("unregister '%s' cdev(%d, %d)\n", dev_name(mu->dev),
                                           MAJOR(mu->char_dev.dev),
                                           MINOR(mu->char_dev.dev));

    device_destroy(mudaq_class, mu->char_dev.dev);
    DEBUG("Destroyed device");
    minor_release(MINOR(mu->char_dev.dev));
    DEBUG("Released minor number");
    cdev_del(&mu->char_dev);
    DEBUG("Deleted character device");
}


/**
 * mudaq pci device handling
 */

static int __devinit mudaq_pci_probe(struct pci_dev * pdev, const struct pci_device_id * pid)
{
    int rv;
    struct mudaq * mu;

    if ( ( rv = mudaq_alloc(&mu) ) < 0) goto fail;
    DEBUG("Allocated mudaq");
    mu->pci_dev = pdev;
    if ( ( rv = pci_enable_device(pdev) ) < 0 ) goto out_free;
    DEBUG("Enabled device");
    if ( ( rv = mudaq_setup_mmio( pdev, mu ) ) < 0 ) goto out_disable;
    DEBUG("Setup mmio");
    if ( ( rv = mudaq_setup_dma( pdev, mu ) ) < 0 ) goto out_mmio;
    DEBUG("Setup dma");
    if ( ( rv = mudaq_register(mu) ) < 0 ) goto out_dma;
    DEBUG("Registered mudaq");
    pci_set_drvdata(pdev, mu);
    if ( ( rv = mudaq_setup_msi(mu) ) < 0 ) goto out_unregister;
    DEBUG("Setup MSI interrupts");

    INFO("Device setup finished");

    mu->dma->flag = false; // initially no dma mapping from ioctl present

    return 0;

 out_unregister:
    mudaq_unregister( mu ) ;
 out_dma:
    mudaq_clear_dma( pdev, mu );
 out_mmio:
    mudaq_clear_mmio( pdev, mu );
 out_disable:
    pci_disable_device( pdev );
 out_free:
    mudaq_free( mu) ;
 fail:
    return rv;
}

static void __devexit mudaq_pci_remove(struct pci_dev * pdev)
{
  struct mudaq * mu  = (struct mudaq *)pci_get_drvdata(pdev);

  if ( mu->dma->flag == true)
    mudaq_free_dma( mu );
  mudaq_clear_msi( mu );
  mudaq_unregister( mu );
  mudaq_clear_dma( pdev, mu );
  mudaq_clear_mmio( pdev, mu );
  pci_disable_device( pdev );
  mudaq_free( mu );

  INFO("Device removed");
}

static struct pci_driver mudaq_pci_driver = {
    .name =     DRIVER_NAME,
    .id_table = PCI_DEVICE_IDS,
    .probe =    mudaq_pci_probe,
    .remove =   __devexit_p(mudaq_pci_remove),
};

MODULE_DEVICE_TABLE(pci, PCI_DEVICE_IDS);


/**
 * module init and exit
 */

static int __init mudaq_init(void)
{
    int rv;
    dev_t first;

    /* create the device class */
    mudaq_class = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(mudaq_class)) {
        ERROR("could not create device class\n");
        rv = PTR_ERR(mudaq_class);
        goto fail;
    }

    /* allocate a major number and some minor numbers */
    rv = alloc_chrdev_region(&first, 0, MAX_NUM_DEVICES, DRIVER_NAME);
    if (rv < 0) {
        ERROR("could not allocate major number\n");
        goto fail_chrdev;
    }
    major = MAJOR(first);
    DEBUG("allocated major number %d\n", major);

    /* register the pci driver */
    rv = pci_register_driver(&mudaq_pci_driver);
    if (rv < 0) {
        ERROR("could not register pci driver\n");
        goto fail_pci;
    }

    DEBUG("module initialized\n");
    return 0;

fail_pci:
    unregister_chrdev_region(first, MAX_NUM_DEVICES);
fail_chrdev:
    class_destroy(mudaq_class);
fail:
    return rv;
}

static void __exit mudaq_exit(void)
{
    pci_unregister_driver(&mudaq_pci_driver);
    unregister_chrdev_region(MKDEV(major, 0), MAX_NUM_DEVICES);
    class_destroy(mudaq_class);

    DEBUG("module removed\n");
}

module_init(mudaq_init);
module_exit(mudaq_exit);

MODULE_DESCRIPTION("mu3e pcie readout board driver");
MODULE_AUTHOR("Moritz Kiehn <kiehn@physi.uni-heidelberg.de>");
MODULE_LICENSE("GPL");