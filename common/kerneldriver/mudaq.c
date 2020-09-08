/*
 * a driver for the mu3e pcie readout board
 *
 * @author  Moritz Kiehn <kiehn@physi.uni-heidelberg.de>
 * @date    2013-10-28
 *
 * modified to work with streaming DMA and not using the uio functions anymore by
 * @author Dorothea vom Bruch <vombruch@physi.uni-heidelberg.de>
 * @date   2015-08-05
 * Some code is based on uio driver (drivers/uio/uio.c), partially exactly the same code
 */

#include <linux/module.h>

#include "mudaq.h"

#include "../include/mudaq_device_constants.h"
#include "../include/mudaq_registers.h"

#include "dmabuf/dmabuf_chrdev.h"

#include <linux/version.h>

#include <linux/pci.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 11, 0) // add `signal.h` file
#include <linux/sched/signal.h>
#endif

#define ERROR(fmt, ...) \
    printk(KERN_ERR   "[%s/%s] " pr_fmt(fmt), THIS_MODULE->name, __FUNCTION__, ##__VA_ARGS__)
#define INFO(fmt, ...) \
    printk(KERN_INFO  "[%s/%s] " pr_fmt(fmt), THIS_MODULE->name, __FUNCTION__, ##__VA_ARGS__)
#define DEBUG(fmt, ...) \
    printk(KERN_DEBUG "[%s/%s] " pr_fmt(fmt), THIS_MODULE->name, __FUNCTION__, ##__VA_ARGS__)

//
// module-wide global variables
//

static struct chrdev* chrdev_mudaq;
static struct chrdev* chrdev_dmabuf;

static DEFINE_IDA(minor_ida);

static
const struct pci_device_id PCI_DEVICE_IDS[] = {
    { PCI_DEVICE(0x1172, 0x0004), },
    { 0, },
};

static
int wrap_ring(int int1, int int2, int wrap, int divisor) {
    int result = 0;
    if ((int1 - int2) > 0) {
        result = (int1 - int2) / divisor;
    }
    else if ((int1 - int2) < 0) {
        result = wrap + (int1 - int2) / divisor;
    }
    else if ((int1 - int2) == 0) {
        result = 0;
    }
    return result;
}

//
// mudaq structures and related functions
//

struct mudaq {
    struct device *dev;
    struct pci_dev *pci_dev;
    int minor;
    u32 to_user[2];
    long irq;
    atomic_t event;
    wait_queue_head_t wait;
    struct mudaq_mem *mem;
    struct mudaq_dma *dma;
};

/*
   First four entries of internal_addr, phys_size and phys_addr are for
   the four PCIe BAR regions
   fifth entry is for the DMA control buffer
*/
struct mudaq_mem {
    __iomem u32 *internal_addr[5];
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
    struct sg_table *sgt;
};

/** free the given mudaq struct and all associated memory */
static
void mudaq_free(struct mudaq *mu) {
    if(mu == NULL) return;

    if(mu->mem) { kfree(mu->mem); mu->mem = NULL; }
    if(mu->dma) { kfree(mu->dma); mu->dma = NULL; }
    kfree(mu);
}

/** allocate a new mudaq struct and initialize its state */
static
struct mudaq* mudaq_alloc(void) {
    int error;

    /* allocate memory for the device structure */
    struct mudaq* mu = kzalloc(sizeof(struct mudaq), GFP_KERNEL);
    if(mu == NULL) {
        ERROR("could not allocate memory for 'struct mudaq'\n");
        error = -ENOMEM;
        goto fail;
    }

    mu->mem = kzalloc(sizeof(struct mudaq_mem), GFP_KERNEL);
    if(mu->mem == NULL) {
        ERROR("could not allocate memory for 'struct mudaq_mem'\n");
        error = -ENOMEM;
        goto fail;
    }

    mu->dma = kzalloc(sizeof(struct mudaq_dma), GFP_KERNEL);
    if(mu->dma == NULL) {
        ERROR("could not allocate memory for 'struct mudaq_dma'\n");
        error = -ENOMEM;
        goto fail;
    }

    return mu;

fail:
    mudaq_free(mu);
    return ERR_PTR(error);
}

static
int mudaq_interrupt_control(struct mudaq *info, s32 irq_on) {
    /* no need to do anything. interrupts are activated from userspace */
    DEBUG("Called interrupt control w/ %#x\n", irq_on);
    return 0;
}

static
irqreturn_t mudaq_interrupt_handler(int irq, struct mudaq *mu) {
    /* no need to do anything. just acknowledge that something happened. */
    DEBUG("Received interrupt\n");
    return IRQ_HANDLED;
}

/* Trigger an interrupt event */
static
void mudaq_event_notify(struct mudaq *mu) {
    atomic_inc(&mu->event); // interrupt counter
    wake_up_interruptible(&mu->wait); // wake up read function
}

/* Hardware interrupt handler */
static
irqreturn_t mudaq_interrupt(int irq, void *dev_id) {
    struct mudaq* mu = dev_id;
    irqreturn_t ret = mudaq_interrupt_handler(irq, mu);

    if (ret == IRQ_HANDLED)
        mudaq_event_notify(mu);

    return ret;
}

/* Access registers */
inline
__iomem u32 *mudaq_register_rw(struct mudaq *mu, unsigned index) {
    __iomem u32 *base = mu->mem->internal_addr[0];
    return base + index;
}

inline
__iomem u32 *mudaq_register_ro(struct mudaq *mu, unsigned index) {
    __iomem u32 *base = mu->mem->internal_addr[1];
    return base + index;
}

#define mudaq_write32_test(mu, value, index) ({ \
    void __iomem* addr = mudaq_register_rw((mu), index); \
    u32 a = (value), b; \
    iowrite32(a, addr); \
    b = ioread32(addr); \
    if(a != b) { \
        ERROR("write '%u' at '%s', but read back is '%u'\n", a, #index, b); \
    } \
})

static
void mudaq_deactivate(struct mudaq *mu) {
    mudaq_write32_test(mu, 0, DATAGENERATOR_REGISTER_W);
    mudaq_write32_test(mu, 0, DMA_REGISTER_W);
}

/* Copy dma bus addresses to device registers */
static
void mudaq_set_dma_ctrl_addr(struct mudaq *mu, dma_addr_t ctrl_handle) {
    mudaq_write32_test(mu, ctrl_handle & 0xFFFFFFFF, DMA_CTRL_ADDR_LOW_REGISTER_W);
    mudaq_write32_test(mu, ctrl_handle >> 32, DMA_CTRL_ADDR_HI_REGISTER_W);
}

static
int mudaq_set_dma_data_addr(struct mudaq* mu,
                                    dma_addr_t data_handle,
                                    u32 mem_location,
                                    u32 n_pages) {
    u32 regcontent;
    regcontent = SET_DMA_NUM_PAGES_RANGE(0x0, n_pages);
    regcontent = SET_DMA_RAM_LOCATION_RANGE(regcontent, mem_location);

    mudaq_write32_test(mu, regcontent, DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W);
    // DMA address in firmware is updated only during write to LOW address
    mudaq_write32_test(mu, data_handle >> 32, DMA_DATA_ADDR_HI_REGISTER_W);
    mudaq_write32_test(mu, data_handle & 0xFFFFFFFF, DMA_DATA_ADDR_LOW_REGISTER_W);

    return 0;
}

static
void mudaq_set_dma_n_buffers(struct mudaq* mu, u32 n_buffers) {
    mudaq_write32_test(mu, n_buffers, DMA_NUM_ADDRESSES_REGISTER_W);
}

//
// mudaq device file operations
//

static
ssize_t mudaq_fops_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos) {
    struct mudaq* mu = filp->private_data;
    DECLARE_WAITQUEUE(wait, current);
    size_t retval;
    u32 new_event_count;
    u32 *dma_buf_ctrl;
    int n_interrupts = 0;

    if (!mu->irq) return -EIO;
    if (count != sizeof(mu->to_user)) return -EINVAL;

    add_wait_queue(&mu->wait, &wait); // add read process to wait queue

    do {
        set_current_state(TASK_INTERRUPTIBLE); // mark process as being asleep but interruptible

        new_event_count = atomic_read(&mu->event);
        if (new_event_count != mu->to_user[0]) {       // interrupt occured
            mu->to_user[0] = new_event_count;              // pass interrupt number
            dma_buf_ctrl = mu->mem->internal_addr[4];
            /* How many DMA blocks were transfered (in units of interrupts (64 blocks) )?
             * Offset in ring buffer comes in bytes from FPGA, transform to uint32_t words here
             */
            DEBUG("interrupt number: %d, address: %x%x\n", new_event_count, (int) dma_buf_ctrl[1], (int) dma_buf_ctrl[0]);
            n_interrupts = wrap_ring((int) dma_buf_ctrl[3] >> 2, (int) mu->to_user[1], N_BUFFERS,
                                     PAGES_PER_INTERRUPT * PAGE_SIZE / 4);
            if (n_interrupts != 1) {
                DEBUG("ctrl buffer: %x, previous: %x\n", (int) dma_buf_ctrl[3] >> 2, (int) mu->to_user[1]);
                INFO("Missed %d interrupt(s)\n", n_interrupts);
            }

            mu->to_user[1] = (u32) dma_buf_ctrl[3] >> 2;  // position in ring buffer last written to + 1 (in words)
            if (copy_to_user(buf, &(mu->to_user), sizeof(mu->to_user)))
                retval = -EFAULT;
            else {
                mu->to_user[0] = new_event_count;             // save event_count in case copying to user space fails
                retval = count;
            }
            break;
        }

        if (filp->f_flags & O_NONBLOCK) { // check if user requested non-blocking I/O
            retval = -EAGAIN;
            break;
        }
        if (signal_pending(current)) { // restart system when receiving interrupt
            retval = -ERESTARTSYS;
            break;
        }
        schedule(); // check whether interrupt occured while going through above operations

    } while (1);

    __set_current_state(TASK_RUNNING); // process is able to run
    remove_wait_queue(&mu->wait, &wait); // remove process from wait queue so that it can only be awakened once

    return retval;

}

static
ssize_t mudaq_fops_write(struct file *filp, const char __user *buf, size_t count, loff_t *f_pos) {
    struct mudaq* mu = filp->private_data;
    s32 irq_on;
    ssize_t retval;

    if (!mu->irq) return -EIO;
    if (count != sizeof(s32)) return -EINVAL;

    if (copy_from_user(&irq_on, buf, count) != 0) return -EFAULT;

    retval = mudaq_interrupt_control(mu, irq_on);
    if(retval != 0) {
        return -EIO;
    }

    return sizeof(s32);
}

static
int mudaq_fops_mmap(struct file *filp, struct vm_area_struct *vma) {
    struct mudaq* mu = filp->private_data;
    int index = (int) vma->vm_pgoff;
    unsigned long requested_pages = 0, actual_pages;
    int rv = 0;

    DEBUG("Mapping for index %d\n", index);

    /* use the requested page offset as an index to select the pci region that
       should be mapped. readout board has 4 accessible memory regions
       registers rw, registers ro, memory rw and memory ro
       in addition, we have one dma ctrl buffer */
    if ((index < 0) || (index > 4)) {
        DEBUG("invalid mmap memory index %d\n", index);
        return -EINVAL;
    }

    /* only allow mapping of the whole selected memory.
       WARNING actual size must not be page-aligned
       But minimum virtual address space of vma
       is one page */
    actual_pages = PAGE_ALIGN(mu->mem->phys_size[index]) >> PAGE_SHIFT;
    requested_pages = vma_pages(vma);

    if (requested_pages != actual_pages) {
        ERROR("invalid mmap pages requested. requested %lu actual %lu\n",
              requested_pages, actual_pages);
        rv = -EINVAL;
        goto out;
    }
    if (mu->mem->phys_addr[index] + mu->mem->phys_size[index] < mu->mem->phys_addr[index]) {
        ERROR("invalid memory settings. phys_addr %d size %d\n",
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

    switch (index) {
    case 0: // rw registers
    case 2: // rw memory
        vma->vm_flags |= VM_WRITE;
        fallthrough;
    case 1: // ro registers
    case 3: // ro memory
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
        return io_remap_pfn_range(
            vma,
            vma->vm_start,
            PHYS_PFN(mu->mem->phys_addr[index]),
            vma->vm_end - vma->vm_start,
            vma->vm_page_prot
        );
    case 4: // dma control buffer
        return dma_mmap_coherent(
            mu->dev,
            vma,
            mu->mem->internal_addr[index],
            mu->mem->bus_addr_ctrl,
            mu->mem->phys_size[index]
        );

    default:
        break;
    }

    /* this should NEVER happen */
    ERROR("something went horribly wrong. run for your lives\n");
    rv = -EINVAL;

out:
    return rv;
}

static
void mudaq_clear_mmio(struct pci_dev *dev, struct mudaq *mu) {
    if(mu == NULL || mu->mem == NULL) return;

    for(int i = 0; i < 4; i++) {
        if(mu->mem->internal_addr[i] == NULL) continue;
        pci_iounmap(dev, mu->mem->internal_addr[i]);
        mu->mem->internal_addr[i] = NULL;
    }

    pci_release_regions(dev);
}

static
int mudaq_setup_mmio(struct pci_dev *pdev, struct mudaq *mu) {
    const int bars[] = {0, 1, 2, 3};
    const char *names[] = {"registers_rw", "registers_ro", "memory_rw", "memory_ro"};
    int rv = 0;

    /* request access to pci BARs */
    rv = pci_request_regions(pdev, THIS_MODULE->name);
    if (rv < 0) {
        ERROR("could not request memory regions\n");
        goto out;
    }

    /* Map PCI bar regions to __iomem addresses (addressable from kernel)  */
    for (int i = 0; i < 4; i++) {
        mu->mem->phys_addr[i] = pci_resource_start(pdev, bars[i]);
        mu->mem->phys_size[i] = pci_resource_len(pdev, bars[i]);
        mu->mem->internal_addr[i] = pci_iomap(pdev, bars[i], mu->mem->phys_size[i]);
        if (mu->mem->internal_addr[i] == NULL) {
            ERROR("pci_iomap failed for '%s'\n", names[i]);
            rv = -ENODEV;
            goto fail_unmap;
        }
        DEBUG("Bar %d: at %x with size %d at internal address %lx\n", bars[i], mu->mem->phys_addr[i],
              mu->mem->phys_size[i], (long unsigned) mu->mem->internal_addr[i]);
    }

    return 0;

fail_unmap:
    mudaq_clear_mmio(pdev, mu);
out:
    return rv;
}

static
void mudaq_clear_dma(struct pci_dev *pdev, struct mudaq *mu) {
    /* deactivate any readout activity before removing the dma memory */
    mudaq_deactivate(mu);
    mudaq_set_dma_ctrl_addr(mu, 0x0);

    dma_free_coherent(&pdev->dev, mu->mem->phys_size[4], mu->mem->internal_addr[4], mu->mem->bus_addr_ctrl);
}

/** setup coherent dma
 *
 * set dma masks and create one DMA buffer for control
 *
 * WARNING
 * assumes that the read/write registers are already mapped
 */
static
int mudaq_setup_dma(struct pci_dev *pdev, struct mudaq *mu) {
    int rv;
    dma_addr_t ctrl_addr;
    void *ctrl_internal;
    size_t ctrl_size = MUDAQ_BUFFER_CTRL_SIZE;
    const char *name = "dma_ctrl";

    if ((rv = pci_set_dma_mask(pdev, DMA_BIT_MASK(64))) < 0) return rv;
    if ((rv = pci_set_consistent_dma_mask(pdev, DMA_BIT_MASK(64))) < 0) return rv;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 0, 0) // remove `dma_zalloc_coherent` function
    ctrl_internal = dma_alloc_coherent(&pdev->dev, ctrl_size, &ctrl_addr, GFP_KERNEL);
#else
    ctrl_internal = dma_zalloc_coherent(&pdev->dev, ctrl_size, &ctrl_addr, GFP_KERNEL);
#endif

    if (ctrl_internal == NULL) {
        ERROR("could not allocate dma control buffer\n");
        rv = -ENOMEM;
        goto out;
    }

    mu->mem->bus_addr_ctrl = ctrl_addr;
    mu->mem->phys_size[4] = ctrl_size;
    mu->mem->internal_addr[4] = ctrl_internal;

//    DEBUG("Addr: %d , Size: %d, Internal Add %d\n",int(ctrl_addr) int(ctrl_size) int(ctrl_internal));
    DEBUG("setup dma for '%s' dma_addr %#zx size %d bytes\n",
          name, (size_t) mu->mem->bus_addr_ctrl, mu->mem->phys_size[4]);

    /* deactivate any readout activity to always start in a consistent state
       before setting up the dma addresses */
    mudaq_deactivate(mu);
    mudaq_set_dma_ctrl_addr(mu, ctrl_addr);
    return 0;

out:
    return rv;
}

static
void mudaq_clear_msi(struct mudaq *mu) {
    /* release irq */
    devm_free_irq(mu->dev, mu->irq, mu);
    DEBUG("Freed irq\n");

    /* disable MSI */
    pci_disable_msi(mu->pci_dev);
    pci_clear_mwi(mu->pci_dev);
}

static
int mudaq_setup_msi(struct mudaq *mu) {
    int rv;
    pci_set_master(mu->pci_dev);

    /* enable message-write-invalidate transactions for cache coherency. */
    if ((rv = pci_set_mwi(mu->pci_dev)) < 0) return rv;

    /* use message-based interrupts */
    if ((rv = pci_enable_msi(mu->pci_dev)) < 0) goto out_clear_mwi;
    mu->irq = mu->pci_dev->irq;

    /* request irq */
    init_waitqueue_head(&mu->wait);
    atomic_set(&mu->event, 0);
    rv = devm_request_irq(mu->dev, mu->irq, mudaq_interrupt, 0, THIS_MODULE->name, mu);
    if (rv) {
        DEBUG("Failed to initialize irq, error: %d\n", rv);
        goto out_clear_msi;
    }

    return 0;

out_clear_msi:
    pci_disable_msi(mu->pci_dev);
out_clear_mwi:
    pci_clear_mwi(mu->pci_dev);
    return rv;
}

static
void mudaq_free_dma(struct mudaq *mu) {
    if(mu == NULL) return;

    mudaq_deactivate(mu);
    mudaq_set_dma_n_buffers(mu, 0);
    for(int i = 0; i < 4096; i++) {
        mudaq_set_dma_data_addr(mu, 0, i, 0);
    }

    if(mu->dma == NULL) return;

    if(mu->dma->sgt != NULL) {
        dma_unmap_sg(&mu->pci_dev->dev, mu->dma->sgt->sgl, mu->dma->sgt->nents, DMA_FROM_DEVICE);
        sg_free_table(mu->dma->sgt);
        kfree(mu->dma->sgt);
        mu->dma->sgt = NULL;
    }

    for(int i = 0; i < mu->dma->npages; i++) {
        if (PageReserved(mu->dma->pages[i]))
            INFO("Page %d is in reserved space\n", i);
        if (!PageReserved(mu->dma->pages[i]))
            SetPageDirty(mu->dma->pages[i]);
//        page_cache_release( mu->dma->pages[i] );
        put_page(mu->dma->pages[i]);
    }
    mu->dma->npages = 0;

    if(mu->dma->pages != NULL) {
        kfree(mu->dma->pages);
        mu->dma->pages = NULL;
    }

    INFO("Freed pinned DMA buffer in kernel space\n");

    mu->dma->flag = false;
}

/* release function is called when a process closes the device file */
static
int mudaq_fops_release(struct inode *inode, struct file *filp) {
    struct mudaq *mu = filp->private_data;
    if (mu->dma->flag == true)
        mudaq_free_dma(mu);

    return 0;
}

static
int mudaq_fops_open(struct inode *inode, struct file *filp) {
    struct mudaq* mu = container_of(inode->i_cdev, struct chrdev_device, cdev)->private_data;
    if (mu == NULL) return -ENODEV;
    filp->private_data = mu;

    return 0;
}

static
long mudaq_fops_ioctl(struct file *filp,
                      unsigned int cmd,    /* magic and sequential number for ioctl */
                      unsigned long ioctl_param) {
    int retval = 0;
    struct mudaq* mu = filp->private_data;
    int err = 0;
    int i_list;
    struct scatterlist *sg;
    u32 new_event_count;
    void __user* user_buffer = (void __user*)ioctl_param;

    mu->dma->flag = false; // in case something goes wrong

    /*
     * Extract type and number bitfields from IOCTL cmd
     * Return error in case of wrong cmd's
     * type = magic, number = sequential number
     */

    if (_IOC_TYPE(cmd) != MUDAQ_IOC_TYPE) {
        retval = -ENOTTY;
        goto fail;
    }
    if (_IOC_NR(cmd) > MUDAQ_IOC_NR) {
        retval = -ENOTTY;
        goto fail;
    }

    /*
     * Verify that address of parameter does not point to kernel space memory
     * access_ok checks for this
     *    returns 1 for access ok
     *            0 for access not ok
     * VERIFY_READ  = read user space memory
     * VERIFY_WRITE = write to user space memory
     * _IOC_READ    = read kernel space memory
     * _IOC_WRITE   = write data to kernel space memory
     * https://github.com/zfsonlinux/zfs/issues/8261 --> kernel > 5.0
     */

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 0, 0) // remove `type` argument
    err = !access_ok(user_buffer, _IOC_SIZE(cmd));
#else
    if (_IOC_DIR(cmd) & _IOC_READ) {
        err = !access_ok(VERIFY_WRITE, user_buffer, _IOC_SIZE(cmd));
    }
    else if (_IOC_DIR(cmd) & _IOC_WRITE) {
        err = !access_ok(VERIFY_READ, user_buffer, _IOC_SIZE(cmd));
    }
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
        new_event_count = atomic_read(&mu->event);
        retval = copy_to_user(user_buffer, &new_event_count, sizeof(new_event_count));
        if (retval > 0) {
            ERROR("copy_to_user failed with error %d \n", retval);
            goto fail;
        }
        break;
    case MAP_DMA : { // receive a pointer to a virtual address in user space
        struct mesg msg;
        retval = copy_from_user(&msg, user_buffer, sizeof(msg));
        if (retval > 0) {
            ERROR("copy_from_user failed with error %d \n", retval);
            goto fail;
        }
        INFO("Received following virtual address: 0x%0llx \n", (u64) msg.address);
        if(!PAGE_ALIGNED(msg.address)) {
            pr_err("address is not page aligned\n");
            return -EINVAL;
        }
        INFO("Size of allocated memory: %zu bytes\n", msg.size);
        if(msg.size != MUDAQ_DMABUF_DATA_LEN) {
            pr_err("invalid size\n");
            return -EINVAL;
        }

        /* allocate memory for page pointers */
        mu->dma->pages = kzalloc(sizeof(long unsigned) * N_PAGES, GFP_KERNEL);
        if(mu->dma->pages == NULL) {
            DEBUG("Error allocating memory for page table\n");
            retval = -ENOMEM;
            goto fail;
        }

        /* allocate memory for scatter gather table containing scatter gather lists chained to each other */
        mu->dma->sgt = kzalloc(sizeof(struct sg_table), GFP_KERNEL);
        if(mu->dma->sgt == NULL) {
            DEBUG("Error allocating memory for scatter gather table\n");
            retval = -ENOMEM;
            goto fail;
        }

        DEBUG("Allocated memory\n");

        /* get pages in kernel space from user space address */
        down_read(&current->mm->mmap_sem); //lock the memory management structure for our process

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 6, 0) // add `get_user_pages_remote` function
        retval = get_user_pages_remote(
#else
        retval = get_user_pages(
#endif
            current,
            current->mm,
            (unsigned long)msg.address,
            N_PAGES,
            FOLL_WRITE, // write
#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 9, 0) // remove `force` flag
            0, // do not force access
#endif
            mu->dma->pages,
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 10, 0) // add `locked` flag
            NULL, NULL
#else
            NULL
#endif
        );

        up_read(&current->mm->mmap_sem);  // unlock

        if(retval < 0) {
            ERROR("Error %d while getting user pages \n", retval);
            mu->dma->npages = 0;
            goto fail;
        }
        DEBUG("Number of pages received: %d\n", retval);
        mu->dma->npages = retval;
        if(mu->dma->npages != N_PAGES) {
            pr_err("get_user_pages != N_PAGES\n");
            goto fail;
        }

        /* allocate scatter gather table */
        retval = sg_alloc_table_from_pages(mu->dma->sgt, mu->dma->pages, mu->dma->npages, 0,
                                           PAGE_SIZE * mu->dma->npages, GFP_KERNEL);
        if (retval < 0) {
            ERROR("Could not set scatter gather table\n");
            goto fail;
        }

        DEBUG("Found %d discontinuous pieces in memory buffer\n", mu->dma->sgt->nents);

        /* FPGA can only take 4096 addresses
         * => check for this limit
         */
        if (mu->dma->sgt->nents > 4096) {
            ERROR("Number of DMA addresses: %d > 4096! Too much for FPGA\n", mu->dma->sgt->nents);
            retval = -ENOMEM;
            goto fail;
        }

        /** Map scatter gather lists to DMA addresses
         * calculate number of pages per list
         * pass address and number of pages to FPGA
         */
        retval = dma_map_sg(&mu->pci_dev->dev, mu->dma->sgt->sgl, mu->dma->sgt->nents, DMA_FROM_DEVICE);
        if(retval == 0) {
            ERROR("Could not map sg list\n");
            retval = -EFAULT;
            goto fail;
        }

        mudaq_deactivate(mu); // deactivate readout before setting dma addresses

        for_each_sg(mu->dma->sgt->sgl, sg, mu->dma->sgt->nents, i_list) {
            dma_addr_t dma_addr = sg_dma_address(sg);
            int dma_pages = sg_dma_len(sg) >> PAGE_SHIFT;
            DEBUG("At %d: address %llx, length in pages: %d\n", i_list, dma_addr, dma_pages);
            mu->dma->bus_addrs[i_list] = dma_addr;
            mu->dma->n_pages[i_list] = dma_pages;
            mudaq_set_dma_data_addr(mu, dma_addr, i_list, dma_pages);
        }
        mudaq_set_dma_n_buffers(mu, mu->dma->sgt->nents);

        INFO("Setup mapping for pinned DMA data buffer\n");

        mu->to_user[1] = 0;      // reset offset in ring buffer
        mu->dma->flag = true;  // flag to release pages and free memory when removing device

        break;
    }

    default:
        return -EINVAL;
    }

    return 0;

fail:
    mudaq_free_dma(mu);
    return retval;
}

static const
struct file_operations mudaq_fops = {
    .owner          = THIS_MODULE,
    .read           = mudaq_fops_read,
    .write          = mudaq_fops_write,
    .mmap           = mudaq_fops_mmap,
    .open           = mudaq_fops_open,
    .unlocked_ioctl = mudaq_fops_ioctl,
    .release        = mudaq_fops_release,
};

//
// register / unregister mudaq device with the kernel
//

/** unregister the mudaq device */
static
void mudaq_unregister(struct mudaq *mu) {
    struct chrdev_device* chrdev_device;
    struct dmabuf* dmabuf;

    if(mu == NULL || mu->minor < 0) return;

    chrdev_device = &chrdev_dmabuf->devices[mu->minor];
    dmabuf = chrdev_device->private_data;
    chrdev_device_del(chrdev_device);

    mudaq_deactivate(mu);
    mudaq_set_dma_n_buffers(mu, 0);
    for(int i = 0; i < 4096; i++) {
        mudaq_set_dma_data_addr(mu, 0, i, 0);
    }

    dmabuf_free(dmabuf);

    chrdev_device_del(&chrdev_mudaq->devices[mu->minor]);

    ida_free(&minor_ida, mu->minor);
    DEBUG("Released minor number\n");
}

/* register the mudaq device. device is live after successful call. */
static
int mudaq_register(struct mudaq *mu) {
    int error = 0;
    int minor;
    struct chrdev_device* chrdev_device;
    struct dmabuf* dmabuf;

    minor = ida_alloc_range(&minor_ida, 0, MAX_NUM_DEVICES - 1, GFP_KERNEL);
    if(minor < 0) goto err_out;

    chrdev_device = chrdev_device_add(chrdev_mudaq, minor, &mudaq_fops, &mu->pci_dev->dev, mu);
    if(IS_ERR_OR_NULL(chrdev_device)) {
        error = PTR_ERR(chrdev_device);
        goto err_out;
    }
    mu->dev = chrdev_device->device;

    dmabuf = dmabuf_alloc(&mu->pci_dev->dev, MUDAQ_DMABUF_DATA_LEN);
    if(IS_ERR_OR_NULL(dmabuf)) {
        error = PTR_ERR(dmabuf);
        goto err_out;
    }

    chrdev_device = chrdev_device_add(chrdev_dmabuf, minor, &dmabuf_chrdev_fops, &mu->pci_dev->dev, dmabuf);
    if(IS_ERR_OR_NULL(chrdev_device)) {
        error = PTR_ERR(chrdev_device);
        goto err_out;
    }

    mudaq_deactivate(mu);

    // setup device DMA tables using dmabuf entries
    {
        int i = 0;
        struct dmabuf_entry* entry;
        list_for_each_entry(entry, &dmabuf->entries, list_head) {
            M_INFO("set dma entry %d: dma_addr = %llx, dma_pages = %lu\n", i, entry->dma_handle, entry->size >> PAGE_SHIFT);
            mudaq_set_dma_data_addr(mu, entry->dma_handle, i, entry->size >> PAGE_SHIFT);
            i++;
        }
        mudaq_set_dma_n_buffers(mu, i + 1);
    }
    mu->to_user[1] = 0;

    return 0;

err_out:
    mudaq_unregister(mu);
    return error;
}

//
// mudaq pci device handling
//

static
void mudaq_pci_remove(struct pci_dev *pdev) {
    struct mudaq* mu = pci_get_drvdata(pdev);

    if (mu->dma->flag == true)
        mudaq_free_dma(mu);
    mudaq_clear_msi(mu);
    mudaq_unregister(mu);
    mudaq_clear_dma(pdev, mu);
    mudaq_clear_mmio(pdev, mu);
    pci_disable_device(pdev);
    mudaq_free(mu);

    INFO("Device removed\n");
}

static
int mudaq_pci_probe(struct pci_dev *pdev, const struct pci_device_id *pid) {
    int rv;

    struct mudaq* mu = mudaq_alloc();
    if(IS_ERR_OR_NULL(mu)) {
        rv = PTR_ERR(mu);
        mu = NULL;
        goto fail;
    }
    DEBUG("Allocated mudaq\n");

    mu->pci_dev = pdev;
    if ((rv = pci_enable_device(pdev)) < 0) goto out_free;
    DEBUG("Enabled device\n");
    if ((rv = mudaq_setup_mmio(pdev, mu)) < 0) goto out_disable;
    DEBUG("Setup mmio\n");
    if ((rv = mudaq_setup_dma(pdev, mu)) < 0) goto out_mmio;
    DEBUG("Setup dma\n");
    if ((rv = mudaq_register(mu)) < 0) goto out_dma;
    DEBUG("Registered mudaq\n");
    pci_set_drvdata(pdev, mu);
    if ((rv = mudaq_setup_msi(mu)) < 0) goto out_unregister;
    DEBUG("Setup MSI interrupts\n");

    INFO("Device setup finished\n");

    mu->dma->flag = false; // initially no dma mapping from ioctl present

    return 0;

out_unregister:
    mudaq_unregister(mu);
out_dma:
    mudaq_clear_dma(pdev, mu);
out_mmio:
    mudaq_clear_mmio(pdev, mu);
out_disable:
    pci_disable_device(pdev);
out_free:
    mudaq_free(mu);
fail:
    return rv;
}

static
struct pci_driver mudaq_pci_driver = {
    .name =     THIS_MODULE->name,
    .id_table = PCI_DEVICE_IDS,
    .probe =    mudaq_pci_probe,
    .remove =   mudaq_pci_remove,
};

MODULE_DEVICE_TABLE(pci, PCI_DEVICE_IDS);

//
// module init and exit
//

static
void __exit mudaq_exit(void) {
    pci_unregister_driver(&mudaq_pci_driver);

    chrdev_free(chrdev_dmabuf);
    chrdev_free(chrdev_mudaq);

    DEBUG("module removed\n");
}

static
int __init mudaq_init(void) {
    int error;

    chrdev_mudaq = chrdev_alloc("mudaq", MAX_NUM_DEVICES);
    if(IS_ERR_OR_NULL(chrdev_mudaq)) {
        error = PTR_ERR(chrdev_mudaq);
        goto err_out;
    }
    chrdev_dmabuf = chrdev_alloc("mudaq_dmabuf", MAX_NUM_DEVICES);
    if(IS_ERR_OR_NULL(chrdev_dmabuf)) {
        error = PTR_ERR(chrdev_dmabuf);
        goto err_out;
    }

    /* register the pci driver */
    error = pci_register_driver(&mudaq_pci_driver);
    if(error) {
        ERROR("could not register pci driver\n");
        goto err_out;
    }

    DEBUG("module initialized\n");

    return 0;

err_out:
    chrdev_free(chrdev_dmabuf);
    chrdev_free(chrdev_mudaq);
    return error;
}

module_exit(mudaq_exit);
module_init(mudaq_init);

MODULE_DESCRIPTION("mu3e pcie readout board driver");
MODULE_AUTHOR("Moritz Kiehn <kiehn@physi.uni-heidelberg.de>");
MODULE_LICENSE("GPL");
