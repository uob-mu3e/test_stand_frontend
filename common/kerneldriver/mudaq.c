/*
 * a driver for the mu3e pcie readout board
 *
 * @author  Moritz Kiehn <kiehn@physi.uni-heidelberg.de>
 * @date    2013-10-28
 *
 * modified to work with streaming DMA and not using the uio functions anymore
 * by
 * @author Dorothea vom Bruch <vombruch@physi.uni-heidelberg.de>
 * @date   2015-08-05
 * Some code is based on uio driver (drivers/uio/uio.c), partially exactly the
 * same code
 */

#include "mudaq.h"

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/uaccess.h>
#include <linux/version.h>

#include "../include/mudaq_device_constants.h"
#include "../include/mudaq_registers.h"

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 11, 0)  // add `signal.h` file
#include <linux/sched/signal.h>
#endif

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 4, 0)
#define fallthrough \
    do {            \
    } while (0)
#endif

#define ERROR(fmt, ...)                                                      \
    printk(KERN_ERR "[%s/%s] " pr_fmt(fmt), THIS_MODULE->name, __FUNCTION__, \
           ##__VA_ARGS__)
#define INFO(fmt, ...)                                                        \
    printk(KERN_INFO "[%s/%s] " pr_fmt(fmt), THIS_MODULE->name, __FUNCTION__, \
           ##__VA_ARGS__)
#define DEBUG(fmt, ...)                                                        \
    printk(KERN_DEBUG "[%s/%s] " pr_fmt(fmt), THIS_MODULE->name, __FUNCTION__, \
           ##__VA_ARGS__)

//
// module-wide global variables
//

static const struct pci_device_id PCI_DEVICE_IDS[] = {
    {
        PCI_DEVICE(0x1172, 0x0004),
    },
    {
        0,
    },
};

#include <linux/miscdevice.h>

static DEFINE_IDA(mudaq_ida);

//
// mudaq structures and related functions
//

struct mudaq {
    struct pci_dev* pci_dev;
    u32 to_user[2];
    long irq;
    atomic_t event;
    wait_queue_head_t wait;
    struct mudaq_mem* mem;
    struct mudaq_dma* dma;

    int minor;
    struct dmabuf* dmabuf;
    struct miscdevice misc_mudaq;
    struct miscdevice misc_dmabuf;
};

static int dmabuf_fops_open(struct inode* inode, struct file* file) {
    struct mudaq* mudaq =
        container_of(file->private_data, struct mudaq, misc_dmabuf);
    file->private_data = mudaq->dmabuf;
    return 0;
}

#include "dmabuf/dmabuf_fops.h"

/*
   First four entries of internal_addr, phys_size and phys_addr are for
   the four PCIe BAR regions
   fifth entry is for the DMA control buffer
*/
struct mudaq_mem {
    __iomem u32* internal_addr[5];
    u32 phys_size[5];
    u32 phys_addr[5];
    dma_addr_t bus_addr_ctrl;
};

struct mudaq_dma {
    bool flag;
    struct page** pages;
    dma_addr_t bus_addrs[N_PAGES];
    int n_pages[N_PAGES];
    int npages;
    struct sg_table* sgt;
};

/** free the given mudaq struct and all associated memory */
static void mudaq_free(struct mudaq* mu) {
    if (IS_ERR_OR_NULL(mu)) return;

    if (!IS_ERR_OR_NULL(mu->mem)) {
        kfree(mu->mem);
        mu->mem = NULL;
    }
    if (!IS_ERR_OR_NULL(mu->dma)) {
        kfree(mu->dma);
        mu->dma = NULL;
    }
    kfree(mu);
}

/** allocate a new mudaq struct and initialize its state */
static struct mudaq* mudaq_alloc(void) {
    int error;

    /* allocate memory for the device structure */
    struct mudaq* mu = kzalloc(sizeof(struct mudaq), GFP_KERNEL);
    if (IS_ERR_OR_NULL(mu)) {
        ERROR("could not allocate memory for 'struct mudaq'\n");
        error = -ENOMEM;
        goto fail;
    }
    mu->minor = -1;
    mu->misc_mudaq.minor = MISC_DYNAMIC_MINOR;
    mu->misc_dmabuf.minor = MISC_DYNAMIC_MINOR;

    mu->mem = kzalloc(sizeof(struct mudaq_mem), GFP_KERNEL);
    if (IS_ERR_OR_NULL(mu->mem)) {
        ERROR("could not allocate memory for 'struct mudaq_mem'\n");
        error = -ENOMEM;
        goto fail;
    }

    mu->dma = kzalloc(sizeof(struct mudaq_dma), GFP_KERNEL);
    if (IS_ERR_OR_NULL(mu->dma)) {
        ERROR("could not allocate memory for 'struct mudaq_dma'\n");
        error = -ENOMEM;
        goto fail;
    }

    return mu;

fail:
    mudaq_free(mu);
    return ERR_PTR(error);
}

static int mudaq_interrupt_control(struct mudaq* info, s32 irq_on) {
    /* no need to do anything. interrupts are activated from userspace */
    DEBUG("Called interrupt control w/ %#x\n", irq_on);
    return 0;
}

static irqreturn_t mudaq_interrupt_handler(int irq, struct mudaq* mu) {
    /* no need to do anything. just acknowledge that something happened. */
    DEBUG("Received interrupt\n");
    return IRQ_HANDLED;
}

/* Trigger an interrupt event */
static void mudaq_event_notify(struct mudaq* mu) {
    atomic_inc(&mu->event);            // interrupt counter
    wake_up_interruptible(&mu->wait);  // wake up read function
}

/* Hardware interrupt handler */
static irqreturn_t mudaq_interrupt(int irq, void* dev_id) {
    struct mudaq* mu = dev_id;
    irqreturn_t ret = mudaq_interrupt_handler(irq, mu);

    if (ret == IRQ_HANDLED) mudaq_event_notify(mu);

    return ret;
}

/* Access registers */
inline __iomem u32* mudaq_register_rw(struct mudaq* mu, unsigned index) {
    __iomem u32* base;
    if (IS_ERR_OR_NULL(mu) || IS_ERR_OR_NULL(mu->mem)) return NULL;
    base = mu->mem->internal_addr[0];
    return base + index;
}

inline __iomem u32* mudaq_register_ro(struct mudaq* mu, unsigned index) {
    __iomem u32* base;
    if (IS_ERR_OR_NULL(mu) || IS_ERR_OR_NULL(mu->mem)) return NULL;
    base = mu->mem->internal_addr[1];
    return base + index;
}

// write to register then read back and check
#define mudaq_write32_test(mu, value, index)                            \
    ({                                                                  \
        void __iomem* addr = mudaq_register_rw((mu), index);            \
        if (!IS_ERR_OR_NULL(addr)) {                                    \
            u32 a = (value), b;                                         \
            iowrite32(a, addr);                                         \
            b = ioread32(addr);                                         \
            if (a != b) {                                               \
                ERROR("write '%u' at '%s', but read back is '%u'\n", a, \
                      #index, b);                                       \
            }                                                           \
        }                                                               \
    })

static void mudaq_deactivate(struct mudaq* mu) {
    mudaq_write32_test(mu, 0, DATAGENERATOR_REGISTER_W);
    mudaq_write32_test(mu, 0, DMA_REGISTER_W);
}

/* Copy dma bus addresses to device registers */
static void mudaq_set_dma_ctrl_addr(struct mudaq* mu, dma_addr_t ctrl_handle) {
    mudaq_write32_test(mu, ctrl_handle & 0xFFFFFFFF,
                       DMA_CTRL_ADDR_LOW_REGISTER_W);
    mudaq_write32_test(mu, ctrl_handle >> 32, DMA_CTRL_ADDR_HI_REGISTER_W);
}

static int mudaq_set_dma_data_addr(struct mudaq* mu, dma_addr_t data_handle,
                                   u32 mem_location, u32 n_pages) {
    u32 regcontent;
    regcontent = SET_DMA_NUM_PAGES_RANGE(0x0, n_pages);
    regcontent = SET_DMA_RAM_LOCATION_RANGE(regcontent, mem_location);

    mudaq_write32_test(mu, regcontent, DMA_RAM_LOCATION_NUM_PAGES_REGISTER_W);
    // DMA address in firmware is updated only during write to LOW address
    mudaq_write32_test(mu, data_handle >> 32, DMA_DATA_ADDR_HI_REGISTER_W);
    mudaq_write32_test(mu, data_handle & 0xFFFFFFFF,
                       DMA_DATA_ADDR_LOW_REGISTER_W);

    return 0;
}

static void mudaq_set_dma_n_buffers(struct mudaq* mu, u32 n_buffers) {
    mudaq_write32_test(mu, n_buffers, DMA_NUM_ADDRESSES_REGISTER_W);
}

static void mudaq_clear_mmio(struct pci_dev* dev, struct mudaq* mu) {
    if (IS_ERR_OR_NULL(mu) || IS_ERR_OR_NULL(mu->mem)) return;

    for (int i = 0; i < 4; i++) {
        if (IS_ERR_OR_NULL(mu->mem->internal_addr[i])) continue;
        pci_iounmap(dev, mu->mem->internal_addr[i]);
        mu->mem->internal_addr[i] = NULL;
    }

    pci_release_regions(dev);
}

static int mudaq_setup_mmio(struct pci_dev* pdev, struct mudaq* mu) {
    const int bars[] = {0, 1, 2, 3};
    const char* names[] = {"registers_rw", "registers_ro", "memory_rw",
                           "memory_ro"};
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
        mu->mem->internal_addr[i] =
            pci_iomap(pdev, bars[i], mu->mem->phys_size[i]);
        if (IS_ERR_OR_NULL(mu->mem->internal_addr[i])) {
            ERROR("pci_iomap failed for '%s'\n", names[i]);
            rv = -ENODEV;
            goto fail_unmap;
        }
        DEBUG("Bar %d: at %x with size %d at internal address %lx\n", bars[i],
              mu->mem->phys_addr[i], mu->mem->phys_size[i],
              (long unsigned)mu->mem->internal_addr[i]);
    }

    return 0;

fail_unmap:
    mudaq_clear_mmio(pdev, mu);
out:
    return rv;
}

static void mudaq_clear_dma(struct pci_dev* pdev, struct mudaq* mu) {
    /* deactivate any readout activity before removing the dma memory */
    mudaq_deactivate(mu);
    mudaq_set_dma_ctrl_addr(mu, 0x0);

    dma_free_coherent(&pdev->dev, mu->mem->phys_size[4],
                      mu->mem->internal_addr[4], mu->mem->bus_addr_ctrl);
}

/** setup coherent dma
 *
 * set dma masks and create one DMA buffer for control
 *
 * WARNING
 * assumes that the read/write registers are already mapped
 */
static int mudaq_setup_dma(struct pci_dev* pdev, struct mudaq* mu) {
    int rv;
    dma_addr_t ctrl_addr;
    void* ctrl_internal;
    size_t ctrl_size = MUDAQ_BUFFER_CTRL_SIZE;
    const char* name = "dma_ctrl";

    if ((rv = pci_set_dma_mask(pdev, DMA_BIT_MASK(64))) < 0) return rv;
    if ((rv = pci_set_consistent_dma_mask(pdev, DMA_BIT_MASK(64))) < 0)
        return rv;

#if LINUX_VERSION_CODE >= \
    KERNEL_VERSION(5, 0, 0)  // remove `dma_zalloc_coherent` function
    ctrl_internal =
        dma_alloc_coherent(&pdev->dev, ctrl_size, &ctrl_addr, GFP_KERNEL);
#else
    ctrl_internal =
        dma_zalloc_coherent(&pdev->dev, ctrl_size, &ctrl_addr, GFP_KERNEL);
#endif

    if (IS_ERR_OR_NULL(ctrl_internal)) {
        ERROR("could not allocate dma control buffer\n");
        rv = -ENOMEM;
        goto out;
    }

    mu->mem->bus_addr_ctrl = ctrl_addr;
    mu->mem->phys_size[4] = ctrl_size;
    mu->mem->internal_addr[4] = ctrl_internal;

    //    DEBUG("Addr: %d , Size: %d, Internal Add %d\n",int(ctrl_addr)
    //    int(ctrl_size) int(ctrl_internal));
    DEBUG("setup dma for '%s' dma_addr %#zx size %d bytes\n", name,
          (size_t)mu->mem->bus_addr_ctrl, mu->mem->phys_size[4]);

    /* deactivate any readout activity to always start in a consistent state
       before setting up the dma addresses */
    mudaq_deactivate(mu);
    mudaq_set_dma_ctrl_addr(mu, ctrl_addr);
    return 0;

out:
    return rv;
}

static void mudaq_clear_msi(struct mudaq* mu) {
    /* release irq */
    devm_free_irq(mu->misc_mudaq.this_device, mu->irq, mu);
    DEBUG("Freed irq\n");

    /* disable MSI */
    pci_disable_msi(mu->pci_dev);
    pci_clear_mwi(mu->pci_dev);
}

static int mudaq_setup_msi(struct mudaq* mu) {
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
    rv = devm_request_irq(mu->misc_mudaq.this_device, mu->irq, mudaq_interrupt,
                          0, THIS_MODULE->name, mu);
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

static void mudaq_free_dma(struct mudaq* mu) {
    if (IS_ERR_OR_NULL(mu)) return;

    mudaq_deactivate(mu);
    mudaq_set_dma_n_buffers(mu, 0);
    for (int i = 0; i < 4096; i++) {
        mudaq_set_dma_data_addr(mu, 0, i, 0);
    }

    if (IS_ERR_OR_NULL(mu->dma)) return;

    if (!IS_ERR_OR_NULL(mu->dma->sgt)) {
        dma_unmap_sg(&mu->pci_dev->dev, mu->dma->sgt->sgl, mu->dma->sgt->nents,
                     DMA_FROM_DEVICE);
        sg_free_table(mu->dma->sgt);
        kfree(mu->dma->sgt);
        mu->dma->sgt = NULL;
    }

    for (int i = 0; i < mu->dma->npages; i++) {
        if (PageReserved(mu->dma->pages[i]))
            INFO("Page %d is in reserved space\n", i);
        if (!PageReserved(mu->dma->pages[i])) SetPageDirty(mu->dma->pages[i]);
        //        page_cache_release( mu->dma->pages[i] );
        put_page(mu->dma->pages[i]);
    }
    mu->dma->npages = 0;

    if (!IS_ERR_OR_NULL(mu->dma->pages)) {
        kfree(mu->dma->pages);
        mu->dma->pages = NULL;
    }

    INFO("Freed pinned DMA buffer in kernel space\n");

    mu->dma->flag = false;
}

#include "mudaq_fops.h"

//
// register / unregister mudaq device with the kernel
//

/** unregister the mudaq device */
static void mudaq_unregister(struct mudaq* mu) {
    if (IS_ERR_OR_NULL(mu) || mu->minor < 0) return;

    mudaq_deactivate(mu);
    mudaq_set_dma_n_buffers(mu, 0);
    for (int i = 0; i < 4096; i++) {
        mudaq_set_dma_data_addr(mu, 0, i, 0);
    }

    if (mu->misc_mudaq.minor != MISC_DYNAMIC_MINOR) {
        misc_deregister(&mu->misc_mudaq);
        mu->misc_mudaq.minor = MISC_DYNAMIC_MINOR;
    }
    if (!IS_ERR_OR_NULL(mu->misc_mudaq.name)) {
        kfree(mu->misc_mudaq.name);
        mu->misc_mudaq.name = NULL;
    }
    if (mu->misc_dmabuf.minor != MISC_DYNAMIC_MINOR) {
        misc_deregister(&mu->misc_dmabuf);
        mu->misc_dmabuf.minor = MISC_DYNAMIC_MINOR;
    }
    if (!IS_ERR_OR_NULL(mu->misc_dmabuf.name)) {
        kfree(mu->misc_dmabuf.name);
        mu->misc_dmabuf.name = NULL;
    }

    dmabuf_free(mu->dmabuf);
    mu->dmabuf = NULL;

    ida_free(&mudaq_ida, mu->minor);
    mu->minor = -1;
    DEBUG("Released minor number\n");
}

/* register the mudaq device. device is live after successful call. */
static int mudaq_register(struct mudaq* mu) {
    int error = 0;

    mu->minor = ida_simple_get(&mudaq_ida, 0, MAX_NUM_DEVICES - 1, GFP_KERNEL);
    if (mu->minor < 0) goto err_out;

    mu->dmabuf = dmabuf_alloc(&mu->pci_dev->dev, MUDAQ_DMABUF_DATA_LEN);
    if (IS_ERR_OR_NULL(mu->dmabuf)) {
        error = PTR_ERR(mu->dmabuf);
        goto err_out;
    }

    mu->misc_mudaq.minor = MISC_DYNAMIC_MINOR;
    mu->misc_mudaq.name =
        kasprintf(GFP_KERNEL, "%s%d", THIS_MODULE->name, mu->minor);
    if (IS_ERR_OR_NULL(mu->misc_mudaq.name)) goto err_out;
    mu->misc_mudaq.fops = &mudaq_fops;
    mu->misc_mudaq.parent = &mu->pci_dev->dev;
    error = misc_register(&mu->misc_mudaq);
    if (error != 0) {
        mu->misc_mudaq.minor = MISC_DYNAMIC_MINOR;
        goto err_out;
    }

    mu->misc_dmabuf.minor = MISC_DYNAMIC_MINOR;
    mu->misc_dmabuf.name =
        kasprintf(GFP_KERNEL, "%s%d_dmabuf", THIS_MODULE->name, mu->minor);
    if (IS_ERR_OR_NULL(mu->misc_dmabuf.name)) goto err_out;
    mu->misc_dmabuf.fops = &dmabuf_fops;
    mu->misc_dmabuf.parent = &mu->pci_dev->dev;
    error = misc_register(&mu->misc_dmabuf);
    if (error != 0) {
        mu->misc_dmabuf.minor = MISC_DYNAMIC_MINOR;
        goto err_out;
    }

    mudaq_deactivate(mu);

    // setup device DMA tables using dmabuf entries
    {
        int i = 0;
        struct dmabuf_entry* entry;
        list_for_each_entry(entry, &mu->dmabuf->entries, list_head) {
            M_INFO("set dma entry %d: dma_addr = %llx, dma_pages = %lu\n", i,
                   entry->dma_handle, entry->size >> PAGE_SHIFT);
            mudaq_set_dma_data_addr(mu, entry->dma_handle, i,
                                    entry->size >> PAGE_SHIFT);
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

static void mudaq_pci_remove(struct pci_dev* pdev) {
    struct mudaq* mu = pci_get_drvdata(pdev);

    if (mu->dma->flag == true) mudaq_free_dma(mu);
    mudaq_clear_msi(mu);
    mudaq_unregister(mu);
    mudaq_clear_dma(pdev, mu);
    mudaq_clear_mmio(pdev, mu);
    pci_disable_device(pdev);
    mudaq_free(mu);

    INFO("Device removed\n");
}

static int mudaq_pci_probe(struct pci_dev* pdev,
                           const struct pci_device_id* pid) {
    int rv;

    struct mudaq* mu = mudaq_alloc();
    if (IS_ERR_OR_NULL(mu)) {
        rv = PTR_ERR(mu);
        goto fail;
    }
    DEBUG("Allocated mudaq\n");

    pci_set_drvdata(pdev, mu);
    mu->pci_dev = pdev;

    if ((rv = pci_enable_device(pdev)) < 0) goto out_free;
    DEBUG("Enabled device\n");
    if ((rv = mudaq_setup_mmio(pdev, mu)) < 0) goto out_disable;
    DEBUG("Setup mmio\n");
    if ((rv = mudaq_setup_dma(pdev, mu)) < 0) goto out_mmio;
    DEBUG("Setup dma\n");
    if ((rv = mudaq_register(mu)) < 0) goto out_dma;
    DEBUG("Registered mudaq\n");
    if ((rv = mudaq_setup_msi(mu)) < 0) goto out_unregister;
    DEBUG("Setup MSI interrupts\n");

    M_INFO("Device setup finished\n");

    mu->dma->flag = false;  // initially no dma mapping from ioctl present

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

static struct pci_driver mudaq_pci_driver = {
    .name = THIS_MODULE->name,
    .id_table = PCI_DEVICE_IDS,
    .probe = mudaq_pci_probe,
    .remove = mudaq_pci_remove,
};

MODULE_DEVICE_TABLE(pci, PCI_DEVICE_IDS);

//
// module init and exit
//

static void __exit mudaq_exit(void) {
    pci_unregister_driver(&mudaq_pci_driver);

    DEBUG("module removed\n");
}

static int __init mudaq_init(void) {
    int error;

    // register pci driver
    error = pci_register_driver(&mudaq_pci_driver);
    if (error) {
        ERROR("could not register pci driver\n");
        goto err_out;
    }

    DEBUG("module initialized\n");

    return 0;

err_out:
    return error;
}

module_exit(mudaq_exit);
module_init(mudaq_init);

MODULE_DESCRIPTION("mu3e pcie readout board driver");
MODULE_AUTHOR("Moritz Kiehn <kiehn@physi.uni-heidelberg.de>");
MODULE_LICENSE("GPL");
