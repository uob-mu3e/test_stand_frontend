//

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
            mu->misc_mudaq.this_device,
            vma,
            mu->mem->internal_addr[index],
            mu->mem->bus_addr_ctrl,
            mu->mem->phys_size[index]
        );

    default:
        /* this should NEVER happen */
        ERROR("something went horribly wrong. run for your lives\n");
        rv = -EINVAL;
        break;
    }

out:
    return rv;
}


static
int mudaq_fops_release(struct inode *inode, struct file *filp) {
    struct mudaq *mu = filp->private_data;
    if (mu->dma->flag == true)
        mudaq_free_dma(mu);

    return 0;
}

static
int mudaq_fops_open(struct inode* inode, struct file* file) {
    struct mudaq* mudaq;
    if(file->private_data == NULL) return -EFAULT;
    mudaq = container_of(file->private_data, struct mudaq, misc_mudaq);
    file->private_data = mudaq;
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

    if (_IOC_TYPE(cmd) != MUDAQ_IOC_TYPE) {
        retval = -ENOTTY;
        goto fail;
    }
    if (_IOC_NR(cmd) > MUDAQ_IOC_NR) {
        retval = -ENOTTY;
        goto fail;
    }

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
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 8, 0)
        down_read(&current->mm->mmap_lock);
#else
        down_read(&current->mm->mmap_sem); //lock the memory management structure for our process
#endif

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 6, 0) // add `get_user_pages_remote` function
        retval = get_user_pages_remote(
#else
        retval = get_user_pages(
#endif
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 9, 0) // remove `task_struct` argument
            current,
#endif
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

#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 8, 0)
        up_read(&current->mm->mmap_lock);
#else
        up_read(&current->mm->mmap_sem);  // unlock
#endif

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
