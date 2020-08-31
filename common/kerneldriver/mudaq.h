/*
 * mudaq ioctl definitions.
 */

#ifndef MUDAQ_H
#define MUDAQ_H

#include <linux/ioctl.h>
#include <linux/types.h>

//#define MUDAQ_DMABUF_DATA_LEN 4

#define MUDAQ_IOC_TYPE 102
#define MUDAQ_IOC_NR 4

/**
 * configuration
 */
#define                             DRIVER_NAME "mudaq"
static const char *                 CLASS_NAME = "mudaq";
static const char *                 DEVICE_NAME_TEMPLATE = "mudaq%d";
static const int                    MAX_NUM_DEVICES = 8;

struct mesg {
    volatile void * address;
    size_t size;
};

/* Declare IOC functions */

/** Copy virtual user space address of DMA data memory buffer to kernel space, 
 * map and send bus addresses to FPGA via registers:  sequential # = 1
 *
 * Data type = int
 */
#define MAP_DMA                       _IOW(MUDAQ_IOC_TYPE, 0, struct mesg)

/** Request current interrupt number from driver
 */
#define REQUEST_INTERRUPT_COUNTER     _IOR(MUDAQ_IOC_TYPE, 1, int)

#endif // MUDAQ_H
