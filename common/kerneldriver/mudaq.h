/*
 * Header file with ioctl definitions.
 * 
 * Declareations need to be in header file because they have to be accessible
 * by both the kernel module (mudaq.c) and the process calling ioctl.
 */

#ifndef MUDAQ_H
#define MUDAQ_H

#include <sys/ioctl.h>

//#define MUDAQ_DMABUF_DATA_LEN 4

#define MAGIC_NR 102
#define IOC_MAXNR 4

/**
 * configuration
 */
#define                             DRIVER_NAME "mudaq"
static const char *                 CLASS_NAME = "mudaq";
static const char *                 DEVICE_NAME_TEMPLATE = "mudaq%d";
static const unsigned               MAX_NUM_DEVICES = 8;

/* Declare IOC functions */

/** Copy virtual user space address of DMA data memory buffer to kernel space, 
 * map and send bus addresses to FPGA via registers:  sequential # = 1
 *
 * Data type = int
 */
#define MAP_DMA                       _IOW(MAGIC_NR, 0, int)

/** Request current interrupt number from driver
 */
#define REQUEST_INTERRUPT_COUNTER     _IOR(MAGIC_NR, 1, int)

struct mesg {
  volatile void * address;
  size_t size;
};

#endif
