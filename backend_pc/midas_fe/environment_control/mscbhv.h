/********************************************************************\
 
 Name:         mscbhv.c
 Created by:   Stefan Ritt
 Adapted by:   Andreas Knecht
 
 Contents:     MSCB SCS-2001 based High Voltage Device Driver
 
 $Id: mscbhv.c 2753 2005-10-07 14:55:31Z ritt $
 
 \********************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include "midas.h"
#include "mscb.h"

extern void mfe_error(const char *error);

/*---- globals -----------------------------------------------------*/

typedef struct {
    char mscb_device[NAME_LENGTH];
    char pwd[NAME_LENGTH];
    BOOL debug;
    int  address;
    int  channels;
    int  offset;
} MSCBHV_SETTINGS;

typedef struct {
    unsigned char control;
    float u_demand;
    float u_meas;
    float i_meas;
    unsigned char status;
    unsigned char trip_cnt;
    
    float ramp_up;
    float ramp_down;
    float u_limit;
    float i_limit;
    float ri_limit;
    unsigned char trip_max;
    unsigned char trip_time;
    
    unsigned char cached;
} MSCB_NODE_VARS;

typedef struct {
    MSCBHV_SETTINGS settings;
    int fd;
    MSCB_NODE_VARS *node_vars;
} MSCBHV_INFO;

INT mscbhv_read_all(MSCBHV_INFO * info, int i);

/*---- device driver routines --------------------------------------*/

typedef INT(func_t) (INT cmd, ...);

INT mscbhv_init(HNDLE hkey, MSCBHV_INFO **pinfo, INT channels, func_t *bd)
{
    int  i, status, size, block_address, block_channels;
    HNDLE hDB, hsubkey;
    MSCBHV_INFO *info;
    MSCB_INFO node_info;
    KEY key, adrkey;
    
    
    /* allocate info structure */
    info = (MSCBHV_INFO *) calloc(1, sizeof(MSCBHV_INFO));
    info->node_vars = (MSCB_NODE_VARS *) calloc(channels, sizeof(MSCB_NODE_VARS));
    *pinfo = info;
    
    cm_get_experiment_database(&hDB, NULL);
    
    /* retrieve device name */
    db_get_key(hDB, hkey, &key);
    
    /* create MSCBHV settings record */
    size = sizeof(info->settings.mscb_device);
    strcpy(info->settings.mscb_device, "usb0");
    status = db_get_value(hDB, hkey, "MSCB Device", &info->settings.mscb_device, &size, TID_STRING, TRUE);
    if (status != DB_SUCCESS)
        return FE_ERR_ODB;
    
    size = sizeof(info->settings.pwd);
    info->settings.pwd[0] = 0;
    status = db_get_value(hDB, hkey, "MSCB Pwd", &info->settings.pwd, &size, TID_STRING, TRUE);
    if (status != DB_SUCCESS)
        return FE_ERR_ODB;
    
    block_channels = channels;
    status = db_find_key(hDB, hkey, "Block address", &hsubkey);
    if (status == DB_SUCCESS) {
        db_get_key(hDB, hsubkey, &adrkey);
        for (i=0 ; i<adrkey.num_values ; i++) {
            size = sizeof(block_address);
            status = db_find_key(hDB, hkey, "Block address", &hsubkey);
            assert(status == DB_SUCCESS);
            status = db_get_data_index(hDB, hsubkey, &block_address, &size, i, TID_INT);
            assert(status == DB_SUCCESS);
            size = sizeof(block_channels);
            db_find_key(hDB, hkey, "Block channels", &hsubkey);
            assert(status == DB_SUCCESS);
            status = db_get_data_index(hDB, hsubkey, &block_channels, &size, i, TID_INT);
            assert(status == DB_SUCCESS);
            
            info->settings.address = block_address;
            
        }
    } else {
        block_address = 0;
        size = sizeof(INT);
        status = db_set_value(hDB, hkey, "Block address", &block_address, size, 1, TID_INT);
        if (status != DB_SUCCESS)
            return FE_ERR_ODB;
        block_channels = channels;
        size = sizeof(INT);
        status = db_set_value(hDB, hkey, "Block channels", &block_channels, size, 1, TID_INT);
        if (status != DB_SUCCESS)
            return FE_ERR_ODB;
        
        info->settings.address = block_address;
    }
    
    size = sizeof(info->settings.offset);
    status = db_get_value(hDB, hkey, "Block Offset", &info->settings.offset, &size, TID_INT, TRUE);
    if (status != DB_SUCCESS)
        return FE_ERR_ODB;
    
    
    /* open device on MSCB */
    info->fd = mscb_init(info->settings.mscb_device, NAME_LENGTH, info->settings.pwd, info->settings.debug);
    if (info->fd < 0) {
        cm_msg(MERROR, "mscbhv_init",
               "Cannot access MSCB submaster at \"%s\". Check power and connection.",
               info->settings.mscb_device);
        return FE_ERR_HW;
    }
    
    /* check first node */
    status = mscb_info(info->fd, info->settings.address, &node_info);
    if (status != MSCB_SUCCESS) {
        cm_msg(MERROR, "mscbhv_init",
               "Cannot access HV node at address \"%d\". Please check cabling and power.",
               info->settings.address);
        return FE_ERR_HW;
    }
    
    if (strcmp(node_info.node_name, "SCS-2001") != 0) {
        cm_msg(MERROR, "mscbhv_init",
               "Found unexpected node \"%s\" at address \"%d\".",
               node_info.node_name, info->settings.address);
        return FE_ERR_HW;
    }
    
    if (node_info.revision < 3518 || node_info.revision < 10) {
        cm_msg(MERROR, "mscbhv_init",
               "Found node \"%d\" with old firmware %d (SVN revistion >= 3518 required)",
               info->settings.address, node_info.revision);
        return FE_ERR_HW;
    }
    
    
    /* read all values from HV devices */
    for (i=0 ; i<channels ; i++) {
        
        if (i % 10 == 0)
            printf("%s: %d\r", key.name, i);
        
        status = mscbhv_read_all(info, i);
        
        if (status != FE_SUCCESS)
            return status;
    }
    printf("%s: %d\n", key.name, i);
    
    return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT mscbhv_read_all(MSCBHV_INFO * info, int i)
{
    int size, status;
    unsigned char buffer1[256], buffer2[256], buffer3[256], *pbuf;
    char str[256];
    
    int offset = info->settings.offset;
    int index = floor(i/offset)*3*offset + (i % offset);
    
    size = sizeof(buffer1);
    status = mscb_read(info->fd, info->settings.address, index, buffer1, &size);
    
    if (status != MSCB_SUCCESS) {
        sprintf(str, "Error reading MSCB HV at \"%s:%d\".",
                info->settings.mscb_device, info->settings.address);
        mfe_error(str);
        return FE_ERR_HW;
    }
    
    size = sizeof(buffer2);
    status = mscb_read(info->fd, info->settings.address, index+offset, buffer2, &size);
    
    if (status != MSCB_SUCCESS) {
        sprintf(str, "Error reading MSCB HV at \"%s:%d\".",
                info->settings.mscb_device, info->settings.address);
        mfe_error(str);
        return FE_ERR_HW;
    }
    
    size = sizeof(buffer3);
    status = mscb_read(info->fd, info->settings.address, index+2*offset, buffer3, &size);
    
    if (status != MSCB_SUCCESS) {
        sprintf(str, "Error reading MSCB HV at \"%s:%d\".",
                info->settings.mscb_device, info->settings.address);
        mfe_error(str);
        return FE_ERR_HW;
    }
    
    /* decode variables from buffer */
    pbuf = buffer1;
    info->node_vars[i].u_demand = *((float *)pbuf);

    pbuf = buffer2;
    info->node_vars[i].u_meas = *((float *)pbuf);

    pbuf = buffer3;
    info->node_vars[i].i_meas = *((float *)pbuf);
    
    /* mark voltage/current as valid in cache */
    info->node_vars[i].cached = 1;
    
    return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT mscbhv_exit(MSCBHV_INFO * info)
{
  if(info){ 
    mscb_exit(info->fd);
    free(info->node_vars);
    free(info);
    }
    
    return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT mscbhv_set(MSCBHV_INFO * info, INT channel, float value)
{
    int offset = info->settings.offset;
    int index = floor(channel/offset)*3*offset + (channel % offset);

    mscb_write(info->fd, info->settings.address, index, &value, 4);
    
    return FE_SUCCESS;
}

/*----------------------------------------------------------------------------*/

INT mscbhv_get(MSCBHV_INFO * info, INT channel, float *pvalue)
{
    int size, status;
    unsigned char buffer2[256], buffer3[256], *pbuf;
    char str[256];
    
    int offset = info->settings.offset;
    int index = floor(channel/offset)*3*offset + (channel % offset);
   
    /* check if value was previously read by mscbhv_read_all() */
    if (info->node_vars[channel].cached) {
        *pvalue = info->node_vars[channel].u_meas;
        info->node_vars[channel].cached = 0;
        return FE_SUCCESS;
    }
    
    size = sizeof(buffer2);
    status = mscb_read(info->fd, info->settings.address, index+offset, buffer2, &size);
    
    if (status != MSCB_SUCCESS) {
        sprintf(str, "Error reading MSCB HV at \"%s:%d\".",
                //info->settings.mscb_device, info->settings.address[i]);
                info->settings.mscb_device, info->settings.address);
        mfe_error(str);
        return FE_ERR_HW;
    }
    
    size = sizeof(buffer3);
    status = mscb_read(info->fd, info->settings.address, index+2*offset, buffer3, &size);
    
    if (status != MSCB_SUCCESS) {
        sprintf(str, "Error reading MSCB HV at \"%s:%d\".",
                info->settings.mscb_device, info->settings.address);
        mfe_error(str);
        return FE_ERR_HW;
    }
    
    /* decode variables from buffer */
    pbuf = buffer2;
    info->node_vars[channel].u_meas = *((float *)pbuf);

    pbuf = buffer3;
    info->node_vars[channel].i_meas = *((float *)pbuf);
    
    *pvalue = info->node_vars[channel].u_meas;
    return FE_SUCCESS;
}

/*---- device driver entry point -----------------------------------*/

INT mscbhv(INT cmd, ...)
{
    va_list argptr;
    HNDLE hKey;
    INT channel, status;
    float value, *pvalue;
    int *pivalue;
    MSCBHV_INFO *info;
    //char *name;
    
    va_start(argptr, cmd);
    status = FE_SUCCESS;
    
    switch (cmd) {
        case CMD_INIT: {
            hKey = va_arg(argptr, HNDLE);
            MSCBHV_INFO **pinfo = va_arg(argptr, MSCBHV_INFO **);
            channel = va_arg(argptr, INT);
            va_arg(argptr, DWORD);
            func_t* bd = va_arg(argptr, func_t *);
            status = mscbhv_init(hKey, pinfo, channel, bd);
            break; }
            
        case CMD_EXIT:
            info = va_arg(argptr, MSCBHV_INFO *);
            status = mscbhv_exit(info);
            break;
            
        case CMD_SET:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            value = (float) va_arg(argptr, double);
            status = mscbhv_set(info, channel, value);
            break;
            
        case CMD_GET:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            pvalue = va_arg(argptr, float *);
            status = mscbhv_get(info, channel, pvalue);
            break;
            
        case CMD_GET_DEMAND:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            pvalue = va_arg(argptr, float *);
            *pvalue = info->node_vars[channel].u_demand;
            break;
            
        case CMD_GET_CURRENT:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            pvalue = va_arg(argptr, float *);
            *pvalue = info->node_vars[channel].i_meas;
            break;
            
        case CMD_SET_CURRENT_LIMIT:
        case CMD_SET_VOLTAGE_LIMIT:
            // Not available
            status = FE_SUCCESS;
            break;
            
        case CMD_GET_LABEL:
            status = FE_SUCCESS;
            break;
            
        case CMD_SET_LABEL:
            status = FE_SUCCESS;
            break;
            
        case CMD_GET_THRESHOLD:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            pvalue = va_arg(argptr, float *);
            *pvalue = 0.1f;
            break;
            
        case CMD_GET_THRESHOLD_CURRENT:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            pvalue = va_arg(argptr, float *);
            *pvalue = 1;
            break;
            
        case CMD_GET_THRESHOLD_ZERO:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            pvalue = va_arg(argptr, float *);
            *pvalue = 5;
            break;
            
        case CMD_GET_VOLTAGE_LIMIT:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            pvalue = va_arg(argptr, float *);
            // Default value
            *pvalue = 200.;
            break;
            
        case CMD_GET_CURRENT_LIMIT:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            pvalue = va_arg(argptr, float *);
            // Default value
            *pvalue = 100.;
            break;
            
        case CMD_GET_RAMPUP:
        case CMD_GET_RAMPDOWN:
        case CMD_GET_TRIP_TIME:
            // Not available
            status = FE_SUCCESS;
            break;
            
        case CMD_SET_RAMPUP:
        case CMD_SET_RAMPDOWN:
        case CMD_SET_TRIP_TIME:
            // Not available
            status = FE_SUCCESS;
            break;
            
        case CMD_GET_TRIP:
        case CMD_GET_STATUS:
        case CMD_GET_TEMPERATURE:
            info = va_arg(argptr, MSCBHV_INFO *);
            channel = va_arg(argptr, INT);
            pivalue = va_arg(argptr, INT *);
            *pivalue = 0; // not implemented for the moment...
            status = FE_SUCCESS;
            break;
            
        case CMD_START:
        case CMD_STOP:
            break;
            
        default:
            cm_msg(MERROR, "mscbhv device driver", "Received unknown command %d", cmd);
            status = FE_ERR_DRIVER;
            break;
    }
    
    va_end(argptr);
    
    return status;
}

/*------------------------------------------------------------------*/
