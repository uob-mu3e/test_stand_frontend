
/********************************************************************\
 
 Name:         environment_control.cpp
 Created by: Luigi Vigani   
        Stefan Ritt
 		Frederik Wauters
 		Andreas Knecht
 
 Contents:     Slow control frontend for the mu3e  experiment, environment and pixel temperature sensors on 2 SCS-3000
  
 \********************************************************************/

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "mscb.h"
#include "midas.h"
#include "mfe.h"
#include "device/mscbdev.h"
#include "class/multi.h"

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "SC Frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = TRUE;

/* a frontend status page is displayed with this frequency in ms    */
INT display_period = 0;//1000;

/* maximum event size produced by this frontend */
INT max_event_size = 5*100000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 2 *5* 100000;

/* Overwrite equipment struct in ODB from values in code*/
BOOL equipment_common_overwrite = FALSE;

/*-- Equipment list ------------------------------------------------*/

/* device driver list */

DEVICE_DRIVER multi_driver_env[] = {
    {"ADC", mscbdev, 0, NULL, DF_INPUT | DF_MULTITHREAD},
    {""}
};

DEVICE_DRIVER multi_driver_pix[] = {
    {"ADC", mscbdev, 0, NULL, DF_INPUT | DF_MULTITHREAD},
    {""}
};

EQUIPMENT equipment[] = {
        
    {"Environment",              /* equipment name */
        {150, 0,                      /* event ID, trigger mask */
            "SYSTEM",                  /* event buffer */
            EQ_SLOW,                   /* equipment type */
            0,                         /* event source */
            "MIDAS",                   /* format */
            TRUE,                      /* enabled */
            RO_ALWAYS,        /* read when running and on transitions */
            60000,                     /* produce event every 60 sec */
            1000,                      /* read one event every second */
            0,                         /* number of sub events */
            1,                         /* log history every event */
            "", "", ""} ,
        cd_multi_read,              /* readout routine */
        cd_multi,                   /* class driver main routine */
        multi_driver_env,               /* device driver list */
        NULL,                       /* init string */
    },
    
    {"Pixel-SC-Temperature",              /* equipment name */
        {151, 0,                      /* event ID, trigger mask */
            "SYSTEM",                  /* event buffer */
            EQ_SLOW,                   /* equipment type */
            0,                         /* event source */
            "MIDAS",                   /* format */
            TRUE,                      /* enabled */
            RO_ALWAYS,        /* read when running and on transitions */
            60000,                     /* produce event every 60 sec */
            1000,                      /* read one event every second */
            0,                         /* number of sub events */
            1,                         /* log history every event */
            "", "", ""} ,
        cd_multi_read,              /* readout routine */
        cd_multi,                   /* class driver main routine */
        multi_driver_pix,               /* device driver list */
        NULL,                       /* init string */
    },

    {""}
};

/*-- Dummy routines ------------------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
{
    return 1;
};
INT interrupt_configure(INT cmd, INT source, PTYPE adr)
{
    return 1;
};

/*-- Function to define MSCB variables in a convenient way ---------*/

void mscb_define(const char *submaster,const char *equipment,const char *devname, DEVICE_DRIVER *driver,
                 int address, unsigned char var_index, char *name, double threshold)
{
    int i, dev_index, chn_index, chn_total;
    char str[256];
    float f_threshold;
    HNDLE hDB;
    
    cm_get_experiment_database(&hDB, NULL);
    
    if (submaster && submaster[0]) {
        char dev_name[500];
        int size_dev_name = sizeof(dev_name);

        sprintf(str, "/Equipment/%s/Settings/Devices/%s/Device", equipment, devname);
        int has_subname = db_get_value(hDB, 0, str, dev_name, &size_dev_name, TID_STRING, TRUE);

        if (has_subname != DB_SUCCESS) { //Variable not set, first time...
            sprintf(str, "/Equipment/%s/Settings/Devices/%s/Device", equipment, devname);
            db_set_value(hDB, 0, str, submaster, 32, 1, TID_STRING);

        }

        sprintf(str, "/Equipment/%s/Settings/Devices/%s/Pwd", equipment, devname);
        int has_subpwd = db_get_value(hDB, 0, str, dev_name, &size_dev_name, TID_STRING, TRUE);

        if (has_subpwd != DB_SUCCESS) { //Variable not set, first time...
            sprintf(str, "/Equipment/%s/Settings/Devices/%s/Pwd", equipment, devname);
            db_set_value(hDB, 0, str, "mu3e", 32, 1, TID_STRING);
        }
    }
    
    /* find device in device driver */
    for (dev_index=0 ; driver[dev_index].name[0] ; dev_index++)
        if (equal_ustring(driver[dev_index].name, devname))
            break;
    
    if (!driver[dev_index].name[0]) {
        cm_msg(MERROR, "mscb_define", "Device \"%s\" not present in device driver list", devname);
        return;
    }
    
    /* count total number of channels */
    for (i=chn_total=0 ; i<=dev_index ; i++)
        chn_total += driver[i].channels;
    
    chn_index = driver[dev_index].channels;
    sprintf(str, "/Equipment/%s/Settings/Devices/%s/MSCB Address", equipment, devname);
    db_set_value_index(hDB, 0, str, &address, sizeof(int), chn_index, TID_INT, TRUE);
    sprintf(str, "/Equipment/%s/Settings/Devices/%s/MSCB Index", equipment, devname);
    db_set_value_index(hDB, 0, str, &var_index, sizeof(char), chn_index, TID_BYTE, TRUE);
    
    if (threshold != -1) {
        sprintf(str, "/Equipment/%s/Settings/Update Threshold", equipment);
        f_threshold = (float) threshold;
        db_set_value_index(hDB, 0, str, &f_threshold, sizeof(float), chn_total, TID_FLOAT, TRUE);
    }
    
    if (name && name[0]) {
        sprintf(str, "/Equipment/%s/Settings/Names Input", equipment);
        db_set_value_index(hDB, 0, str, name, 32, chn_total, TID_STRING, TRUE);
    }
    
    /* increment number of channels for this driver */
    driver[dev_index].channels++;
}

/*-- Error dispatcher causing communiction alarm -------------------*/

void scfe_error(const char *error)
{
    char str[256];
    
    strlcpy(str, error, sizeof(str));
    cm_msg(MERROR, "scfe_error", "%s", str);
    al_trigger_alarm("MSCB", str, "MSCB Alarm", "Communication Problem", AT_INTERNAL);
}

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
    HNDLE hDB;
    //int i, im, iv;
    //char str[80];
    
    cm_get_experiment_database(&hDB, NULL);
    
    /* set error dispatcher for alarm functionality */
    mfe_set_error(scfe_error);
    
    /* set maximal retry count */
    mscb_set_max_retry(100);
    
    /* SCS-3000 environment */
    printf("Initializing Environment...\n");
    for (unsigned int i=0 ; i<71 ; i++)
        mscb_define("mscb400.mu3e", "Environment", "ADC", multi_driver_env, 65535, i, NULL, 0.0005);
    
    /* SCS-3000 pixels */
    printf("Initializing Pixel-SC-Temperature...\n");
    for (unsigned int j = 0; j < 48; j++)
        mscb_define("mscb334.mu3e", "Pixel-SC-Temperature", "ADC", multi_driver_pix, 0, j, NULL, 0.0005);

    return CM_SUCCESS;
}

/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{
    return CM_SUCCESS;
}

/*-- Begin of Run --------------------------------------------------*/

INT begin_of_run(INT run_number, char *error)
{
    return CM_SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT run_number, char *error)
{
    return CM_SUCCESS;
}

/*-- Pause Run -----------------------------------------------------*/

INT pause_run(INT run_number, char *error)
{
    return CM_SUCCESS;
}

/*-- Resuem Run ----------------------------------------------------*/

INT resume_run(INT run_number, char *error)
{
    return CM_SUCCESS;
}

/*-- Frontend Loop -------------------------------------------------*/

INT frontend_loop()
{
    ss_sleep(10);
    return CM_SUCCESS;
}

/*------------------------------------------------------------------*/
