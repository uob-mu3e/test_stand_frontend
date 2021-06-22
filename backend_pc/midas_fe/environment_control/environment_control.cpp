
/********************************************************************\
 
 Name:         sc_fe.c
 Created by:   Stefan Ritt
 		Frederik Wauters
 		Andreas Knecht
 
 Contents:     Slow control frontend for the muX  experiment
 
 $Id: sc_fe.c 21520 2014-11-03 12:03:56Z ritt $
 
 \********************************************************************/

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "mscb.h"
#include "midas.h"
#include "mfe.h"
#include "class/hv.h"
#include "device/mscbdev.h"
#include "class/multi.h"
#include "mscbhv.h"
#include "device/mscbhvr.h"

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


// New HV power supply with "many" channels
DEVICE_DRIVER hv_driver[] = {
    {"64xHV-PSI", mscbhv, 0, NULL, DF_MULTITHREAD},
    {""}
};

DEVICE_DRIVER cfd_driver[] = {
    {"CFD-950", mscbdev, 0, NULL, DF_INPUT | DF_MULTITHREAD},
    {""}
};

DEVICE_DRIVER pressure_driver[] = {
    {"PFEIFFER", mscbdev, 0, NULL, DF_INPUT | DF_MULTITHREAD},
    {""}
};

DEVICE_DRIVER multi_driver_env[] = {
    {"ADC", mscbdev, 0, NULL, DF_INPUT | DF_MULTITHREAD},
    {""}
};

DEVICE_DRIVER multi_driver_pix[] = {
    {"ADC", mscbdev, 0, NULL, DF_INPUT | DF_MULTITHREAD},
    {""}
};

// HV device with hardware ramp and 1 channel per node
//DEVICE_DRIVER poshv_driver[] = {
//   {"PMT", mscbhvr, 0, NULL, DF_HW_RAMP | DF_PRIO_DEVICE | DF_MULTITHREAD},
//   {""}
//};

EQUIPMENT equipment[] = {
    
    //{"SiPM_HV",                       /* equipment name */
    //    {10, 0,                     /* event ID, trigger mask */
    //        "SYSTEM",                  /* event buffer */
    //        EQ_SLOW,                   /* equipment type */
    //        0,                         /* event source */
    //        "MIDAS",                   /* format */
    //        TRUE,                      /* enabled */
    //        RO_ALWAYS,
    //        60000,                     /* produce event every 60 sec */
    //        1000,                      /* read one event every second */
    //        0,                         /* number of sub events */
    //        1,                        /* log history every 10 seconds event */
    //        "", "", ""} ,
    //    cd_hv_read,                 /* readout routine */
    //    cd_hv,                      /* class driver main routine */
    //    hv_driver,                  /* device driver list */
    //    NULL,                       /* init string */
    //},
    
    
    //    {"CFD",                     /* equipment name */
    //    {11, 0,                     /* event ID, trigger mask */
    //     "SYSTEM",                  /* event buffer */
    //     EQ_SLOW,                   /* equipment type */
    //     0,                         /* event source */
    //     "MIDAS",                   /* format */
    //     TRUE,                      /* enabled */
    //     RO_ALWAYS,
    //     60000,                     /* read every 60 sec */
    //     1000,                      /* read one event every second */
    //     0,                         /* number of sub events */
    //     1,                         /* log history every second */
    //     "", "", ""} ,
    //    cd_multi_read,              /* readout routine */
    //    cd_multi,                   /* class driver main routine */
    //    cfd_driver,                 /* device driver list */
    //    NULL,                       /* init string */
    //    },
    
    //   {"Pressures",                /* equipment name */
    //    {12, 0,                     /* event ID, trigger mask */
    //     "SYSTEM",                  /* event buffer */
    //     EQ_SLOW,                   /* equipment type */
    //     0,                         /* event source */
    //     "MIDAS",                   /* format */
    //     TRUE,                      /* enabled */
    //     RO_ALWAYS,
    //     60000,                     /* read every 60 sec */
    //     1000,                      /* read one event every second */
    //     0,                         /* number of sub events */
    //     1,                         /* log history every second */
    //    "", "", ""} ,
    //    cd_multi_read,              /* readout routine */
    //    cd_multi,                   /* class driver main routine */
    //    pressure_driver,            /* device driver list */
    //    NULL,                       /* init string */
    //    },
    
    {"Environment",              /* equipment name */
        {13, 0,                      /* event ID, trigger mask */
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
        {13, 0,                      /* event ID, trigger mask */
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

     //  {"Pos_HV",                       /* equipment name */
      //  {16, 0,                     /* event ID, trigger mask */
      //   "SYSTEM",                  /* event buffer */
       //  EQ_SLOW,                   /* equipment type */
      //   0,                         /* event source */
     //    "MIDAS",                   /* format */
     //    TRUE,                      /* enabled */
     //    RO_ALWAYS,
     //    60000,                     /* produce event every 60 sec */
     //    1000,                      /* read one event every second */
     //    0,                         /* number of sub events */
      //   10,                        /* log history every 10 seconds event */
      //   "", "", ""} ,
     //   cd_hv_read,                 /* readout routine */
      //  cd_hv,                      /* class driver main routine */
      //  poshv_driver,                  /* device driver list */
      //  NULL,                       /* init string */
      //  },
    
    
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
        sprintf(str, "/Equipment/%s/Settings/Devices/%s/Device", equipment, devname);
        db_set_value(hDB, 0, str, submaster, 32, 1, TID_STRING);
        sprintf(str, "/Equipment/%s/Settings/Devices/%s/Pwd", equipment, devname);
        db_set_value(hDB, 0, str, "meg", 32, 1, TID_STRING);
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


/*-- Function to define MSCB HV variables in a convenient way ------*/

void mscbhv_define(const char *submaster,const char *equipment,const char *devname, DEVICE_DRIVER *driver,
                   int block_addr, int block_channels, int index, int block_offset, double th_measured, double th_current)
{
    int i, dev_index, chn_total;
    char str[256];
    float f_threshold;
    HNDLE hDB;
    
    cm_get_experiment_database(&hDB, NULL);
    
    if (submaster && submaster[0]) {
        sprintf(str, "/Equipment/%s/Settings/Devices/%s/MSCB Device", equipment, devname);
        db_set_value(hDB, 0, str, submaster, 32, 1, TID_STRING);
        sprintf(str, "/Equipment/%s/Settings/Devices/%s/MSCB Pwd", equipment, devname);
        db_set_value(hDB, 0, str, "meg", 32, 1, TID_STRING);
    }
    
    /* find device in device driver */
    for (dev_index=0 ; driver[dev_index].name[0] ; dev_index++)
        if (equal_ustring(driver[dev_index].name, devname))
            break;
    
    if (!driver[dev_index].name[0]) {
        cm_msg(MERROR, "mscbhvr_define", "Device \"%s\" not present in device driver list", devname);
        return;
    }
    
    /* count total number of channels */
    for (i=chn_total=0 ; i<=dev_index ; i++)
        chn_total += driver[i].channels;
    
    sprintf(str, "/Equipment/%s/Settings/Devices/%s/Block Address", equipment, devname);
    db_set_value_index(hDB, 0, str, &block_addr, sizeof(int), index, TID_INT, TRUE);
    sprintf(str, "/Equipment/%s/Settings/Devices/%s/Block Channels", equipment, devname);
    db_set_value_index(hDB, 0, str, &block_channels, sizeof(int), index, TID_INT, TRUE);
    sprintf(str, "/Equipment/%s/Settings/Devices/%s/Block Offset", equipment, devname);
    db_set_value_index(hDB, 0, str, &block_offset, sizeof(int), index, TID_INT, TRUE);
    
    sprintf(str, "/Equipment/%s/Settings/Update Threshold Measured", equipment);
    for (i=0 ; i<block_channels ; i++) {
        f_threshold = (float) th_measured;
        db_set_value_index(hDB, 0, str, &f_threshold, sizeof(float), chn_total+i, TID_FLOAT, TRUE);
    }
    
    sprintf(str, "/Equipment/%s/Settings/Update Threshold Current", equipment);
    for (i=0 ; i<block_channels ; i++) {
        f_threshold = (float) th_current;
        db_set_value_index(hDB, 0, str, &f_threshold, sizeof(float), chn_total+i, TID_FLOAT, TRUE);
    }
    
    /* increment number of channels for this driver */
    driver[dev_index].channels += block_channels;
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
    

    
    /* HV */
    // New SiPM power supply with many channels
    //mscbhv_define("mscb267.psi.ch", "SiPM_HV", "64xHV-PSI", hv_driver, 1, 64, 0, 8, 0.01, 0.01);
    
    
    /* CFD */
    
    //   for (i=0 ; i<8 ; i++)
    //      mscb_define("mscb045.psi.ch", "CFD", "CFD-950", cfd_driver, 1, 11+i, NULL, 0.01);
    //   for (i=0 ; i<8 ; i++)
    //      mscb_define("mscb045.psi.ch", "CFD", "CFD-950", cfd_driver, 2, 11+i, NULL, 0.01);
    //   for (i=0 ; i<8 ; i++)
    //      mscb_define("mscb045.psi.ch", "CFD", "CFD-950", cfd_driver, 3, 11+i, NULL, 0.01);
    //   for (i=0 ; i<8 ; i++)
    //      mscb_define("mscb045.psi.ch", "CFD", "CFD-950", cfd_driver, 4, 11+i, NULL, 0.01);
    
    /* Pfeiffer */
    //for (i=1 ; i<=6 ; i++) {
    //   sprintf(str, "P%d", i);
    //   mscb_define("mscb094.psi.ch", "Pressures", "PFEIFFER", pressure_driver, 100, i, NULL, 0.001);
    //}
    
    /* SCS-3000 environment */
    printf("Initializing Environment...\n");
    for (unsigned int i=0 ; i<71 ; i++)
        mscb_define("mscb400.mu3e", "Environment", "ADC", multi_driver_env, 65535, i, NULL, 0.0005);
    
    /* SCS-3000 pixels */
    printf("Initializing Pixel-SC-Temperature...\n");
    for (unsigned int j = 0; j < 48; j++)
        mscb_define("mscb334.mu3e", "Pixel-SC-Temperature", "ADC", multi_driver_pix, 0, j, NULL, 0.0005);

    /* HV from Konrad*/
    //mscbhvr_define("mscb210.psi.ch", "Pos_HV", "PMT", poshv_driver, 0, 20, 0, 0.1, 0.1);

    
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
