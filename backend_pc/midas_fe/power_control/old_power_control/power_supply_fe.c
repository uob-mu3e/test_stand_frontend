/********************************************************************\

  Name:         power_supply_fe.c
  Created by:   Frederik Wauters

  Contents:     Derived from the example slow control frontend.c

  $Id$

\********************************************************************/

#include <stdio.h>
#include "midas.h"
#include "mfe.h"
#include "hv.h"
#include "hmp4040dev.h"
#include "keithley2611Bdev.h"
#include "lv.h"
//#include "scpi.h"

/* make frontend functions callable from the C framework */

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "MuPix Voltage Supply";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = TRUE;

/* a frontend status page is displayed with this frequency in ms    */
INT display_period = 1000;

/* maximum event size produced by this frontend */
INT max_event_size = 10000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 10 * 10000;

/*-- Function declarations -----------------------------------------*/

INT frontend_init();
INT frontend_exit();
INT begin_of_run(INT run_number, char *error);
INT end_of_run(INT run_number, char *error);
INT pause_run(INT run_number, char *error);
INT resume_run(INT run_number, char *error);
INT frontend_loop();

/*-- Equipment list ------------------------------------------------*/


DEVICE_DRIVER lv_driver[] = {
   {"LVDevice", hmp4040dev, 4, NULL, DF_INPUT | DF_OUTPUT },
   {"LVDSDevice", hmp4040dev, 4, NULL, DF_INPUT | DF_OUTPUT },
   {""}
};

DEVICE_DRIVER hv_driver[] = {
   {"HV Device", keithley2611Bdev, 1, NULL, DF_REPORT_CHSTATE | DF_HW_RAMP | DF_REPORT_STATUS},
   {""}
};

// |

EQUIPMENT equipment[] = {

   {"LV",                       /* equipment name */
    {30, 0,                       /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_SLOW,                   /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS,        /* read when running and on transitions */
     10000,                     /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", ""} ,
    cd_lv_read,                 /* readout routine */
    cd_lv,                      /* class driver main routine */
    lv_driver,                  /* device driver list */
    NULL,                       /* init string */
    },
    
    {"HV",                       /* equipment name */
    {30, 0,                       /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_SLOW,                   /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS,        /* read when running and on transitions */
     10000,                     /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", ""} ,
    cd_hv_read,                 /* readout routine */
    cd_hv,                      /* class driver main routine */
    hv_driver,                  /* device driver list */
    NULL,                       /* init string */
    },

   {""}
};

/*-- Dummy routines ------------------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
{
   return 1;
}

INT interrupt_configure(INT cmd, INT source, POINTER_T adr)
{
   return 1;
}

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{  
   printf("Initializing power supply fe\n");
   return CM_SUCCESS;
}

/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{
   printf("exiting power supply fe\n");
   return CM_SUCCESS;
}

/*-- Frontend Loop -------------------------------------------------*/

INT frontend_loop()
{
   ss_sleep(100);
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

/*-- Resume Run ----------------------------------------------------*/

INT resume_run(INT run_number, char *error)
{
   return CM_SUCCESS;
}

/*------------------------------------------------------------------*/
