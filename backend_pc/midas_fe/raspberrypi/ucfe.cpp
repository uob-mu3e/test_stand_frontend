/********************************************************************\

  Name:         ucfe.cpp
  Created by:   Marius Koeppel June 2018

  Contents:     Slow control front-end for Raspberry Pi 
                computer for usb server and clock configuration. 

                The custom page rpi.html can be used to control
                configure the usb server and to program the clock 
                configuration.

                /Custom/Path = /<path to rpi.html>
                /Custom/RPi = rpi.html

                Befor compiling this program, you have to install
                the I2C package on the Raspberry Pi with:

                $ sudo apt-get install libi2c-dev i2c-tools
                
\********************************************************************/

#include <stdio.h>
#include "midas.h"

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "SC Frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

/* a frontend status page is displayed with this frequency in ms    */
INT display_period = 1000;

/* maximum event size produced by this frontend */
INT max_event_size = 10000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 10 * 10000;

/*-- Function declarations -----------------------------------------*/
INT read_sc_event(char *pevent, INT off);


/*-- Equipment list ------------------------------------------------*/
EQUIPMENT equipment[] = {
   {"RaspberryPi",              /* equipment name */
    {4, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                   /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read when running and on transitions */
     1000,                     /* read every 1 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", ""},
    read_sc_event,              /* readout routine */
    },

   {""}
};


/*-- Dummy routines ------------------------------------------------*/

INT poll_event(char *pevent, INT off)
{
   return 1;
};
INT interrupt_configure(INT cmd, INT source[], POINTER_T adr)
{
   return 1;
};

/*-- Readout routines ------------------------------------------------*/
INT read_sc_event(INT idx)
{
    return 0;
}

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
   return CM_SUCCESS;
}

/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{
   return CM_SUCCESS;
}

/*-- Frontend Loop -------------------------------------------------*/

INT frontend_loop()
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

/*-- Resume Run ----------------------------------------------------*/

INT resume_run(INT run_number, char *error)
{
   return CM_SUCCESS;
}

/*------------------------------------------------------------------*/
