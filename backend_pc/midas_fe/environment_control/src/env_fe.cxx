/********************************************************************\

  Name:         env_fe.c
  Created by:   Stefan Ritt

  Contents:     Slow control frontend for Mu3e experiment environment

\********************************************************************/

#include <cstdio>
#include <cstring>
#include "mscb.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h"
#include "class/multi.h"
#include "class/generic.h"
#include "device/mscbdev.h"
#include "device/mdevice.h"

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "Environment Frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/*-- Equipment list ------------------------------------------------*/

BOOL equipment_common_overwrite = TRUE;

EQUIPMENT equipment[] = {

   {"Environment",            // equipment name
      {150, 0,                // event ID, trigger mask
         "SYSTEM",            // event buffer
         EQ_SLOW,             // equipment type
         0,                   // event source
         "MIDAS",             // format
         TRUE,                // enabled
         RO_ALWAYS,
         60000,               // read full event every 60 sec
         1000,                // read one value every 1000 msec
         0,                   // number of sub events
         1,                   // log history every second
         "", "", ""} ,
      cd_multi_read,          // readout routine
      cd_multi
    },

   {"Pixel Temperatures",     // equipment name
     {151, 0,                 // event ID, trigger mask
       "SYSTEM",              // event buffer
       EQ_SLOW,               // equipment type
       0,                     // event source
       "MIDAS",               // format
       TRUE,                  // enabled
       RO_ALWAYS,
       60000,                 // read full event every 60 sec
       1000,                  // read one value every 1000 msec
       0,                     // number of sub events
       1,                     // log history every second
       "", "", ""} ,
     cd_multi_read,           // readout routine
     cd_multi
   },

  {""}
};

/*-- Error dispatcher causing communiction alarm -------------------*/

void hpfe_error(const char *error)
{
   char str[256];

   strlcpy(str, error, sizeof(str));
   cm_msg(MERROR, "hpfe_error", "%s", str);
   al_trigger_alarm("MSCB", str, "MSCB Alarm", "Communication Problem", AT_INTERNAL);
}

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
   /* set error dispatcher for alarm functionality */
   mfe_set_error(hpfe_error);

   /* set maximal retry count */
   mscb_set_max_retry(100);
   midas::odb::delete_key("/Equipment/Environment/Settings");

   /*---- set correct ODB device addresses ----*/

   // O2 sensors
   mdevice_mscb env("Environment", "Input", DF_INPUT, "mscb400.mu3e", "", 100);
   env.define_var(1,  3, "Env_US2 - O2-7-top",       0.001, 10.0, 0.009);
   env.define_var(1, 19, "Env_US3 - O2-6-central",   0.001, 10.0, 0.042);
   env.define_var(1, 18, "Env_DS3 - O2-11-bottom",   0.001, 10.0, 0.082);

   // Temperature
   env.define_var(1,  2, "PCMini52_T",               0.005,  20.0, 0.000);
   env.define_var(1,  5, "LM35_Fibre_US",            0.005, 100.0, 0.000);
   env.define_var(1,  6, "LM35_FEC1_US",             0.005, 100.0, 0.000);
   env.define_var(1,  7, "LM35_top_US",              0.005, 100.0, 0.000);
   env.define_var(1, 13, "LM35_Fibre_DS",            0.005, 100.0, 0.000);
   env.define_var(1, 14, "LM35_FEC1_DS",             0.005, 100.0, 0.000);
   env.define_var(1, 15, "LM35_top_DS",              0.005, 100.0, 0.000);

   // Humidity
   env.define_var(1,  1, "PCMini52_RH",              0.005, 20.0, 0.000);
   env.define_var(1,  4, "HIH4040_top_US",           0.005, 32.5905, 26.0724);
   env.define_var(1, 12, "HIH4040_top_DS",           0.005, 32.5905, 26.0724);
   env.define_var(1, 20, "HIH4040_bottom_US",        0.005, 32.5905, 26.0724);
   env.define_var(1, 21, "HIH4040_central_US",       0.005, 32.5905, 26.0724);
   env.define_var(1, 29, "HIH4040_bottom_DS",        0.005, 32.5905, 26.0724);
   env.define_var(1, 30, "HIH4040_central_DS",       0.005, 32.5905, 26.0724);

   // Water
   env.define_var(1,  0, "Water_US",                 0.005, 1.0, 0.000);
   env.define_var(1,  9, "Water_DS",                 0.005, 1.0, 0.000);

   // define associated history panels
   env.define_panel("Oxygen",         0,  2);
   env.define_panel("Tempreature",    3,  9);
   env.define_panel("Humidity",      10, 16);
   env.define_panel("Water",         17, 18);

   // Pixel Temperatures
   mdevice_mscb pix("Pixel Temperatures", "Input", DF_INPUT, "mscb334.mu3e", "", 100);
   pix.define_var(0,  0, "US L0-0" , 0.005, -165.9, -228);
   pix.define_var(0,  1, "US L0-1" , 0.005, -165.9, -228);
   pix.define_var(0, 39, "US L0-2" , 0.005, -165.9, -228);
   pix.define_var(0,  3, "US L0-3" , 0.005, -165.9, -228);
   pix.define_var(0,  4, "US L0-4" , 0.005, -165.9, -228);
   pix.define_var(0,  5, "US L0-5" , 0.005, -165.9, -228);
   pix.define_var(0,  6, "US L0-6" , 0.005, -165.9, -228);
   pix.define_var(0,  7, "US L0-7" , 0.005, -165.9, -228);
   pix.define_var(0, 16, "US L1-0" , 0.005, -165.9, -228);
   pix.define_var(0, 17, "US L1-1" , 0.005, -165.9, -228);
   pix.define_var(0, 18, "US L1-2" , 0.005, -165.9, -228);
   pix.define_var(0, 19, "US L1-3" , 0.005, -165.9, -228);
   pix.define_var(0, 20, "US L1-4" , 0.005, -165.9, -228);
   pix.define_var(0, 21, "US L1-5" , 0.005, -165.9, -228);
   pix.define_var(0, 22, "US L1-6" , 0.005, -165.9, -228);
   pix.define_var(0, 23, "US L1-7" , 0.005, -165.9, -228);
   pix.define_var(0, 32, "US L1-8" , 0.005, -165.9, -228);
   pix.define_var(0, 33, "US L1-9" , 0.005, -165.9, -228);
   pix.define_var(0,  8, "DS L0-0" , 0.005, -165.9, -228);
   pix.define_var(0,  9, "DS L0-1" , 0.005, -165.9, -228);
   pix.define_var(0, 10, "DS L0-2" , 0.005, -165.9, -228);
   pix.define_var(0, 11, "DS L0-3" , 0.005, -165.9, -228);
   pix.define_var(0, 12, "DS L0-4" , 0.005, -165.9, -228);
   pix.define_var(0, 13, "DS L0-5" , 0.005, -165.9, -228);
   pix.define_var(0, 14, "DS L0-6" , 0.005, -165.9, -228);
   pix.define_var(0, 15, "DS L0-7" , 0.005, -165.9, -228);
   pix.define_var(0, 24, "DS L1-0" , 0.005, -165.9, -228);
   pix.define_var(0, 25, "DS L1-1" , 0.005, -165.9, -228);
   pix.define_var(0, 26, "DS L1-2" , 0.005, -165.9, -228);
   pix.define_var(0, 27, "DS L1-3" , 0.005, -165.9, -228);
   pix.define_var(0, 28, "DS L1-4" , 0.005, -165.9, -228);
   pix.define_var(0, 29, "DS L1-5" , 0.005, -165.9, -228);
   pix.define_var(0, 30, "DS L1-6" , 0.005, -165.9, -228);
   pix.define_var(0, 31, "DS L1-7" , 0.005, -165.9, -228);
   pix.define_var(0, 40, "DS L1-8" , 0.005, -165.9, -228);
   pix.define_var(0, 41, "DS L1-9" , 0.005, -165.9, -228);

   // define associated history panels
   env.define_panel("Tempreatures US L0",  0,   7);
   env.define_panel("Tempreatures US L1",  8,  17);
   env.define_panel("Tempreatures DS L0", 18,  24);
   env.define_panel("Tempreatures DS L1", 25,  34);

   return CM_SUCCESS;
}
