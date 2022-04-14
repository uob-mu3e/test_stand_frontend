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
     {203, 0,                 // event ID, trigger mask
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

   // SCS3000 Port 0
   mdevice_mscb env("Environment", "Input", DF_INPUT, "mscb400.mu3e", "", 100);
   env.set_threshold(0.001);
   env.define_var(1,  0, "US1 - Water");
   env.define_var(1,  1, "US1 - PCMini52_RH");
   env.define_var(1,  2, "US1 - PCMini52_T");
   env.define_var(1,  3, "US2 - O2-7-top");
   env.define_var(1,  4, "US2 - HIH4040_top");
   env.define_var(1,  5, "US2 - LM35_Fibre");
   env.define_var(1,  6, "US2 - LM35_FEC1");
   env.define_var(1,  7, "US2 - LM35_top");

   // SCS3000 Port 1
   env.define_var(1,  8, "DS1 - Water");
   env.define_var(1,  9, "NC 1-1");
   env.define_var(1, 10, "DS2 - O2-4-top");
   env.define_var(1, 11, "DS2 - HIH4040_top");
   env.define_var(1, 12, "DS2 - LM35_Fibre");
   env.define_var(1, 13, "DS2 - LM35_FEC1");
   env.define_var(1, 14, "DS2 - LM35_top");
   env.define_var(1, 15, "NC 1-7");

   // SCS3000 Port 2
   env.define_var(1, 16, "US3 - O2-11-bottom");
   env.define_var(1, 17, "US3 - O2-6-central");
   env.define_var(1, 18, "US3 - HIH4040_bottom");
   env.define_var(1, 19, "US3 - HIH4040_central");
   env.define_var(1, 20, "NC 2-4");
   env.define_var(1, 21, "US6 - O2_3");
   env.define_var(1, 22, "US6 - LM35_3");
   env.define_var(1, 23, "US6 - HIH_3");

   // SCS3000 Port 3
   env.define_var(1, 24, "DS3 - O2-11-bottom");
   env.define_var(1, 25, "DS3 - O2-6-central");
   env.define_var(1, 26, "DS3 - HIH4040_bottom");
   env.define_var(1, 27, "DS3 - HIH4040_central");
   env.define_var(1, 28, "NC 2-4");
   env.define_var(1, 29, "US6 - LM35_6");
   env.define_var(1, 30, "US6 - HIH_6");
   env.define_var(1, 31, "US6 - O2_6");

   // define associated history panels
   env.define_panel("Oxygen",      {"US2 - O2-7-top",
                                    "DS2 - O2-4-top",
                                    "US3 - O2-11-bottom",
                                    "US3 - O2-6-central",
                                    "US6 - O2_3",
                                    "DS3 - O2-11-bottom",
                                    "DS3 - O2-6-central",
                                    "US6 - O2_6"});
   env.define_panel("Tempreature", {"US2 - LM35_Fibre",
                                    "US2 - LM35_FEC1",
                                    "US2 - LM35_top",
                                    "DS2 - LM35_Fibre",
                                    "DS2 - LM35_FEC1",
                                    "DS2 - LM35_top",
                                    "US6 - LM35_3",
                                    "US6 - LM35_6"});
   env.define_panel("Humidity",    {"US2 - HIH4040_top",
                                    "DS2 - HIH4040_top",
                                    "US3 - HIH4040_bottom",
                                    "US3 - HIH4040_central",
                                    "US6 - HIH_3",
                                    "DS3 - HIH4040_bottom",
                                    "DS3 - HIH4040_central",
                                    "US6 - HIH_6"});
   env.define_panel("Water",       {"US1 - Water",
                                    "DS1 - Water"});

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
