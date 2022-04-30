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
    
    
   {"Water",            // equipment name
      {153, 0,                // event ID, trigger mask
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

   //~ {"Pixel Temperatures",     // equipment name
     //~ {203, 0,                 // event ID, trigger mask
       //~ "SYSTEM",              // event buffer
       //~ EQ_SLOW,               // equipment type
       //~ 0,                     // event source
       //~ "MIDAS",               // format
       //~ TRUE,                  // enabled
       //~ RO_ALWAYS,
       //~ 60000,                 // read full event every 60 sec
       //~ 1000,                  // read one value every 1000 msec
       //~ 0,                     // number of sub events
       //~ 1,                     // log history every second
       //~ "", "", ""} ,
     //~ cd_multi_read,           // readout routine
     //~ cd_multi
   //~ },

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

   mdevice_mscb env("Environment", "Input", DF_INPUT, "mscb400.mu3e", "", 100);
   env.set_threshold(0.005);
   
   //current mapping in https://elog.psi.ch/elogs/Testbeams/3965
   
   //sensor calibration https://elog.psi.ch/elogs/Mu3e+Services/58
   // T = LM35
   //RH = HIH4010
   //O2 = Alphasense O2A2, ~100 uA @ 21% over 100 Ohm * INA gain, see also https://elog.psi.ch/elogs/Testbeams/3441
   
   // SCS3000 msc400 Port 0
   env.define_var(1,  0, "US_bottom_Water");
   env.define_var(1,  1, "US_central_PCMini52_T");
   env.define_var(1,  2, "US_central_PCMini52_RH");
   env.define_var(1,  3, "US_top_O2");
   env.define_var(1,  4, "US_top_RH");
   env.define_var(1,  5, "US_FEC0_T");
   
   // SCS3000 msc400 Port 1
   env.define_var(1,  9, "DS_bottom_Water");
   env.define_var(1,  11, "DS_top_O2");
   env.define_var(1,  12, "DS_top_RH");
   env.define_var(1,  14, "DS_FEC2_T");
   env.define_var(1,  15, "DS_top_T");
   
   // SCS3000 msc400 Port 2
   env.define_var(1,  18, "US_bottom_O2");
   env.define_var(1,  19, "US_central_O2");
   env.define_var(1,  20, "US_bottom_RH");
   env.define_var(1,  21, "US_central_RH");
   
   // SCS3000 msc400 Port 3
   env.define_var(1,  27, "DS_bottom_O2");
   env.define_var(1,  28, "DS_central_O2");
   env.define_var(1,  29, "DS_bottom_RH");
   env.define_var(1,  30, "DS_central_RH");
   

   // define associated history panels
   env.define_panel("Oxygen",      {"US_top_O2",
                                    "DS_top_O2",
                                    "US_bottom_O2",
                                    "US_central_O2",
                                    "DS_central_O2"});
   env.define_panel("Temperature", {"US_central_PCMini52_T",
                                    "US_FEC0_T",
                                    "DS_FEC2_T",
                                    "DS_top_T"});
   env.define_panel("Humidity",    {"US_central_PCMini52_RH",
                                    "US_top_RH",
                                    "DS_top_RH",
                                    "US_bottom_RH",
                                    "US_central_RH",
                                    "DS_bottom_RH",
                                    "DS_central_RH"});
   env.define_panel("Water",       {"US_bottom_Water",
                                    "DS_bottom_Water"});
                                    
   mdevice_mscb water("Water", "Input", DF_INPUT, "mscb334.mu3e", "", 100);
   water.set_threshold(0.005);
   
   water.define_var(1,  0, "US_In_P");
   water.define_var(1,  1, "DS_In_P");
   
   water.define_var(1,  3, "US_Out_P");
   water.define_var(1,  4, "DS_Out_P");
   
   water.define_var(1,  5, "US_In_T");
   water.define_var(1,  6, "DS_In_T");
   
   water.define_var(1,  7, "US_Out_T");
   
   water.define_var(1,  16, "DS_Out_T");
   
   water.define_var(1,  17, "US_In_Flow");
   water.define_var(1,  18, "DS_In_Flow");

   water.define_panel("Pressure",    {"US_In_P",
                                      "US_Out_P",
                                      "DS_In_P",
                                      "DS_Out_P"});
   water.define_panel("Temperature", {"US_In_T",
                                      "US_Out_T",
                                      "DS_In_T",
                                      "DS_Out_T"});
   water.define_panel("Flow",        {"US_In_Flow",
                                      "DS_In_Flow"});


   // Pixel Temperatures
   //mdevice_mscb pix("Pixel Temperatures", "Input", DF_INPUT, "mscb334.mu3e", "", 100);
   //~ pix.define_var(0,  0, "US L0-0" , 0.005, -165.9, -228);
   //~ pix.define_var(0,  1, "US L0-1" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 39, "US L0-2" , 0.005, -165.9, -228);
   //~ pix.define_var(0,  3, "US L0-3" , 0.005, -165.9, -228);
   //~ pix.define_var(0,  4, "US L0-4" , 0.005, -165.9, -228);
   //~ pix.define_var(0,  5, "US L0-5" , 0.005, -165.9, -228);
   //~ pix.define_var(0,  6, "US L0-6" , 0.005, -165.9, -228);
   //~ pix.define_var(0,  7, "US L0-7" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 16, "US L1-0" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 17, "US L1-1" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 18, "US L1-2" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 19, "US L1-3" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 20, "US L1-4" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 21, "US L1-5" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 22, "US L1-6" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 23, "US L1-7" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 32, "US L1-8" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 33, "US L1-9" , 0.005, -165.9, -228);
   //~ pix.define_var(0,  8, "DS L0-0" , 0.005, -165.9, -228);
   //~ pix.define_var(0,  9, "DS L0-1" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 10, "DS L0-2" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 11, "DS L0-3" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 12, "DS L0-4" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 13, "DS L0-5" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 14, "DS L0-6" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 15, "DS L0-7" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 24, "DS L1-0" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 25, "DS L1-1" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 26, "DS L1-2" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 27, "DS L1-3" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 28, "DS L1-4" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 29, "DS L1-5" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 30, "DS L1-6" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 31, "DS L1-7" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 40, "DS L1-8" , 0.005, -165.9, -228);
   //~ pix.define_var(0, 41, "DS L1-9" , 0.005, -165.9, -228);

   // define associated history panels
   //~ env.define_panel("Tempreatures US L0",  0,   7);
   //~ env.define_panel("Tempreatures US L1",  8,  17);
   //~ env.define_panel("Tempreatures DS L0", 18,  24);
   //~ env.define_panel("Tempreatures DS L1", 25,  34);

   return CM_SUCCESS;
}
