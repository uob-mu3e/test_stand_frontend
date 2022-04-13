/********************************************************************\

  Name:         hv_fe.c
  Created by:   Stefan Ritt

  Contents:     Slow control frontend for Mu3e High Voltage

\********************************************************************/

#include <cstdio>
#include <cstring>
#include "mscb.h"
#include "midas.h"
#include "odbxx.h"
#include "mfe.h"
#include "class/hv.h"
#include "device/mscbhv4.h"
#include "device/mscbdev.h"
#include "device/mdevice.h"
#include "mdevice_mscbhv4.h"

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "SC Frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/*-- Equipment list ------------------------------------------------*/

BOOL equipment_common_overwrite = TRUE;

EQUIPMENT equipment[] = {

   {"MuPix HV",                 // equipment name
      {140, 0,                  // event ID, trigger mask
         "SYSTEM",              // event buffer
         EQ_SLOW,               // equipment type
         0,                     // event source
         "MIDAS",               // format
         TRUE,                  // enabled
         RO_ALWAYS,
         60000,                 // read full event every 60 sec
         10,                    // read one value every 10 msec
         0,                     // number of sub events
         1,                     // log history every second
         "", "", ""} ,
      cd_hv_read,               // readout routine
      cd_hv,                    // class driver main routine
   },

   {"SciFi HV",                 // equipment name
      {141, 0,                  // event ID, trigger mask
         "SYSTEM",              // event buffer
         EQ_SLOW,               // equipment type
         0,                     // event source
         "MIDAS",               // format
         TRUE,                  // enabled
         RO_ALWAYS,
         60000,                 // read full event every 60 sec
         10,                    // read one value every 10 msec
         0,                     // number of sub events
         1,                     // log history every second
         "", "", ""} ,
      cd_hv_read,               // readout routine
      cd_hv,                    // class driver main routine
   },

   {""}
};

/*-- Error dispatcher causing communiction alarm -------------------*/

void fe_error(const char *error)
{
   char str[256];

   strlcpy(str, error, sizeof(str));
   cm_msg(MERROR, "fe_error", "%s", str);
   al_trigger_alarm("MSCB", str, "MSCB Alarm", "Communication Problem", AT_INTERNAL);
}

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
   /* set error dispatcher for alarm functionality */
   mfe_set_error(fe_error);

   /* set maximal retry count */
   mscb_set_max_retry(100);
   midas::odb::delete_key("/Equipment/MuPix HV/Settings");
   midas::odb::delete_key("/Equipment/SciFi HV/Settings");

   /*---- set correct ODB device addresses ----*/

   mdevice_mscbhv4 mupix("MuPix HV", "MuPix HV", "mscb394");
   mupix.define_box(1, {"U0-0", "U0-1", "U0-2", "U0-3"});
   mupix.define_box(2, {"U1-0", "U1-1", "U1-2", "U1-3"});
   mupix.define_box(3, {"U2-0", "U2-1", "U2-2", "U2-3"});
   mupix.define_box(4, {"U3-0", "U3-1", "U3-2", "U3-3"});
   mupix.define_box(5, {"D0-0", "D0-1", "D0-2", "D0-3"});
   mupix.define_box(6, {"D1-0", "D1-1", "D1-2", "D1-3"});
   mupix.define_box(7, {"D2-0", "D2-1", "D2-2", "D2-3"});
   mupix.define_box(8, {"D3-0", "D3-1", "D3-2", "D3-3"});

   mdevice_mscbhv4 scifi("SciFi HV", "SciFi HV", "mscb394");
   scifi.define_box(20, {"Module 2U", "Module 1U", "", ""});
   scifi.define_box(21, {"", "", "Module 2D", "Module 1D"});

   return CM_SUCCESS;
}
