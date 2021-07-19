/********************************************************************\

  Name:         sc_fe.c
  Created by:   Stefan Ritt

  Contents:     Slow control frontend for Mu3e
                MSCB devices and PSI beamline

\********************************************************************/

#include <cstdio>
#include <cstring>
#include "mscb.h"
#include "midas.h"
#include "mfe.h"
#include "class/multi.h"
#include "class/generic.h"
#include "device/mscbdev.h"
#include "device/mscbhvr.h"
#include "odbxx.h"
#include "history.h"

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "Pixel_Temperatures";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = TRUE;

/* a frontend status page is displayed with this frequency in ms    */
INT display_period = 0;

/* maximum event size produced by this frontend */
INT max_event_size = 10000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 10 * 10000;

/*-- Equipment list ------------------------------------------------*/

/* device driver list */
DEVICE_DRIVER mscb_driver[] = {
   {"Input", mscbdev, 0, nullptr, DF_INPUT | DF_MULTITHREAD},
   {"Output", mscbdev, 0, nullptr, DF_OUTPUT | DF_MULTITHREAD},
   {""}
};

BOOL equipment_common_overwrite = TRUE;

EQUIPMENT equipment[] = {

   {"Pixel_Temperatures",            /* equipment name */
    {151, 0,                     /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_SLOW,                   /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS,
     60000,                     /* read full event every 60 sec */
     1000,                       /* read one value every 1000 msec */
     0,                         /* number of sub events */
     1,                         /* log history every second */
     "", "", ""} ,
    cd_multi_read,              /* readout routine */
    cd_multi,                   /* class driver main routine */
    mscb_driver,                /* device driver list */
    nullptr,                       /* init string */
    },

  {""}
};

/*-- Dummy routines ------------------------------------------------*/

INT poll_event(__attribute__((unused)) INT source, __attribute__((unused)) INT count, __attribute__((unused))BOOL test)
{
   return 1;
};

INT interrupt_configure(__attribute__((unused)) INT cmd, __attribute__((unused)) INT source, __attribute__((unused)) PTYPE adr)
{
   return 1;
};

/*-- Function to define MSCB variables in a convenient way ---------*/

void mscb_define(std::string eq, std::string devname, DEVICE_DRIVER *driver,
                 std::string submaster, int address, unsigned char var_index, std::string name,
                 double threshold, double factor, double offset)
{
   midas::odb::set_debug(false);
   midas::odb dev = {
           {"Device",       std::string(255, '\0')},
           {"Pwd",          std::string(31, '\0')},
           {"MSCB Address", 0},
           {"MSCB Index",   (UINT8) 0}
   };
   dev.connect("/Equipment/" + eq + "/Settings/Devices/" + devname);

   if (!submaster.empty()) {
      if (dev["Device"] == std::string(""))
         dev["Device"] = submaster;
      else if (dev["Device"] != submaster) {
         cm_msg(MERROR, "mscb_define", "Device \"%s\" defined with different submasters", devname.c_str());
         return;
      }
   } else
      if (dev["Device"] == std::string("")) {
         cm_msg(MERROR, "mscb_define", "Device \"%s\" defined without submaster name", devname.c_str());
         return;
      }

   // find device in device driver
   int dev_index;
   for (dev_index=0 ; driver[dev_index].name[0] ; dev_index++)
      if (equal_ustring(driver[dev_index].name, devname.c_str()))
         break;

   if (!driver[dev_index].name[0]) {
      cm_msg(MERROR, "mscb_define", "Device \"%s\" not present in device driver list", devname.c_str());
      return;
   }

   // count total number of channels
   int chn_total = 0;
   for (int i=0 ; i<=dev_index ; i++)
      chn_total += driver[i].channels;

   int chn_index = driver[dev_index].channels;

   dev.set_auto_enlarge_array(true);
   dev["MSCB Address"][chn_index] = address;
   dev["MSCB Index"][chn_index] = var_index;

   midas::odb settings;
   settings.connect("/Equipment/" + eq + "/Settings");
   settings.set_auto_enlarge_array(true);
   settings.set_preserve_string_size(true);

   if (driver[dev_index].flags & DF_INPUT) {
      if (chn_index == 0)
         settings["Update Threshold"] = (float) threshold;
      else
         settings["Update Threshold"][chn_index] = (float) threshold;
   }

   std::string fn(devname + " Factor");
   std::string on(devname + " Offset");
   if (chn_index == 0) {
      settings[fn] = (float) factor;
      settings[on] = (float) offset;
   } else {
      settings[fn][chn_index] = (float) factor;
      settings[on][chn_index] = (float) offset;
   }

   if (!name.empty()) {
      std::string sk = "Names " + devname;
      if (chn_index == 0) {
         settings[sk] = std::string(31, ' ');
         settings[sk] = name;
      } else
         settings[sk][chn_index] = &name;
   }

   // increment number of channels for this driver
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
   /* set error dispatcher for alarm functionality */
   mfe_set_error(scfe_error);

   /* set maximal retry count */
   mscb_set_max_retry(100);
   midas::odb::delete_key("/Equipment/Pixel_Temperatures/Settings");

   /*---- set correct ODB device addresses ----*/

   // Inputs (Factors and Offsets  from Luigi: 0.228-0.3318*x*1000)
   // Pixel_Temperatures
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0,  0, "US L0-0" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0,  1, "US L0-1" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 39, "US L0-2" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0,  3, "US L0-3" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0,  4, "US L0-4" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0,  5, "US L0-5" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0,  6, "US L0-6" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0,  7, "US L0-7" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 16, "US L1-0" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 17, "US L1-1" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 18, "US L1-2" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 19, "US L1-3" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 20, "US L1-4" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 21, "US L1-5" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 22, "US L1-6" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 23, "US L1-7" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 32, "US L1-8" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 33, "US L1-9" , 0.005, -165.9, -228);  
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0,  8, "DS L0-0" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0,  9, "DS L0-1" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 10, "DS L0-2" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 11, "DS L0-3" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 12, "DS L0-4" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 13, "DS L0-5" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 14, "DS L0-6" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 15, "DS L0-7" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 24, "DS L1-0" , 0.005, -165.9, -228);
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 25, "DS L1-1" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 26, "DS L1-2" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 27, "DS L1-3" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 28, "DS L1-4" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 29, "DS L1-5" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 30, "DS L1-6" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 31, "DS L1-7" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 40, "DS L1-8" , 0.005, -165.9, -228); 
   mscb_define("Pixel_Temperatures", "Input", mscb_driver, "mscb334.mu3e", 0, 41, "DS L1-9" , 0.005, -165.9, -228); 

   // History Panels
   // Pixel_Temperatures
   hs_define_panel("Environment", "Pixel_Temp",{"Pixel_Temperatures:US L0-0","Pixel_Temperatures:US L0-1",
						"Pixel_Temperatures:US L0-2","Pixel_Temperatures:US L0-3",
						"Pixel_Temperatures:US L0-4","Pixel_Temperatures:US L0-5",
						"Pixel_Temperatures:US L0-6","Pixel_Temperatures:US L0-7",
						"Pixel_Temperatures:US L1-0","Pixel_Temperatures:US L1-1",
						"Pixel_Temperatures:US L1-2","Pixel_Temperatures:US L1-3","Pixel_Temperatures:US L1-4",
						"Pixel_Temperatures:US L1-5","Pixel_Temperatures:US L1-6",
						"Pixel_Temperatures:US L1-7","Pixel_Temperatures:US L1-7","Pixel_Temperatures:US L1-9",
						"Pixel_Temperatures:DS L0-0","Pixel_Temperatures:DS L0-1",
						"Pixel_Temperatures:DS L0-2","Pixel_Temperatures:DS L0-3",
						"Pixel_Temperatures:DS L0-4","Pixel_Temperatures:DS L0-5",
						"Pixel_Temperatures:DS L0-6","Pixel_Temperatures:DS L0-7",
						"Pixel_Temperatures:DS L1-0","Pixel_Temperatures:DS L1-1",
						"Pixel_Temperatures:DS L1-2","Pixel_Temperatures:DS L1-3","Pixel_Temperatures:DS L1-4",
						"Pixel_Temperatures:DS L1-5","Pixel_Temperatures:DS L1-6",
						"Pixel_Temperatures:DS L1-7","Pixel_Temperatures:DS L1-7","Pixel_Temperatures:DS L1-9"
						});


   return CM_SUCCESS;
}

/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{
   return CM_SUCCESS;
}

/*-- Begin of Run --------------------------------------------------*/

INT begin_of_run(__attribute__((unused)) INT rn, __attribute__((unused)) char *error)
{
   return CM_SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(__attribute__((unused)) INT rn, __attribute__((unused)) char *error)
{
   return CM_SUCCESS;
}

/*-- Pause Run -----------------------------------------------------*/

INT pause_run(__attribute__((unused)) INT rn, __attribute__((unused)) char *error)
{
   return CM_SUCCESS;
}

/*-- Resuem Run ----------------------------------------------------*/

INT resume_run(__attribute__((unused)) INT rn, __attribute__((unused)) char *error)
{
   return CM_SUCCESS;
}

/*-- Frontend Loop -------------------------------------------------*/

INT frontend_loop()
{  
   /* don't eat up all CPU time in main thread */
   ss_sleep(100);

   return CM_SUCCESS;
}

/*------------------------------------------------------------------*/

