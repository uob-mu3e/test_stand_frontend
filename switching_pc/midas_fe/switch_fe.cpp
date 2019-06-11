/********************************************************************\

  Name:         taken from switch_fe.cpp
  Created by:   Stefan Ritt
  Updated by:   Marius Koeppel

  Contents:     Code for switching front-end to illustrate
                manual generation of slow control events
                and hardware updates via cm_watch().

                The values of

                /Equipment/Switching SC/Settings/Active
                /Equipment/Switching SC/Settings/Delay

                are propagated to hardware when the ODB value changes.

                The SC Commands

                /Equipment/Switching SC/Settings/Write
                /Equipment/Switching SC/Settings/Read

                can be set to TRUE to trigger a specific action
                in this front-end.

                For a real program, the "TODO" lines have to be 
                replaced by actual hardware acces.

                Custom page slow control
                -----------

                The custom page "sc.html" in this directory can be
                used to control the settins of this frontend. To
                do so, set "/Custom/Path" in the ODB to this 
                directory and create a string

                /Custom/Slow Control = sc.html

                then click on "Slow Control" on the left lower corner
                in the web status page.

\********************************************************************/

#include <stdio.h>
#include <cassert>
#include "midas.h"
#include "mfe.h"

#include "mudaq_device.h"

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "SW Frontend";
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

/* DMA Buffer and related */
mudaq::DmaMudaqDevice * mup;

/*-- Function declarations -----------------------------------------*/

INT read_sc_event(char *pevent, INT off);
void sc_settings_changed(HNDLE, HNDLE, int, void *);
void sc_variables_changed(HNDLE hDB, HNDLE hKey, INT, void *);

/*-- Equipment list ------------------------------------------------*/

/* Default values for /Equipment/Switching SC/Settings */
const char *sc_settings_str[] = {
"Active = BOOL : 1",
"Delay = INT : 0",
"Write = BOOL : 0",
"Read = BOOL : 0",
"Read WM = BOOL : 0",
"Read RM = BOOL : 0",
"Read Test = BOOL : 0",
"[32] Temp0",
"[32] Temp1",
"[32] Temp2",
"[32] Temp3",
nullptr
};

EQUIPMENT equipment[] = {

   {"Switching",                /* equipment name */
    {2, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,               /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read always and update ODB */
     1000,                      /* read every 1 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", ""} ,
     read_sc_event,             /* readout routine */
   },

   {""}
};


/*-- Dummy routines ------------------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
{
   return 1;
};

INT interrupt_configure(INT cmd, INT source, POINTER_T adr)
{
   return 1;
};

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
   HNDLE hKeySC;
   HNDLE hVarSC;

   // create Settings structure in ODB
   db_create_record(hDB, 0, "Equipment/Switching/Settings", strcomb(sc_settings_str));
   db_find_key(hDB, 0, "/Equipment/Switching", &hKeySC);
   assert(hKeySC);

   db_watch(hDB, hKeySC, sc_settings_changed, nullptr);

   // set default values of variables
   db_create_key(hDB, 0, "Equipment/Switching/Variables/FPGA_ID", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/SC_LENGTH", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/SC_StartAdd", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/SC_WM_MEM_Start", TID_INT);

   db_create_key(hDB, 0, "Equipment/Switching/Variables/length_test", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/start_test", TID_INT);

   // add custom page to ODB
   db_create_key(hDB, 0, "Custom/Switching&", TID_STRING);
   const char * name = "sc.html";
   db_set_value(hDB,0,"Custom/Switching&", name, sizeof(name), 1, TID_STRING);

   // open mudaq
   mup = new mudaq::DmaMudaqDevice("/dev/mudaq0");
   if ( !mup->open() ) {
       cm_msg(MERROR, "frontend_init" , "Could not open device");
       return FE_ERR_DRIVER;
   }

   if ( !mup->is_ok() ) {
       cm_msg(MERROR, "frontend_init", "Mudaq is not ok");
       return FE_ERR_DRIVER;
   }

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

/*--- Read Slow Control Event to be put into data stream --------*/

INT read_sc_event(char *pevent, INT off)
{
   bk_init(pevent);

   mudaq::DmaMudaqDevice & mu = *mup;

   if (mu.read_memory_rw(0) == 0) {
      return CM_SUCCESS; // no new data
   }

   unsigned int sc_length = mu.read_memory_rw(0);
   unsigned int sc_start_add = mu.read_memory_rw(1);

   int *pdata;
   bk_create(pevent, "WM", TID_INT, (void **)&pdata);

   for (unsigned int i = sc_start_add; i < sc_start_add + sc_length; i++) {
       *pdata++ = mu.read_memory_rw(i);
   }

   mu.write_memory_rw(0, 0);

   bk_close(pevent, pdata);

   return bk_size(pevent);
}

/*--- Called whenever settings have changed ------------------------*/

void sc_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *)
{
   KEY key;

   db_get_key(hDB, hKey, &key);

   mudaq::DmaMudaqDevice & mu = *mup;

   if (std::string(key.name) == "Active") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "sc_settings_changed", "Set active to %d", value);
      // TODO: propagate to hardware
   }

   if (std::string(key.name) == "Delay") {
      INT value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_INT);
      cm_msg(MINFO, "sc_settings_changed", "Set delay to %d", value);
      // TODO: propagate to hardware
   }

   if (std::string(key.name) == "Write") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if (value) {
         cm_msg(MINFO, "sc_settings_changed", "Execute write");

          uint32_t FPGA_ID    = 0xFFF;
          uint32_t arr[] = { 1, 2, 3, 4, 0xAFFEAFFE };
          uint32_t *data = arr;
          uint16_t length     = 5;
          uint32_t startaddr  = 0x1;
          uint32_t mem_start = 0x2;

          for(int i = 0; i < 1; i++){
              mu.FEB_write(FPGA_ID, data, length, startaddr, mem_start);
              usleep(1000);
          }

         value = FALSE; // reset flag in ODB
         db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }

   if (std::string(key.name) == "Read") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if (value) {
         cm_msg(MINFO, "sc_settings_changed", "Execute read");

         INT FPGA_ID, size_fpga;
         INT Length, size_length;
         INT SC_StartAdd, size_start;
         INT SC_WM_MEM_START, size_mem;

         size_fpga = sizeof(FPGA_ID);
         size_length = sizeof(Length);
         size_start = sizeof(SC_StartAdd);
         size_mem = sizeof(SC_WM_MEM_START);

         char str_fpga[128];
         char str_length[128];
         char str_start[128];
         char str_mem[128];

         sprintf(str_fpga,"Equipment/Switching/Variables/FPGA_ID");
         sprintf(str_length,"Equipment/Switching/Variables/SC_LENGTH");
         sprintf(str_start,"Equipment/Switching/Variables/SC_StartAdd");
         sprintf(str_mem,"Equipment/Switching/Variables/SC_WM_MEM_Start");

         db_get_value(hDB, 0, str_fpga, &FPGA_ID, &size_fpga, TID_INT, 0);
         db_get_value(hDB, 0, str_length, &Length, &size_length, TID_INT, 0);
         db_get_value(hDB, 0, str_start, &SC_StartAdd, &size_start, TID_INT, 0);
         db_get_value(hDB, 0, str_mem, &SC_WM_MEM_START, &size_mem, TID_INT, 0);

         INT new_SC_WM_MEM_START = SC_WM_MEM_START + 4;
         INT new_size_mem;
         new_size_mem = sizeof(new_SC_WM_MEM_START);

         db_set_value(hDB, 0, str_mem, &new_SC_WM_MEM_START, new_size_mem, 1, TID_INT);

         for(int i = 0; i < 1; i++){
             mu.FEB_read((uint32_t) FPGA_ID, (uint16_t) Length, (uint32_t) SC_StartAdd, (uint32_t) SC_WM_MEM_START);
             usleep(1000);
         }

         value = FALSE; // reset flag in ODB
         db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }

    if (std::string(key.name) == "Read Test") {
        BOOL value;
        int size = sizeof(value);
        db_get_data(hDB, hKey, &value, &size, TID_BOOL);
        if (value) {
            cm_msg(MINFO, "sc_settings_changed", "Execute read");

            INT length_test, start_test, size_length, size_start;

            size_length = sizeof(length_test);
            size_start = sizeof(start_test);

            char str_length[128];
            char str_start[128];

            sprintf(str_length,"Equipment/Switching/Variables/length_test");
            sprintf(str_start,"Equipment/Switching/Variables/start_test");

            db_get_value(hDB, 0, str_length, &length_test, &size_length, TID_INT, 0);
            db_get_value(hDB, 0, str_start, &start_test, &size_start, TID_INT, 0);

            mu.write_memory_rw(0, length_test);
            mu.write_memory_rw(1, start_test);

            value = FALSE; // reset flag in ODB
            db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
        }
    }
}
