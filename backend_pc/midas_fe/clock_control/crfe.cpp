/********************************************************************\

  Name:         crfe.c
  Created by:   Stefan Ritt

  Contents:     Code for modern slow control front-end "Clock and Reset"
                to illustrate manual generation of slow control
                events and hardware updates via cm_watch().

                The values of

                /Equipment/Clock Reset/Settings/Active
                /Equipment/Clock Reset/Settings/Delay

                are propagated to hardware when the ODB value chanes.

                The triggers

                /Equipment/Clock Reset/Settings/Reset Trigger
                /Equipment/Clock Reset/Settings/Sync Trigger

                can be set to TRUE to trigger a specific action
                in this front-end.

                For a real program, the "TODO" lines have to be 
                replaced by actual hardware acces.

                Custom page
                -----------

                The custom page "cr.html" in this directory can be
                used to control the settins of this frontend. To
                do so, set "/Custom/Path" in the ODB to this 
                directory and create a string

                /Custom/Clock Reset = cr.html

                then click on "Clock Reset" on the left lower corner
                in the web status page.

\********************************************************************/

#include <stdio.h>
#include <cassert>
#include <iostream>
#include "midas.h"
#include "mfe.h"


#include "clockboard.h"
#include "reset_protocol.h"

using std::cout;
using std::endl;
using std::hex;


/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "CR Frontend";
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


// Clock board interface
clockboard * cb;

/*-- Function declarations -----------------------------------------*/

INT read_cr_event(char *pevent, INT off);
void cr_settings_changed(HNDLE, HNDLE, int, void *);

/*-- Equipment list ------------------------------------------------*/

/* Default values for /Equipment/Clock Reset/Settings */
const char *cr_settings_str[] = {
"Active = BOOL : 1",
"IP = STRING : [16] 10.32.113.218",
"PORT = INT : 50001",
"Run Prepare = BOOL : 0",
"Sync = BOOL : 0",
"Start Run = BOOL : 0",
"End Run = BOOL : 0",
"Abort Run = BOOL : 0",
"Start Link Test = BOOL : 0",
"Stop Link Test = BOOL : 0",
"Start Sync Test = BOOL : 0",
"Stop Sync Test = BOOL : 0",
"Test Sync = BOOL : 0",
"Reset = BOOL : 0",
"Stop Reset = BOOL : 0",
"Enable = BOOL : 0",
"Disable = BOOL : 0",
"Address = BOOL : 0",
"Payload = INT : 0",
"Names CRT1 = STRING[4] :",
"[32] Motherboard Current",
"[32] Motherboard Voltage",
"[32] RX Firefly Temp",
"[32] TX Firefly Temp",
nullptr
};

EQUIPMENT equipment[] = {

   {"Clock Reset",              /* equipment name */
    {10, 0,                     /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,               /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read always and update ODB */
     10000,                     /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", ""} ,
    read_cr_event,              /* readout routine */
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
   HNDLE hKey;

   // create Settings structure in ODB
   db_create_record(hDB, 0, "Equipment/Clock Reset/Settings", strcomb(cr_settings_str));
   db_find_key(hDB, 0, "/Equipment/Clock Reset", &hKey);
   assert(hKey);

   db_watch(hDB, hKey, cr_settings_changed, nullptr);

   // add custom page to ODB
   db_create_key(hDB, 0, "Custom/Clock and Reset&", TID_STRING);
   const char * name = "cr.html";
   db_set_value(hDB,0,"Custom/Clock and Reset&",name, sizeof(name), 1,TID_STRING);

   char ip[256];// = "10.32.113.218";
   int size = 256;
   if(!(db_get_value(hDB, hKey, "Settings/IP", ip, &size, TID_STRING, false)== DB_SUCCESS))
       return CM_DB_ERROR;
   int port;// = 50001;
   size =sizeof(port);
   if(!(db_get_value(hDB, hKey, "settings/PORT", &port, &size, TID_INT, false)==DB_SUCCESS))
           return CM_DB_ERROR;

   cout << "IP: " << ip << " port: " << port << endl;

   cb = new clockboard(ip, port);

   if(!cb->isConnected())
        return CM_TIMEOUT;

   cb->init_clockboard();


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

/*--- Read Clock and Reset Event to be put into data stream --------*/

INT read_cr_event(char *pevent, INT off)
{
   bk_init(pevent);

   float *pdata;
   bk_create(pevent, "CRT1", TID_FLOAT, (void **)&pdata);

   *pdata++ = cb->read_mother_board_current();
   *pdata++ = cb->read_mother_board_voltage();
   *pdata++ = cb->read_rx_firefly_temp();
   *pdata++ = cb->read_tx_firefly_temp();

   bk_close(pevent, pdata);

   return bk_size(pevent);
}

/*--- Called whenever settings have changed ------------------------*/

void cr_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *)
{
   KEY key;

   db_get_key(hDB, hKey, &key);

    if (std::string(key.name) == "Active") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "cr_settings_changed", "Set active to %d", value);
      // TODO: propagate to hardware
   }

   auto it = cb->reset_protocol.commands.find(std::string(key.name));

   if(it != cb->reset_protocol.commands.end()){
       // Easy case are commands without payload
       if(!(it->second.has_payload)){
           BOOL value;
           int size = sizeof(value);
           db_get_data(hDB, hKey, &value, &size, TID_BOOL);
           if (value) {
              cm_msg(MINFO, "cr_settings_changed", "Execute %s", key.name);
              cb->write_command(key.name);
              value = FALSE; // reset flag in ODB
              db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
           }
           // here the case with payload
       } else {
           // Run prepare needs the run number
           if (std::string(key.name) == "Run Prepare") {
              BOOL value;
              int size = sizeof(value);
              db_get_data(hDB, hKey, &value, &size, TID_BOOL);
              if (value) {
                 cm_msg(MINFO, "cr_settings_changed", "Execute Run Prepare");
                 int run;
                 int size = sizeof(run);
                 db_get_value(hDB, 0, "/Runinfo/Run number", &run, &size, TID_INT, false);
                 cb->write_command(key.name,run);
                 value = FALSE; // reset flag in ODB
                 db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
              }
           } else {
               // Take the payload from the payload ODB field
               BOOL value;
               int size = sizeof(value);
               db_get_data(hDB, hKey, &value, &size, TID_BOOL);
               if (value) {
                    cm_msg(MINFO, "cr_settings_changed", "Execute %s", key.name);
                    int payload;
                    int size = sizeof(payload);
                    db_get_value(hDB, 0, "/Equipment/Clock Reset/Settings/Payload", &payload, &size, TID_INT, false);
                    cb->write_command(key.name,payload);
                    value = FALSE; // reset flag in ODB
                    db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
               }
           }
       }
   }


}
