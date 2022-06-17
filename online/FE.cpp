/********************************************************************\

  Name:         frontend.c
  Created by:   Midas template adapted by Bristol students and A. Loreti

  Contents:     Slow control Bristol
                

\********************************************************************/

#include "AS.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "midas.h"
#include <dlfcn.h>

#include "mfe.h"

#include "/software/mu3e/midas_installations/midas_v1/mu3eSummer22/packages/midas/include/odbxx.h"

#include <iostream>
#include <fstream>
#include <python2.7/Python.h>
#include <fstream>

#include <sstream>
#include <limits>
#include <string>
#include <typeinfo>

using namespace std;


void Com(float setpoint){
string SP = to_string(setpoint);
string command = "s"+SP;
ofstream ard("/dev/ttyACM0"); //opens arduino
if (ard){
	ard << command << '\n';
}
}


/*-- Globals -------------------------------------------------------*/
INT serial_port = setting_arduino();

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "Temperature and humidity";                   //The frontend name

/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

/* a frontend status page is displayed with this frequency in ms */
INT display_period = 1000;

/* maximum event size produced by this frontend */
INT max_event_size = 10000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 100 * 10000;

/*-- Function declarations -----------------------------------------*/

INT frontend_init(void);
INT frontend_exit(void);
INT begin_of_run(INT run_number, char *error);
INT end_of_run(INT run_number, char *error);
INT pause_run(INT run_number, char *error);
INT resume_run(INT run_number, char *error);
INT frontend_loop(void);

INT interrupt_configure(INT cmd, INT source, POINTER_T adr);
INT poll_event(INT source, INT count, BOOL test);
INT read_periodic_event(char *pevent, INT off);


/*-- Equipment list ------------------------------------------------*/
BOOL equipment_common_overwrite = TRUE;
EQUIPMENT equipment[] = {

   {"ArduinoTestStation",              /* equipment name */
      {9, 0,                 /* event ID, trigger mask */		//The event ID that needs to be unique to each of the 'Equipments' used
         "SYSTEM",           /* event buffer */
         EQ_PERIODIC,        /* equipment type */
         0,                  /* event source */
         "MIDAS",            /* format */
         TRUE,               /* enabled */
         RO_RUNNING | RO_TRANSITIONS |   /* read when running and on transitions */
         RO_ODB,             /* and update ODB */
         1000,               /* read every sec */
         0,                  /* stop run after this event limit */
         0,                  /* number of sub events */
         1,               /* log history */				//Boolean to determine whether the history log data will be taken for a particular variable or not. MUST be a 1 or 0.
         "", "", "",},
      read_periodic_event,   /* readout routine */
   },
   {""}

};

/********************************************************************\
              Callback routines for system transitions

  These routines are called whenever a system transition like start/
  stop of a run occurs. The routines are called on the following
  occations:

  frontend_init:  When the frontend program is started. This routine
                  should initialize the hardware.

  frontend_exit:  When the frontend program is shut down. Can be used
                  to releas any locked resources like memory, commu-
                  nications ports etc.

  begin_of_run:   When a new run is started. Clear scalers, open
                  rungates, etc.

  end_of_run:     Called on a request to stop a run. Can send
                  end-of-run event and close run gates.

  pause_run:      When a run is paused. Should disable trigger events.

  resume_run:     When a run is resumed. Should enable trigger events.
\********************************************************************/

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
   /* put any hardware initialization here */
   /* print message and return FE_ERR_HW if frontend should not be started */
   unsigned char data[] = "m";    // sets human readable output
   write_data(serial_port, data, sizeof(data));
   
   return SUCCESS;
}

/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{
   close(serial_port);
   return SUCCESS;
}

/*-- Begin of Run --------------------------------------------------*/

INT begin_of_run(INT run_number, char *error)
{
    ifstream myReadFile;
    string setpoint;

    myReadFile.open("setpoint.txt");
    //myReadFile >> setpoint;
    getline(myReadFile, setpoint);
    //cout << setpoint;
    
    unsigned char data[] = "s15";
    
    //strcpy(data, setpoint.c_str());

    //unsigned char* data_uchar = reinterpret_cast<unsigned char*>(data);
    write_data(serial_port, data, sizeof(data));
    
    /* put here clear scalers etc. */
   return SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT run_number, char *error)
{
   return SUCCESS;
}

/*-- Pause Run -----------------------------------------------------*/

INT pause_run(INT run_number, char *error)
{
    
    //midas::odb exp("/");
    //exp["set_point"] = 10;
    
    //float setpoint=exp["set_point"];
    //cout << "The setpoint is: " << setpoint << endl;
    
    //char hostname[20];
    //char str[128];
    //int size,status;
    //size = sizeof(hostname); 
    //sprintf(str,"/setpoint",equipment[FIFO].name); 
    //status = db_get_value(hDB, 0, str, hostname, &size, TID_STRING, FALSE);


  //  ifstream myReadFile;
   // string setpoint;

    //myReadFile.open("setpoint.txt");
    //myReadFile >> setpoint;
   //getline(myReadFile, setpoint);
    //cout << setpoint;

    //char* data = new char[setpoint.size() + 1];
    //strcpy(data, setpoint.c_str());

   // unsigned char data_1[10];
  //  strcpy((char*)data_1, setpoint.c_str());

    //unsigned char* data_uchar = reinterpret_cast<unsigned char*>(data);
  //  write_data(serial_port, data_1, sizeof(data_1));
    //unsigned char data2[] = "s15";
    //write_data(serial_port, data2, sizeof(data2));
   return SUCCESS;
}

/*-- Resume Run ----------------------------------------------------*/

INT resume_run(INT run_number, char *error)
{
   return SUCCESS;
}

/*-- Frontend Loop -------------------------------------------------*/

INT frontend_loop()
{
   /* if frontend_call_loop is true, this routine gets called when
      the frontend is idle or once between every event */
   return SUCCESS;
}

/*------------------------------------------------------------------*/

/********************************************************************\

  Readout routines for different events

\********************************************************************/

/*-- Event readout -------------------------------------------------*/
//THE FRONTEND CODE WHICH CONTROLS AND TAKES DATA FROM THE SENSORS

/*-- Trigger event routines ----------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
/* Polling routine for events. Returns TRUE if event
   is available. If test equals TRUE, don't return. The test
   flag is used to time the polling */
{
   int i;
   DWORD flag;

   for (i = 0; i < count; i++) {
      /* poll hardware and set flag to TRUE if new event is available */
      flag = TRUE;

      if (flag)
         if (!test)
            return TRUE;
   }

   return 0;
}

/*-- Interrupt configuration ---------------------------------------*/

INT interrupt_configure(INT cmd, INT source, POINTER_T adr)
{
   switch (cmd) {
   case CMD_INTERRUPT_ENABLE:
      break;
   case CMD_INTERRUPT_DISABLE:
      break;
   case CMD_INTERRUPT_ATTACH:
      break;
   case CMD_INTERRUPT_DETACH:
      break;
   }
   return SUCCESS;
}


/*-- Periodic event ------------------------------------------------*/
INT read_periodic_event(char *pevent, INT off)
{
   float *pdata;  //Pointer to the data

   /* init bank structure */
   bk_init(pevent);
   //bk_init32a(pevent);
 
	//Reads Arduino
   vector<double> v;
   
    while(v.size()!=7){
      
      read_data1(serial_port, v);
      
      //Fills midas data banks
      if(v.size()==7){
               
         //cout << v[0]<< '\t' <<v[1]<< '\t' <<v[2]<< '\t' << v[4] << '\t' << v[5] << '\t' << v[6] << '\t' << endl;
         
         midas::odb exp("/Equipment/ArduinoTestStation/Variables");
         Com(exp["_S_"]);
         

         /* create SCLR bank */
         bk_create(pevent, "_T_", TID_FLOAT, (void **)&pdata);
         *pdata++=(float)v[0];
         bk_close(pevent, pdata);

         /* create SCLR bank */
         bk_create(pevent, "_F_", TID_FLOAT, (void **)&pdata);
         *pdata++=(float)v[1];
         bk_close(pevent, pdata);

         /* create SCLR bank */
         bk_create(pevent, "_P_", TID_FLOAT, (void **)&pdata);
         *pdata++=(float)v[2];
         bk_close(pevent, pdata);
         
         /* create SCLR bank */
         bk_create(pevent, "_A_", TID_FLOAT, (void **)&pdata);
         *pdata++=(float)v[3];
         bk_close(pevent, pdata);  

         /* create SCLR bank */
         //bk_create(pevent, "_S_", TID_FLOAT, (void**)&pdata);
         //*pdata++ = (float)v[4];
         //bk_close(pevent, pdata);

         /* create SCLR bank */
         bk_create(pevent, "_RH_", TID_FLOAT, (void**)&pdata);
         *pdata++ = (float)v[5];
         bk_close(pevent, pdata);

	       /* create SCLR bank */
         bk_create(pevent, "_AT_", TID_FLOAT, (void**)&pdata);
         *pdata++ = (float)v[6];
         bk_close(pevent, pdata);


         

    }
  }
      
   v.clear();
   

   return bk_size(pevent);
    
}
