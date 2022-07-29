/********************************************************************\

  Name:         teststand-frontend.cpp
  Created by:   Midas template adapted by Bristol students and A. Loreti
  Contents:     Slow control Bristol

\********************************************************************/

#include <dlfcn.h>
#include <math.h>
#include <python2.7/Python.h>
#include <stdio.h>
#include <stdlib.h>

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <typeinfo>

#include "arduino-station.h"
#include "mfe.h"
#include "midas.h"
#include "odbxx.h"


//-- Globals -------------------------------------------------------
// serial_port: open and configure arduino
// *frontend_name: pointer variable for name of client as seen by other MIDAS
// clients *frontend_file_name: the frontend file name, don't change it
// frontend_call_loop: frontend_loop is called periodically if this variable is
// TRUE display_period: a frontend status page is displayed with this frequency
// in ms max_event_size: maximum event size produced by this frontend
// max_event_size_frag: maximum event size for fragmented events (EQ_FRAGMENTED)
// event_buffer_size: buffer size to hold events
INT serial_port = setting_arduino();
const char *frontend_name = "Temperature and humidity";
const char *frontend_file_name = __FILE__;
BOOL frontend_call_loop = FALSE;
INT display_period = 1000;
INT max_event_size = 10000;
INT max_event_size_frag = 5 * 1024 * 1024;
INT event_buffer_size = 100 * 10000;

//-- Function declarations -----------------------------------------
void send_command_ard(float value, std::string command);
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

//-- Equipment list ------------------------------------------------
BOOL equipment_common_overwrite = TRUE;
EQUIPMENT equipment[] = {
    {
        "ArduinoTestStation",  // equipment name
        {
            9,            // unique event ID
            0,            // trigger mask
            "SYSTEM",     // event buffer
            EQ_PERIODIC,  // equipment type
            0,            // event source
            "MIDAS",      // format
            TRUE,         // enabled
            RO_RUNNING | RO_TRANSITIONS |
                RO_ODB,  // read when running, transitions, update ODB
            1000,        // read every sec
            0,           // stop run after this event limit
            0,           // number of sub events
            1,           // log history (bool)
            "",          // device driver list
            "",
            "",
        },
        read_periodic_event,  // readout routine function name
    },
    {""}
};

//-- General functions ------------------------------------------------
// send_command_ard()
// - access the device file for arduino and send a command through tty
// - e.g. s15, v12, c2.5
void send_command_ard(float value, std::string command) {
    command = command + std::__cxx11::to_string(value);
    std::ofstream ard("/dev/ttyACM0");
    if (ard) ard << command << '\n';
    return;
}

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

//-- Frontend Init -------------------------------------------------
// - Toggles human readable output
// - Put any hardware initialization here
// - Print message and return FE_ERR_HW if frontend should not be started
INT frontend_init() {
    unsigned char data[] = "r";
    write_data(serial_port, data, sizeof(data));
    return SUCCESS;
}

//-- Frontend Exit -------------------------------------------------
INT frontend_exit() {
    close(serial_port);
    return SUCCESS;
}

//-- Begin of Run --------------------------------------------------
// - Put in here anything to run at beginning of the run
// - ODB variables are already set with odbedit in a shell script, avoid doing
// it twice
// - it is also easier to set odb variables with shell script (no need to recompile)
INT begin_of_run(INT run_number, char *error) { return SUCCESS; }

//-- End of Run ----------------------------------------------------
INT end_of_run(INT run_number, char *error) { return SUCCESS; }

//-- Pause Run -----------------------------------------------------
INT pause_run(INT run_number, char *error) { return SUCCESS; }

//-- Resume Run ----------------------------------------------------
INT resume_run(INT run_number, char *error) { return SUCCESS; }

//-- Frontend Loop -------------------------------------------------
// - if frontend_call_loop is true, this routine gets called when
//   the frontend is idle or once between every event
INT frontend_loop() { return SUCCESS; }

//------------------------------------------------------------------

/********************************************************************\

  Readout routines for different events

\********************************************************************/

//-- Event readout -------------------------------------------------
// THE FRONTEND CODE WHICH CONTROLS AND TAKES DATA FROM THE SENSORS

//-- Trigger event routines ----------------------------------------
// Polling routine for events. Returns TRUE if event
// is available. If test equals TRUE, don't return. The test
// flag is used to time the polling
INT poll_event(INT source, INT count, BOOL test) {
    DWORD flag;
    for (int i = 0; i < count; i++) {
        flag = TRUE;
        if (flag)
            if (!test) return TRUE;
    }
    return 0;
}

//-- Interrupt configuration ---------------------------------------
INT interrupt_configure(INT cmd, INT source, POINTER_T adr) {
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

//-- Periodic event ------------------------------------------------
// - Init bank structure (commented out bk_init32a(pevent);)
// - Reads Arduino into vector
// - Fills midas data banks
// - Create SCLR bank for each variable
// - Read odb variables (S)etpoint, (V)oltage, (C)urrent
// every loop and send to arduino to adjust (is this expensive if the value
// hasn't been changed?)
INT read_periodic_event(char *pevent, INT off) {
    float *pdata;
    bk_init(pevent);
    std::vector<double> data_stream;
    const int data_stream_size = 7;
    while (data_stream.size() != data_stream_size) {
        read_data1(serial_port, data_stream);
        if (data_stream.size() == data_stream_size) {
            midas::odb exp("/Equipment/ArduinoTestStation/Variables");

            send_command_ard(exp["_S_"], "s");
            send_command_ard(exp["_V_"], "v");
            send_command_ard(exp["_C_"], "c");

            bk_create(pevent, "_T_", TID_FLOAT, (void **)&pdata);
            *pdata++ = (float)data_stream[0];
            bk_close(pevent, pdata);

            bk_create(pevent, "_F_", TID_FLOAT, (void **)&pdata);
            *pdata++ = (float)data_stream[1];
            bk_close(pevent, pdata);

            bk_create(pevent, "_P_", TID_FLOAT, (void **)&pdata);
            *pdata++ = (float)data_stream[2];
            bk_close(pevent, pdata);

            bk_create(pevent, "_A_", TID_FLOAT, (void **)&pdata);
            *pdata++ = (float)data_stream[3];
            bk_close(pevent, pdata);

            bk_create(pevent, "_RH_", TID_FLOAT, (void **)&pdata);
            *pdata++ = (float)data_stream[5];
            bk_close(pevent, pdata);

            bk_create(pevent, "_AT_", TID_FLOAT, (void **)&pdata);
            *pdata++ = (float)data_stream[6];
            bk_close(pevent, pdata);
        }
    }
    data_stream.clear();
    return bk_size(pevent);
}
