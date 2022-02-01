/********************************************************************\

  Name:         febcratefe.c
  Created by:   Niklaus Berger

  Contents:     Front-end for controlling the FEB cartes and reading slow control
                information from them

\********************************************************************/

#include <stdio.h>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include "midas.h"
#include "mfe.h"
#include "history.h"
#include "odbxx.h"
#include "mscb.h"
#include "link_constants.h"
#include "mu3ebanks.h"


using std::cout;
using std::endl;
using std::hex;
using std::string;
using std::to_string;
using std::vector;

using midas::odb;

/* Start address of power in the crate controller - TODO: Move to an appropriate header*/
constexpr uint8_t CC_POWER_OFFSET = 5;
constexpr uint8_t CC_VT_READOUT_START = 1;

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "Febcrate Frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

/* Overwrite equipment struct in ODB from values in code*/
BOOL equipment_common_overwrite = FALSE;

/* a frontend status page is displayed with this frequency in ms    */
INT display_period = 0;

/* maximum event size produced by this frontend */
INT max_event_size = 10000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 10 * 10000;

/* Local state of FEB power*/
std::array<uint8_t, N_FEBCRATES*MAX_FEBS_PER_CRATE> febpower{};


/*-- Function declarations -----------------------------------------*/

INT read_febcrate_sc_event(char *pevent, INT off);

void febpower_changed(odb o);

void setup_odb();
void setup_watches();
void setup_history();
void setup_alarms();

INT init_crates();

/*-- Equipment list ------------------------------------------------*/
EQUIPMENT equipment[] = {
    {"FEBCrates",                    /* equipment name */
    {114, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                 /* equipment type */
     0,                         /* event source crate 0, all stations */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,   /* read during run transitions and update ODB */
     10000,                      /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", "",},
     read_febcrate_sc_event,          /* readout routine */
    },
   {""}
};


/*-- Dummy routines ------------------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
{
   return 1;
}

INT interrupt_configure(INT cmd, INT source, POINTER_T adr)
{
   return 1;
}

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
     #ifdef MY_DEBUG
        odb::set_debug(true);
    #endif

    set_equipment_status(equipment[0].name, "Initializing...", "var(--myellow)");

    for(size_t i =0; i < febpower.size(); i++)
        febpower[i] = 0;

    setup_odb();
    setup_watches();
    setup_history();
    setup_alarms();

    //init feb crates
    INT status = init_crates();
    if (status != SUCCESS)
        return FE_ERR_DRIVER;

    set_equipment_status(equipment[0].name, "OK", "var(--mgreen)");
    return CM_SUCCESS;
}

INT init_crates() {
    odb febpower_odb("/Equipment/FEBCrates/Variables/FEBPower");
    febpower_changed(febpower_odb);
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

/*--- Read Slow Control Event from crate controllers to be put into data stream --------*/
INT read_febcrate_sc_event(char *pevent, INT off){
    bk_init(pevent);
    float *pdata;
    bk_create(pevent, "SCFC", TID_FLOAT, (void **)&pdata);
    odb crates("/Equipment/FEBCrates/Settings");
    for(uint32_t i = 0; i < N_FEBCRATES; i++){
        *pdata++ = i;
        std::string mscb = crates["CrateControllerMSCB"][i];
        char cstr[256]; //not good...
        strcpy(cstr, mscb.c_str());
        if(mscb.empty()){
            for(uint32_t j= 0; j < per_crate_SCFC_size-1; j++)// -1 as index is already written
                *pdata++ = 0;
        } else {
            uint16_t node = crates["CrateControllerNode"][i];
            int fd = mscb_init(cstr, sizeof(cstr), nullptr, 0);
            if (fd < 0) {
               cm_msg(MINFO, "read_febcrate_sc_event", "Cannot connect to node: %d", node);
               for(uint32_t j= 0; j < per_crate_SCFC_size-1; j++)// -1 as index is already written
                   *pdata++ = 0;
               continue;
            }
            float data;
            int size = sizeof(float);
            for(int k=0; k < 4; k++){
                mscb_read(fd, node, CC_VT_READOUT_START+k, &data, &size);
                *pdata++ = data;
            }
            for(uint32_t j= 0; j < MAX_FEBS_PER_CRATE; j++)
                *pdata++ = 0;
        }
    }
    bk_close(pevent,pdata);
    return bk_size(pevent);

    return 0;
}


void febpower_changed(odb o)
{
    cm_msg(MINFO, "febpower_changed()" , "Febpower!");
    std::vector<uint8_t> power_odb = o;
    odb crates("/Equipment/FEBCrates/Settings");
    for(size_t i =0; i < febpower.size(); i++){
        if(febpower[i] != power_odb[i]){
            uint16_t crate = crates["FEBCrate"][i];
            uint16_t slot = crates["FEBSlot"][i];
            std::string mscb = crates["CrateControllerMSCB"][crate];
            char cstr[256]; //not good...
            strcpy(cstr, mscb.c_str());
            uint16_t node = crates["CrateControllerNode"][crate];
            int fd = mscb_init(cstr, sizeof(cstr), nullptr, 0);
            if (fd < 0) {
               cm_msg(MINFO, "read_febcrate_sc_event", "Cannot connect to node: %d", node);
               return;
            }
            uint8_t power = power_odb[i];
            if(power){
                cm_msg(MINFO, "febpower_changed", "Switching on FEB %d in crate %d", slot, crate);
            }
            else {
                cm_msg(MINFO, "febpower_changed", "Switching off FEB %d in crate %d", slot, crate); 
            }

            mscb_write(fd, node, slot+CC_POWER_OFFSET,&power,sizeof(power));
            febpower[i] = power;
        }
    }
}

// ODB Setup //////////////////////////////
void setup_odb(){

    // TODO: This has to go somewhere else
    std::array<uint16_t, MAX_N_FRONTENDBOARDS> arr;
    arr.fill(255);

    odb crate_settings = {
        {"CrateControllerMSCB", std::array<std::string, N_FEBCRATES>{}},
        {"CrateControllerNode", std::array<uint16_t, N_FEBCRATES>{}},
        {"FEBCrate", arr},
        {"FEBSlot", arr},
        {"Names SCFC", std::array<std::string, per_crate_SCFC_size*N_FEBCRATES>()}
    };

    crate_settings.connect("/Equipment/FEBCrates/Settings");
    // Why is the line above needed? Not having it realiably crashes the program
    // when writing the names below
     create_scfc_names_in_odb(crate_settings);

    crate_settings.connect("/Equipment/FEBCrates/Settings");

    odb crate_variables = {
        {"FEBPower", std::array<uint8_t, N_FEBCRATES*MAX_FEBS_PER_CRATE>{}},
        {"SCFC", std::array<float, per_crate_SCFC_size*N_FEBCRATES>()}
    };

    crate_variables.connect("/Equipment/FEBCrates/Variables");


    // add custom page to ODB
    odb custom("/Custom");
    custom["FEBcrates&"] = "crates.html";
}

void setup_history(){

}

void setup_alarms(){

}


void setup_watches(){
    // watch for changes in the FEB powering state
    odb febpower_odb("/Equipment/FEBCrates/Variables/FEBPower");
    febpower_odb.watch(febpower_changed);
}