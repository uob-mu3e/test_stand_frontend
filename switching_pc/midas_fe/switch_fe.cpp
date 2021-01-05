#define FEB_ENABLE_REGISTER_LOW_W FEB_ENABLE_REGISTER_W
#define RUN_NR_ACK_REGISTER_LOW_R RUN_NR_ACK_REGISTER_R
/********************************************************************\

  Name:         taken from switch_fe.cpp
  Created by:   Stefan Ritt
  Updated by:   Marius Koeppel, Konrad Briggl, Lukas Gerritzen

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

		Scifi-Related actions:
		/Equipment/Switching SC/Settings/SciFiConfig: triggers a configuration of all ASICs 
		Mupix-Related actions:
		/Equipment/Switching SC/Settings/MupixConfig: triggers a configuration of all ASICs 
		/Equipment/Switching SC/Settings/MupixBoard: triggers a configuration of all Mupix motherboards 

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
#include <switching_constants.h>
#include <history.h>
#include "midas.h"
#include "odbxx.h"
#include "mfe.h"
#include "string.h"
#include "mudaq_device_scifi.h"
#include "mudaq_dummy.h"

//Slow control for mutrig/scifi ; mupix
#include "mutrig_midasodb.h"
#include "mupix_midasodb.h"
#include "link_constants.h"
#include "SciFi_FEB.h"
#include "Tiles_FEB.h"
#include "mupix_FEB.h"

#include "missing_hardware.h"

using namespace std;
using midas::odb;

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "SW Frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

/* Overwrite equipment struct in ODB from values in code*/
BOOL equipment_common_overwrite = FALSE;

/* a frontend status page is displayed with this frequency in ms    */
//INT display_period = 1000;
INT display_period = 0;

/* maximum event size produced by this frontend */
INT max_event_size = 10000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 10 * 10000;

int switch_id = 0; // TODO to be loaded from outside

INT status;

/* DMA Buffer and related */ 
#ifdef NO_SWITCHING_BOARD
    mudaq::MudaqDevice * mup;
#else
    mudaq::MudaqDevice * mup;
#endif


/*-- Function declarations -----------------------------------------*/

INT read_sc_event(char *pevent, INT off);
INT read_WMEM_event(char *pevent, INT off);
INT read_scifi_sc_event(char *pevent, INT off);
INT read_scitiles_sc_event(char *pevent, INT off);
INT read_mupix_sc_event(char *pevent, INT off);
void sc_settings_changed(HNDLE, HNDLE, int, void *);
void switching_board_mask_changed(odb o);
void frontend_board_mask_changed(odb o);

uint64_t get_link_active_from_odb(odb o); //throws
void set_feb_enable(uint64_t enablebits);
uint64_t get_runstart_ack();
uint64_t get_runend_ack();
void print_ack_state();

void setup_odb();
void setup_watches();

INT init_mudaq(auto & mu);
INT init_scifi(auto & mu);
INT init_scitiles(auto & mu);
INT init_mupix(auto & mu);


/*-- Equipment list ------------------------------------------------*/
enum EQUIPMENT_ID {Switching=0,SciFi,SciTiles,Mupix};
EQUIPMENT equipment[] = {

   {"Switching",                /* equipment name */
    {110, 0,                    /* event ID, trigger mask */
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
   {"SciFi",                    /* equipment name */
    {111, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                 /* equipment type */
     0,                         /* event source crate 0, all stations */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read always and update ODB */
     10000,                      /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", "",},
     read_scifi_sc_event,          /* readout routine */
    },
   {"SciTiles",                    /* equipment name */
    {112, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                 /* equipment type */
     0,                         /* event source crate 0, all stations */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,        /* read always and update ODB */
     10000,                      /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", "",},
     read_scitiles_sc_event,          /* readout routine */
    },
    {"Mupix",                    /* equipment name */
    {113, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                 /* equipment type */
     0,                         /* event source crate 0, all stations */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,   /* read during run transitions and update ODB */
     1000,                      /* read every 1 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", "",},
     read_mupix_sc_event,          /* readout routine */
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

    HNDLE hKeySC;

    // create Settings structure in ODB
    setup_odb();
    setup_watches();
    
    // open mudaq
    #ifdef NO_SWITCHING_BOARD
        mup = new dummy_mudaq::DummyDmaMudaqDevice("/dev/mudaq0");
    #else
        mup = new mudaq::DmaMudaqDevice("/dev/mudaq0");
    #endif       
        
    status = init_mudaq(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;

    //set link enables so slow control can pass
    odb cur_links_odb("/Equipment/Links/Settings/LinkMask");
    try{ set_feb_enable(get_link_active_from_odb(cur_links_odb)); }
    catch(...){ return FE_ERR_ODB;}
    
    //init scifi
    status = init_scifi(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;
    
    //init scitiles
    status = init_scitiles(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;

    //init mupix
    status = init_mupix(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;

    // TODO: Define history panels
    // --- SciFi panels created in mutrig::midasODB::setup_db, below
    
    // Set our transition sequence. The default is 500. Setting it
    // to 400 means we are called BEFORE most other clients.
    cm_set_transition_sequence(TR_START, 400);

    // Set our transition sequence. The default is 500. Setting it
    // to 600 means we are called AFTER most other clients.
    cm_set_transition_sequence(TR_STOP, 600);

    return CM_SUCCESS;
}

// ODB Setup //////////////////////////////
void setup_odb(){

    /* Default values for /Equipment/Switching/Settings */
    odb sc_settings = {
            {"Active", true},
            {"Delay", false},
            {"Write", false},
            {"Single Write", false},
            {"Read", false},
            {"Read WM", false},
            {"Read RM", false},
            {"Reset SC Main", false},
            {"Reset SC Secondary", false},
            {"Clear WM", false},
            {"Last RM ADD", false},
            {"MupixConfig", false},
            {"MupixBoard", false},
            {"SciFiConfig", false},
            {"SciTilesConfig", false},
            {"Reset Bypass Payload", 0},
            {"Reset Bypass Command", 0},
    };

    sc_settings.connect("/Equipment/Switching/Settings", true);

    /* Default values for /Equipment/Switching/Variables */
    odb sc_variables = {
            {"FPGA_ID_READ", 0},
            {"START_ADD_READ", 0},
            {"LENGTH_READ", 0},

            {"FPGA_ID_WRITE", 0},
            {"DATA_WRITE", 0},
            {"DATA_WRITE_SIZE", 0},

            {"START_ADD_WRITE", 0},
            {"SINGLE_DATA_WRITE", 0},

            {"RM_START_ADD", 0},
            {"RM_LENGTH", 0},
            {"RM_DATA", 0},

            {"WM_START_ADD", 0},
            {"WM_LENGTH", 0},
            {"WM_DATA", 0},
    };

    sc_variables.connect("/Equipment/Switching/Variables");

    // add custom page to ODB
    odb custom("/Custom");
    custom["Switching&"] = "sc.html";
    
    // setup odb for switching board
    odb swb_varibles("/Equipment/Switching/Variables");
    swb_varibles["Merger Timeout All FEBs"] = 0;

    // TODO: not sure at the moment we have a midas frontend for three feb types but 
    // we need to have different swb at the final experiment so maybe one needs to take
    // things apart later. For now we put this "common" FEB variables into the generic
    // switching path
    hs_define_panel("Switching", "All FEBs", {"Switching:Merger Timeout All FEBs"});

}

void setup_watches(){

    // watch if this switching board is enabled
    odb switch_mask("/Equipment/Links/Settings/SwitchingBoardMask");
    switch_mask.watch(frontend_board_mask_changed);

    // watch if this links are enabled
    odb links_odb("/Equipment/Links/Settings/LinkMask");
    links_odb.watch(switching_board_mask_changed);

}

void switching_board_mask_changed(odb o) {

    string name = o.get_name();

    cm_msg(MINFO, "switching_board_mask_changed", "Switching board masking changed");
    cm_msg(MINFO, "switching_board_mask_changed", "With name %s and odb %s", name, o);

    INT switching_board_mask[MAX_N_SWITCHINGBOARDS];
    int size = sizeof(INT)*MAX_N_FRONTENDBOARDS;

    //db_get_data(hDB, hKey, &switching_board_mask, &size, TID_INT);

    BOOL value = switching_board_mask[switch_id] > 0 ? true : false;

    for(int i = 0; i < 2; i++) {
        char str[128];
        sprintf(str,"Equipment/%s/Common/Enabled", equipment[i].name);
        db_set_value(hDB,0,str, &value, sizeof(value), 1, TID_BOOL);

        cm_msg(MINFO, "switching_board_mask_changed", "Set Equipment %s enabled to %d", equipment[i].name, value);
    }

    SciFiFEB::Instance()->RebuildFEBsMap();
    MupixFEB::Instance()->RebuildFEBsMap();

}

void frontend_board_mask_changed(odb o) {
    try{
        set_feb_enable(get_link_active_from_odb(o));
        SciFiFEB::Instance()->RebuildFEBsMap();
        MupixFEB::Instance()->RebuildFEBsMap();
    }catch(...){}
}

INT init_mudaq(auto & mu) {

    
    if ( !mu.open() ) {
        cm_msg(MERROR, "frontend_init" , "Could not open device");
        return FE_ERR_DRIVER;
    }

    if ( !mu.is_ok() ) {
        cm_msg(MERROR, "frontend_init", "Mudaq is not ok");
        return FE_ERR_DRIVER;
    }

    mu.FEBsc_resetMain();
    mu.FEBsc_resetSecondary();

    return SUCCESS;
}

INT init_scifi(auto & mu) {

    // SciFi setup part
    set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Initializing...", "var(--myellow)");
    SciFiFEB::Create(mu, equipment[EQUIPMENT_ID::SciFi].name, "/Equipment/SciFi"); //create FEB interface signleton for scifi
    SciFiFEB::Instance()->SetSBnumber(switch_id);
    status=mutrig::midasODB::setup_db("/Equipment/SciFi",SciFiFEB::Instance());
    if(status != SUCCESS){
        set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Start up failed", "var(--mred)");
        return status;
    }
    //init all values on FEB
    SciFiFEB::Instance()->WriteAll();
    SciFiFEB::Instance()->WriteFEBID();

    set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Ok", "var(--mgreen)");
    
    //set custom page
    odb custom("/Custom");
    custom["SciFi-ASICs&"] = "mutrigTdc.html";

    return SUCCESS;
}

INT init_scitiles(auto & mu) {
    
    //SciTiles setup part
    set_equipment_status(equipment[EQUIPMENT_ID::SciTiles].name, "Initializing...", "var(--myellow)");
    TilesFEB::Create(mu, equipment[EQUIPMENT_ID::SciTiles].name, "/Equipment/SciTiles"); //create FEB interface signleton for scitiles
    status=mutrig::midasODB::setup_db("/Equipment/SciTiles", TilesFEB::Instance());
    if(status != SUCCESS){
        set_equipment_status(equipment[EQUIPMENT_ID::SciTiles].name, "Start up failed", "var(--mred)");
        return status;
    }
    //init all values on FEB
    TilesFEB::Instance()->WriteAll();
    TilesFEB::Instance()->WriteFEBID();

    set_equipment_status(equipment[EQUIPMENT_ID::SciTiles].name, "Ok", "var(--mgreen)");

    //set custom page
    odb custom("/Custom");
    custom["SciTiles-ASICs&"] = "tile_custompage.html";
    
    return SUCCESS;
}

INT init_mupix(auto & mu) {

    //Mupix setup part
    set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Initializing...", "var(--myellow)");
    MupixFEB::Create(mu, equipment[EQUIPMENT_ID::Mupix].name, "/Equipment/Mupix"); //create FEB interface signleton for mupix
    MupixFEB::Instance()->SetSBnumber(switch_id);
    status=mupix::midasODB::setup_db("/Equipment/Mupix", MupixFEB::Instance(), true);
    if(status != SUCCESS){
        set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Start up failed", "var(--mred)");
        return status;
    }
    //init all values on FEB
    MupixFEB::Instance()->WriteFEBID();
    
    set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Ok", "var(--mgreen)");
   
    // setup odb rate counters for each feb
    char set_str[255];
    odb rate_counters("/Equipment/Mupix/Variables");
    for(int i = 0; i < MupixFEB::Instance()->getNFPGAs(); i++){
        sprintf(set_str, "merger rate FEB%d", i);
        rate_counters[set_str] = 0;
        sprintf(set_str, "hit ena rate FEB%d", i);
        rate_counters[set_str] = 0;
        sprintf(set_str, "reset phase FEB%d", i);
        rate_counters[set_str] = 0;
        sprintf(set_str, "TX reset%d", i);
        rate_counters[set_str] = 0;
    }
    //end of Mupix setup part
    
    // Define history panels for each FEB Mupix
    for(int i = 0; i < MupixFEB::Instance()->getNFPGAs(); i++){
        sprintf(set_str, "FEB%d", i);
        hs_define_panel("Mupix", set_str, {"Mupix:merger rate " + string(set_str),
                                           "Mupix:hit ena rate " + string(set_str),
                                           "Mupix:reset phase " + string(set_str),
              //                             "Mupix:TX reset " + string(set_str),
                                           });
    }
    
    
    //TODO: set custom page
    //odb custom("/Custom");
    //custom["Mupix&"] = "mupix_custompage.html";

    return SUCCESS;
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
   int status;
try{
   set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Starting Run", "var(--morange)");
   set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Starting Run", "var(--morange)");

   /* Set new run number */
   mup->write_register(RUN_NR_REGISTER_W, run_number);
   /* Reset acknowledge/end of run seen registers before start of run */
   uint32_t start_setup = 0;
   start_setup = SET_RESET_BIT_RUN_START_ACK(start_setup);
   start_setup = SET_RESET_BIT_RUN_END_ACK(start_setup);
   mup->write_register_wait(RESET_REGISTER_W, start_setup, 1000);
   mup->write_register(RESET_REGISTER_W, 0x0);

   /* get link active from odb. */
    odb cur_links_odb("/Equipment/Links/Settings/LinkMask");
    uint64_t link_active_from_odb = get_link_active_from_odb(cur_links_odb);

   //configure ASICs for SciFi
   status=SciFiFEB::Instance()->ConfigureASICs();
   if(status!=SUCCESS){
      cm_msg(MERROR,"switch_fe","ASIC configuration failed");
      return CM_TRANSITION_CANCELED;
   }

   //configure ASICs for Tiles
   status=TilesFEB::Instance()->ConfigureASICs();
   if(status!=SUCCESS){
      cm_msg(MERROR,"switch_fe","ASIC configuration failed");
      return CM_TRANSITION_CANCELED;
   }

   //configure Pixel sensors
   //status=MupixFEB::Instance()->ConfigureASICs();
   //if(status!=SUCCESS){
   //   cm_msg(MERROR,"switch_fe","ASIC configuration failed");
   //   return CM_TRANSITION_CANCELED;
   //}


   //last preparations
   SciFiFEB::Instance()->ResetAllCounters();

   HNDLE hKey;
   char ip[256];
   int size = 256;
   if(db_find_key(hDB, 0, "/Equipment/Clock Reset", &hKey) != DB_SUCCESS){
       cm_msg(MERROR,"switch_fe","could not find CRFE, is CRFE running ?", run_number);
       return CM_TRANSITION_CANCELED;
   }else if(db_get_value(hDB, hKey, "Settings/IP", ip, &size, TID_STRING, false)!= DB_SUCCESS) {
       cm_msg(MERROR,"switch_fe","could not find CRFE IP, is CRFE running ?", run_number);
       return CM_TRANSITION_CANCELED;
   }

   if(string(ip)=="0.0.0.0"){
       /* send run prepare signal from here */
       cm_msg(MINFO,"switch_fe","Bypassing CRFE for run transition");
       DWORD valueRB = run_number;
       mup->FEBsc_write(mup->FEBsc_broadcast_ID, &valueRB,1,0xfff5,true); //run number
       valueRB= (1<<8) | 0x10;
       mup->FEBsc_write(mup->FEBsc_broadcast_ID, &valueRB,1,0xfff4,true); //run prep command
       valueRB= 0xbcbcbcbc;
       mup->FEBsc_write(mup->FEBsc_broadcast_ID, &valueRB,1,0xfff5,true); //reset payload
       valueRB= 0;//(1<<8) | 0x00;
       mup->FEBsc_write(mup->FEBsc_broadcast_ID, &valueRB,1,0xfff4,true); //reset command
   }else{
       /* send run prepare signal via CR system */
       INT value = 1;
       cm_msg(MINFO,"switch_fe","Using CRFE for run transition");
       db_set_value_index(hDB,0,"Equipment/Clock Reset/Run Transitions/Request Run Prepare",
                          &value, sizeof(value), switch_id, TID_INT, false);
   }

   uint16_t timeout_cnt=300;
   uint64_t link_active_from_register;
   printf("Waiting for run prepare acknowledge from all FEBs\n");
   do{
      timeout_cnt--;
      link_active_from_register = get_runstart_ack();
      printf("%u  %lx  %lx\n",timeout_cnt,link_active_from_odb, link_active_from_register);
      usleep(10000);
   }while( (link_active_from_register & link_active_from_odb) != link_active_from_odb && (timeout_cnt > 0));

   if(timeout_cnt==0) {
      cm_msg(MERROR,"switch_fe","Run number mismatch on run %d", run_number);
      print_ack_state();
      return CM_TRANSITION_CANCELED;
   }

   set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Scintillating...", "lightBlue");
   set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Running...", "lightGreen");
   return CM_SUCCESS;
}catch(...){return CM_TRANSITION_CANCELED;}
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT run_number, char *error)
{
try{
   /* get link active from odb */
    odb cur_links_odb("/Equipment/Links/Settings/LinkMask");
    uint64_t link_active_from_odb = get_link_active_from_odb(cur_links_odb);

   printf("end_of_run: Waiting for stop signals from all FEBs\n");
   uint16_t timeout_cnt = 50;
   uint64_t stop_signal_seen = get_runend_ack();
   printf("Stop signal seen from 0x%16lx, expect stop signals from 0x%16lx\n", stop_signal_seen, link_active_from_odb);
   while( (stop_signal_seen & link_active_from_odb) != link_active_from_odb &&
         timeout_cnt > 0) {
      usleep(1000);
      stop_signal_seen = get_runend_ack();
      printf("%u:  Stop signal seen from %16lx, expect stop signals from %16lx\n", timeout_cnt,stop_signal_seen, link_active_from_odb);
      timeout_cnt--;
   };
      printf("%u:  Stop signal seen from %16lx, expect stop signals from %16lx\n", timeout_cnt,stop_signal_seen, link_active_from_odb);

   if(timeout_cnt==0) {
      cm_msg(MERROR,"switch_fe","End of run marker only found for frontends %16lx", stop_signal_seen);
      cm_msg(MINFO,"switch_fe","... Expected to see from frontends %16lx", link_active_from_odb);
      print_ack_state();
      set_equipment_status(equipment[EQUIPMENT_ID::Switching].name, "Not OK", "var(--mred)");
      set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Transition interrupted", "var(--morange)");
      set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Transition interrupted", "var(--morange)");
      return CM_TRANSITION_CANCELED;
   }

//   printf("Waiting for buffers to empty\n");
//   timeout_cnt = 0;
//   while(! mup->read_register_ro(0/* TODO Buffer Empty */) &&
//         timeout_cnt++ < 50) {
//      timeout_cnt++;
//      usleep(1000);
//   };
//
//   if(timeout_cnt>=50) {
//      cm_msg(MERROR,"switch_fe","Buffers on Switching Board %d not empty at end of run", switch_id);
//      set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Not OK", "var(--mred)");
//      return CM_TRANSITION_CANCELED;
//   }
//   printf("Buffers all empty\n");

   printf("EOR successful\n");


   set_equipment_status(equipment[EQUIPMENT_ID::Switching].name, "Ok", "var(--mgreen)");
   set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Ok", "var(--mgreen)");
   set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Ok", "var(--mgreen)");
   return CM_SUCCESS;
}catch(...){return CM_TRANSITION_CANCELED;}
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
    // get mudaq
    #ifdef MY_DEBUG
        dummy_mudaq::DummyMudaqDevice & mu = *mup;
    #else
        mudaq::MudaqDevice & mu = *mup;
    #endif
        
    // get odb
    // TODO: at the moment the timeout is a counter for all FEBs
    odb merger_timeout_cnt("/Equipment/Switching/Variables");
    auto merger_timeout_all = mu.read_register_ro(0x26);
    merger_timeout_cnt["Merger Timeout All FEBs"] = merger_timeout_all;
    
    // create bank, pdata
    bk_init(pevent);
    DWORD *pdata;
    bk_create(pevent, "SWB0", TID_DWORD, (void **)&pdata);
    
    *pdata++ = merger_timeout_all;
    
    bk_close(pevent,pdata);
    return bk_size(pevent);

    // TODO why do we do this?
    while(mup->FEBsc_get_packet()){};
    //TODO: make this a switch
    mup->FEBsc_dump_packets();
    //return 0;
    //return mup->FEBsc_write_bank(pevent,off);
}

/*--- Read Slow Control Event from SciFi to be put into data stream --------*/

INT read_scifi_sc_event(char *pevent, INT off){
	static int i=0;
    printf("Reading Scifi FEB status data from all FEBs %d\n",i++);
    //TODO: Make this more proper: move this to class driver routine and make functions not writing to ODB all the time (only on update).
    //Add readout function for this one that gets data from class variables and writes midas banks
    SciFiFEB::Instance()->ReadBackAllCounters();
    SciFiFEB::Instance()->ReadBackAllRunState();
    SciFiFEB::Instance()->ReadBackAllDatapathStatus();
    return 0;
}

/*--- Read Slow Control Event from SciTiles to be put into data stream --------*/

INT read_scitiles_sc_event(char *pevent, INT off){
	static int i=0;
    printf("Reading SciTiles FEB status data from all FEBs %d\n",i++);
    //TODO: Make this more proper: move this to class driver routine and make functions not writing to ODB all the time (only on update).
    //Add readout function for this one that gets data from class variables and writes midas banks
    TilesFEB::Instance()->ReadBackAllCounters();
    TilesFEB::Instance()->ReadBackAllRunState();
    TilesFEB::Instance()->ReadBackAllDatapathStatus();
    return 0;
}

/*--- Read Slow Control Event from Mupix to be put into data stream --------*/

INT read_mupix_sc_event(char *pevent, INT off){
    // get odb
    odb rate_cnt("/Equipment/Mupix/Variables");
    uint32_t HitsEnaRate;
    uint32_t MergerRate;
    uint32_t ResetPhase;
    uint32_t TXReset;
    char set_str[255];
    static int i = 0;
 
    bk_init(pevent);
    DWORD *pdata;
    bk_create(pevent, "FECN", TID_WORD, (void **) &pdata);
    printf("Reading MuPix FEB status data from all FEBs %d\n", i++);
    MupixFEB::Instance()->ReadBackAllRunState();
    for(int i = 0; i < MupixFEB::Instance()->getNFPGAs(); i++){
        HitsEnaRate = MupixFEB::Instance()->ReadBackHitsEnaRate(i);
        MergerRate = MupixFEB::Instance()->ReadBackMergerRate(i);
        ResetPhase = MupixFEB::Instance()->ReadBackResetPhase(i);
        TXReset = MupixFEB::Instance()->ReadBackTXReset(i);


        sprintf(set_str, "hit ena rate FEB%d", i);
        // TODO: change hex value
        rate_cnt[set_str] = 0x7735940 - HitsEnaRate;
        
        sprintf(set_str, "merger rate FEB%d", i);
        rate_cnt[set_str] = MergerRate;

        sprintf(set_str, "reset phase FEB%d", i);
        rate_cnt[set_str] = ResetPhase;

        sprintf(set_str, "TX reset FEB%d", i);
        rate_cnt[set_str] = TXReset;

        *pdata++ = HitsEnaRate;
        *pdata++ = MergerRate;
        *pdata++ = ResetPhase; 
        *pdata++ = TXReset;
    }
    
    bk_close(pevent,pdata);
    return bk_size(pevent);
}

/*--- helper functions ------------------------*/

BOOL sc_settings_changed_hepler(const char *key_name, HNDLE hDB, HNDLE hKey, DWORD type){
    BOOL value;
    int size = sizeof(value);
    db_get_data(hDB, hKey, &value, &size, type);
    //if(value) cm_msg(MINFO, "sc_settings_changed", "trigger for key=\"%s\"", key_name);
    return value;
}

void set_odb_flag_false(const char *key_name, HNDLE hDB, HNDLE hKey, DWORD type){
    //cm_msg(MINFO, "sc_settings_changed", "reseting odb flag of key \"\"", key_name);
    BOOL value = FALSE; // reset flag in ODB
    db_set_data(hDB, hKey, &value, sizeof(value), 1, type);
}

INT get_odb_value_by_string(const char *key_name){
    INT ODB_DATA, SIZE_ODB_DATA;
    SIZE_ODB_DATA = sizeof(ODB_DATA);
    db_get_value(hDB, 0, key_name, &ODB_DATA, &SIZE_ODB_DATA, TID_INT, 0);
    return ODB_DATA;
}

/*--- Called whenever settings have changed ------------------------*/

void sc_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *)
{
   KEY key;

   db_get_key(hDB, hKey, &key);

    #ifdef MY_DEBUG
        dummy_mudaq::DummyMudaqDevice & mu = *mup;
    #else
        mudaq::MudaqDevice & mu = *mup;
    #endif


   if (string(key.name) == "Active") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      cm_msg(MINFO, "sc_settings_changed", "Set active to %d", value);
      // TODO: propagate to hardware
   }

   if (string(key.name) == "Delay") {
      INT value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_INT);
      cm_msg(MINFO, "sc_settings_changed", "Set delay to %d", value);
      // TODO: propagate to hardware
   }

   if (string(key.name) == "Reset SC Main" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
       mu.FEBsc_resetMain();
       set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
   }

   if (string(key.name) == "Reset SC Secondary" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
       mu.FEBsc_resetSecondary();
       set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
   }

   if (string(key.name) == "Write" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
       INT FPGA_ID=get_odb_value_by_string("Equipment/Switching/Variables/FPGA_ID_WRITE");
       INT START_ADD=get_odb_value_by_string("Equipment/Switching/Variables/START_ADD_WRITE");
       INT DATA_WRITE_SIZE=get_odb_value_by_string("Equipment/Switching/Variables/DATA_WRITE_SIZE");

       uint32_t DATA_ARRAY[DATA_WRITE_SIZE];
       for (int i = 0; i < DATA_WRITE_SIZE; i++) {
           char STR_DATA[128];
           sprintf(STR_DATA,"Equipment/Switching/Variables/DATA_WRITE[%d]", i);
           DATA_ARRAY[i] = get_odb_value_by_string(STR_DATA);
       }

       uint32_t *data = DATA_ARRAY;

       int count=0;
       while(count < 3){
           if(mu.FEBsc_write((uint32_t) FPGA_ID, data, (uint16_t) DATA_WRITE_SIZE, (uint32_t) START_ADD)!=-1) break;
           count++;
      }
      if(count==3) 
	      cm_msg(MERROR,"switch_fe","Tried 4 times to send a slow control write packet but did not succeed");
      
      set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
   }

   if (string(key.name) == "Read" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
       INT FPGA_ID=get_odb_value_by_string("Equipment/Switching/Variables/FPGA_ID_READ");
       INT LENGTH=get_odb_value_by_string("Equipment/Switching/Variables/LENGTH_READ");
       INT START_ADD=get_odb_value_by_string("Equipment/Switching/Variables/START_ADD_READ");

       uint32_t data[LENGTH];
       int count=0;
       while(count < 3){
           if(mu.FEBsc_read((uint32_t) FPGA_ID, data, (uint16_t) LENGTH, (uint32_t) START_ADD)>=0)
                break;
           count++;
       }
       if(count==3) 
           cm_msg(MERROR,"switch_fe","Tried 4 times to get a slow control read response but did not succeed");

       set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
   }

   if (string(key.name) == "Single Write" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
        INT FPGA_ID=get_odb_value_by_string("Equipment/Switching/Variables/FPGA_ID_WRITE");
        INT DATA=get_odb_value_by_string("Equipment/Switching/Variables/SINGLE_DATA_WRITE");
        INT START_ADD=get_odb_value_by_string("Equipment/Switching/Variables/START_ADD_WRITE");

        uint32_t data_arr[1] = {0};
        data_arr[0] = (uint32_t) DATA;
        uint32_t *data = data_arr;
        mu.FEBsc_write((uint32_t) FPGA_ID, data, (uint16_t) 1, (uint32_t) START_ADD);

        set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }

    if (string(key.name) == "Read WM" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
        INT WM_START_ADD=get_odb_value_by_string("Equipment/Switching/Variables/WM_START_ADD");
        INT WM_LENGTH=get_odb_value_by_string("Equipment/Switching/Variables/WM_LENGTH");
        INT WM_DATA;
        INT SIZE_WM_DATA = sizeof(WM_DATA);

        HNDLE key_WM_DATA;
        db_find_key(hDB, 0, "Equipment/Switching/Variables/WM_DATA", &key_WM_DATA);
        db_set_num_values(hDB, key_WM_DATA, WM_LENGTH);
        for (int i = 0; i < WM_LENGTH; i++) {
            WM_DATA = mu.read_memory_rw((uint32_t) WM_START_ADD + i);
            db_set_value_index(hDB, 0, "Equipment/Switching/Variables/WM_DATA", &WM_DATA, SIZE_WM_DATA, i, TID_INT, FALSE);
        }

        set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }

    if (string(key.name) == "Read RM" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
        cm_msg(MINFO, "sc_settings_changed", "Execute Read RM");

        INT RM_START_ADD=get_odb_value_by_string("Equipment/Switching/Variables/RM_START_ADD");
        INT RM_LENGTH=get_odb_value_by_string("Equipment/Switching/Variables/RM_LENGTH");
        INT RM_DATA;
        INT SIZE_RM_DATA = sizeof(RM_DATA);

        HNDLE key_WM_DATA;
        db_find_key(hDB, 0, "Equipment/Switching/Variables/RM_DATA", &key_WM_DATA);
        db_set_num_values(hDB, key_WM_DATA, RM_LENGTH);
        for (int i = 0; i < RM_LENGTH; i++) {
            RM_DATA = mu.read_memory_ro((uint32_t) RM_START_ADD + i);
            db_set_value_index(hDB, 0, "Equipment/Switching/Variables/RM_DATA", &RM_DATA, SIZE_RM_DATA, i, TID_INT, FALSE);
        }

        set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }

    if (string(key.name) == "Last RM ADD" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
        INT LAST_RM_ADD, SIZE_LAST_RM_ADD;
        SIZE_LAST_RM_ADD = sizeof(LAST_RM_ADD);
        char STR_LAST_RM_ADD[128];
        sprintf(STR_LAST_RM_ADD,"Equipment/Switching/Variables/LAST_RM_ADD");
        INT NEW_LAST_RM_ADD = mu.read_register_ro(MEM_WRITEADDR_LOW_REGISTER_R);
        INT SIZE_NEW_LAST_RM_ADD;
        SIZE_NEW_LAST_RM_ADD = sizeof(NEW_LAST_RM_ADD);
        db_set_value(hDB, 0, STR_LAST_RM_ADD, &NEW_LAST_RM_ADD, SIZE_NEW_LAST_RM_ADD, 1, TID_INT);
        
        set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }
/*
    if (string(key.name) == "Read MALIBU File" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
        INT FPGA_ID, SIZE_FPGA_ID;
        INT START_ADD, SIZE_START_ADD;
        INT PCIE_MEM_START, SIZE_PCIE_MEM_START;

        SIZE_FPGA_ID = sizeof(FPGA_ID);
        SIZE_START_ADD = sizeof(START_ADD);
        SIZE_PCIE_MEM_START = sizeof(PCIE_MEM_START);

        char STR_FPGA_ID[128];
        char STR_START_ADD[128];
        char STR_PCIE_MEM_START[128];

        sprintf(STR_FPGA_ID,"Equipment/Switching/Variables/FPGA_ID_WRITE");
        sprintf(STR_START_ADD,"Equipment/Switching/Variables/START_ADD_WRITE");
        sprintf(STR_PCIE_MEM_START,"Equipment/Switching/Variables/PCIE_MEM_START");

        db_get_value(hDB, 0, "Equipment/Switching/Variables/FPGA_ID_WRITE", &FPGA_ID, &SIZE_FPGA_ID, TID_INT, 0);
        db_get_value(hDB, 0, STR_START_ADD, &START_ADD, &SIZE_START_ADD, TID_INT, 0);
        db_get_value(hDB, 0, STR_PCIE_MEM_START, &PCIE_MEM_START, &SIZE_PCIE_MEM_START, TID_INT, 0);

        uint32_t DATA_ARRAY[256];
        uint32_t n = 0;
        uint32_t w = 0;
        for(int i = 0; i < sizeof(stic3_config_ALL_OFF); i++) {
            if(i%4 == 0) { w = 0; n++; }
            w |= stic3_config_ALL_OFF[i] << (i % 4 * 8);
            if(i%4 == 3) {
                DATA_ARRAY[i/4] = w;
            }
        }

        INT NEW_PCIE_MEM_START = PCIE_MEM_START + 5 + n;

        uint32_t *data = DATA_ARRAY;
        mu.FEB_write((uint32_t) FPGA_ID, data, (uint16_t) n, (uint32_t) START_ADD, (uint32_t) PCIE_MEM_START);

        uint32_t data_arr[1] = {START_ADD};
        mu.FEB_write((uint32_t) FPGA_ID, data_arr, (uint16_t) 1, (uint32_t) 0xFFF1, (uint32_t) NEW_PCIE_MEM_START);
        data_arr[0] = { 0x01100000 + (0xFFFF & n)};

        NEW_PCIE_MEM_START = NEW_PCIE_MEM_START + 6;

        mu.FEB_write((uint32_t) FPGA_ID, data_arr, (uint16_t) 1, (uint32_t) 0xFFF0, (uint32_t) NEW_PCIE_MEM_START);

        NEW_PCIE_MEM_START = NEW_PCIE_MEM_START + 6;
        INT SIZE_NEW_PCIE_MEM_START;
        SIZE_NEW_PCIE_MEM_START = sizeof(NEW_PCIE_MEM_START);
        db_set_value(hDB, 0, STR_PCIE_MEM_START, &NEW_PCIE_MEM_START, SIZE_NEW_PCIE_MEM_START, 1, TID_INT);

        
	set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }
*/
    if (string(key.name) == "SciFiConfig" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
          int status=SciFiFEB::Instance()->ConfigureASICs();
          if(status!=SUCCESS){ 
         	//TODO: what to do? 
          }
	  set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }
    if (string(key.name) == "SciTilesConfig" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
          int status=TilesFEB::Instance()->ConfigureASICs();
          if(status!=SUCCESS){ 
         	//TODO: what to do? 
          }
	  set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }
    if (string(key.name) == "MupixConfig" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
          int status=MupixFEB::Instance()->ConfigureASICs();
          if(status!=SUCCESS){ 
         	//TODO: what to do? 
          }
	  set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }
    if (string(key.name) == "MupixBoard" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
          int status=MupixFEB::Instance()->ConfigureBoards();
          if(status!=SUCCESS){
            //TODO: what to do?
          }
      set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }
    if (string(key.name) == "Reset Bypass Command") {
          DWORD command, payload;
          int size = sizeof(DWORD);
          db_get_data(hDB, hKey, &command, &size, TID_DWORD);
	  if((command&0xff) == 0) return;
          int status = db_get_value(hDB, 0, "/Equipment/Switching/Settings/Reset Bypass Payload", &payload, &size, TID_DWORD, false);

	  printf("Reset Bypass Command %8.8x, payload %8.8x\n",command,payload);
          //first send payload
          status=mup->FEBsc_write(mup->FEBsc_broadcast_ID, &payload,1,0xfff5,false);
	  //do not expect a reply here, for example during sync no data is returned (in reset state)
          status=mup->FEBsc_write(mup->FEBsc_broadcast_ID, &command,1,0xfff4,false);
          if(status!=SUCCESS){/**/}
	  //reset last command & payload
	  payload=0xbcbcbcbc;
          status=mup->FEBsc_write(mup->FEBsc_broadcast_ID, &payload,1,0xfff5,false);
          command=0;//value&(1<<8);
          status=mup->FEBsc_write(mup->FEBsc_broadcast_ID, &command,1,0xfff4,false);
          if(status!=SUCCESS){/**/}
	  //reset odb flag
          command=command&(1<<8);
          db_set_data(hDB, hKey, &command, size, 1, TID_DWORD);
    }

}

//--------------- Link related settings
//

uint64_t get_link_active_from_odb(odb o){

   /* get link active from odb */
   uint64_t link_active_from_odb = 0;
   for(int link = 0; link < MAX_LINKS_PER_SWITCHINGBOARD; link++) {
      int offset = MAX_LINKS_PER_SWITCHINGBOARD * switch_id;
      int cur_mask = o[offset + link];
      if((cur_mask == FEBLINKMASK::ON) || (cur_mask == FEBLINKMASK::DataOn)){
        //a standard FEB link (SC and data) is considered enabled if RX and TX are. 
	    //a secondary FEB link (only data) is enabled if RX is.
	    //Here we are concerned only with run transitions and slow control, the farm frontend may define this differently.
        link_active_from_odb += (1 << link);
      }
   }
   return link_active_from_odb;
}

void set_feb_enable(uint64_t enablebits){
   //mup->write_register(FEB_ENABLE_REGISTER_HIGH_W, enablebits >> 32); TODO make 64 bits
   mup->write_register(FEB_ENABLE_REGISTER_LOW_W,  enablebits & 0xFFFFFFFF);
}

uint64_t get_runstart_ack(){
   uint64_t reg = mup->read_register_ro(RUN_NR_ACK_REGISTER_LOW_R);
//   reg |= mup->read_register_ro(RUN_NR_ACK_REGISTER_HIGH_R) << 32; TODO make 64 bits
   return reg;
}
uint64_t get_runend_ack(){
   uint64_t reg = mup->read_register_ro(RUN_STOP_ACK_REGISTER_R);
//   reg |= mup->read_register_ro(RUN_STOP_ACK_REGISTER_HIGH_R) << 32; TODO make 64 bits
   return reg;
}

void print_ack_state(){
   uint64_t link_active_from_register = get_runstart_ack();
   for(int i = 0; i < MAX_LINKS_PER_SWITCHINGBOARD; i++) {
      //if ((link_active_from_register >> i) & 0x1){
         mup->write_register_wait(RUN_NR_ADDR_REGISTER_W, uint32_t(i), 1000);
         uint32_t val=mup->read_register_ro(RUN_NR_REGISTER_R);
         cm_msg(MINFO,"switch_fe","Switching board %d, Link %d: PREP_ACK=%u STOP_ACK=%u RNo=0x%8.8x", switch_id, i, (val>>25)&1, (val>>24)&1,(val>>0)&0xffffff);
      //}
   }
}


