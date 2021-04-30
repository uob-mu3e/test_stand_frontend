#define FEB_ENABLE_REGISTER_LOW_W FEB_ENABLE_REGISTER_W
#define RUN_NR_ACK_REGISTER_LOW_R RUN_NR_ACK_REGISTER_R
/********************************************************************\

  Name:         taken from switch_fe.cpp
  Created by:   Stefan Ritt
  Updated by:   Marius Koeppel, Konrad Briggl, Lukas Gerritzen, Niklaus Berger

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
#include "mudaq_device.h"
#include "mudaq_dummy.h"

#include "FEBSlowcontrolInterface.h"
#include "DummyFEBSlowcontrolInterface.h"
#include "feblist.h"


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

const int switch_id = 0; // TODO to be loaded from outside (on compilation?)
const int per_fe_SSFE_size = 26;
const int per_crate_SCFC_size = 21;

/* Inteface to the PCIe FPGA */
mudaq::MudaqDevice * mup;

/* Abstraction for talking to the FEBs via the PCIe FPGA or MSCB (to be implemented) */
FEBSlowcontrolInterface * feb_sc;

/* Lists of the active FEBs */
FEBList * feblist;

/* FEB classes */
MuFEB       * mufeb;
MupixFEB    * mupixfeb;
SciFiFEB    * scififeb;
TilesFEB    * tilefeb;


/*-- Function declarations -----------------------------------------*/

INT read_sc_event(char *pevent, INT off);
INT read_WMEM_event(char *pevent, INT off);
INT read_scifi_sc_event(char *pevent, INT off);
INT read_scitiles_sc_event(char *pevent, INT off);
INT read_mupix_sc_event(char *pevent, INT off);
INT read_febcrate_sc_event(char *pevent, INT off);

void sc_settings_changed(odb o);
void switching_board_mask_changed(odb o);
void frontend_board_mask_changed(odb o);

uint64_t get_link_active_from_odb(odb o); //throws
void set_feb_enable(uint64_t enablebits);
uint64_t get_runstart_ack();
uint64_t get_runend_ack();
void print_ack_state();

void setup_odb();
void setup_watches();
void setup_history();
void setup_alarms();

INT init_mudaq(mudaq::MudaqDevice&  mu);
INT init_crates();
INT init_febs(mudaq::MudaqDevice&  mu);
INT init_scifi(mudaq::MudaqDevice&  mu);
INT init_scitiles(mudaq::MudaqDevice& mu);
INT init_mupix(mudaq::MudaqDevice& mu);



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
     FALSE,                      /* enabled */
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
     FALSE,                      /* enabled */
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
     FALSE,                      /* enabled */
     RO_ALWAYS | RO_ODB,   /* read during run transitions and update ODB */
     1000,                      /* read every 1 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", "",},
     read_mupix_sc_event,          /* readout routine */
    },
    {"FEBCrates",                    /* equipment name */
    {110, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                 /* equipment type */
     0,                         /* event source crate 0, all stations */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_ALWAYS | RO_ODB,   /* read during run transitions and update ODB */
     10000,                      /* read every 1 sec */
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

    // create Settings structure in ODB
    cout << "Setting up ODB" << endl;
    setup_odb();


    cout << "Opening Mudaq" << endl;
    // open mudaq
    #ifdef NO_SWITCHING_BOARD
        mup = new mudaq::DummyDmaMudaqDevice("/dev/mudaq0");
    #else
        mup = new mudaq::DmaMudaqDevice("/dev/mudaq0");
    #endif       
        
    INT status = init_mudaq(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;


    cout << "Setting link enables" << endl;
    //set link enables so slow control can pass
    odb cur_links_odb("/Equipment/Links/Settings/LinkMask");
    try{
        set_feb_enable(get_link_active_from_odb(cur_links_odb)); }
    catch(...){ return FE_ERR_ODB;}
    
    cout << "Creating FEB List" << endl;
    // Create the FEB List
    feblist = new FEBList(switch_id);

    //init feb crates
    status = init_crates();
    if (status != SUCCESS)
        return FE_ERR_DRIVER;

    //init febs (general)
    status = init_febs(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;


    /*
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
    */
    // TODO: Define generic history panels

    // Subdetector specific panels should be created created in subdet::midasODB::setup_db
    // functions called from above

    // TODO: Same for alarms
    
    // Set our transition sequence. The default is 500. Setting it
    // to 400 means we are called BEFORE most other clients.
    cm_set_transition_sequence(TR_START, 400);

    // Set our transition sequence. The default is 500. Setting it
    // to 600 means we are called AFTER most other clients.
    cm_set_transition_sequence(TR_STOP, 600);


    cout << "Setting up Watches" << endl;
    setup_watches();

    return CM_SUCCESS;
}

// ODB Setup //////////////////////////////
void setup_odb(){

   // midas::odb::set_debug(true);

    string namestr;
    string bankname;
    if(switch_id == 0){
        namestr = "Names SCFE";
        bankname = "SCFE";
    }
    if(switch_id == 1){
        namestr = "Names SUFE";
        bankname = "SUFE";
    }
    if(switch_id == 2){
        namestr = "Names SDFE";
        bankname = "SDFE";
    }
    if(switch_id == 3){
        namestr = "Names SFFE";
        bankname = "SFFE";
    }

    /* Default values for /Equipment/Switching/Settings */
    odb settings = {
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
            {"Load Firmware", false},
            {"Firmware File",""},
            {"Firmware FEB ID",0},
            // For this, switch_id has to be known at compile time (calls for a preprocessor macro, I guess)
            {namestr.c_str(), std::array<std::string, per_fe_SSFE_size*N_FEBS[switch_id]>()},
            {"Names SCFC", std::array<std::string, per_crate_SCFC_size*N_FEBCRATES>()}
    };

    int bankindex = 0;

    for(int i=0; i < N_FEBS[switch_id]; i++){
        string feb = "FEB" + to_string(i);
        string * s = new string(feb);
        (*s) += " Index";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Arria Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " MAX Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " SI1 Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " SI2 Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " ext Arria Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " DCDC Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 1.1";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 1.8";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 2.5";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 3.3";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 20";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 Voltage";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 RX1 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 RX2 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 RX3 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 RX4 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly1 Alarms";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 Temperature";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 Voltage";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 RX1 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 RX2 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 RX3 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 RX4 Power";
        settings[namestr][bankindex++] = s;
        s = new string(feb);
        (*s) += " Firefly2 Alarms";
        settings[namestr][bankindex++] = s;
    }

    bankindex = 0;
    for(int i=0; i < N_FEBCRATES; i++){
        string feb = "Crate" + to_string(i);
        string * s = new string(feb);
        (*s) += " Index";
        settings["Names SCFC"][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 20";
        settings["Names SCFC"][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 3.3";
        settings["Names SCFC"][bankindex++] = s;
        s = new string(feb);
        (*s) += " Voltage 5";
        settings["Names SCFC"][bankindex++] = s;
        s = new string(feb);
        (*s) += " CC Temperature";
        settings["Names SCFC"][bankindex++] = s;
        for(int j=0; j < MAX_FEBS_PER_CRATE; j++){
            s = new string(feb);
            (*s) += "FEB" + to_string(j) + "Temperature";
            settings["Names SCFC"][bankindex++] = s;
        }
    }

    settings.print();

    settings.connect("/Equipment/Switching/Settings", true);

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

            {"Merger Timeout All FEBs", 0},

            {bankname.c_str(),std::array<float, per_fe_SSFE_size*N_FEBS[switch_id]>{}}
    };

    sc_variables.connect("/Equipment/Switching/Variables");

    odb firmware_variables = {
        {"Arria V Firmware Version", std::array<uint32_t, MAX_N_FRONTENDBOARDS>{}},
        {"Max 10 Firmware Version", std::array<uint32_t, MAX_N_FRONTENDBOARDS>{}},
        {"FEB Version", std::array<uint32_t, MAX_N_FRONTENDBOARDS>{20}}
    };

    firmware_variables.connect("/Equipment/Switching/Variables/FEBFirmware");

    // add custom page to ODB
    odb custom("/Custom");
    custom["Switching&"] = "sc.html";
    custom["Febs&"] = "febs.html";
    
    // TODO: not sure at the moment we have a midas frontend for three feb types but 
    // we need to have different swb at the final experiment so maybe one needs to take
    // things apart later. For now we put this "common" FEB variables into the generic
    // switching path
    hs_define_panel("Switching", "All FEBs", {"Switching:Merger Timeout All FEBs"});

}

void setup_watches(){
    //UI watch
    odb sc_variables("/Equipment/Switching/Settings");
    sc_variables.watch(sc_settings_changed);

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

    vector<INT> switching_board_mask = o;

    BOOL value = switching_board_mask[switch_id] > 0 ? true : false;

    for(int i = 0; i < 2; i++) {        
        char str[128];
        sprintf(str,"Equipment/%s/Common/Enabled", equipment[i].name);
        odb enabled(str);
        enabled = value;
        //db_set_value(hDB,0,str, &value, sizeof(value), 1, TID_BOOL);
        cm_msg(MINFO, "switching_board_mask_changed", "Set Equipment %s enabled to %d", equipment[i].name, value);
    }

    feblist->RebuildFEBList();
    mufeb->ReadFirmwareVersionsToODB();
}

void frontend_board_mask_changed(odb o) {
    feblist->RebuildFEBList();
    mufeb->ReadFirmwareVersionsToODB();
}

INT init_mudaq(mudaq::MudaqDevice &mu) {

    
    if ( !mu.open() ) {
        cm_msg(MERROR, "frontend_init" , "Could not open device");
        return FE_ERR_DRIVER;
    }

    if ( !mu.is_ok() ) {
        cm_msg(MERROR, "frontend_init", "Mudaq is not ok");
        return FE_ERR_DRIVER;
    }
#ifdef NO_SWITCHING_BOARD
    feb_sc = new DummyFEBSlowcontrolInterface(mu);
#else
    feb_sc = new FEBSlowcontrolInterface(mu);
#endif
    return SUCCESS;
}

INT init_crates() {
    return SUCCESS;
}

INT init_febs(mudaq::MudaqDevice & mu) {

    // SciFi setup part
    set_equipment_status(equipment[EQUIPMENT_ID::Switching].name, "Initializing...", "var(--myellow)");
    mufeb = new  MuFEB(*feb_sc,
                        feblist->getFEBs(),
                        feblist->getSciFiFEBMask(),
                        equipment[EQUIPMENT_ID::Switching].name,
                        "/Equipment/SciFi",
                        switch_id); //create FEB interface signleton for scifi

    //init all values on FEB
    mufeb->WriteFEBID();

    // Get all the relevant firmware versions
    mufeb->ReadFirmwareVersionsToODB();

    set_equipment_status(equipment[EQUIPMENT_ID::Switching].name, "Ok", "var(--mgreen)");

    return SUCCESS;
}


INT init_scifi(mudaq::MudaqDevice & mu) {

    // SciFi setup part
    set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Initializing...", "var(--myellow)");
    scififeb = new SciFiFEB(*feb_sc,
                     feblist->getSciFiFEBs(),
                     feblist->getSciFiFEBMask(),
                     equipment[EQUIPMENT_ID::SciFi].name,
                     "/Equipment/SciFi",
                      switch_id); //create FEB interface signleton for scifi

    int status=mutrig::midasODB::setup_db("/Equipment/SciFi",scififeb);
    if(status != SUCCESS){
        set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Start up failed", "var(--mred)");
        return status;
    }
    //init all values on FEB
    scififeb->WriteAll();
    scififeb->WriteFEBID();

    set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Ok", "var(--mgreen)");
    
    //set custom page
    odb custom("/Custom");
    custom["SciFi-ASICs&"] = "mutrigTdc.html";

    return SUCCESS;
}

INT init_scitiles(mudaq::MudaqDevice & mu) {

    
    //SciTiles setup part
    set_equipment_status(equipment[EQUIPMENT_ID::SciTiles].name, "Initializing...", "var(--myellow)");
    tilefeb = new TilesFEB(*feb_sc,
                     feblist->getTileFEBs(),
                     feblist->getTileFEBMask(),
                     equipment[EQUIPMENT_ID::SciTiles].name,
                     "/Equipment/SciTiles",
                      switch_id); //create FEB interface signleton for scitiles
    int status=mutrig::midasODB::setup_db("/Equipment/SciTiles", tilefeb);
    if(status != SUCCESS){
        set_equipment_status(equipment[EQUIPMENT_ID::SciTiles].name, "Start up failed", "var(--mred)");
        return status;
    }
    //init all values on FEB
    tilefeb->WriteAll();
    tilefeb->WriteFEBID();

    set_equipment_status(equipment[EQUIPMENT_ID::SciTiles].name, "Ok", "var(--mgreen)");

    //set custom page
    odb custom("/Custom");
    custom["SciTiles-ASICs&"] = "tile_custompage.html";
    
    return SUCCESS;
}


INT init_mupix(mudaq::MudaqDevice & mu) {


    //Mupix setup part
    set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Initializing...", "var(--myellow)");
    mupixfeb = new MupixFEB(*feb_sc,
                     feblist->getPixelFEBs(),
                     feblist->getPixelFEBMask(),
                     equipment[EQUIPMENT_ID::Mupix].name,
                     "/Equipment/Mupix",
                     switch_id); //create FEB interface signleton for mupix

    int status=mupix::midasODB::setup_db("/Equipment/Mupix", mupixfeb, true);
    if(status != SUCCESS){
        set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Start up failed", "var(--mred)");
        return status;
    }
    //init all values on FEB
    mupixfeb->WriteFEBID();
    
    set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Ok", "var(--mgreen)");
   
    // setup odb rate counters for each feb
    // TODO: That should probably go into setup_db
    char set_str[255];
    odb rate_counters("/Equipment/Mupix/Variables");
    for(uint i = 0; i < mupixfeb->getNFPGAs(); i++){
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
    for(uint i = 0; i < mupixfeb->getNFPGAs(); i++){
        sprintf(set_str, "FEB%d", i);
        hs_define_panel("Mupix", set_str, {"Mupix:merger rate " + string(set_str),
                                           "Mupix:hit ena rate " + string(set_str),
                                           "Mupix:reset phase " + string(set_str)
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
try{ // TODO: What can throw here?? Why?? Is there another way to handle this??
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
   status=scififeb->ConfigureASICs();
   if(status!=SUCCESS){
      cm_msg(MERROR,"switch_fe","ASIC configuration failed");
      return CM_TRANSITION_CANCELED;
   }

   //configure ASICs for Tiles
   status=tilefeb->ConfigureASICs();
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
   scififeb->ResetAllCounters();


   // TODO: Switch to odbxx here
   HNDLE hKey;
   char ip[256];
   int size = 256;
   if(db_find_key(hDB, 0, "/Equipment/Clock Reset", &hKey) != DB_SUCCESS){
       cm_msg(MERROR,"switch_fe","could not find CRFE, is CRFE running ?");
       return CM_TRANSITION_CANCELED;
   }else if(db_get_value(hDB, hKey, "Settings/IP", ip, &size, TID_STRING, false)!= DB_SUCCESS) {
       cm_msg(MERROR,"switch_fe","could not find CRFE IP, is CRFE running ?");
       return CM_TRANSITION_CANCELED;
   }

   if(string(ip)=="0.0.0.0"){
       /* send run prepare signal from here */
       cm_msg(MINFO,"switch_fe","Bypassing CRFE for run transition");
       // TODO: Get rid of hardcoded adresses here!
       DWORD valueRB = run_number;
       feb_sc->FEB_register_write(FEBSlowcontrolInterface::ADDRS::BROADCAST_ADDR, RESET_PAYLOAD_REGISTER_RW, valueRB); //run number
       valueRB= ((1<<RESET_BYPASS_BIT_ENABLE) |(1<<RESET_BYPASS_BIT_REQUEST)) | 0x10;
       feb_sc->FEB_register_write(FEBSlowcontrolInterface::ADDRS::BROADCAST_ADDR, RUN_STATE_RESET_BYPASS_REGISTER_RW, valueRB); //run prep command
       valueRB= 0xbcbcbcbc;
       feb_sc->FEB_register_write(FEBSlowcontrolInterface::ADDRS::BROADCAST_ADDR, RESET_PAYLOAD_REGISTER_RW, valueRB); //reset payload
       valueRB= (1<<RESET_BYPASS_BIT_ENABLE) | 0x00;
       feb_sc->FEB_register_write(FEBSlowcontrolInterface::ADDRS::BROADCAST_ADDR, RUN_STATE_RESET_BYPASS_REGISTER_RW, valueRB); //reset command
   }else{
       /* send run prepare signal via CR system */
       // TODO: Move to odbxx
       feb_sc->FEB_register_write(FEBSlowcontrolInterface::ADDRS::BROADCAST_ADDR, RUN_STATE_RESET_BYPASS_REGISTER_RW, 0); // disable reset bypass for all connected febs
       INT value = 1;
       cm_msg(MINFO,"switch_fe","Using CRFE for run transition");
       db_set_value_index(hDB,0,"Equipment/Clock Reset/Run Transitions/Request Run Prepare",
                          &value, sizeof(value), switch_id, TID_INT, false);
   }

   // TODO: Can we do better than than a hardcoded timeout count?
   // Here we do need a timout, maybe set via ODB??
   // Also, this should never take 30s!!!
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

/*--- Read Slow Control Event from crate controllers to be put into data stream --------*/
INT read_febcrate_sc_event(char *pevent, INT off){
    return 0;
}

/*--- Read Slow Control Event from FEBsto be put into data stream --------*/
INT read_sc_event(char *pevent, INT off)
{    
    cout << "Reading FEB SC" << endl;

    string bankname;
    if(switch_id == 0){
        bankname = "SCFE";
    }
    if(switch_id == 1){
        bankname = "SUFE";
    }
    if(switch_id == 2){
        bankname = "SDFE";
    }
    if(switch_id == 3){
        bankname = "SFFE";
    }

    // create bank, pdata
    bk_init(pevent);
    DWORD *pdata;
    bk_create(pevent, bankname.c_str(), TID_FLOAT, (void **)&pdata);
    pdata = mufeb->fill_SSFE(pdata);
    bk_close(pevent,pdata);
    return bk_size(pevent);


}

/*--- Read Slow Control Event from SciFi to be put into data stream --------*/

INT read_scifi_sc_event(char *pevent, INT off){
	static int i=0;
    printf("Reading Scifi FEB status data from all FEBs %d\n",i++);
    //TODO: Make this more proper: move this to class driver routine and make functions not writing to ODB all the time (only on update).
    //Add readout function for this one that gets data from class variables and writes midas banks
    scififeb->ReadBackAllCounters();
    scififeb->ReadBackAllRunState();
    scififeb->ReadBackAllDatapathStatus();
    return 0;
}

/*--- Read Slow Control Event from SciTiles to be put into data stream --------*/

INT read_scitiles_sc_event(char *pevent, INT off){
	static int i=0;
    printf("Reading SciTiles FEB status data from all FEBs %d\n",i++);
    //TODO: Make this more proper: move this to class driver routine and make functions not writing to ODB all the time (only on update).
    //Add readout function for this one that gets data from class variables and writes midas banks
    tilefeb->ReadBackAllCounters();
    tilefeb->ReadBackAllRunState();
    tilefeb->ReadBackAllDatapathStatus();
    return 0;
}

/*--- Read Slow Control Event from Mupix to be put into data stream --------*/

INT read_mupix_sc_event(char *pevent, INT off){
    // get odb11:29:52.162 2021/03/01 [SW Frontend,INFO] Setting FEBID of Central:Board1: Link0, SB0.0 to (feb1)-feb0
    odb rate_cnt("/Equipment/Mupix/Variables");
    uint32_t HitsEnaRate;
    uint32_t MergerRate;
    uint32_t ResetPhase;
    uint32_t TXReset;
    char set_str[255];
    static int i = 0;
 
    // TODO: sort generic ro and mupix specific ro
    bk_init(pevent);
    DWORD *pdata;
    bk_create(pevent, "FECN", TID_WORD, (void **) &pdata);
    printf("Reading MuPix FEB status data from all FEBs %d\n", i++);
    uint32_t d;
    feb_sc->FEB_read(0,0,d);
    printf("%i\n", d);

    mupixfeb->ReadBackAllRunState();
    for(uint i = 0; i < mupixfeb->getNFPGAs(); i++){
        HitsEnaRate =mupixfeb->ReadBackHitsEnaRate(i);
        MergerRate = mupixfeb->ReadBackMergerRate(i);
        ResetPhase = mupixfeb->ReadBackResetPhase(i);
        TXReset = mupixfeb->ReadBackTXReset(i);


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

//TODO: Get rid of this...
INT get_odb_value_by_string(const char *key_name){
    INT ODB_DATA, SIZE_ODB_DATA;
    SIZE_ODB_DATA = sizeof(ODB_DATA);
    db_get_value(hDB, 0, key_name, &ODB_DATA, &SIZE_ODB_DATA, TID_INT, 0);
    return ODB_DATA;
}

/*--- Called whenever settings have changed ------------------------*/

void sc_settings_changed(odb o)
{
    std::string name = o.get_name();

    printf("%s\n",name.c_str());

#ifdef MY_DEBUG
    dummy_mudaq::DummyMudaqDevice & mu = *mup;
#else
    mudaq::MudaqDevice & mu = *mup;
#endif

    if (name == "Active") {
        bool value = o;
        cm_msg(MINFO, "sc_settings_changed", "Set active to %d", value);
        // TODO: propagate to hardware
    }

    if (name == "Delay") {
        INT value = o;
        cm_msg(MINFO, "sc_settings_changed", "Set delay to %d", value);
        // TODO: propagate to hardware
    }

    if (name == "Reset SC Main" && o) {
        bool value = o;
        if(value){
             feb_sc->FEBsc_resetMain();
             o = false;
        }
    }

    if (name == "Reset SC Secondary" && o) {
        bool value = o;
        if(value){
            feb_sc->FEBsc_resetSecondary();
             o = false;
        }
    }


   if (name == "Write" && o) {
       odb fpgaid("Equipment/Switching/Variables/FPGA_ID_WRITE");
       odb startaddr("Equipment/Switching/Variables/START_ADD_WRITE");
       odb writesize("Equipment/Switching/Variables/DATA_WRITE_SIZE");
       odb dataarray("Equipment/Switching/Variables/DATA_WRITE");
       vector<uint32_t> dataarrayv = dataarray;
       feb_sc->FEB_write(fpgaid, startaddr, dataarrayv);
        o = false;
   }

   if (name == "Read" && o) {
       odb fpgaid("Equipment/Switching/Variables/FPGA_ID_READ");
       odb startaddr("Equipment/Switching/Variables/START_ADD_READ");
       odb readsize("Equipment/Switching/Variables/LENGTH_READ");
       vector<uint32_t> data(readsize);

       feb_sc->FEB_read(fpgaid, startaddr, data);
       // TODO: Do something with the value we read...
       o = false;
   }

   if (name == "Single Write" && o) {
        odb fpgaid("Equipment/Switching/Variables/FPGA_ID_WRITE");
        odb datawrite("Equipment/Switching/Variables/SINGLE_DATA_WRITE");
        odb startaddr("Equipment/Switching/Variables/START_ADD_WRITE");

        uint32_t data = datawrite;
        feb_sc->FEB_write(fpgaid, startaddr, data);
        o = false;
    }

    if (name == "Read WM" && o) {
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

        o = false;
    }

    if (name == "Read RM" && o) {
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

        o = false;
    }

    if (name == "Last RM ADD" && o) {

        char STR_LAST_RM_ADD[128];
        sprintf(STR_LAST_RM_ADD,"Equipment/Switching/Variables/LAST_RM_ADD");
        INT NEW_LAST_RM_ADD = mu.read_register_ro(MEM_WRITEADDR_LOW_REGISTER_R);
        INT SIZE_NEW_LAST_RM_ADD;
        SIZE_NEW_LAST_RM_ADD = sizeof(NEW_LAST_RM_ADD);
        db_set_value(hDB, 0, STR_LAST_RM_ADD, &NEW_LAST_RM_ADD, SIZE_NEW_LAST_RM_ADD, 1, TID_INT);
        
        o = false;
    }

    if (name == "SciFiConfig" && o) {
          int status=scififeb->ConfigureASICs();
          if(status!=SUCCESS){ 
         	//TODO: what to do? 
          }
       o = false;
    }
    if (name == "SciTilesConfig" && o) {
          int status=tilefeb->ConfigureASICs();
          if(status!=SUCCESS){ 
         	//TODO: what to do? 
          }
      o = false;
    }
    if (name == "MupixConfig" && o) {
          int status=mupixfeb->ConfigureASICs();
          if(status!=SUCCESS){ 
         	//TODO: what to do? 
          }
      o = false;
    }
    if (name == "MupixBoard" && o) {
          int status=mupixfeb->ConfigureBoards();
          if(status!=SUCCESS){
            //TODO: what to do?
          }
      o = false;
    }
    if (name == "Reset Bypass Command") {
         uint32_t command = o;

	  if((command&0xff) == 0) return;
      uint32_t payload = odb("/Equipment/Switching/Settings/Reset Bypass Payload");

	  printf("Reset Bypass Command %8.8x, payload %8.8x\n",command,payload);

        // TODO: get rid of hardcoded addresses
        feb_sc->FEB_register_write(FEBSlowcontrolInterface::ADDRS::BROADCAST_ADDR, 0xf5, payload);
        feb_sc->FEB_register_write(FEBSlowcontrolInterface::ADDRS::BROADCAST_ADDR, 0xf4, command);
        // reset payload and command TODO: Is this needed?
        payload=0xbcbcbcbc;
        command=0;
        feb_sc->FEB_register_write(FEBSlowcontrolInterface::ADDRS::BROADCAST_ADDR, 0xf5, payload);
        feb_sc->FEB_register_write(FEBSlowcontrolInterface::ADDRS::BROADCAST_ADDR, 0xf4, command);
        //reset odb flag
          command=command&(1<<8);
          o = command;
    }

    if (name == "Load Firmware" && o) {
        printf("Load firmware triggered");
        string fname = odb("/Equipment/Switching/Settings/Firmware File");
        uint32_t id = odb("/Equipment/Switching/Settings/Firmware FEB ID");
       mufeb->LoadFirmware(fname,id);
       o = false;
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
   //uint64_t link_active_from_register = get_runstart_ack();
   for(int i = 0; i < MAX_LINKS_PER_SWITCHINGBOARD; i++) {
      //if ((link_active_from_register >> i) & 0x1){
         mup->write_register_wait(RUN_NR_ADDR_REGISTER_W, uint32_t(i), 1000);
         uint32_t val=mup->read_register_ro(RUN_NR_REGISTER_R);
         cm_msg(MINFO,"switch_fe","Switching board %d, Link %d: PREP_ACK=%u STOP_ACK=%u RNo=0x%8.8x", switch_id, i, (val>>25)&1, (val>>24)&1,(val>>0)&0xffffff);
      //}
   }
}


