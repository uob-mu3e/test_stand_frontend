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
#include <cstring>

#include <switching_constants.h>
#include <a10_counters.h>
#include <mu3ebanks.h>

#include <history.h>
#include "midas.h"
#include "odbxx.h"
#include "mfe.h"
#include "string.h"
#include "mudaq_device.h"
#include "mudaq_dummy.h"
#include "mscb.h"

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

/* Start address of power in the crate controller - TODO: Move to an appropriate header*/
const uint8_t CC_POWER_OFFSET = 5;
const uint8_t CC_VT_READOUT_START = 1;

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
INT max_event_size = 20000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 10 * 10000;

constexpr int switch_id = 0; // TODO to be loaded from outside (on compilation?)


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

/* Local state of FEB power*/
std::array<uint8_t, N_FEBCRATES*MAX_FEBS_PER_CRATE> febpower{};
/* Local state of sorter delays */
std::array<uint32_t, N_FEBS[switch_id]> sorterdelays{};

/*-- Function declarations -----------------------------------------*/

INT read_sc_event(char *pevent, INT off);
INT read_WMEM_event(char *pevent, INT off);
INT read_scifi_sc_event(char *pevent, INT off);
INT read_scitiles_sc_event(char *pevent, INT off);
INT read_mupix_sc_event(char *pevent, INT off);
INT read_febcrate_sc_event(char *pevent, INT off);

DWORD * fill_SSCN(DWORD *);

void sc_settings_changed(odb o);
void switching_board_mask_changed(odb o);
void frontend_board_mask_changed(odb o);
void febpower_changed(odb o);
void sorterdelays_changed(odb o);

uint64_t get_link_active_from_odb(odb o); //throws
void set_feb_enable(uint64_t enablebits);
uint64_t get_runstart_ack();
uint64_t get_runend_ack();
void print_ack_state();
uint32_t read_counters(uint32_t write_value);

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
     "", "", ""},
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
    for(size_t i =0; i < febpower.size(); i++)
        febpower[i] = 0;


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

    //init SC

    //init feb crates
    status = init_crates();
    if (status != SUCCESS)
        return FE_ERR_DRIVER;

    //init febs (general)
    status = init_febs(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;


    //init scifi
    cm_msg(MINFO, "switch_fe", "Calling init_scifi");
    cm_msg(MINFO, "switch_fe", "TEST");
    status = init_scifi(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;
    
    /*
    //init scitiles
    status = init_scitiles(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;*/

    //init mupix
    status = init_mupix(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;

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


    cm_msg(MINFO, "frontend_init()", "Setting up Watches");
    setup_watches();

    return CM_SUCCESS;
}

// ODB Setup //////////////////////////////
void setup_odb(){

   // midas::odb::set_debug(true);

    string namestr          = ssfenames[switch_id];
    string bankname         = ssfe[switch_id];
    string cntnamestr       = sscnnames[switch_id];
    string cntbankname      = sscn[switch_id];
    string sorternamestr    = sssonames[switch_id];
    string sorterbankname   = ssso[switch_id];

    // For this, switch_id has to be known at compile time (calls for a preprocessor macro, I guess)
    // Also: How do we do this for four switching boards? Larger array or 4 equipments?
    std::array<uint16_t, N_FEBS[switch_id]> zeroarr;
    zeroarr.fill(0);

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
            {"Sorter Delay", zeroarr},
            // For this, switch_id has to be known at compile time (calls for a preprocessor macro, I guess)
            {namestr.c_str(), std::array<std::string, per_fe_SSFE_size*N_FEBS[switch_id]>()},
            {cntnamestr.c_str(), std::array<std::string, num_swb_counters_per_feb*N_FEBS[switch_id]+4>()},
            {sorternamestr.c_str(), std::array<std::string, per_fe_SSSO_size*N_FEBS[switch_id]>()}
    };

    create_ssfe_names_in_odb(settings,switch_id);
    create_ssso_names_in_odb(settings,switch_id);
    create_sscn_names_in_odb(settings,switch_id);


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

            {bankname.c_str(), std::array<float, per_fe_SSFE_size*N_FEBS[switch_id]>{}},
            {cntbankname.c_str(), std::array<int, num_swb_counters_per_feb*N_FEBS[switch_id]+4>()},
            {sorterbankname.c_str(), std::array<int, per_fe_SSSO_size*N_FEBS[switch_id]>{}}
    };

    sc_variables.connect("/Equipment/Switching/Variables");

    odb firmware_variables = {
        {"Arria V Firmware Version", std::array<uint32_t, MAX_N_FRONTENDBOARDS>{}},
        {"Max 10 Firmware Version", std::array<uint32_t, MAX_N_FRONTENDBOARDS>{}},
        {"FEB Version", std::array<uint32_t, MAX_N_FRONTENDBOARDS>{20}},
    };

    firmware_variables.connect("/Equipment/Switching/Variables/FEBFirmware");

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

    cout << "Setting crate variables" << endl;

    odb crate_variables = {
        {"FEBPower", std::array<uint8_t, N_FEBCRATES*MAX_FEBS_PER_CRATE>{}},
        {"SCFC", std::array<float, per_crate_SCFC_size*N_FEBCRATES>()}
    };

    crate_variables.connect("/Equipment/FEBCrates/Variables");


    // add custom page to ODB
    odb custom("/Custom");
    custom["Switching&"] = "sc.html";
    custom["Febs&"] = "febs.html";
    custom["FEBcrates&"] = "crates.html";
    custom["DAQcounters&"] = "daqcounters.html";

    // Inculde the line below to set up the FEBs and their mapping for the 2021 integration run
//#include "odb_feb_mapping_integration_run_2021.h"

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
    switch_mask.watch(switching_board_mask_changed);

    // watch if the mapping of FEBs to crates changed
    odb crates("/Equipment/FEBCrates/Settings");
    crates.watch(frontend_board_mask_changed);

    // watch if this links are enabled
    odb links_odb("/Equipment/Links/Settings/LinkMask");
    links_odb.watch(frontend_board_mask_changed);

    // watch for changes in the FEB powering state
    odb febpower_odb("/Equipment/FEBCrates/Variables/FEBPower");
    febpower_odb.watch(febpower_changed);

    // Watch for sorter delay changes
    odb sorterdelay_settings("/Equipment/Switching/Settings/Sorter Delay");
    sorterdelay_settings.watch(sorterdelays_changed);
}

void switching_board_mask_changed(odb o) {

    string name = o.get_name();
    cm_msg(MINFO, "switching_board_mask_changed", "Switching board masking changed");
    cm_msg(MINFO, "switching_board_mask_changed", "For INT run we enable MuPix and SciFi Equipment");

    vector<INT> switching_board_mask = o;

    BOOL value = switching_board_mask[switch_id] > 0 ? true : false;

    for(int i = 0; i < 4; i++) {
        // for int run
        if (i == 2) continue;
        char str[128];
        sprintf(str, "/Equipment/%s/Common", equipment[i].name);
        //TODO: Can we avoid this silly back and forth casting?
        odb enabled(std::string(str).c_str());
        enabled["Enabled"] = value;
        cm_msg(MINFO, "switching_board_mask_changed", "Set Equipment %s enabled to %d", equipment[i].name, value);
    }

    feblist->RebuildFEBList();
    mufeb->ReadFirmwareVersionsToODB();
}

void frontend_board_mask_changed(odb o) {
    cm_msg(MINFO, "frontend_board_mask_changed", "Frontend board masking changed");
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
    odb febpower_odb("/Equipment/FEBCrates/Variables/FEBPower");
    febpower_changed(febpower_odb);
    return SUCCESS;
}

INT init_febs(mudaq::MudaqDevice & mu) {

    odb sorterdelays_odb("/Equipment/Switching/Settings/Sorter Delay");
    sorterdelays_changed(sorterdelays_odb);

    // switching setup part
    set_equipment_status(equipment[EQUIPMENT_ID::Switching].name, "Initializing...", "var(--myellow)");
    mufeb = new  MuFEB(*feb_sc,
                        feblist->getFEBs(),
                        feblist->getFEBMask(),
                        equipment[EQUIPMENT_ID::Switching].name,
                        "/Equipment/SciFi",
                        switch_id);

    //init all values on FEB
    mufeb->WriteFEBID();

    // Get all the relevant firmware versions
    mufeb->ReadFirmwareVersionsToODB();

    // Init sorter delays
    odb sorterdelay_odb("/Equipment/Switching/Settings/Sorter Delay");
    sorterdelays_changed(sorterdelay_odb);

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

    
    cm_msg(MINFO,"switch_fe","Setting up ODB for SciFi");
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
   status=mupixfeb->ConfigureASICs();
   if(status!=SUCCESS){
      cm_msg(MERROR,"switch_fe","ASIC configuration failed");
      return CM_TRANSITION_CANCELED;
   }


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
      set_equipment_status(equipment[EQUIPMENT_ID::Switching].name, "Not Ok", "var(--mred)");
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
//      set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Not Ok", "var(--mred)");
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
               std::cout << "Cannot connect to " << node << std::endl;
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

/*--- Read Slow Control Event from FEBs to be put into data stream --------*/
INT read_sc_event(char *pevent, INT off)
{    
    cm_msg(MINFO, "switch_fe::read_sc_event()" , "Reading FEB SC");

    string bankname = ssfe[switch_id];
    string counterbankname = sscn[switch_id];
    string sorterbankname = ssso[switch_id];

    // create bank, pdata
    bk_init(pevent);
    DWORD *pdata;

    bk_create(pevent, bankname.c_str(), TID_FLOAT, (void **)&pdata);
    pdata = mufeb->fill_SSFE(pdata);
    bk_close(pevent,pdata);

    bk_create(pevent, counterbankname.c_str(), TID_INT, (void **)&pdata);
    pdata = fill_SSCN(pdata);
    bk_close(pevent, pdata);

    bk_create(pevent, sorterbankname.c_str(), TID_INT, (void **)&pdata);
    pdata = mufeb->fill_SSSO(pdata);
    bk_close(pevent, pdata);

    return bk_size(pevent);


}

/*--- Read Counters from SWBs to be put into data stream --------*/
DWORD * fill_SSCN(DWORD * pdata)
{
    // TODO: Could we get this from the feblist?
    odb cur_links_odb("/Equipment/Links/Settings/LinkMask");
    std::bitset<64> cur_link_active_from_odb = get_link_active_from_odb(cur_links_odb);

    // first read general counters
    *pdata++ = read_counters(SWB_STREAM_FIFO_FULL_PIXEL_CNT);
    *pdata++ = read_counters(SWB_BANK_BUILDER_IDLE_NOT_HEADER_PIXEL_CNT);
    *pdata++ = read_counters(SWB_BANK_BUILDER_RAM_FULL_PIXEL_CNT);
    *pdata++ = read_counters(SWB_BANK_BUILDER_TAG_FIFO_FULL_PIXEL_CNT);

    for(uint32_t i=0; i < N_FEBS[switch_id]; i++){

        *pdata++ = i;
        *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? (read_counters(SWB_LINK_FIFO_ALMOST_FULL_PIXEL_CNT | (i << 8))) : 0);
        *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? (read_counters(SWB_LINK_FIFO_FULL_PIXEL_CNT | (i << 8))) : 0);
        *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? (read_counters(SWB_SKIP_EVENT_PIXEL_CNT | (i << 8))) : 0);
        *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? (read_counters(SWB_EVENT_PIXEL_CNT | (i << 8))) : 0);
        *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? (read_counters(SWB_SUB_HEADER_PIXEL_CNT | (i << 8))) : 0);
        // TODO: What is the magic number here?
        *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? (0x7735940 - mufeb->ReadBackMergerRate(i)) : 0);
        *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? mufeb->ReadBackResetPhase(i) : 0);
        *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? mufeb->ReadBackTXReset(i) : 0);
    }
    return pdata;
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
    cm_msg(MINFO, "Mupix::read_mupix_sc_event()" , "Reading MuPix FEB SC");

//     create banks with LVDS counters
    string bankname = "PSLL";
    bk_init(pevent);
    DWORD *pdata;
    bk_create(pevent, bankname.c_str(), TID_INT, (void **)&pdata);
    pdata = mupixfeb->fill_PSLL(pdata, feblist->getPixelFEBs().size());
    bk_close(pevent, pdata);

//     TODO: implement bank PSLM
//     TODO: implement bank PSSH

    return bk_size(pevent);
}

//TODO: Get rid of this...
INT get_odb_value_by_string(const char *key_name){
    INT ODB_DATA, SIZE_ODB_DATA;
    SIZE_ODB_DATA = sizeof(ODB_DATA);
    db_get_value(hDB, 0, key_name, &ODB_DATA, &SIZE_ODB_DATA, TID_INT, 0);
    return ODB_DATA;
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
               std::cout << "Cannot connect to " << node << std::endl;
               return;
            }
            uint8_t power = power_odb[i];
           if(power)
               cout << "Switching on FEB " << slot << " in crate " << crate << endl;
           else
               cout << "Switching off FEB " << slot << " in crate " << crate << endl;

            mscb_write(fd, node, slot+CC_POWER_OFFSET,&power,sizeof(power));
            febpower[i] = power;
        }
    }
}

void sorterdelays_changed(odb o)
{
    cm_msg(MINFO, "sorterdelays_changed()" , "Sorterdelays!");
    std::vector<uint32_t> delays_odb = o;
    for(size_t i =0; i < sorterdelays.size(); i++){
        if(sorterdelays[i] != delays_odb[i]){
            mufeb->WriteSorterDelay(switch_id*MAX_FEBS_PER_SWITCHINGBOARD+i, delays_odb[i]);
            sorterdelays[i] = delays_odb[i];
        }
    }
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
              cm_msg(MERROR, "SciFiConfig" , "ASIC Configuration failed.");
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
   for(uint32_t link = 0; link < MAX_LINKS_PER_SWITCHINGBOARD; link++) {
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
   for(uint32_t i = 0; i < MAX_LINKS_PER_SWITCHINGBOARD; i++) {
      //if ((link_active_from_register >> i) & 0x1){
         mup->write_register_wait(RUN_NR_ADDR_REGISTER_W, uint32_t(i), 1000);
         uint32_t val=mup->read_register_ro(RUN_NR_REGISTER_R);
         cm_msg(MINFO,"switch_fe","Switching board %d, Link %d: PREP_ACK=%u STOP_ACK=%u RNo=0x%8.8x", switch_id, i, (val>>25)&1, (val>>24)&1,(val>>0)&0xffffff);
      //}
   }
}

// -- Helper functions
uint32_t read_counters(uint32_t write_value)
{
    mup->write_register(SWB_COUNTER_REGISTER_W, write_value);
    return mup->read_register_ro(SWB_COUNTER_REGISTER_R);
}

