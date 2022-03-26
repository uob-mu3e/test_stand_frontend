//#define FEB_ENABLE_REGISTER_LOW_W FEB_ENABLE_REGISTER_W
//#define RUN_NR_ACK_REGISTER_LOW_R RUN_NR_ACK_REGISTER_R


/********************************************************************\

  Name:         taken from switch_fe.cpp
  Created by:   Stefan Ritt
  Updated by:   Marius Koeppel, Konrad Briggl, Lukas Gerritzen, Niklaus Berger

  Contents:     Code for switching front-end

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
using namespace mu3ebanks;

/*-- Globals -------------------------------------------------------*/

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
INT read_link_sc_event(char *pevent, INT off);
//INT read_WMEM_event(char *pevent, INT off);
INT read_scifi_sc_event(char *pevent, INT off);
INT read_scitiles_sc_event(char *pevent, INT off);
INT read_mupix_sc_event(char *pevent, INT off);

DWORD * fill_SSCN(DWORD *);
DWORD * fill_SSPL(DWORD *);

void sc_settings_changed(odb o);
void switching_board_mask_changed(odb o);
void frontend_board_mask_changed(odb o);
void sorterdelays_changed(odb o);
void scifi_settings_changed(odb o);

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
INT init_febs();
INT init_scifi();
INT init_scitiles();
INT init_mupix();

// Here we choose which switching board we are
#include "OneSwitchingBoard.inc"
//#include "CentralSwitchingBoard.inc"
// Others to be written

/* Local state of sorter delays */
std::array<uint32_t, N_FEBS[switch_id]> sorterdelays{};


/*-- Dummy routines ------------------------------------------------*/

INT poll_event(INT, INT, BOOL)
{
    return 1;
}

INT interrupt_configure(INT, INT, POINTER_T)
{
    return 1;
}

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
    set_equipment_status(equipment[EQUIPMENT_ID::Switching].name, "Initializing...", "var(--myellow)");

    #ifdef MY_DEBUG
        odb::set_debug(true);
    #endif

    // create Settings structure in ODB
    setup_odb();


    // open mudaq
    #ifdef NO_SWITCHING_BOARD
        mup = new mudaq::DummyDmaMudaqDevice("/dev/mudaq0");
    #else
        mup = new mudaq::DmaMudaqDevice("/dev/mudaq0");
    #endif       
        
    INT status = init_mudaq(*mup);
    if (status != SUCCESS)
        return FE_ERR_DRIVER;

    //init febs (general)
    status = init_febs();
    if (status != SUCCESS)
        return FE_ERR_DRIVER;


    //init scifi
    if constexpr(has_scifi){
        status = init_scifi();
        if (status != SUCCESS)
            return FE_ERR_DRIVER;
    }

    
    //init scitiles
    if constexpr(has_tiles){
        status = init_scitiles();
        if (status != SUCCESS)
            return FE_ERR_DRIVER;
    }

    //init mupix
    if constexpr(has_pixels){
        status = init_mupix();
        if (status != SUCCESS)
            return FE_ERR_DRIVER;
    }

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
    setup_history();

    set_equipment_status(equipment[EQUIPMENT_ID::Switching].name, "OK", "var(--mgreen)");
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

    /* TODO: This needs a cleanup! */
    /* Default values for /Equipment/SwitchingX/Settings */
    odb settings = {
            {"Sorter Delay", zeroarr},
            // For this, switch_id has to be known at compile time (calls for a preprocessor macro or some constexpr magic, I guess)
            {namestr.c_str(), std::array<std::string, per_fe_SSFE_size*N_FEBS[switch_id]>()},
            {cntnamestr.c_str(), std::array<std::string, num_swb_counters_per_feb*N_FEBS[switch_id]+4>()},
            {sorternamestr.c_str(), std::array<std::string, per_fe_SSSO_size*N_FEBS[switch_id]>()}
    };

    create_ssfe_names_in_odb(settings,switch_id);
    create_ssso_names_in_odb(settings,switch_id);
    create_sscn_names_in_odb(settings,switch_id);

    string path_s = "/Equipment/" + eq_name + "/Settings";
    settings.connect(path_s, true);

    /* Clean up, move subdetector specific stuff */
    odb commands = {
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
            {"MupixChipToConfigure", 999}, // 999 means all
            {"MupixSetTDACConfig", false},
            {"MupixBoard", false},
            {"Sorter Zero Suppression Mupix", false},
            {"SciFiConfig", false},
            {"SciFiAllOff", false},
            {"SciFiTDCTest", false},
            {"SciTilesConfig", false},
            {"Reset Bypass Payload", 0},
            {"Reset Bypass Command", 0},
            {"Load Firmware", false},
            {"Firmware File",""},
            {"Firmware FEB ID",0},
            {"Firmware Is Emergency Image", false}
    };


    string path_c = "/Equipment/" + eq_name + "/Commands";
    commands.connect(path_c, true);


    /* Default values for /Equipment/SwitchingX/Variables */
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

    string path2 = "/Equipment/" + eq_name + "/Variables";
    sc_variables.connect(path2);

    string pllnamestring    = ssplnames[switch_id];
    string pllbankname      = sspl[switch_id];

    std::array<uint32_t, N_FEBS[switch_id]> verarray;
    verarray.fill(20);
    odb firmware_variables = {
        {"Arria V Firmware Version", std::array<uint32_t, N_FEBS[switch_id]>{}},
        {"Max 10 Firmware Version", std::array<uint32_t, N_FEBS[switch_id]>{}},
        {"FEB Version", verarray}
    };
    string path3 = "/Equipment/" + link_eq_name + "/Variables/FEBFirmware";
    firmware_variables.connect(path3);

    std::array<uint32_t, N_FEBS[switch_id]> febarray;
    febarray.fill(255);
    odb link_settings = {
        {"LinkMask", std::array<uint32_t,N_FEBS[switch_id]>{}},
        {"LinkFEB", febarray},
        {"FEBType", std::array<uint32_t, N_FEBS[switch_id]>{}},
        {"FEBName", std::array<std::string, N_FEBS[switch_id]>{}},
        {pllnamestring.c_str(), std::array<std::string, ssplsize>{}}
    };
    
    
    string path_ls = "/Equipment/" + link_eq_name + "/Settings";
    link_settings.connect(path_ls);

    create_sspl_names_in_odb(link_settings,switch_id);

    odb link_variables = {
        {"LinkStatus", std::array<uint32_t, N_FEBS[switch_id]>{}},
        {"BypassEnabled", std::array<uint32_t,N_FEBS[switch_id]>{}},
        {"RunState", std::array<uint32_t, N_FEBS[switch_id]>{}},
        {pllbankname.c_str(), std::array<uint32_t, ssplsize>{}}
    };
    string path_lv = "/Equipment/" + link_eq_name + "/Variables";
    link_variables.connect(path_lv);

    odb datapath_variables = {
            {"PLL locked", std::array<uint32_t, N_FEBS[switch_id]>{}},
            {"Buffer full", std::array<uint32_t, N_FEBS[switch_id]>{}},
            {"Frame desync", std::array<uint32_t, N_FEBS[switch_id]>{}},
            {"DPA locked", std::array<uint32_t, N_FEBS[switch_id]>{}},
            {"RX ready", std::array<uint32_t, N_FEBS[switch_id]>{}}
    };
    string path_dp = "/Equipment/" + link_eq_name + "/Variables/FEB datapath status";
    datapath_variables.connect(path_dp);

    // add custom pages to ODB
    odb custom("/Custom");
    custom["Switching&"] = "sc.html";
    custom["Links"] = "links.html";
    custom["Febs&"] = "febs.html";
    custom["DAQcounters&"] = "daqcounters.html";
    custom["Data Flow&"] = "dataflow.html";

    // Inculde the line below to set up the FEBs and their mapping for the 2021 integration run
    //#include "odb_feb_mapping_integration_run_2021.h"

    // Inculde the line below to set up the FEBs and their mapping for 2021 EDM run
    //#include "odb_feb_mapping_edm_run_2021.h"


}

void setup_history(){

    hs_define_panel(eq_name.c_str(), "All FEBs", {"Switching:Merger Timeout All FEBs"});

    //TODO: The 12 is the integration run number used to reduce clutter
    for(uint i= 0; i < 12; i++){
        std::string name("FEB"+std::to_string(i));
        std::vector<std::string> tnames;
        tnames.push_back(std::string(eq_name.c_str() + name + std::string(" Arria Temperature")));
        tnames.push_back(std::string(eq_name.c_str() + name + std::string(" MAX Temperature")));
        tnames.push_back(std::string(eq_name.c_str() + name + std::string(" SI1 Temperature")));
        tnames.push_back(std::string(eq_name.c_str() + name + std::string(" SI2 Temperature")));
        tnames.push_back(std::string(eq_name.c_str() + name + std::string(" ext Arria Temperature")));
        tnames.push_back(std::string(eq_name.c_str() + name + std::string(" DCDC Temperature")));
        tnames.push_back(std::string(eq_name.c_str() + name + std::string(" Firefly1 Temperature")));

       std::vector<std::string> vnames;
       vnames.push_back(std::string(eq_name.c_str() + name + std::string(" Voltage 1.1")));
       vnames.push_back(std::string(eq_name.c_str() + name + std::string(" Voltage 1.8")));
       vnames.push_back(std::string(eq_name.c_str() + name + std::string(" Voltage 2.5")));
       vnames.push_back(std::string(eq_name.c_str() + name + std::string(" Voltage 3.3")));
       vnames.push_back(std::string(eq_name.c_str() + name + std::string(" Voltage 20")));
       vnames.push_back(std::string(eq_name.c_str() + name + std::string(" Firefly1 Voltage")));

       std::vector<std::string> pnames;
       pnames.push_back(std::string(eq_name.c_str() + name + std::string(" Firefly1 RX1 Power")));
       pnames.push_back(std::string(eq_name.c_str() + name + std::string(" Firefly1 RX2 Power")));
       pnames.push_back(std::string(eq_name.c_str() + name + std::string(" Firefly1 RX3 Power")));
       pnames.push_back(std::string(eq_name.c_str() + name + std::string(" Firefly1 RX4 Power")));

       hs_define_panel(eq_name.c_str(),std::string(name + std::string(" Temperatures")).c_str(),tnames);
       hs_define_panel(eq_name.c_str(),std::string(name + std::string(" Voltages")).c_str(),vnames);
       hs_define_panel(eq_name.c_str(),std::string(name + std::string(" RX Power")).c_str(),pnames);
    }
}


void setup_watches(){
    //UI watch
    string path_c = "/Equipment/" + std::string(eq_name) + "/Commands";
    odb sc_variables(path_c);
    sc_variables.watch(sc_settings_changed);

    // watch if this switching board is enabled
    odb switch_mask("/Equipment/Clock Reset/Settings/SwitchingBoardMask");
    switch_mask.watch(switching_board_mask_changed);

    // watch if the mapping of FEBs to crates changed
    odb crates("/Equipment/FEBCrates/Settings");
    crates.watch(frontend_board_mask_changed);

    // watch if this links are enabled
    string path_l = "/Equipment/" + std::string(link_eq_name) + "/Settings/LinkMask";
    odb links_odb(path_l);
    links_odb.watch(frontend_board_mask_changed);

    // Watch for sorter delay changes
    string path_sd = "/Equipment/" + std::string(eq_name) + "/Settings/Sorter Delay";
    odb sorterdelay_settings(path_sd);
    sorterdelay_settings.watch(sorterdelays_changed);
}

void switching_board_mask_changed(odb o) {

    vector<INT> switching_board_mask = o;
    BOOL value = switching_board_mask[switch_id] > 0 ? true : false;

    std::string path = "/Equipment/" + std::string(equipment[0].name) + "/Common/Enabled";

    cm_msg(MINFO, "switching_board_mask_changed", "Switching board masking changed");
    cm_msg(MINFO, "switching_board_mask_changed", "For INT run we enable MuPix and SciFi Equipment");

    odb enabled(path);
    if(enabled == value) // no change
        return;
    
    for(int i = 0; i < NEQUIPMENT; i++) {
        std::string path = "/Equipment/" + std::string(equipment[i].name) + "/Common/Enabled";
        odb enabled(path);
        enabled = value;
        cm_msg(MINFO, "switching_board_mask_changed", "Set Equipment %s enabled to %d", equipment[i].name, value);
    }

    feblist->RebuildFEBList();
    mufeb->ReadFirmwareVersionsToODB();
}

void frontend_board_mask_changed(odb) {
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

INT init_febs() {

    //set link enables so slow control can pass
    string path_l = "/Equipment/" + link_eq_name + "/Settings/LinkMask";
    odb cur_links_odb(path_l);
    set_feb_enable(get_link_active_from_odb(cur_links_odb));
    
    // Create the FEB List
    feblist = new FEBList(switch_id, link_eq_name);

    //init SC
    feb_sc->FEBsc_resetSecondary();


    // switching setup part
    mufeb = new  MuFEB(*feb_sc,
                        feblist->getActiveFEBs(),
                        feblist->getFEBMask(),
                        equipment[EQUIPMENT_ID::Switching].name,
                        equipment[EQUIPMENT_ID::Links].name,
                        switch_id);

    //init all values on FEB
    mufeb->WriteFEBIDs();

    // Get all the relevant firmware versions
    mufeb->ReadFirmwareVersionsToODB();

    // Init sorter delays -- Should be subdetector specific??
    string path_sd = "/Equipment/" + std::string(eq_name) + "/Settings/Sorter Delay";
    odb sorterdelay_odb(path_sd);
    sorterdelays_changed(sorterdelay_odb);

    return SUCCESS;
}


INT init_scifi() {

    // SciFi setup part
    set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Initializing...", "var(--myellow)");
    scififeb = new SciFiFEB(*feb_sc,
                     feblist->getSciFiFEBs(),
                     feblist->getSciFiFEBMask(),
                     equipment[EQUIPMENT_ID::Switching].name,
                     equipment[EQUIPMENT_ID::Links].name,
                     equipment[EQUIPMENT_ID::SciFi].name,
                      switch_id); //create FEB interface signleton for scifi

    
    int status=mutrig::midasODB::setup_db("/Equipment/" + scifi_eq_name,*scififeb);
    if(status != SUCCESS){
        set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Start up failed", "var(--mred)");
        return status;
    }
    //init all values on FEB
    scififeb->WriteAll();
    scififeb->WriteFEBIDs();

    
    //set custom page
    odb custom("/Custom");
    custom["SciFi-ASICs"] = "mutrigTdc.html";
    //

    // setup watches
    if ( scififeb->GetNumASICs() != 0 ){
        odb scifi_setting("/Equipment/" + scifi_eq_name + "/Settings/Daq");
        scifi_setting.watch(scifi_settings_changed);
    }

    set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Ok", "var(--mgreen)");

    return SUCCESS;
}

INT init_scitiles() {

    
    //SciTiles setup part
    set_equipment_status(equipment[EQUIPMENT_ID::Tiles].name, "Initializing...", "var(--myellow)");
    tilefeb = new TilesFEB(*feb_sc,
                     feblist->getTileFEBs(),
                     feblist->getTileFEBMask(),
                     equipment[EQUIPMENT_ID::Switching].name,
                     equipment[EQUIPMENT_ID::Links].name,
                     equipment[EQUIPMENT_ID::Tiles].name,
                      switch_id); //create FEB interface signleton for scitiles
    int status=mutrig::midasODB::setup_db("/Equipment/" + tile_eq_name, *tilefeb);
    if(status != SUCCESS){
        set_equipment_status(equipment[EQUIPMENT_ID::Tiles].name, "Start up failed", "var(--mred)");
        return status;
    }
    //init all values on FEB
    tilefeb->WriteAll();
    tilefeb->WriteFEBIDs();

    set_equipment_status(equipment[EQUIPMENT_ID::Tiles].name, "Ok", "var(--mgreen)");

    //set custom page
    odb custom("/Custom");
    custom["SciTiles-ASICs&"] = "tile_custompage.html";
    
    return SUCCESS;
}


INT init_mupix() {


    //Mupix setup part
    set_equipment_status(equipment[EQUIPMENT_ID::Pixels].name, "Initializing...", "var(--myellow)");
    mupixfeb = new MupixFEB(*feb_sc,
                     feblist->getPixelFEBs(),
                     feblist->getPixelFEBMask(),
                     equipment[EQUIPMENT_ID::Switching].name,
                     equipment[EQUIPMENT_ID::Links].name,
                     equipment[EQUIPMENT_ID::Pixels].name,
                     switch_id); //create FEB interface signleton for mupix

    int status=mupix::midasODB::setup_db("/Equipment/" + pixel_eq_name, *mupixfeb, switch_id, true, false);//true);
    if(status != SUCCESS){
        set_equipment_status(equipment[EQUIPMENT_ID::Pixels].name, "Start up failed", "var(--mred)");
        return status;
    }
    //init all values on FEB
    mupixfeb->WriteFEBIDs();
    
    set_equipment_status(equipment[EQUIPMENT_ID::Pixels].name, "Ok", "var(--mgreen)");

    //TODO: set custom page
    odb custom("/Custom");
    custom["Pixel Control"] = "pixel_tracker.html";
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

INT begin_of_run(INT run_number, char *)
{
    for(int i = 0; i < NEQUIPMENT; i++) 
        set_equipment_status(equipment[i].name, "Starting Run", "var(--morange)");

try{ // TODO: What can throw here?? Why?? Is there another way to handle this??

   /* Set new run number */
   mup->write_register(RUN_NR_REGISTER_W, run_number);
   /* Reset acknowledge/end of run seen registers before start of run */
   uint32_t start_setup = 0;
   start_setup = SET_RESET_BIT_RUN_START_ACK(start_setup);
   start_setup = SET_RESET_BIT_RUN_END_ACK(start_setup);
   mup->write_register_wait(RESET_REGISTER_W, start_setup, 1000);
   mup->write_register(RESET_REGISTER_W, 0x0);

   /* get link active from odb. */
    string path_l = "/Equipment/" + std::string(link_eq_name) + "/Settings/LinkMask";
    odb cur_links_odb(path_l);
    uint64_t link_active_from_odb = get_link_active_from_odb(cur_links_odb);

   //last preparations
   mufeb->ResetAllCounters();


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
       feb_sc->FEB_broadcast(RESET_PAYLOAD_REGISTER_RW, valueRB); //run number
       valueRB= ((1<<RESET_BYPASS_BIT_ENABLE) |(1<<RESET_BYPASS_BIT_REQUEST)) | 0x10;
       feb_sc->FEB_broadcast(RUN_STATE_RESET_BYPASS_REGISTER_RW, valueRB); //run prep command
       valueRB= 0xbcbcbcbc;
       feb_sc->FEB_broadcast(RESET_PAYLOAD_REGISTER_RW, valueRB); //reset payload
       valueRB= (1<<RESET_BYPASS_BIT_ENABLE) | 0x00;
       feb_sc->FEB_broadcast(RUN_STATE_RESET_BYPASS_REGISTER_RW, valueRB); //reset command
   }else{
       /* send run prepare signal via CR system */
       // TODO: Move to odbxx
       feb_sc->FEB_broadcast(RUN_STATE_RESET_BYPASS_REGISTER_RW, 0); // disable reset bypass for all connected febs
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
   // TODO: test this part of checking the run number
   do{
    timeout_cnt--;
    link_active_from_register = get_runstart_ack();
    printf("%u  %lx  %lx\n",timeout_cnt, link_active_from_odb, link_active_from_register);
    usleep(10000);
   }while( (link_active_from_register & link_active_from_odb) != link_active_from_odb && (timeout_cnt > 0));

   if(timeout_cnt==0) {
      cm_msg(MERROR,"switch_fe","Run number mismatch on run %d", run_number);
      print_ack_state();
      return CM_TRANSITION_CANCELED;
   }

   set_equipment_status(equipment[EQUIPMENT_ID::SciFi].name, "Scintillating...", "mblue");
   set_equipment_status(equipment[EQUIPMENT_ID::Pixels].name, "Running...", "mgreen");
   return CM_SUCCESS;
}catch(...){return CM_TRANSITION_CANCELED;}
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT, char *)
{

try{
    /* get link active from odb. */
   string path_l = "/Equipment/" + std::string(link_eq_name) + "/Settings/LinkMask";
   odb cur_links_odb(path_l);
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
      set_equipment_status(equipment[EQUIPMENT_ID::Pixels].name, "Transition interrupted", "var(--morange)");
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
   set_equipment_status(equipment[EQUIPMENT_ID::Pixels].name, "Ok", "var(--mgreen)");
   return CM_SUCCESS;
}catch(...){return CM_TRANSITION_CANCELED;}
}

/*-- Pause Run -----------------------------------------------------*/

INT pause_run(INT, char *)
{
   return CM_SUCCESS;
}

/*-- Resume Run ----------------------------------------------------*/

INT resume_run(INT, char *)
{
   return CM_SUCCESS;
}



/*--- Read Slow Control Event from FEBs to be put into data stream --------*/
INT read_sc_event(char *pevent, INT)
{    
    // Do this in link SC?
    /*auto vec = mufeb->CheckLinks(N_FEBS[switch_id]);
    string path_l = "/Equipment/" + std::string(link_eq_name) + "/Variables/LinkStatus";
    odb linkstatus_odb(path_l);
    linkstatus_odb = vec;*/

    //cm_msg(MINFO, "switch_fe::read_sc_event()" , "Reading FEB SC");
    mufeb->ReadBackAllRunState();

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
    std::bitset<64> cur_link_active_from_odb = feblist->getLinkMask();

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
        if(feblist->getFEBatPort(i)){
            auto feb = feblist->getFEBatPort(i).value();
            if(feb.GetLinkStatus().LinkIsOK()){
                *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? mufeb->ReadBackMergerRate(feb) : 0);
                *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? mufeb->ReadBackResetPhase(feb) : 0);
                *pdata++ = (cur_link_active_from_odb.test(i) == 1 ? mufeb->ReadBackTXReset(feb) : 0);
            } else {
                *pdata++ = 0;
                *pdata++ = 0;
                *pdata++ = 0;
            }
        } else {
            *pdata++ = 0;
            *pdata++ = 0;
            *pdata++ = 0;
        }
    }
    return pdata;
}

/*--- Read Slow Control Event from Link status to be put into data stream --------*/
INT read_link_sc_event(char *pevent, INT)
{    
    auto vec = mufeb->CheckLinks(N_FEBS[switch_id]);
    string path_l = "/Equipment/" + std::string(link_eq_name) + "/Variables/LinkStatus";
    odb linkstatus_odb(path_l);
    linkstatus_odb = vec;

    string pllbankname = sspl[switch_id];

    // create bank, pdata
    bk_init(pevent);
    DWORD *pdata;

    bk_create(pevent, pllbankname.c_str(), TID_UINT32, (void **)&pdata);
    pdata = fill_SSPL(pdata);
    bk_close(pevent,pdata);

    return bk_size(pevent);
}

DWORD * fill_SSPL(DWORD * pdata)
{
    *pdata++ = mup->read_register_ro(CNT_PLL_156_REGISTER_R);
    *pdata++ = mup->read_register_ro(CNT_PLL_250_REGISTER_R);
    //TODO: Uncomment once registers are defined
    *pdata++ = 0;//mup->read_register_ro(LINK_LOCKED_LOW_REGISTER_R);
    *pdata++ = 0;//mup->read_register_ro(LINK_LOCKED_HIGH_REGISTER_R);

    return pdata;
}


/*--- Read Slow Control Event from SciFi to be put into data stream --------*/

INT read_scifi_sc_event(char *pevent, INT){

    //TODO: Make this more proper: move this to class driver routine and make functions not writing to ODB all the time (only on update).
    //Add readout function for this one that gets data from class variables and writes midas banks
    scififeb->ReadBackAllCounters();
    scififeb->ReadBackAllRunState();
    scififeb->ReadBackAllDatapathStatus();
    return 0;
}

/*--- Read Slow Control Event from SciTiles to be put into data stream --------*/

INT read_scitiles_sc_event(char *pevent, INT){
    
    //TODO: Make this more proper: move this to class driver routine and make functions not writing to ODB all the time (only on update).
    //Add readout function for this one that gets data from class variables and writes midas banks
    tilefeb->ReadBackAllCounters();
    tilefeb->ReadBackAllRunState();
    tilefeb->ReadBackAllDatapathStatus();
    return 0;
}

/*--- Read Slow Control Event from Mupix to be put into data stream --------*/

INT read_mupix_sc_event(char *pevent, INT){
    //cm_msg(MINFO, "Mupix::read_mupix_sc_event()" , "Reading MuPix FEB SC");
    
    // create banks with LVDS counters & status
    bk_init(pevent);
    DWORD *pdata;
    string lvdsbankname = psls[switch_id];

    bk_create(pevent, lvdsbankname.c_str(), TID_INT, (void **)&pdata);
    pdata = mupixfeb->fill_PSLS(pdata);
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


// TODO: Should go to the MuFEB (or MuPixFEB)
void sorterdelays_changed(odb o)
{
    cm_msg(MINFO, "sorterdelays_changed()" , "Sorterdelays!");
    std::vector<uint32_t> delays_odb = o;
    assert(delays_odb.size() >= feblist->getActiveFEBs().size());
    // TODO: Fix use of raw index!
    for(uint32_t i=0; i < N_FEBS[switch_id]; i++){
        if(sorterdelays[i] != delays_odb[i]){
            auto feb = feblist->getFEBatPort(i);
            if(feb)
                mufeb->WriteSorterDelay(feb.value(), delays_odb[i]);
            sorterdelays[i] = delays_odb[i];
        }
    }
}

// TODO: this is also done in the mutrig class via a lambda function
// but this is not really working at the moment change later
void scifi_settings_changed(odb o)
{
    std::string name = o.get_name();
    bool value = o;

    if (value)
        cm_msg(MINFO, "MutrigFEB::on_settings_changed", "Setting changed (%s)", name.c_str());

    if ( name == "reset_datapath" && o ) {
        if (value) {
            for ( auto FEB : scififeb->getFEBs() ) {
                if (!FEB.IsScEnabled()) continue; //skip disabled
                if (FEB.SB_Number() != scififeb->getSB_number()) continue; //skip commands not for me
                
                scififeb->DataPathReset(FEB);
            }
            o = false;
        }
    }

    if ( name == "reset_asics" && o ) {
        if (value) {
            for ( auto FEB : scififeb->getFEBs() ) {
                if (!FEB.IsScEnabled()) continue; //skip disabled
                if (FEB.SB_Number() != scififeb->getSB_number()) continue; //skip commands not for me
                scififeb->chipReset(FEB);
            }
            o = false;
        }
    }

    if ( name == "reset_lvds" && o ) {
        if (value) {
            for ( auto FEB : scififeb->getFEBs() ) {
                if (!FEB.IsScEnabled()) continue; //skip disabled
                if (FEB.SB_Number() != scififeb->getSB_number()) continue; //skip commands not for me
                scififeb->LVDS_RX_Reset(FEB);
            }
            o = false;
        }
    }

    if ( name == "reset_counters" && o ) {
        if (value) {
            scififeb->ResetAllCounters();
            o = false;
        }
    }
}


/*--- Called whenever settings have changed ------------------------*/

void sc_settings_changed(odb o)
{
    std::string name = o.get_name();

#ifdef MY_DEBUG
    dummy_mudaq::DummyMudaqDevice & mu = *mup;
#else
    mudaq::MudaqDevice & mu = *mup;
#endif

    /* Unfortunately we cannot do a string matching case statement.
    Nested if would work, but be very nested. We choose ifs with returns*/


    if (name == "Reset SC Main" && o) {
        bool value = o;
        if(value){
             feb_sc->FEBsc_resetMain();
             o = false;
        }
        return;
    }

    if (name == "Reset SC Secondary" && o) {
        bool value = o;
        if(value){
            feb_sc->FEBsc_resetSecondary();
             o = false;
        }
        return;
    }


   if (name == "Write" && o) {
       odb fpgaid("Equipment/Switching/Variables/FPGA_ID_WRITE");
       odb startaddr("Equipment/Switching/Variables/START_ADD_WRITE");
       odb writesize("Equipment/Switching/Variables/DATA_WRITE_SIZE");
       odb dataarray("Equipment/Switching/Variables/DATA_WRITE");
       vector<uint32_t> dataarrayv = dataarray;
        auto feb = feblist->getFEBatPort(fpgaid);
        if(feb)
            feb_sc->FEB_write(feb.value(), startaddr, dataarrayv);
        else
            cm_msg(MERROR, "sc_settings_changed - Write", "FEB does not exist");
        o = false;
        return;
   }

   if (name == "Read" && o) {
       odb fpgaid("Equipment/Switching/Variables/FPGA_ID_READ");
       odb startaddr("Equipment/Switching/Variables/START_ADD_READ");
       odb readsize("Equipment/Switching/Variables/LENGTH_READ");
       vector<uint32_t> data(readsize);
        auto feb = feblist->getFEBatPort(fpgaid);
        if(feb)
            feb_sc->FEB_read(feb.value(), startaddr, data);
        else
            cm_msg(MERROR, "sc_settings_changed - Read", "FEB does not exist");
       // TODO: Do something with the value we read...
       o = false;
       return;
   }

   if (name == "Single Write" && o) {
        odb fpgaid("Equipment/Switching/Variables/FPGA_ID_WRITE");
        odb datawrite("Equipment/Switching/Variables/SINGLE_DATA_WRITE");
        odb startaddr("Equipment/Switching/Variables/START_ADD_WRITE");

        uint32_t data = datawrite;
        auto feb = feblist->getFEBatPort(fpgaid);
        if(feb)
            feb_sc->FEB_write(feb.value(), startaddr, data);
        else
            cm_msg(MERROR, "sc_settings_changed - Single Write", "FEB does not exist");
        o = false;
        return;
    }

    if (name == "Read WM" && o) {
        INT WM_START_ADD=get_odb_value_by_string("Equipment/Switching/Variables/WM_START_ADD");
        INT WM_LENGTH=get_odb_value_by_string("Equipment/Switching/Variables/WM_LENGTH");
        INT WM_DATA;
        INT SIZE_WM_DATA = sizeof(WM_DATA);

        // TODO: Change to ODBXX
        HNDLE key_WM_DATA;
        db_find_key(hDB, 0, "Equipment/Switching/Variables/WM_DATA", &key_WM_DATA);
        db_set_num_values(hDB, key_WM_DATA, WM_LENGTH);
        for (int i = 0; i < WM_LENGTH; i++) {
            WM_DATA = mu.read_memory_rw((uint32_t) WM_START_ADD + i);
            db_set_value_index(hDB, 0, "Equipment/Switching/Variables/WM_DATA", &WM_DATA, SIZE_WM_DATA, i, TID_INT, FALSE);
        }

        o = false;
        return;
    }

    if (name == "Read RM" && o) {
        cm_msg(MINFO, "sc_settings_changed", "Execute Read RM");

        INT RM_START_ADD=get_odb_value_by_string("Equipment/Switching/Variables/RM_START_ADD");
        INT RM_LENGTH=get_odb_value_by_string("Equipment/Switching/Variables/RM_LENGTH");
        INT RM_DATA;
        INT SIZE_RM_DATA = sizeof(RM_DATA);
        // TODO: Change to ODBXX
        HNDLE key_WM_DATA;
        db_find_key(hDB, 0, "Equipment/Switching/Variables/RM_DATA", &key_WM_DATA);
        db_set_num_values(hDB, key_WM_DATA, RM_LENGTH);
        for (int i = 0; i < RM_LENGTH; i++) {
            RM_DATA = mu.read_memory_ro((uint32_t) RM_START_ADD + i);
            db_set_value_index(hDB, 0, "Equipment/Switching/Variables/RM_DATA", &RM_DATA, SIZE_RM_DATA, i, TID_INT, FALSE);
        }

        o = false;
        return;
    }

    if (name == "Last RM ADD" && o) {

        char STR_LAST_RM_ADD[128];
        sprintf(STR_LAST_RM_ADD,"Equipment/Switching/Variables/LAST_RM_ADD");
        INT NEW_LAST_RM_ADD = mu.read_register_ro(MEM_WRITEADDR_LOW_REGISTER_R);
        INT SIZE_NEW_LAST_RM_ADD;
        SIZE_NEW_LAST_RM_ADD = sizeof(NEW_LAST_RM_ADD);
        db_set_value(hDB, 0, STR_LAST_RM_ADD, &NEW_LAST_RM_ADD, SIZE_NEW_LAST_RM_ADD, 1, TID_INT);
        
        o = false;
        return;
    }

    if (name == "SciFiConfig" && o) {
          int status=scififeb->ConfigureASICs();
          if(status!=SUCCESS){ 
              cm_msg(MERROR, "SciFiConfig" , "ASIC Configuration failed.");
         	//TODO: what to do? 
          }
       o = false;
       return;
    }
    if (name == "SciFiAllOff" && o) {
        cm_msg(MERROR, "SciFiAllOff", "Configuring all SciFi ASICs in All Off mode.");
        int status=scififeb->ConfigureASICsAllOff();
        if(status!=SUCCESS){
            cm_msg(MERROR, "SciFiAllOff" , "ASIC all off configuration failed. Return value was %d, expected %d.", status, SUCCESS);
            //TODO: what to do?
        }
       o = false;
       return;
    }
    if (name == "SciFiTDCTest") {
          int status=scififeb->ChangeTDCTest(o);
          if(status!=SUCCESS){
              cm_msg(MERROR, "SciFiConfig" , "Changing SciFi test pulses failed");
          }
          return;
    }
    if (name == "SciTilesConfig" && o) {
          int status=tilefeb->ConfigureASICs();
          if(status!=SUCCESS){
         	//TODO: what to do?
          }
      o = false;
      return;
    }
    if (name == "MupixConfig" && o) {
          int status=mupixfeb->ConfigureASICs();
          if(status!=SUCCESS){ 
         	//TODO: what to do? 
          }
      o = false;
      return;
    }
    if (name == "Reset Bypass Command") {
         uint32_t command = o;

	  if((command&0xff) == 0) return;
      uint32_t payload = odb("/Equipment/Switching/Settings/Reset Bypass Payload");

	  cm_msg(MINFO, "sc_settings_changed", "Reset Bypass Command %d, payload %d", command, payload);

        // TODO: get rid of hardcoded addresses
        feb_sc->FEB_broadcast(0xf5, payload);
        feb_sc->FEB_broadcast(0xf4, command);
        // reset payload and command TODO: Is this needed?
        payload=0xbcbcbcbc;
        command=0;
        feb_sc->FEB_broadcast(0xf5, payload);
        feb_sc->FEB_broadcast(0xf4, command);
        //reset odb flag
          command=command&(1<<8);
          o = command;
          return;
    }
    if (name == "Sorter Zero Suppression Mupix") {
        if (o) {
            cm_msg(MINFO, "sc_settings_changed", "Sorter Zero Suppression Mupix on");
            feb_sc->FEB_broadcast(MP_SORTER_ZERO_SUPPRESSION_REGISTER_W, 0x1);
        } else {
            cm_msg(MINFO, "sc_settings_changed", "Sorter Zero Suppression Mupix off");
            feb_sc->FEB_broadcast(MP_SORTER_ZERO_SUPPRESSION_REGISTER_W, 0x0);
        }
        return;
    }
    if (name == "Load Firmware" && o) {
        cm_msg(MINFO, "sc_settings_changed", "Load firmware triggered");
        string fname = odb("/Equipment/Switching/Settings/Firmware File");
        uint32_t id = odb("/Equipment/Switching/Settings/Firmware FEB ID");
        bool emergency = odb("/Equipment/Switching/Settings/Firmware Is Emergency Image");

        auto feb = feblist->getFEBatPort(id);
        if(feb)
            mufeb->LoadFirmware(fname,feb.value(), emergency);
        else
            cm_msg(MERROR, "sc_settings_changed - Single Write", "FEB does not exist");
        o = false;
        return;
    }

}

//--------------- Link related settings
//

uint64_t get_link_active_from_odb(odb o){

   /* get link active from odb */
   uint64_t link_active_from_odb = 0;
   for(uint32_t link = 0; link < MAX_LINKS_PER_SWITCHINGBOARD; link++) {
      int cur_mask = o[link];
      if((cur_mask == FEBLINKMASK::ON) || (cur_mask == FEBLINKMASK::DataOn)){
        //a standard FEB link (SC and data) is considered enabled if RX and TX are. 
	    //a secondary FEB link (only data) is enabled if RX is.
        link_active_from_odb += (1 << link);
      }
   }
   return link_active_from_odb;
}

void set_feb_enable(uint64_t enablebits){
   //mup->write_register(FEB_ENABLE_REGISTER_HIGH_W, enablebits >> 32); TODO make 64 bits
   mup->write_register(FEB_ENABLE_REGISTER_W,  enablebits & 0xFFFFFFFF);
}

uint64_t get_runstart_ack(){
   uint64_t reg = mup->read_register_ro(RUN_NR_ACK_REGISTER_R);
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

