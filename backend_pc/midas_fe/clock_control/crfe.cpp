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
#include <string>
#include <vector>
#include "midas.h"
#include "mfe.h"
#include "history.h"
#include "odbxx.h"


#include "clockboard_bypass.h"
#include "clockboard_a10.h"
#include "clockboard.h"
#include "reset_protocol.h"
#include "link_constants.h"

#include "missing_hardware.h"

using std::cout;
using std::endl;
using std::hex;
using std::string;
using std::to_string;
using std::vector;

using midas::odb;


/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "CR Frontend";
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

/* Size of the CRT1 bank in DWORDS*/
const INT CRT1SIZE = 100;

// Clock board interface
clockboard * cb;
/*-- Function declarations -----------------------------------------*/

INT read_cr_event(char *pevent, INT off);
INT read_link_event(char *pevent, INT off);
void cr_settings_changed(odb);
void link_settings_changed(odb);
void prepare_run_on_request(odb);

void setup_odb();
void setup_watches();
void setup_history();
void setup_alarms();

/*-- Equipment list ------------------------------------------------*/
EQUIPMENT equipment[] = {

   {"Clock Reset",              /* equipment name */
    {101, 0,                     /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,               /* equipment type */
     0,                         /* event source */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_RUNNING | RO_STOPPED | RO_ODB,        /* read while running and stopped but not at transitions and update ODB */
     10000,                     /* read every 10 sec */
     0,                         /* stop run after this event limit */
     0,                         /* number of sub events */
     1,                         /* log history every event */
     "", "", ""} ,
    read_cr_event,              /* readout routine */
   },

    {"Links",              /* equipment name */
     {103, 0,                     /* event ID, trigger mask */
      "SYSTEM",                  /* event buffer */
      EQ_PERIODIC,               /* equipment type */
      0,                         /* event source */
      "MIDAS",                   /* format */
      TRUE,                      /* enabled */
      RO_RUNNING | RO_STOPPED | RO_ODB,        /* read while running and stopped but not at transitions and update ODB */
      10000,                     /* read every 10 sec */
      0,                         /* stop run after this event limit */
      0,                         /* number of sub events */
      1,                         /* log history every event */
      "", "", ""} ,
     read_link_event,              /* readout routine */
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
    setup_odb();
    setup_watches();
    setup_history();
    setup_alarms();

    odb settings;
    settings.connect("/Equipment/Clock Reset/Settings");
    std::string ip = settings["IP"];
    int port       = settings["Port"];

    cout << "IP: " << ip << " port: " << port << endl;



    #ifdef NO_CLOCK_BOX
        cm_msg(MINFO, "frontend_init", "Using clock board bypass for reset commands to FEB");
        cb = new clockboard_bypass(ip, port);
    #else
        #ifdef A10_EMULATED_CLOCK_BOX
            cb = new clockboard_a10(ip, port);
        #else
            if(ip=="0.0.0.0"){
                cm_msg(MINFO, "frontend_init", "Using clock board bypass for reset commands to FEB");
                 cb = new clockboard_bypass(ip, port);
            }else{
                cb = new clockboard(ip, port);
            }
        #endif;
    #endif

   if(!cb->isConnected())
        return CM_TIMEOUT;

   int clkdisable = settings["TX_CLK_MASK"];
   int rstdisable = settings["TX_RST_MASK"];
   int clkinvert  = settings["TX_CLK_INVERT_MASK"];
   int rstinvert  = settings["TX_RST_INVERT_MASK"];

   cb->init_clockboard(clkinvert, rstinvert, clkdisable, rstdisable);

   odb variables;
   variables.connect("/Equipment/Clock Reset/Variables");
   // check which daughter cards are equipped
   uint32_t daughters = cb->daughters_present();
   variables["Daughters Present"] = daughters;

   // check which fireflys are present
   vector<uint32_t> ffs(clockboard::MAXNDAUGHTER,0);
   for(uint8_t i=0; i < clockboard::MAXNDAUGHTER; i++){
       if(daughters & (1<<i)){
           for(uint8_t j =0; j < clockboard::MAXFIREFLYPERDAUGTHER; j++){
               ffs[i] |=  ((uint32_t)(cb->firefly_present(i,j))) << j;
           }
        }
    }

   variables["Fireflys Present"] = ffs;

   //Set our transition sequence. The default is 500.
   cm_set_transition_sequence(TR_START, 500);

    //Set our transition sequence. The default is 500. Setting it
    // to 100 means we are called BEFORE most other clients.
    cm_set_transition_sequence(TR_STOP, 100);

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
    // set equipment status for status web page
    set_equipment_status("Clock Reset", "Starting run", "yellowLight");
    cb->write_command("Sync");

    cb->write_command("Start Run");
    set_equipment_status("Clock Reset", "Ok", "greenLight");

   return CM_SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT run_number, char *error)
{

   cb->write_command("End Run");
   set_equipment_status("Clock Reset", "Run stopped", "yellowLight");
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

INT read_cr_event(char *pevent, INT off [[maybe_unused]])
{
#ifdef NO_CLOCK_BOX
        return 0;
#endif;

#ifdef A10_EMULATED_CLOCK_BOX
    return 0;
#endif;

    if(!cb->isConnected()){
        // terminate!
        cm_msg(MERROR, "read_cr_event", "Connection to clock board lost");
        cm_disconnect_experiment();
        return 0;
   }

    odb variables;
    variables.connect("/Equipment/Clock Reset/Variables");

    uint32_t daughters = variables["Daughters Present"];
    vector<uint32_t> fireflys = variables["Fireflys Present"];

   bk_init32a(pevent);

   float *pdata;
   bk_create(pevent, "CRT1", TID_FLOAT, (void **)&pdata);

   *pdata++ = cb->read_mother_board_current();
   *pdata++ = cb->read_mother_board_voltage();

   *pdata++= ((cb->read_rx_firefly_alarms()) << 16) + cb->read_rx_firefly_los();
   *pdata++ = cb->read_rx_firefly_temp();
   *pdata++ = cb->read_rx_firefly_voltage();

   *pdata++= ((cb->read_tx_clk_firefly_alarms()) << 16) + cb->read_tx_clk_firefly_lf();
   *pdata++ = cb->read_tx_clk_firefly_temp();
   *pdata++ = cb->read_tx_clk_firefly_voltage();

   *pdata++= ((cb->read_tx_rst_firefly_alarms()) << 16) + cb->read_tx_rst_firefly_lf();
   *pdata++ = cb->read_tx_rst_firefly_temp();
   *pdata++ = cb->read_tx_rst_firefly_voltage();

   *pdata++ = cb->read_fan_current();

   for(uint8_t i=0;i < clockboard::MAXNDAUGHTER; i++){
       if(daughters & (1<<i)){
           *pdata++ = cb->read_daughter_board_current(i);
           *pdata++ = cb->read_daughter_board_voltage(i);
           for(uint8_t j=0;j < clockboard::MAXFIREFLYPERDAUGTHER; j++){
               if(fireflys[i] & (1<<j)){
                   *pdata++= ((cb->read_tx_firefly_alarms(i,j)) << 16) + cb->read_tx_firefly_lf(i,j);
                   *pdata++ = cb->read_tx_firefly_temp(i,j);
                   *pdata++ = cb->read_tx_firefly_voltage(i,j);
               } else {
                   *pdata++ = -1.0;
                   *pdata++ = -1.0;
                   *pdata++ = -1.0;
               }
           }
       } else {
           *pdata++ = -1.0;
           *pdata++ = -1.0;
           for(uint8_t j=0;j < clockboard::MAXFIREFLYPERDAUGTHER; j++){
               *pdata++ = -1.0;
               *pdata++ = -1.0;
               *pdata++ = -1.0;
           }
       }
   }

   bk_close(pevent, pdata);

   return bk_size(pevent);
}

/*--- Link status Event to be put into data stream --------*/

INT read_link_event(char *pevent, INT off [[maybe_unused]])
{
    bk_init32a(pevent);
    return bk_size(pevent);
}

/*--- Called whenever settings have changed ------------------------*/

void cr_settings_changed(odb o)
{
   std::string name = o.get_name();

   BOOL addressed = false;

    if (name == "Active") {
        bool value = o;
        cm_msg(MINFO, "cr_settings_changed", "Set active to %d", value);
        // TODO: propagate to hardware
    }

    if (name == "RX_MASK") {
      int value = o;
      cm_msg(MINFO, "cr_settings_changed", "Set RX_MASK to %x", value);
      cb->disable_rx_channels(value);
   }

   if (name == "TX_MASK") {
        vector<int> value = o;
        for(int i=0; i < clockboard::MAXFIREFLY; i++){
            cm_msg(MINFO, "cr_settings_changed", "Set TX_MASK[%i] to %x", i, value[i]);
            cb->disable_tx_channels(i/3,i%3,value[i]);
        }
   }

    if (name == "TX_CLK_MASK") {
      int value = o;
      cm_msg(MINFO, "cr_settings_changed", "Set TX_CLK_MASK to %x", value);
      cb->disable_tx_clk_channels(value);
   }

    if (name == "TX_RST_MASK") {
      int value = o;
      cm_msg(MINFO, "cr_settings_changed", "Set TX_RST_MASK to %x", value);
      cb->disable_tx_rst_channels(value);
   }

    if (name == "TX_CLK_INVERT_MASK") {
      int value = o;
      cm_msg(MINFO, "cr_settings_changed", "Set TX_CLK_INVERT_MASK to %x", value);
      cb->invert_tx_clk_channels(value);
   }

    if (name == "TX_RST_INVERT_MASK") {
      int value = o;
      cm_msg(MINFO, "cr_settings_changed", "Set TX_RST_INVERT_MASK to %x", value);
      cb->invert_tx_rst_channels(value);
   }

    if (name == "Addressed") {
      addressed = o;
      cm_msg(MINFO, "cr_settings_changed", "Set addressed to %d", addressed);
      // TODO: propagate to hardware
   }

   auto it = cb->reset_protocol.commands.find(name);

   if(it != cb->reset_protocol.commands.end()){

       odb settings("/Equipment/Clock Reset/Settings");
       addressed = settings["Addressed"];
       int address =0;
       if(addressed){
           address = settings["FEBAddress"];
           cm_msg(MINFO, "cr_settings_changed", "address reset command to addr:%d", address);
       }

       // Easy case are commands without payload
       if(!(it->second.has_payload)){
           BOOL value = o;
           if (value) {
              cm_msg(MINFO, "cr_settings_changed", "Execute1 %s", name.c_str());
              if(address)
                cb->write_command(name,0,address);
              else
                 cb->write_command(name);
              o = false; //TODO: Check if this works...
           }
       } else {
           // Run prepare needs the run number
           if (name == "Run Prepare") {
              BOOL value = o;
              if (value) {
                 cm_msg(MINFO, "cr_settings_changed", "Execute Run Prepare");
                 odb r("/Runinfo/Run number");
                 int run = r;
                 if(address)
                    cb->write_command(name,run,address);
                 else
                    cb->write_command(name,run);
                 o = false; //TODO: Check if this works...
              }
           } else {
               // Take the payload from the payload ODB field
               BOOL value = o;
               if (value) {
                    cm_msg(MINFO, "cr_settings_changed", "Execute2 %s", name.c_str());
                    odb p("/Equipment/Clock Reset/Settings/Payload");
                    int payload = p;
                    if(address)
                       cb->write_command(name,payload,address);
                     else
                       cb->write_command(name,payload);
                    o = false; //TODO: Check if this works...
               }
           }
       }
   }
}

void link_settings_changed(odb o)
{

   std::string name = o.get_name();

   if (name == "SwitchingBoardMask") {
      vector<INT> value = o;
      cm_msg(MINFO, "link_settings_changed", "Set Switching Board Mask to %d %d %d %d",
             value[0], value[1], value[2], value[3]);
   }

    if (name == "LinkMask") {
      vector<INT> value = o;
      cm_msg(MINFO, "link_settings_changed", "Seting Link Board Mask");

      //A FEB is only disabled if both SC and datataking are disabled. Typically these settings are linked,
      //here we do not enforce any kind of consistency.
      for(uint32_t i = 0; i < MAX_N_FRONTENDBOARDS; i++){
          if(value[i] == FEBLINKMASK::OFF){
              cb->write_command("Disable",0,i);
          } else {
              cb->write_command("Enable",0,i);
          }
      }
   }
}

void prepare_run_on_request(odb o){

    cm_msg(MINFO, "prepare_run_on_request", "Execute Run Prepare on request called");

    vector<int> request = o;
    
    bool norequest = true;
    for(uint32_t i=0; i < request.size(); i++){
        if(request[i])
            norequest = false;
    }

    if(norequest)
        return;




    odb a("/Equipment/Links/Settings/SwitchingBoardMask");
    vector<int> active = a;

    bool allok = true;
    bool notalloff = false;
    for(uint32_t i=0; i < MAX_N_SWITCHINGBOARDS; i++){
        printf("%i : %i : %i\n", i, request[i], active[i]);
        allok = allok && ((request[i] > 0) || (active[i] == 0));
        notalloff = notalloff || active[i];
    }

    if(!notalloff) return;

    if(allok && notalloff){
        cm_msg(MINFO, "prepare_run_on_request", "Execute Run Prepare on request");
        odb r("/Runinfo/Run number");
        int run = r;

        cb->write_command("Run Prepare",run);

        // reset requests
        for(uint32_t i=0; i < MAX_N_SWITCHINGBOARDS; i++){
            active[i] =0;
        }

        odb rrp("/Equipment/Clock Reset/Run Transitions/Request Run Prepare");
        rrp = active;

    } else {
        cm_msg(MINFO, "prepare_run_on_request", "Waiting for more switching boards");
    }
}

// ODB Setup //////////////////////////////
void setup_odb(){

    odb settings = {
        {"Active" , true},
        {"IP", "192.168.0.220"},
        {"Port", 50001},
        {"N_READBACK", 4},
        {"TX_CLK_MASK", 0x0AA},
        {"TX_RST_MASK",  0xAA0},
        {"TX_CLK_INVERT_MASK",  0x0A00},
        {"TX_RST_INVERT_MASK",  0x0000},
        {"RX_MASK", 0},
        {"TX_MASK", std::array<int,clockboard::MAXFIREFLY>{}},
        {"TX_INVERT_MASK",std::array<int,clockboard::MAXFIREFLY>{}},
        {"Run Prepare", false},
        {"Sync", false},
        {"Start Run", false},
        {"End Run", false},
        {"Abort Run", false},
        {"Start Link Test", false},
        {"Stop Link Test", false},
        {"Start Sync Test", false},
        {"Stop Sync Test", false},
        {"Test Sync", false},
        {"Reset", false},
        {"Stop Reset", false},
        {"Enable", false},
        {"Disable", false},
        {"Addressed", false},
        {"FEBAddress", 0},
        {"Payload", 0},
        {"Names CRT1",std::array<std::string,CRT1SIZE>()}
    };

    settings["Names CRT1"][0] = "Motherboard Current";
    settings["Names CRT1"][1] = "Motherboard Voltage";
    settings["Names CRT1"][2] = "RX Firefly Alarms";
    settings["Names CRT1"][3] = "RX Firefly Temp";
    settings["Names CRT1"][4] =  "RX Firefly Voltage";
    settings["Names CRT1"][5] = "TX Clk Firefly Alarms";
    settings["Names CRT1"][6] = "TX Clk Firefly Temp";
    settings["Names CRT1"][7] = "TX Clk Firefly Voltage";
    settings["Names CRT1"][8] = "TX Rst Firefly Alarms";
    settings["Names CRT1"][9] = "TX Rst Firefly Temp";
    settings["Names CRT1"][10] = "TX Rst Firefly Voltage";
    settings["Names CRT1"][11] = "Fan Current";

    int daughterstartindex = 12;
    int nffvariables =3;
    int ndaughtervariables = nffvariables * clockboard::MAXFIREFLYPERDAUGTHER + 2;

    for(int i=0; i < clockboard::MAXNDAUGHTER; i++){ // loop over daughters
         string * s =  new string("Daughterboard ");
         (*s) += to_string(i) + string(" Current");
         settings["Names CRT1"][daughterstartindex + i*ndaughtervariables +  0] = s;
         s =  new string("Daughterboard ");
         (*s) += to_string(i) + string(" Voltage");
         settings["Names CRT1"][daughterstartindex + i*ndaughtervariables +  1] = s;
         // loop over fireflys
         for(int j=0; j < clockboard::MAXFIREFLYPERDAUGTHER; j++){
             s =  new string("D ");
             (*s) += to_string(i) + " " + to_string(j) + " Firefly Alarms";
             settings["Names CRT1"][daughterstartindex + i*ndaughtervariables +  2 + j*nffvariables] = s;
             s =  new string("D ");
             (*s) += to_string(i) + " " + to_string(j) + " Firefly Temp";
             settings["Names CRT1"][daughterstartindex + i*ndaughtervariables +  2 + j*nffvariables +1] = s;
             s =  new string("D ");
             (*s) += to_string(i) + " " + to_string(j) + " Firefly Voltage";
             settings["Names CRT1"][daughterstartindex + i*ndaughtervariables +  2 + j*nffvariables +2] = s;
         }
     }
    //cout << "Dump: " <<  settings.dump() << endl;

    settings.connect("/Equipment/Clock Reset/Settings", true);

    odb variables = {
         {"Daughters Present", 0},
         {"Fireflys Present", std::array<int,clockboard::MAXNDAUGHTER>{}},
         {"CRT1", std::array<float,CRT1SIZE>{}}
    };
    variables.connect("/Equipment/Clock Reset/Variables");

    odb linksettings = {
        {"SwitchingBoardMask", std::array<int,MAX_N_SWITCHINGBOARDS>{}},
        {"SwitchingBoardNames", {"Central", "Recurl US", "Recurl DS", "Fibres"}},
        {"LinkMask", std::array<int,MAX_N_FRONTENDBOARDS>{}},
        {"FrontEndBoardType", std::array<int,MAX_N_FRONTENDBOARDS>{}},
        {"FrontEndBoardNames", std::array<std::string,MAX_N_FRONTENDBOARDS>{}}
    };
    linksettings.connect("/Equipment/Links/Settings");

    odb linkvariables = {
       {"SwitchingBoardStatus", std::array<int,MAX_N_SWITCHINGBOARDS>{}},
       {"RXLinkStatus", std::array<int,MAX_N_FRONTENDBOARDS>{}},
       {"TXLinkStatus", std::array<int,MAX_N_FRONTENDBOARDS>{}}
    };
    linkvariables.connect("/Equipment/Links/Variables");

    odb runtransitions;
    runtransitions.connect("/Equipment/Clock Reset/Run Transitions");
    runtransitions["Request Run Prepare"] = std::array<int, MAX_N_SWITCHINGBOARDS>{};

    odb custom;
    custom.connect("/Custom");
    custom["Clock and Reset"] = "cr.html";
    custom["Links"] = "links.html";
}

void setup_watches(){
    odb crodb("/Equipment/Clock Reset/Settings");
    crodb.watch(cr_settings_changed);

    odb linkodb("/Equipment/Links");
    linkodb.watch(link_settings_changed);

    odb rrp("/Equipment/Clock Reset/Run Transitions/Request Run Prepare");
    rrp.watch(prepare_run_on_request);
}

void setup_history(){

   hs_define_panel("Clock Reset","Motherboard",{"Clock Reset:Motherboard Current",
                                                "Clock Reset:Motherboard Voltage",
                                                "Clock Reset:Fan Current"});

   hs_define_panel("Clock Reset","Firefly Temperatures",{"Clock Reset:RX Firefly Temp",
                                                         "Clock Reset:TX Clk Firefly Temp",
                                                         "Clock Reset:TX Rst Firefly Temp"});

   hs_define_panel("Clock Reset","Firefly Voltages",{"Clock Reset:RX Firefly Voltage",
                                                     "Clock Reset:TX Clk Firefly Voltage",
                                                     "Clock Reset:TX Rst Firefly Voltage"});

   vector<string> cnames;
   vector<string> vnames;
   for(int i=0; i < 8; i++){
        cnames.push_back(string("Clock Reset:Daughterboard ") +std::to_string(i)+string(" Current"));
        vnames.push_back(string("Clock Reset:Daughterboard ") +std::to_string(i)+string(" Voltage"));
        vector<string> fnames;
        for(int j=0; j < 3; j++){
            fnames.push_back(string("Clock Reset:D ") + std::to_string(i) + " " + std::to_string(j)+
                                " Firefly Temp");
        }
        hs_define_panel("Clock Reset",(string("Firefly Temperatures Daughter ")+std::to_string(i)).c_str(),fnames);
   }

   hs_define_panel("Clock Reset","Daughterboard Currents",cnames);
   hs_define_panel("Clock Reset","Daughterboard Voltages",vnames);

   for(int i=0; i < 8; i++){
        vector<string> fnames;
        for(int j=0; j < 3; j++){
            fnames.push_back(string("Clock Reset:D ") + std::to_string(i) + " " + std::to_string(j)+
                                " Firefly Voltage");
        }
        hs_define_panel("Clock Reset",(string("Firefly Voltages Daughter ")+std::to_string(i)).c_str(),fnames);
   }
}

void setup_alarms(){
    // To be done
    //al_define_odb_alarm("Firefly temperature",
     //                   "/Equipment/Clock Reset/Variables/CRT1[3] > 65","Alarm","Firefly temperature %s");

}
