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
		/Equipment/Switching SC/Settings/MupixConfig: triggers a configuration of all ASICs 

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
#include "mfe.h"

#include "mudaq_device_scifi.h"

//Slow control for mupix/mupix
#include "mupix_midasodb.h"
#include "link_constants.h"
#include "mupix_FEB.h"
/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "SW Frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

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

/* DMA Buffer and related */
mudaq::DmaMudaqDevice * mup;

/* Use CRFE bypass during run-start transitions, directly send command to FEB*/
#define CRFE_BYPASS

/*-- Function declarations -----------------------------------------*/

INT read_sc_event(char *pevent, INT off);
INT read_WMEM_event(char *pevent, INT off);
INT read_mupix_sc_event(char *pevent, INT off);
void sc_settings_changed(HNDLE, HNDLE, int, void *);
void switching_board_mask_changed(HNDLE, HNDLE, int, void *);
void frontend_board_mask_changed(HNDLE, HNDLE, int, void *);

uint64_t get_link_active_from_odb(); //throws
void set_feb_enable(uint64_t enablebits);
uint64_t get_runstart_ack();
uint64_t get_runend_ack();
void print_ack_state();

/*-- Equipment list ------------------------------------------------*/

/* Default values for /Equipment/Switching SC/Settings */
const char *sc_settings_str[] = {
"Active = BOOL : 1",
"Delay = INT : 0",
"Write = BOOL : 0",
"Single Write = BOOL : 0",
"Read = BOOL : 0",
"Read WM = BOOL : 0",
"Read RM = BOOL : 0",
"Reset SC Master = BOOL : 0",
"Reset SC Slave = BOOL : 0",
"Clear WM = BOOL : 0",
"Last RM ADD = BOOL : 0",
"MupixConfig = BOOL : 0",
"MupixBoard = BOOL : 0",
"[32] Temp0",
"[32] Temp1",
"[32] Temp2",
"[32] Temp3",
nullptr
};

enum EQUIPMENT_ID {Switching=0,Mupix};
EQUIPMENT equipment[] = {

   {"Switching",                /* equipment name */
    {110, 0,                      /* event ID, trigger mask */
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
   {"Mupix",                    /* equipment name */
    {113, 0,                      /* event ID, trigger mask */
     "SYSTEM",                  /* event buffer */
     EQ_PERIODIC,                 /* equipment type */
     0,                         /* event source crate 0, all stations */
     "MIDAS",                   /* format */
     TRUE,                      /* enabled */
     RO_TRANSITIONS | RO_ODB,   /* read during run transitions and update ODB */
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
   HNDLE hKeySC;

   // create Settings structure in ODB
   db_create_record(hDB, 0, "Equipment/Switching/Settings", strcomb(sc_settings_str));
   db_find_key(hDB, 0, "/Equipment/Switching", &hKeySC);
   assert(hKeySC);

   db_watch(hDB, hKeySC, sc_settings_changed, nullptr);

   // set default values of variables
   db_create_key(hDB, 0, "Equipment/Switching/Variables/FPGA_ID_READ", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/START_ADD_READ", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/LENGTH_READ", TID_INT);

   db_create_key(hDB, 0, "Equipment/Switching/Variables/FPGA_ID_WRITE", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/DATA_WRITE", TID_INT); // TODO: why is it possible to address this as an array in js?
   db_create_key(hDB, 0, "Equipment/Switching/Variables/DATA_WRITE_SIZE", TID_INT);

   db_create_key(hDB, 0, "Equipment/Switching/Variables/START_ADD_WRITE", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/SINGLE_DATA_WRITE", TID_INT);

   db_create_key(hDB, 0, "Equipment/Switching/Variables/RM_START_ADD", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/RM_LENGTH", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/RM_DATA", TID_INT);

   db_create_key(hDB, 0, "Equipment/Switching/Variables/WM_START_ADD", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/WM_LENGTH", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/WM_DATA", TID_INT);

    // add custom page to ODB
   db_create_key(hDB, 0, "Custom/Switching&", TID_STRING);
   const char * name = "sc.html";
   db_set_value(hDB,0,"Custom/Switching&", name, sizeof(name), 1, TID_STRING);

   HNDLE hKey;
   // watch if this switching board is enabled
   db_find_key(hDB, 0, "/Equipment/Links/Settings/SwitchingBoardMask", &hKey);
   assert(hKey);
   db_watch(hDB, hKey, switching_board_mask_changed, nullptr);

   // watch if this frontend board is enabled
   db_find_key(hDB, 0, "/Equipment/Links/Settings/LinkMask", &hKey);
   assert(hKey);
   db_watch(hDB, hKey, frontend_board_mask_changed, nullptr);

   // Define history panels
   // --- SciFi panels created in mutrig::midasODB::setup_db, below

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
   mup->FEBsc_resetMaster();
   mup->FEBsc_resetSlave();

   //set link enables so slow control can pass 
   try{ set_feb_enable(get_link_active_from_odb()); }
   catch(...){ return FE_ERR_ODB;}


   //Mupix setup part
   set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Initializing...", "var(--myellow)");
   MupixFEB::Create(*mup,hDB,equipment[EQUIPMENT_ID::Mupix].name,"/Equipment/Mupix"); //create FEB interface signleton for scifi
   MupixFEB::Instance()->SetSBnumber(switch_id);
   int status=mupix::midasODB::setup_db(hDB,"/Equipment/Mupix",MupixFEB::Instance(),true);
   if(status != SUCCESS){
      set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Start up failed", "var(--mred)");
      return status;
   }
   set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Ok", "var(--mgreen)");
   //end of Mupix setup part

   /*
    * Set our transition sequence. The default is 500. Setting it
    * to 400 means we are called BEFORE most other clients.
    */
   cm_set_transition_sequence(TR_START, 400);

   /*
    * Set our transition sequence. The default is 500. Setting it
    * to 600 means we are called AFTER most other clients.
    */
    cm_set_transition_sequence(TR_STOP, 600);



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
try{
   set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Starting Run", "var(--morange)");

   /* Set new run number */
   mup->write_register(RUN_NR_REGISTER_W, run_number);
   /* Reset acknowledge/end of run seen registers before start of run */
   uint32_t start_setup = 0;
   start_setup = SET_RESET_BIT_RUN_START_ACK(start_setup);
   start_setup = SET_RESET_BIT_RUN_END_ACK(start_setup);
   mup->write_register_wait(RESET_REGISTER_W, start_setup, 1000);
   mup->write_register(RESET_REGISTER_W, 0x0);

//   mup->FEBsc_resetMaster(); //KB: needed?
//   mup->FEBsc_resetSlave(); //KB: needed?

   /* get link active from odb. */
   uint64_t link_active_from_odb = get_link_active_from_odb();

   //configure ASICs
   int status=MupixFEB::Instance()->ConfigureASICs();
   if(status!=SUCCESS){
      cm_msg(MERROR,"switch_fe","ASIC configuration failed");
      return CM_TRANSITION_CANCELED;
   }


#ifndef CRFE_BYPASS
   /* send run prepare signal via CR system */
   INT value = 1;
   db_set_value_index(hDB,0,"Equipment/Clock Reset/Run Transitions/Request Run Prepare",
                      &value, sizeof(value), switch_id, TID_INT, false);
#else
   /* direct sending of run prepare, CRFE bypass frontend is not working during run transitions*/
   cm_msg(MINFO,"switch_fe","Bypassing CRFE for run transition");
   DWORD value = run_number;
   mup->FEBsc_write(mup->FEBsc_broadcast_ID, &value,1,0xfff5,true); //run number
   value= (1<<8) | 0x10;
   mup->FEBsc_write(mup->FEBsc_broadcast_ID, &value,1,0xfff4,true); //run prep command
   value= 0xbcbcbcbc;
   mup->FEBsc_write(mup->FEBsc_broadcast_ID, &value,1,0xfff5,true); //reset payload
   value= 0;//(1<<8) | 0x00;
   mup->FEBsc_write(mup->FEBsc_broadcast_ID, &value,1,0xfff4,true); //reset command
#endif

   uint16_t timeout_cnt=300;
   uint64_t link_active_from_register;
   printf("Waiting for run prepare acknowledge from all FEBs\n");
   do{
      timeout_cnt--;
      link_active_from_register = get_runstart_ack();
      printf("%u  %lx  %lx\n",timeout_cnt,link_active_from_odb, link_active_from_register);
      usleep(10000);
   }while(link_active_from_register != link_active_from_odb && (timeout_cnt > 0));

   if(timeout_cnt==0) {
      cm_msg(MERROR,"switch_fe","Run number mismatch on run %d", run_number);
      print_ack_state();
      return CM_TRANSITION_CANCELED;
   }

   set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Scintillating...", "lightBlue");
   return CM_SUCCESS;
}catch(...){return CM_TRANSITION_CANCELED;}
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT run_number, char *error)
{
try{
   /* get link active from odb */
   uint64_t link_active_from_odb = get_link_active_from_odb();

   printf("end_of_run: Waiting for stop signals from all FEBs\n");
   uint16_t timeout_cnt = 50;
   uint64_t stop_signal_seen = get_runend_ack();
   printf("Stop signal seen from 0x%16lx, expect stop signals from 0x%16lx\n", stop_signal_seen, link_active_from_odb);
   while(stop_signal_seen != link_active_from_odb &&
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
      set_equipment_status(equipment[EQUIPMENT_ID::Mupix].name, "Not OK", "var(--mred)");
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
    while(mup->FEBsc_get_packet()){};
    //TODO: make this a switch
    mup->FEBsc_dump_packets();
    return 0;
    //return mup->FEBsc_write_bank(pevent,off);
}

/*--- Read Slow Control Event from Mupix to be put into data stream --------*/

INT read_mupix_sc_event(char *pevent, INT off){
	static int i=0;
    printf("Reading Scifi FEB status data from all FEBs %d\n",i++);
    MupixFEB::Instance()->ReadBackAllRunState();
    return 0;
}

/*--- helper functions ------------------------*/

BOOL sc_settings_changed_hepler(const char *key_name, HNDLE hDB, HNDLE hKey, DWORD type){
    BOOL value;
    int size = sizeof(value);
    db_get_data(hDB, hKey, &value, &size, type);
    if(value)
        cm_msg(MINFO, "sc_settings_changed", "trigger for key=\"%s\"", key_name);
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

   if (std::string(key.name) == "Reset SC Master" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
       mu.FEBsc_resetMaster();
       set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
   }

   if (std::string(key.name) == "Reset SC Slave" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
       mu.FEBsc_resetSlave();
       set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
   }

   if (std::string(key.name) == "Write" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
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

   if (std::string(key.name) == "Read" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
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
       if(count==3) cm_msg(MERROR,"switch_fe","Tried 4 times to get a slow control read response but did not succeed");

       set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
   }

   if (std::string(key.name) == "Single Write" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
        INT FPGA_ID=get_odb_value_by_string("Equipment/Switching/Variables/FPGA_ID_WRITE");
        INT DATA=get_odb_value_by_string("Equipment/Switching/Variables/SINGLE_DATA_WRITE");
        INT START_ADD=get_odb_value_by_string("Equipment/Switching/Variables/START_ADD_WRITE");

	uint32_t data_arr[1] = {0};
        data_arr[0] = (uint32_t) DATA;
        uint32_t *data = data_arr;
        mu.FEBsc_write((uint32_t) FPGA_ID, data, (uint16_t) 1, (uint32_t) START_ADD);

        set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }

    if (std::string(key.name) == "Read WM" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
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

    if (std::string(key.name) == "Read RM" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
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

    if (std::string(key.name) == "Last RM ADD" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
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
    if (std::string(key.name) == "Read MALIBU File" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
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
    if (std::string(key.name) == "MupixConfig" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
          int status=MupixFEB::Instance()->ConfigureASICs();
          if(status!=SUCCESS){ 
         	//TODO: what to do? 
          }
	  set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }
    if (std::string(key.name) == "MupixBoard" && sc_settings_changed_hepler(key.name, hDB, hKey, TID_BOOL)) {
          int status=MupixFEB::Instance()->ConfigureBoards();
          if(status!=SUCCESS){
            //TODO: what to do?
          }
      set_odb_flag_false(key.name,hDB,hKey,TID_BOOL);
    }
    if (std::string(key.name) == "Reset Bypass Command") {
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

uint64_t get_link_active_from_odb(){
   INT frontend_board_active_odb[MAX_N_FRONTENDBOARDS];
   int size = sizeof(INT)*MAX_N_FRONTENDBOARDS;
   int status = db_get_value(hDB, 0, "/Equipment/Links/Settings/LinkMask", frontend_board_active_odb, &size, TID_INT, false);
   if (status != SUCCESS){
      cm_msg(MERROR,"switch_fe","Error getting record for /Equipment/Links/Settings");
      throw;
   }

   /* get link active from odb */
   uint64_t link_active_from_odb = 0;
   for(int link = 0; link < MAX_LINKS_PER_SWITCHINGBOARD; link++) {
      int offset = MAX_LINKS_PER_SWITCHINGBOARD* switch_id;
      if((frontend_board_active_odb[offset + link] == FEBLINKMASK::ON) ||
	(frontend_board_active_odb[offset + link] == FEBLINKMASK::DataOn))
	 //a standard FEB link (SC and data) is considered enabled if RX and TX are. 
	 //a secondary FEB link (only data) is enabled if RX is.
	 //Here we are concerned only with run transitions and slow control, the farm frontend may define this differently.
         link_active_from_odb += (1 << link);
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

void switching_board_mask_changed(HNDLE hDB, HNDLE hKey, INT, void *) {
   printf("switching_board_mask_changed\n");
   INT switching_board_mask[MAX_N_SWITCHINGBOARDS];
   int size = sizeof(INT)*MAX_N_FRONTENDBOARDS;
   db_get_data(hDB, hKey, &switching_board_mask, &size, TID_INT);
   BOOL value = switching_board_mask[switch_id] > 0 ? true : false;

   for(int i = 0; i < 2; i++) {
      char str[128];
      sprintf(str,"Equipment/%s/Common/Enabled", equipment[i].name);
      db_set_value(hDB,0,str, &value, sizeof(value), 1, TID_BOOL);

      cm_msg(MINFO, "switching_board_mask_changed", "Set Equipment %s enabled to %d", equipment[i].name, value);
   }

   MupixFEB::Instance()->RebuildFEBsMap();

}

void frontend_board_mask_changed(HNDLE hDB, HNDLE hKey, INT, void *) {
   try{
      set_feb_enable(get_link_active_from_odb());
      MupixFEB::Instance()->RebuildFEBsMap();
   }catch(...){}
}
