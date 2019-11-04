/********************************************************************\

  Name:         taken from switch_fe.cpp
  Created by:   Stefan Ritt
  adapted by:   Marius Koeppel
  adapted by:   Martin Mueller

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
#include "midas.h"
#include "mfe.h"
#include "mscb.h"

//#include "mudaq_device.h"

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "FEBrw_slow Frontend";
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

/* DMA Buffer and related */
//mudaq::DmaMudaqDevice * mup;

/* Values for reading SC Events */
uint32_t current_ro_idx = 0;

/*-- Function declarations -----------------------------------------*/

INT read_sc_event(char *pevent, INT off);
INT read_WMEM_event(char *pevent, INT off);
void sc_settings_changed(HNDLE, HNDLE, int, void *);

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
"[32] Temp0",
"[32] Temp1",
"[32] Temp2",
"[32] Temp3",
nullptr
};

EQUIPMENT equipment[] = {

   {"FEBrw_slow",                /* equipment name */
    {2, 0,                      /* event ID, trigger mask */
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
   db_create_key(hDB, 0, "Equipment/Switching/Variables/PCIE_MEM_START_READ", TID_INT);

   db_create_key(hDB, 0, "Equipment/Switching/Variables/FPGA_ID_WRITE", TID_INT);
   db_create_key(hDB, 0, "Equipment/Switching/Variables/PCIE_MEM_START_WRITE", TID_INT);
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
   db_create_key(hDB, 0, "Custom/FEBrw_slow&", TID_STRING);
   const char * name = "FEBrw_slow.html";
   db_set_value(hDB,0,"Custom/FEBrw_slow&", name, sizeof(name), 1, TID_STRING);

   // open mudaq
   //mup = new mudaq::DmaMudaqDevice("/dev/mudaq0");
   //if ( !mup->open() ) {
   //    cm_msg(MERROR, "frontend_init" , "Could not open device");
   //    return FE_ERR_DRIVER;
   //}

   //if ( !mup->is_ok() ) {
   //    cm_msg(MERROR, "frontend_init", "Mudaq is not ok");
   //    return FE_ERR_DRIVER;
   //}

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

/*--- Read Slow Control Event to be put into data stream --------*/

INT read_sc_event(char *pevent, INT off)
{
   bk_init(pevent);

   //mudaq::DmaMudaqDevice & mu = *mup;

   // getting a read event
   /*if (mu.read_memory_ro(current_ro_idx) == 0x1e0000bc) {
       uint32_t *pdata;
       bk_create(pevent, "RM_READ", TID_INT, (void **)&pdata);
       *pdata++ =   mu.read_memory_ro(current_ro_idx);
       *pdata++ = mu.read_memory_ro(current_ro_idx + 1); // startadd
       *pdata++ = mu.read_memory_ro(current_ro_idx + 2); // length
       uint32_t event_length = mu.read_memory_ro(current_ro_idx + 2);
       for (uint32_t i = 0; i < event_length; i++) { // getting data
           *pdata++ = mu.read_memory_ro(current_ro_idx + 3 + i);
       }
       bk_close(pevent, pdata);
       current_ro_idx = 3 + event_length;
       std::cout << current_ro_idx << std::endl;
       return bk_size(pevent);
   }*/

   // getting a write event
   /*if (mu.read_memory_ro(current_ro_idx) == 0x1f0000bc) {
       uint32_t *pdata;
       bk_create(pevent, "RM_WRITE", TID_INT, (void **)&pdata);
       *pdata++ =   mu.read_memory_ro(current_ro_idx);
       *pdata++ = mu.read_memory_ro(current_ro_idx + 1); // startadd
       *pdata++ = mu.read_memory_ro(current_ro_idx + 2); // length + acknowledge
       bk_close(pevent, pdata);
       current_ro_idx = 3;
       return bk_size(pevent);
   }*/

    return 0; // ToDo: is this the right thing to do if you don"t have an event?
}

/*--- Read WMEM if button was pressed --------*/

//INT read_WMEM_event(char *pevent, INT off)
//{
//    bk_init(pevent);
//
//    mudaq::DmaMudaqDevice & mu = *mup;
//
//    if (mu.read_memory_rw(0) == 0) {
//        return CM_SUCCESS; // no new data
//    }
//
//    unsigned int sc_length = mu.read_memory_rw(0);
//    unsigned int sc_start_add = mu.read_memory_rw(1);
//
//    int *pdata;
//    bk_create(pevent, "WM", TID_INT, (void **)&pdata);
//
//    for (unsigned int i = sc_start_add; i < sc_start_add + sc_length; i++) {
//        *pdata++ = mu.read_memory_rw(i);
//    }
//
//    mu.write_memory_rw(0, 0);
//
//    bk_close(pevent, pdata);
//
//    return bk_size(pevent);
//}

/*--- Called whenever settings have changed ------------------------*/

void sc_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *)
{
   KEY key;

   db_get_key(hDB, hKey, &key);

   //mudaq::DmaMudaqDevice & mu = *mup;

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

   if (std::string(key.name) == "Reset SC Master") {
       BOOL value;
       int size = sizeof(value);
       db_get_data(hDB, hKey, &value, &size, TID_BOOL);
       cm_msg(MINFO, "sc_settings_changed", "Reset SC Master");
       uint32_t reset_reg = 0;

      // reset_reg = SET_RESET_BIT_SC_MASTER(reset_reg);
      // mu.write_register(RESET_REGISTER_W, reset_reg);
      // mu.write_register(RESET_REGISTER_W, 0x0);

//       for(int i = 0; i <= 64*1024; i++){
//           mu.write_memory_rw(i, 0);
//       }
   }

   if (std::string(key.name) == "Reset SC Slave") {
       BOOL value;
       int size = sizeof(value);
       db_get_data(hDB, hKey, &value, &size, TID_BOOL);
       cm_msg(MINFO, "sc_settings_changed", "Reset SC Slave");
       uint32_t reset_reg = 0;
       reset_reg = SET_RESET_BIT_SC_SLAVE(reset_reg);
      // mu.write_register(RESET_REGISTER_W, reset_reg);
      // mu.write_register(RESET_REGISTER_W, 0x0);
   }

   if (std::string(key.name) == "Write") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if (value) {
         cm_msg(MINFO, "sc_settings_changed", "Execute write");

         INT DATA_WRITE_SIZE, SIZE_DATA_WRITE_SIZE;
         SIZE_DATA_WRITE_SIZE = sizeof(DATA_WRITE_SIZE);
         char STR_DATA_WRITE_SIZE[128];
         sprintf(STR_DATA_WRITE_SIZE,"Equipment/Switching/Variables/DATA_WRITE_SIZE");
         db_get_value(hDB, 0, STR_DATA_WRITE_SIZE, &DATA_WRITE_SIZE, &SIZE_DATA_WRITE_SIZE, TID_INT, 0);

         uint32_t DATA_ARRAY[DATA_WRITE_SIZE];

         INT FPGA_ID, SIZE_FPGA_ID;
         INT DATA, SIZE_DATA;
         INT START_ADD, SIZE_START_ADD;
         INT PCIE_MEM_START, SIZE_PCIE_MEM_START;

         SIZE_FPGA_ID = sizeof(FPGA_ID);
         SIZE_DATA = sizeof(DATA);
         SIZE_START_ADD = sizeof(START_ADD);
         SIZE_PCIE_MEM_START = sizeof(PCIE_MEM_START);

         char STR_FPGA_ID[128];
         char STR_START_ADD[128];
         char STR_PCIE_MEM_START[128];

         sprintf(STR_FPGA_ID,"Equipment/Switching/Variables/FPGA_ID_WRITE");
         sprintf(STR_START_ADD,"Equipment/Switching/Variables/START_ADD_WRITE");
         sprintf(STR_PCIE_MEM_START,"Equipment/Switching/Variables/PCIE_MEM_START_WRITE");

         db_get_value(hDB, 0, STR_FPGA_ID, &FPGA_ID, &SIZE_FPGA_ID, TID_INT, 0);
         db_get_value(hDB, 0, STR_START_ADD, &START_ADD, &SIZE_START_ADD, TID_INT, 0);
         db_get_value(hDB, 0, STR_PCIE_MEM_START, &PCIE_MEM_START, &SIZE_PCIE_MEM_START, TID_INT, 0);

         INT NEW_PCIE_MEM_START = PCIE_MEM_START + 5 + DATA_WRITE_SIZE;
         INT SIZE_NEW_PCIE_MEM_START;
         SIZE_NEW_PCIE_MEM_START = sizeof(NEW_PCIE_MEM_START);

         db_set_value(hDB, 0, STR_PCIE_MEM_START, &NEW_PCIE_MEM_START, SIZE_NEW_PCIE_MEM_START, 1, TID_INT);

         for (int i = 0; i < DATA_WRITE_SIZE; i++) {
             char STR_DATA[128];
             sprintf(STR_DATA,"Equipment/Switching/Variables/DATA_WRITE[%d]", i);
             db_get_value(hDB, 0, STR_DATA, &DATA, &SIZE_DATA, TID_INT, 0);
             DATA_ARRAY[i] = (uint32_t) DATA;
         }

         uint32_t *data = DATA_ARRAY;

        // mu.FEB_write((uint32_t) FPGA_ID, data, (uint16_t) DATA_WRITE_SIZE, (uint32_t) START_ADD, (uint32_t) PCIE_MEM_START);
         printf("feb_write called");
         value = FALSE; // reset flag in ODB
         db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);


         // for testing mscb:      M.M.

         char device[256], password[256], ip[256];
         int debug=0;
          device[0] = password[0] = 0;
          //cmd[0] = 0;

          //strlcpy(device, "mscb263" /*argv[++i]*/);
          //strlcpy(password, "mscb263" /*argv[++i]*/);

          //host2ip(device, ip, sizeof(ip));

         /* open port */
         int fd = mscb_init("mscb263", sizeof(device), "mscb263", debug ? 1 : 0);
        // int fd = mscb_init(device, sizeof(device), password, debug ? 1 : 0);
         unsigned short addr = 44192;

         mscb_addr(fd, MCMD_ADDR_NODE16, addr, 10);

         mscb_write_mem(fd, (unsigned short) addr, 0, START_ADD, &data, DATA_WRITE_SIZE);

      }
   }

   if (std::string(key.name) == "Read") {
      BOOL value;
      int size = sizeof(value);
      db_get_data(hDB, hKey, &value, &size, TID_BOOL);
      if (value) {
         cm_msg(MINFO, "sc_settings_changed", "Execute read");

         INT FPGA_ID, SIZE_FPGA_ID;
         INT LENGTH, SIZE_LENGTH;
         INT START_ADD, SIZE_START_ADD;
         INT PCIE_MEM_START, SIZE_PCIE_MEM_START;

         SIZE_FPGA_ID = sizeof(FPGA_ID);
         SIZE_LENGTH = sizeof(LENGTH);
         SIZE_START_ADD = sizeof(START_ADD);
         SIZE_PCIE_MEM_START = sizeof(PCIE_MEM_START);

         char STR_FPGA_ID[128];
         char STR_LENGTH[128];
         char STR_START_ADD[128];
         char STR_PCIE_MEM_START[128];

         sprintf(STR_FPGA_ID,"Equipment/Switching/Variables/FPGA_ID_READ");
         sprintf(STR_LENGTH,"Equipment/Switching/Variables/LENGTH_READ");
         sprintf(STR_START_ADD,"Equipment/Switching/Variables/START_ADD_READ");
         sprintf(STR_PCIE_MEM_START,"Equipment/Switching/Variables/PCIE_MEM_START_READ");

         db_get_value(hDB, 0, STR_FPGA_ID, &FPGA_ID, &SIZE_FPGA_ID, TID_INT, 0);
         db_get_value(hDB, 0, STR_LENGTH, &LENGTH, &SIZE_LENGTH, TID_INT, 0);
         db_get_value(hDB, 0, STR_START_ADD, &START_ADD, &SIZE_START_ADD, TID_INT, 0);
         db_get_value(hDB, 0, STR_PCIE_MEM_START, &PCIE_MEM_START, &SIZE_PCIE_MEM_START, TID_INT, 0);

         INT NEW_PCIE_MEM_START = PCIE_MEM_START + 5;
         INT SIZE_NEW_PCIE_MEM_START;
         SIZE_NEW_PCIE_MEM_START = sizeof(NEW_PCIE_MEM_START);

         db_set_value(hDB, 0, STR_PCIE_MEM_START, &NEW_PCIE_MEM_START, SIZE_NEW_PCIE_MEM_START, 1, TID_INT);

        // mu.FEB_read((uint32_t) FPGA_ID, (uint16_t) LENGTH, (uint32_t) START_ADD, (uint32_t) PCIE_MEM_START);

         value = FALSE; // reset flag in ODB
         db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
      }
   }

   if (std::string(key.name) == "Single Write") {
        BOOL value;
        int size = sizeof(value);
        db_get_data(hDB, hKey, &value, &size, TID_BOOL);
        if (value) {
            cm_msg(MINFO, "sc_settings_changed", "Execute single write");

            INT FPGA_ID, SIZE_FPGA_ID;
            INT DATA, SIZE_DATA;
            INT START_ADD, SIZE_START_ADD;
            INT PCIE_MEM_START, SIZE_PCIE_MEM_START;

            SIZE_FPGA_ID = sizeof(FPGA_ID);
            SIZE_DATA = sizeof(DATA);
            SIZE_START_ADD = sizeof(START_ADD);
            SIZE_PCIE_MEM_START = sizeof(PCIE_MEM_START);

            char STR_FPGA_ID[128];
            char STR_DATA[128];
            char STR_START_ADD[128];
            char STR_PCIE_MEM_START[128];

            sprintf(STR_FPGA_ID,"Equipment/Switching/Variables/FPGA_ID_WRITE");
            sprintf(STR_DATA,"Equipment/Switching/Variables/SINGLE_DATA_WRITE");
            sprintf(STR_START_ADD,"Equipment/Switching/Variables/START_ADD_WRITE");
            sprintf(STR_PCIE_MEM_START,"Equipment/Switching/Variables/PCIE_MEM_START_WRITE");

            db_get_value(hDB, 0, STR_FPGA_ID, &FPGA_ID, &SIZE_FPGA_ID, TID_INT, 0);
            db_get_value(hDB, 0, STR_DATA, &DATA, &SIZE_DATA, TID_INT, 0);
            db_get_value(hDB, 0, STR_START_ADD, &START_ADD, &SIZE_START_ADD, TID_INT, 0);
            db_get_value(hDB, 0, STR_PCIE_MEM_START, &PCIE_MEM_START, &SIZE_PCIE_MEM_START, TID_INT, 0);

            INT NEW_PCIE_MEM_START = PCIE_MEM_START + 6;
            INT SIZE_NEW_PCIE_MEM_START;
            SIZE_NEW_PCIE_MEM_START = sizeof(NEW_PCIE_MEM_START);

            db_set_value(hDB, 0, STR_PCIE_MEM_START, &NEW_PCIE_MEM_START, SIZE_NEW_PCIE_MEM_START, 1, TID_INT);

            uint32_t data_arr[1] = {0};
            data_arr[0] = (uint32_t) DATA;
            uint32_t *data = data_arr;

            //mu.FEB_write((uint32_t) FPGA_ID, data, (uint16_t) 1, (uint32_t) START_ADD, (uint32_t) PCIE_MEM_START);

            char device[256], password[256], ip[256];
            int debug=0;
            device[0] = password[0] = 0;

            int fd = mscb_init("mscb263", sizeof(device), "mscb263", debug ? 1 : 0);
           // int fd = mscb_init(device, sizeof(device), password, debug ? 1 : 0);
            unsigned short addr = 44192;

            mscb_addr(fd, MCMD_ADDR_NODE16, addr, 10);

            mscb_write_mem(fd, (unsigned short) addr, 0, START_ADD, &DATA, 4);

            value = FALSE; // reset flag in ODB
            db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
        }
    }

    if (std::string(key.name) == "Read WM") {
        BOOL value;
        int size = sizeof(value);
        db_get_data(hDB, hKey, &value, &size, TID_BOOL);
        if (value) {
            cm_msg(MINFO, "sc_settings_changed", "Execute Read WM");

            INT WM_START_ADD, SIZE_WM_START_ADD;
            INT WM_LENGTH, SIZE_WM_LENGTH;
            INT WM_DATA, SIZE_WM_DATA;

            SIZE_WM_START_ADD = sizeof(WM_START_ADD);
            SIZE_WM_LENGTH = sizeof(WM_LENGTH);
            SIZE_WM_DATA = sizeof(WM_DATA);

            char STR_WM_START_ADD[128];
            char STR_WM_LENGTH[128];
            char STR_WM_DATA[128];

            sprintf(STR_WM_START_ADD,"Equipment/Switching/Variables/WM_START_ADD");
            sprintf(STR_WM_LENGTH,"Equipment/Switching/Variables/WM_LENGTH");
            sprintf(STR_WM_DATA,"Equipment/Switching/Variables/WM_DATA");

            db_get_value(hDB, 0, STR_WM_START_ADD, &WM_START_ADD, &SIZE_WM_START_ADD, TID_INT, 0);
            db_get_value(hDB, 0, STR_WM_LENGTH, &WM_LENGTH, &SIZE_WM_LENGTH, TID_INT, 0);

            HNDLE key_WM_DATA;
            db_find_key(hDB, 0, "Equipment/Switching/Variables/WM_DATA", &key_WM_DATA);
            db_set_num_values(hDB, key_WM_DATA, WM_LENGTH);
            for (int i = 0; i < WM_LENGTH; i++) {
                //WM_DATA = mu.read_memory_rw((uint32_t) WM_START_ADD + i);
                db_set_value_index(hDB, 0, STR_WM_DATA, &WM_DATA, SIZE_WM_DATA, i, TID_INT, FALSE);
            }

            value = FALSE; // reset flag in ODB
            db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
        }
    }

    if (std::string(key.name) == "Read RM") {
        BOOL value;
        int size = sizeof(value);
        db_get_data(hDB, hKey, &value, &size, TID_BOOL);
        if (value) {
            cm_msg(MINFO, "sc_settings_changed", "Execute Read RM");

            INT RM_START_ADD, SIZE_RM_START_ADD;
            INT RM_LENGTH, SIZE_RM_LENGTH;
            INT RM_DATA, SIZE_RM_DATA;

            SIZE_RM_START_ADD = sizeof(RM_START_ADD);
            SIZE_RM_LENGTH = sizeof(RM_LENGTH);
            SIZE_RM_DATA = sizeof(RM_DATA);

            char STR_RM_START_ADD[128];
            char STR_RM_LENGTH[128];
            char STR_RM_DATA[128];

            sprintf(STR_RM_START_ADD,"Equipment/Switching/Variables/RM_START_ADD");
            sprintf(STR_RM_LENGTH,"Equipment/Switching/Variables/RM_LENGTH");
            sprintf(STR_RM_DATA,"Equipment/Switching/Variables/RM_DATA");

            db_get_value(hDB, 0, STR_RM_START_ADD, &RM_START_ADD, &SIZE_RM_START_ADD, TID_INT, 0);
            db_get_value(hDB, 0, STR_RM_LENGTH, &RM_LENGTH, &SIZE_RM_LENGTH, TID_INT, 0);

            HNDLE key_WM_DATA;
            db_find_key(hDB, 0, "Equipment/Switching/Variables/RM_DATA", &key_WM_DATA);
            db_set_num_values(hDB, key_WM_DATA, RM_LENGTH);
            for (int i = 0; i < RM_LENGTH; i++) {
               // RM_DATA = mu.read_memory_ro((uint32_t) RM_START_ADD + i);
                db_set_value_index(hDB, 0, STR_RM_DATA, &RM_DATA, SIZE_RM_DATA, i, TID_INT, FALSE);
            }

            value = FALSE; // reset flag in ODB
            db_set_data(hDB, hKey, &value, sizeof(value), 1, TID_BOOL);
        }
    }
}