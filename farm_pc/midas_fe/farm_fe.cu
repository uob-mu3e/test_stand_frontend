#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <unistd.h>

#include "midas.h"
#include "msystem.h"
#include "mcstd.h"
#include "experim.h"
#include "switching_constants.h"
#include "link_constants.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <sstream>
#include <fstream>

#include "mudaq_device.h"
#include "mfe.h"

using namespace std;

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "Stream Frontend";

/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

/* a frontend status page is displayed with this frequency in ms */
INT display_period = 0;

/* DMA Buffer and related */
volatile uint32_t *dma_buf;
size_t dma_buf_size = MUDAQ_DMABUF_DATA_LEN;
uint32_t dma_buf_nwords = dma_buf_size/sizeof(uint32_t);
uint32_t laddr;
uint32_t newdata;
uint32_t readindex;
uint32_t wlen;
uint32_t lastreadindex;
uint32_t lastWritten;
uint32_t lastlastWritten;
uint32_t lastRunWritten;
bool moreevents;
bool firstevent;

/* maximum event size produced by this frontend */
INT max_event_size = dma_buf_nwords;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 32 * max_event_size;

mudaq::DmaMudaqDevice * mup;
mudaq::DmaMudaqDevice::DataBlock block;

/*-- Function declarations -----------------------------------------*/

INT frontend_init();
INT frontend_exit();
INT begin_of_run(INT run_number, char *error);
INT end_of_run(INT run_number, char *error);
INT pause_run(INT run_number, char *error);
INT resume_run(INT run_number, char *error);
INT frontend_loop();

INT read_stream_event(char *pevent, INT off);
INT read_stream_thread(void *param);

INT poll_event(INT source, INT count, BOOL test);
INT interrupt_configure(INT cmd, INT source, POINTER_T adr);

void datagen_settings_changed(HNDLE, HNDLE, int, void *);
void link_active_settings_changed(HNDLE, HNDLE, int, void *);
/*-- Equipment list ------------------------------------------------*/

EQUIPMENT equipment[] = {

   {"Stream",                /* equipment name */
    {1, 0,                   /* event ID, trigger mask */
     "SYSTEM",               /* event buffer */
     EQ_USER,                /* equipment type */
     0,                      /* event source crate 0, all stations */
     "MIDAS",                /* format */
     TRUE,                   /* enabled */
     RO_RUNNING,             /* read only when running */
     100,                    /* poll for 100ms */
     0,                      /* stop run after this event limit */
     0,                      /* number of sub events */
     0,                      /* don't log history */
     "", "", "",},
    NULL,                    /* readout routine */
    },

   {""}
};

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
   cout<<"DMA_BUF_LENGTH: "<<dma_buf_size<<"  dma_buf_nwords: "<<dma_buf_nwords<<endl; 
   HNDLE hDB, hStreamSettings;
   
   set_equipment_status(equipment[0].name, "Initializing...", "var(--myellow)");
   
   // Get database
   cm_get_experiment_database(&hDB, NULL);
   
   // Map /equipment/Stream/Settings (structure defined in experim.h)
   char set_str[255];
   STREAM_SETTINGS_STR(stream_settings_str);

   sprintf(set_str, "/Equipment/Stream/Settings");
   int status = db_create_record(hDB, 0, set_str, strcomb(stream_settings_str));
   status = db_find_key (hDB, 0, set_str, &hStreamSettings);
   if (status != DB_SUCCESS){
      cm_msg(MINFO,"frontend_init","Key %s not found", set_str);
      return status;
   }
   cm_set_transition_sequence(TR_START,300);
   cm_set_transition_sequence(TR_STOP,700);
   // add custom page to ODB
   db_create_key(hDB, 0, "Custom/Farm&", TID_STRING);
   const char * name = "farm.html";
   db_set_value(hDB,0,"Custom/Farm&",name, sizeof(name), 1,TID_STRING);

   HNDLE hKey;

   // create Settings structure in ODB
   db_find_key(hDB, 0, "/Equipment/Stream/Settings/Datagenerator", &hKey);
   assert(hKey);
   db_watch(hDB, hKey, datagen_settings_changed, nullptr);
  
   //link mask changed settings - init & connect to ODB
   db_find_key(hDB, 0, "/Equipment/Links/Settings/LinkMask", &hKey);
   assert(hKey);
   db_watch(hDB, hKey, link_active_settings_changed, nullptr);

   // Allocate memory for the DMA buffer - this can fail!
   if(cudaMallocHost( (void**)&dma_buf, dma_buf_size ) != cudaSuccess){
      cout << "Allocation failed, aborting!" << endl;
      cm_msg(MERROR, "frontend_init" , "Allocation failed, aborting!");
      return FE_ERR_DRIVER;
   }
   
   // initialize to zero
   for (int i = 0; i <  dma_buf_nwords ; i++) {
      (dma_buf)[i] = 0;
   }
   
   mup = new mudaq::DmaMudaqDevice("/dev/mudaq0");
   if ( !mup->open() ) {
      cout << "Could not open device " << endl;
      cm_msg(MERROR, "frontend_init" , "Could not open device");
      return FE_ERR_DRIVER;
   }
   
   if ( !mup->is_ok() )
      return FE_ERR_DRIVER;
   
   cm_msg(MINFO, "frontend_init" , "Mudaq device is ok");
   cout << "Mudaq device is ok " << endl;
   
   // set fpga write pointers
   lastWritten = 0;
   lastlastWritten = 0;
   lastRunWritten = mup->last_written_addr();

   struct mesg user_message;
   user_message.address = dma_buf;
   user_message.size = dma_buf_size;
   
   // map memory to bus addresses for FPGA
   int ret_val = mup->map_pinned_dma_mem( user_message );
   
   if (ret_val < 0) {
      cout << "Mapping failed " << endl;
      cm_msg(MERROR, "frontend_init" , "Mapping failed");
      mup->disable();
      mup->close();
      free( (void *)dma_buf );
      delete mup;
      return FE_ERR_DRIVER;
   }
   
   // switch off and reset DMA for now
   mup->disable();
   
   //update data generator from ODB
   db_find_key(hDB, 0, "/Equipment/Stream/Settings/Datagenerator/Divider", &hKey);
   datagen_settings_changed(hDB,hKey,0,NULL);

   // switch off the data generator (just in case..)
   mup->write_register(DATAGENERATOR_REGISTER_W, 0x0);
   usleep(2000);
   // DMA_CONTROL_W
   mup->write_register(0x5,0x0);

   //set data link enable
   link_active_settings_changed(hDB,hKey,0,NULL);

   usleep(5000);
   
   // create ring buffer for readout thread
   create_event_rb(0);
   
   // create readout thread
   ss_thread_create(read_stream_thread, NULL);
      
   set_equipment_status(equipment[0].name, "Ready for running", "var(--mgreen)");
   
   return SUCCESS;
}

/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{
   if (mup) {
      mup->disable();
      mup->close();
      delete mup;
   }

   // following code crashes the frontend, please fix!
   // free( (void *)dma_buf );
   cudaFreeHost((void *)dma_buf);
   
   return SUCCESS;
}

void datagen_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *)
{
    KEY key;
    db_get_key(hDB, hKey, &key);
    cm_msg(MINFO, "datagen_settings_changed", "Set Farm: %s", key.name);

    if (std::string(key.name) == "Enable") {
       //this is set once we start the run
    }

    if (std::string(key.name) == "Divider") {
       int value;
       int size = sizeof(value);
       db_get_data(hDB, hKey, &value, &size, TID_INT);
       mup->write_register_wait(DATAGENERATOR_DIVIDER_REGISTER_W,value,100);
    }
}

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
   //printf("Data link active: 0x");
   for(int link = MAX_LINKS_PER_SWITCHINGBOARD-1 ; link>=0; link--) {
      int offset = 0;//MAX_LINKS_PER_SWITCHINGBOARD* switch_id;
      if(frontend_board_active_odb[offset + link] & FEBLINKMASK::DataOn)
	 //a standard FEB link (SC and data) is considered enabled if RX and TX are. 
	 //a secondary FEB link (only data) is enabled if RX is.
	 //Here we are concerned only with run transitions and slow control, the farm frontend may define this differently.
         link_active_from_odb += (1 << link);
      //printf("%u",(frontend_board_active_odb[offset + link] & FEBLINKMASK::DataOn?1:0));
   }
   //printf("\n");
   return link_active_from_odb;
}

void set_link_enable(uint64_t enablebits){
   //mup->write_register(DATA_LINK_MASK_REGISTER_HIGH_W, enablebits >> 32); TODO make 64 bits
   mup->write_register(DATA_LINK_MASK_REGISTER_W,  enablebits & 0xFFFFFFFF);
}

void link_active_settings_changed(HNDLE hDB, HNDLE hKey, INT, void *){
    set_link_enable(get_link_active_from_odb());
}



/*-- Begin of Run --------------------------------------------------*/

INT begin_of_run(INT run_number, char *error)
{ 
   set_equipment_status(equipment[0].name, "Starting run", "var(--myellow)");
   
   mudaq::DmaMudaqDevice & mu = *mup;
   
   // Reset last written address used for polling
   laddr = mu.last_written_addr();
   newdata = 0;
   readindex = 0;
   moreevents = false;
   firstevent = true;

   // reset all
   uint32_t reset_reg = 0;
   reset_reg |= 1<<RESET_BIT_EVENT_COUNTER;
   reset_reg |= 1<<RESET_BIT_DATAGEN;

   mu.write_register_wait(RESET_REGISTER_W, reset_reg, 100);
   // Enable register on FPGA for continous readout and enable dma

   mu.enable_continous_readout(0);
   usleep(10);
   mu.write_register_wait(RESET_REGISTER_W, 0x0, 100);

   // Set up data generator: enable only if set in ODB
   HNDLE hKey;
   BOOL value;
   INT size = sizeof(value);
   db_find_key(hDB, 0, "/Equipment/Stream/Settings/Datagenerator/Enable", &hKey);
   db_get_data(hDB, hKey, &value, &size, TID_BOOL);
   uint32_t reg=mu.read_register_rw(DATAGENERATOR_REGISTER_W);
   
   if(value) {
       // TODO: make odb value for slowdown
       mu.write_register_wait(DMA_SLOW_DOWN_REGISTER_W, 0x3E8, 100);
       reg = SET_DATAGENERATOR_BIT_ENABLE(reg);
   }
   
   mu.write_register(DATAGENERATOR_REGISTER_W,reg);
   
   lastWritten = 0;
   lastlastWritten = 0;
   lastRunWritten = mu.last_written_addr();//lastWritten;
   
    //mu.enable_continous_readout(1);
   // Note: link masks are already set during fe_init and via ODB callback

   /*
   // Get ODB settings for this equipment
   HNDLE hDB, hStreamSettings;
   INT status;
   char set_str[256];
   STREAM_SETTINGS settings;  // defined in experim.h
   
   cm_get_experiment_database(&hDB, NULL);
   sprintf(set_str, "/Equipment/Stream/Settings");
   status = db_find_key (hDB, 0, set_str, &hStreamSettings);
   if (status != DB_SUCCESS) {
      cm_msg(MERROR, "begin_of_run", "cannot find stream settings record from ODB");
      return status;
   }
   size = sizeof(settings);
   status = db_get_record(hDB, hStreamSettings, &settings, &size, 0);
   if (status != DB_SUCCESS) {
      cm_msg(MERROR, "begin_of_run", "cannot retrieve stream settings from ODB");
      return status;
   }
   */

   set_equipment_status(equipment[0].name, "Running", "var(--mgreen)");
   
   return SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT run_number, char *error)
{

   mudaq::DmaMudaqDevice & mu = *mup;
   printf("farm_fe: Waiting for buffers to empty\n");
   uint16_t timeout_cnt = 0;
   while(! mu.read_register_ro(BUFFER_STATUS_REGISTER_R) & 1<<0/* TODO right bit */ &&
         timeout_cnt++ < 50) {
      printf("Waiting for buffers to empty %d/50\n", timeout_cnt);
      timeout_cnt++;
      usleep(1000);
   };

   if(timeout_cnt>=50) {
      cm_msg(MERROR,"farm_fe","Buffers on Switching Board not empty at end of run");
      set_equipment_status(equipment[0].name, "Not OK", "var(--mred)");
      //return CM_TRANSITION_CANCELED;
   }else{
      printf("Buffers all empty\n");
   }

   // TODO: Find a better way to see when DMA is finished.

   printf("Waiting for DMA to finish\n");
   usleep(1000); // Wait for DMA to finish
   timeout_cnt = 0;
   while(mu.last_written_addr() != lastlastWritten && //(readindex % dma_buf_nwords) &&
         timeout_cnt++ < 50) {
      printf("Waiting for DMA to finish %d/50\n", timeout_cnt);
      timeout_cnt++;
      usleep(1000);
   };

   if(timeout_cnt>=50) {
      cm_msg(MERROR,"farm_fe","DMA did not finish");
      set_equipment_status(equipment[0].name, "Not OK", "var(--mred)");
//      return CM_TRANSITION_CANCELED;
   }else{
      printf("DMA is finished\n");
   }

    // stop generator
   uint32_t datagen_setup = 0;
   datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
   mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);
   mu.write_register_wait(DMA_SLOW_DOWN_REGISTER_W, 0x0, 100);

   // disable DMA
   mu.disable();

   set_equipment_status(equipment[0].name, "Ready for running", "var(--mgreen)");
   
   return SUCCESS;
}

/*-- Pause Run -----------------------------------------------------*/

INT pause_run(INT run_number, char *error)
{
   mudaq::DmaMudaqDevice & mu = *mup;
   
//   uint32_t datagen_setup = mu.read_register_rw(DATAGENERATOR_REGISTER_W);
//   datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
//   mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup,1000);

   // disable DMA
   mu.disable(); // Marius Koeppel: not sure if this works
   
   set_equipment_status(equipment[0].name, "Paused", "var(--myellow)");
   
   return SUCCESS;
}

/*-- Resume Run ----------------------------------------------------*/

INT resume_run(INT run_number, char *error)
{
   mudaq::DmaMudaqDevice & mu = *mup;
   
//   uint32_t datagen_setup = mu.read_register_rw(DATAGENERATOR_REGISTER_W);
//   datagen_setup = SET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
//   mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup,1000);

   // enable DMA
   mu.enable_continous_readout(0); // Marius Koeppel: not sure if this works
   
   set_equipment_status(equipment[0].name, "Running", "var(--mgreen)");
   
   return SUCCESS;
}

/*-- Frontend Loop -------------------------------------------------*/

INT frontend_loop()
{
   /* if frontend_call_loop is true, this routine gets called when
      the frontend is idle or once between every event */
   return SUCCESS;
}

/*-- Trigger event routines ----------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
/* Polling routine for events. Returns TRUE if event
 is available. If test equals TRUE, don't return. The test
 flag is used to time the polling */
{
   /*
   if(moreevents && !test)
      return 1;
   
   mudaq::DmaMudaqDevice & mu = *mup;
   
   for (int i = 0; i < count; i++) {
      uint32_t addr = mu.last_written_addr();
      if ((addr != laddr) && !test) {
         if (firstevent) {
            newdata = addr;
            firstevent = false;
         } else {
            if(addr > laddr)
               newdata = addr - laddr;
            else
               newdata = 0x10000 - laddr + addr;
         }
         if (newdata > 0x10000) {
            return 0;
         }
         laddr = addr;
         return 1;
      }
   }
   */
   return 0;
}

/*-- Interrupt configuration ---------------------------------------*/

INT interrupt_configure(INT cmd, INT source, POINTER_T adr)
{
   return SUCCESS;
}

/*-- Event readout -------------------------------------------------*/

INT read_stream_event(char *pevent, INT off)
{
   /*
   bk_init(pevent);
   
   DWORD *pdata;
   uint32_t read = 0;
   bk_create(pevent, "HEAD", TID_DWORD, (void **)&pdata);
   
   for (int i =0; i < 8; i ++) {
      *pdata++ = dma_buf[(++readindex)%dma_buf_nwords];
      read++;
   }
   
   bk_close(pevent, pdata);
   newdata -= read;
   
   if (read < newdata && newdata < 0x10000)
      moreevents = true;
   else
      moreevents = false;
   
   return bk_size(pevent);
   */
   return 0;
}

INT check_event(volatile uint32_t * buffer, uint32_t idx, bool rp_before_wp)
{
    // rp_before_wp no event check here ...
    if ( rp_before_wp ) return 0;

    // check if the event is good
    EVENT_HEADER* eh=(EVENT_HEADER*)(&buffer[idx]);
    BANK_HEADER* bh=(BANK_HEADER*)(&buffer[idx+4]);

    if ( eh->event_id != 0x1 ) return -1;
    if ( eh->trigger_mask != 0x0 ) return -1;
    if ( bh->flags != 0x11 ) return -1;

    return 0;
}

INT update_equipment_status(int status, int cur_status, EQUIPMENT *eq)
{
    
    if ( status != DB_SUCCESS ) {
        set_equipment_status(eq[0].name, "Buffer ERROR", "var(--myellow)");
        return -1;
    }

    if ( cur_status != DB_SUCCESS ) {
        set_equipment_status(eq[0].name, "Running", "var(--mgreen)");
    }

    return DB_SUCCESS;

}

/*-- Event readout -------------------------------------------------*/

INT read_stream_thread(void *param)
{
    // we are at mu.last_written_addr() ...  ask for a new bunch of x 256-bit words
    // get mudaq and set readindex to last written addr
    mudaq::DmaMudaqDevice & mu = *mup;

    uint32_t* pdata;
    int status;
    int cur_status = -1;
    wlen = 0;
    readindex = 0;
   
    // variables for dummy data
    // TODO add odb value for dummy data
    uint32_t SERIAL = 0x00000001;
    uint32_t TIME = 0x00000001;
    bool use_dmmy_data = false;

    // tell framework that we are alive
    signal_readout_thread_active(0, TRUE);
   
    // obtain ring buffer for inter-thread data exchange
    int rbh = get_event_rbh(0);
    int rb_status;

    while (is_readout_thread_enabled()) {

        // don't readout events if we are not running
        if (run_state != STATE_RUNNING) {
            set_equipment_status(equipment[0].name, "Not running", "var(--myellow)");
            //ss_sleep(100);
            //TODO: signalling from main thread?
        continue;
        }

        // dummy data
        if (use_dmmy_data == true) {
            uint32_t dma_buf_dummy[48];

            for (int i = 0; i<2; i++) {
            // event header
            dma_buf_dummy[0+i*24] = 0x00000001; // Trigger and Event ID
            dma_buf_dummy[1+i*24] = SERIAL; // Serial number
            dma_buf_dummy[2+i*24] = TIME; // time
            dma_buf_dummy[3+i*24] = 24*4-4*4; // event size
            dma_buf_dummy[4+i*24] = 24*4-4*4; // all bank size
            dma_buf_dummy[5+i*24] = 0x11; // flags
            // bank 0
            dma_buf_dummy[6+i*24] = 0x46454230; // bank name
            dma_buf_dummy[7+i*24] = 0x6; // bank type TID_DWORD
            dma_buf_dummy[8+i*24] = 0x3*4; // data size
            dma_buf_dummy[9+i*24] = 0xAFFEAFFE; // data
            dma_buf_dummy[10+i*24] = 0xAFFEAFFE; // data
            dma_buf_dummy[11+i*24] = 0xAFFEAFFE; // data
            // bank 1
            dma_buf_dummy[12+i*24] = 0x46454231; // bank name
            dma_buf_dummy[13+i*24] = 0x6; // bank type TID_DWORD
            dma_buf_dummy[14+i*24] = 0x3*4; // data size
            dma_buf_dummy[15+i*24] = 0xAFFEAFFE; // data
            dma_buf_dummy[16+i*24] = 0xAFFEAFFE; // data
            dma_buf_dummy[17+i*24] = 0xAFFEAFFE; // data
            // bank 2
            dma_buf_dummy[18+i*24] = 0x2; // bank name
            dma_buf_dummy[19+i*24] = 0x6; // bank type TID_DWORD
            dma_buf_dummy[20+i*24] = 0x3*4; // data size
            dma_buf_dummy[21+i*24] = 0xAFFEAFFE; // data
            dma_buf_dummy[22+i*24] = 0xAFFEAFFE; // data
            dma_buf_dummy[23+i*24] = 0xAFFEAFFE; // data
            SERIAL += 1;
            TIME += 1;
            }

            volatile uint32_t * dma_buf_volatile;
            dma_buf_volatile = dma_buf_dummy;
            copy_n(&dma_buf_volatile[0], 48, pdata); // len in data words            
            rb_increment_wp(rbh, sizeof(dma_buf_dummy)); // in byte length

            continue;
        }

        //if (mu.last_endofevent_addr() == 0) continue;
        //if (mu.last_written_addr() == 0) continue;
        //if (mu.last_endofevent_addr() == lastlastEndOfEvent) continue;
        //if (mu.last_written_addr() == lastlastWritten) continue;
        
        //lastlastEndOfEvent = lastEndOfEvent;
        
        //lastEndOfEvent = mu.last_endofevent_addr();
        
        //if ((lastEndOfEvent+1)*8 > lastlastWritten) continue;
        
        // in the FPGA the endofevent is one off, since it can be that
        // the end of event does not fit into the 4kB anymore so we have
        // to check this here. Also the endofevent is in 256 bit words
        // so we have to multiply by 8 to get to 32 bit words
        // here we check if the lastWritten is aligned with the end of event
        // this is not really a problem but later we need to check somehow if
        // we are at the end of the event so for now we continue if they are equal
        //if (((lastEndOfEvent+1)*8)%dma_buf_nwords == lastWritten) {
        //    ss_sleep(100);
        //    continue;
        //}
        
        // only to make it save that we are at the end. Sometimes it fails and
        // the end of event is off so we would then take the next one
        
        //if (((dma_buf[((lastEndOfEvent+1)*8-1)%dma_buf_nwords] == 0xAFFEAFFE) or
        //        (dma_buf[((lastEndOfEvent+1)*8-1)%dma_buf_nwords] == 0x0000009c)) 
        //        ){
        //    cout << hex << (lastEndOfEvent+1)*8 << " " << lastWritten << " " << dma_buf[(lastEndOfEvent+1)*8] << endl;
        //}

            //cout << hex << lastWritten << endl;
            //cout << hex << lastlastWritten << endl;
        
            lastWritten = mu.last_written_addr();
        
            if (lastWritten == 0) {
                //cout << "last_written" << endl;
                continue;
            }

            if ( lastWritten == lastRunWritten ) {
                continue;
            } else {
                lastRunWritten = 999999999;
            }

            if(lastWritten % dma_buf_nwords == lastlastWritten % dma_buf_nwords) continue;

//            printf("lastlastWritten = 0x%08X\n", lastlastWritten);
//            printf("lastWritten = 0x%08X\n", lastWritten);
//            break;

            if(lastWritten < lastlastWritten) lastWritten += dma_buf_nwords;

//            uint32_t rb_space = rb_get_space(rbh);
            uint32_t offset = lastlastWritten;
//            printf("event: data[0] = 0x%08X\n", dma_buf[(offset + 0) % dma_buf_nwords]);
//            printf("event: data[1] = 0x%08X\n", dma_buf[(offset + 1) % dma_buf_nwords]);
//            printf("event: data[2] = 0x%08X\n", dma_buf[(offset + 2) % dma_buf_nwords]);
//            printf("event: data[3] = 0x%08X\n", dma_buf[(offset + 3) % dma_buf_nwords]);
            while(true) {
                // check enough words for header
                if(lastWritten - offset < 4) break;
                uint32_t eventLength = dma_buf[(offset + 3) % dma_buf_nwords];
                // check enough words for data
                if(lastWritten - offset < 4 + eventLength / 4) break;
                if(eventLength > max_event_size) {
                    printf("ERROR: (eventLength = 0x%08X) > max_event_size\n", eventLength);
                    abort();
                    exit(1);
                }
//                printf("event: id = 0x%08X\n", dma_buf[(offset + 1) % dma_buf_nwords]);
                offset += 4; // header
                offset += eventLength / 4; // data
//                printf("event: offset = 0x%08X, eventLength = 0x%08X, data = 0x%08X\n", offset, eventLength, dma_buf[offset % dma_buf_nwords]);
            }
//            printf("offset = 0x%08X, lastWritten = 0x%08X, lastlastWritten = 0x%08X\n", offset, lastWritten, lastlastWritten);
            lastWritten = offset;
            if(lastWritten > dma_buf_nwords) lastWritten -= dma_buf_nwords;

            if(lastWritten == lastlastWritten) continue;

            // obtain buffer space
            status = rb_get_wp(rbh, (void **)&pdata, 0);
//            printf("rb_get_wp = 0x%08X\n", pdata);

            if (status == DB_TIMEOUT) {
                set_equipment_status(equipment[0].name, "Buffer full", "var(--myellow)");
            }

            if(status != DB_SUCCESS) {
//                cout << "warn: status != DB_SUCCESS, discard data" << endl;
                lastlastWritten = lastWritten;
                continue;
            }

            //printf("lastlastWritten = 0x%08X\n", lastlastWritten);
            //printf("lastWritten = 0x%08X\n", lastWritten);

            // sanity check
            if(lastlastWritten + 3 < dma_buf_nwords and dma_buf[lastlastWritten + 3] > 0x1000) {
                printf("NOT GOOD\n");
            }

            uint32_t wlen = 0;
            if(lastWritten < lastlastWritten) {
                // partial copy when wrapping
                copy_n(&dma_buf[lastlastWritten], dma_buf_nwords - lastlastWritten, pdata);
                wlen += dma_buf_nwords - lastlastWritten;
                lastlastWritten = 0;
            }
            if(lastWritten != lastlastWritten) {
                // complete copy
                copy_n(&dma_buf[lastlastWritten], lastWritten - lastlastWritten, pdata + wlen);
                wlen += lastWritten - lastlastWritten;
                lastlastWritten = lastWritten;
            }
            //printf("wlen = 0x%08X\n", wlen);

            rb_status = rb_increment_wp(rbh, wlen * 4); // in byte length
            if(rb_status != DB_SUCCESS) {
                printf("warn: rb_status != DB_SUCCESS\n");
            }
            cur_status = update_equipment_status(rb_status, cur_status, equipment);
   }
   // tell framework that we finished
   signal_readout_thread_active(0, FALSE);
   return 0;
}
