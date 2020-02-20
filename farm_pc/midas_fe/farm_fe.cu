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
uint32_t lastlastWritten;
uint32_t lastRunWritten;
bool moreevents;
bool firstevent;

/* maximum event size produced by this frontend */
INT max_event_size = dma_buf_size;

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
   for (int i = 0; i < dma_buf_nwords ; i++) {
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

   // empty dma buffer
   for (int i = 0; i < dma_buf_nwords ; i++) {
      (dma_buf)[i] = 0;
   }

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

// check if the event is good
template < typename T >
INT check_event(T* buffer, uint32_t idx) {
    EVENT_HEADER* eh = (EVENT_HEADER*)(buffer + idx);
    BANK_HEADER* bh = (BANK_HEADER*)(eh + 1);

    if ( eh->event_id != 0x1 ) {
        printf("Error: Wrong event id 0x%08X\n", eh->event_id);
        return -1;
    }
    if ( eh->trigger_mask != 0x0 ) {
        printf("Error: Wrong trigger_mask 0x%08X\n", eh->trigger_mask);
        return -1;
    }
    if ( bh->flags != 0x11 ) {
        printf("Error: Wrong flags 0x%08X\n", bh->flags);
        return -1;
    }
    
    uint32_t eventDataSize = eh->data_size; // bytes
    printf("eventDataSize = %u bytes\n", eventDataSize);

    // offset bank relative to event data
    uint32_t bankOffset = 8; // bytes
    // iterate through banks
    while(true) {
        BANK32* b = (BANK32*)(&buffer[idx + 4 + bankOffset / 4]);
	printf("bank: name = %4.4s, data_size = %u bytes, offset = %u bytes\n", b->name, b->data_size, bankOffset);
        bankOffset += sizeof(BANK32) + b->data_size; // bytes
        if(bankOffset > eventDataSize) { sleep(10); return -1; }
        if(bankOffset == eventDataSize) break;
    }

    return 0;
}

int copy_event(uint32_t* dst, volatile uint32_t* src) {
    // copy event header and global bank header to destination
    std::copy_n(src, sizeof(EVENT_HEADER) / 4 + sizeof(BANK_HEADER) / 4, dst);
    // get header for future asjustment
    EVENT_HEADER* eh = (EVENT_HEADER*)(dst);
    BANK_HEADER* bh = (BANK_HEADER*)(eh + 1);

    // start from first bank
    int src_i = 6, dst_i = 6;

    while(true) {
        // get bank
        BANK32* bank = (BANK32*)(src + src_i);
        // copy bank to dst
        std::copy_n((uint32_t*)bank, sizeof(BANK32) / 4 + bank->data_size / 4, dst + dst_i);
        // go to next bank
        src_i += sizeof(BANK32) / 4 + bank->data_size / 4;
        // insert empty word if needed in dst
        dst_i += sizeof(BANK32) / 4 + bank->data_size / 4;
        if(src_i >= sizeof(EVENT_HEADER) / 4 + eh->data_size / 4) break;
        // at this point we expect next bank
        if(bank->data_size % 8) {
            // insert padding word
            dst[dst_i] = 0xFFFFFFFF;
            dst_i += 1;
        }
    }

    // update data_size's
    bh->data_size = dst_i * 4 - sizeof(EVENT_HEADER) - sizeof(BANK_HEADER);
    eh->data_size = dst_i * 4 - sizeof(EVENT_HEADER);

    return dst_i;
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

INT read_stream_thread(void *param) {
    // get mudaq
    mudaq::DmaMudaqDevice & mu = *mup;

    int cur_status = -1;

    // tell framework that we are alive
    signal_readout_thread_active(0, TRUE);

    // obtain ring buffer for inter-thread data exchange
    int rbh = get_event_rbh(0);

    uint32_t max_requested_words = dma_buf_nwords/2;
    // request to read dma_buffer_size/2 (count in blocks of 256 bits)
    mu.write_register_wait(0xC, max_requested_words / (256/32), 100);

    while (is_readout_thread_enabled()) {
        // don't readout events if we are not running
        if (run_state != STATE_RUNNING) {
            set_equipment_status(equipment[0].name, "Not running", "var(--myellow)");
            //ss_sleep(100);
            //TODO: signalling from main thread?
            continue;
        }

        set_equipment_status(equipment[0].name, "Running", "var(--mgreen)");

        // start dma
        mu.enable_continous_readout(0);

        // wait for requested data
        while ( (mu.read_register_ro(0x1C) & 1) == 0 ) {}

        // disable dma
        mu.disable();
        // and get lastWritten
        lastlastWritten = 0;
        uint32_t lastWritten = mu.last_written_addr();
//        printf("lastWritten = 0x%08X\n", lastWritten);

        // print dma_buf content
//        for ( int i = lastWritten - 0x100; i < lastWritten + 0x100; i++) {
//            if(i % 8 == 0) printf("[0x%08X]", i);
//            printf("  %08X", dma_buf[i]);
//            if(i % 8 == 7) printf("\n");
//        } printf("\n");

        // walk events to find end of last event
        if(lastWritten < lastlastWritten) lastWritten += dma_buf_nwords;
        uint32_t offset = lastlastWritten;
        while(true) {
            // check enough space for header
            if(offset + 4 > lastWritten) break;
//            printf("event: offset = 0x%08X, event_id = 0x%08X, data_size = 0x%08X\n", offset, dma_buf[offset % dma_buf_nwords], dma_buf[(offset + 3) % dma_buf_nwords]);
            uint32_t eventLength = 16 + dma_buf[(offset + 3) % dma_buf_nwords];
            if(eventLength > max_requested_words * 4) {
                printf("ERROR: (eventLength = 0x%08X) > max_event_size\n", eventLength);
                break;
            }
            // check enough space for data
            if(offset + eventLength / 4 > lastWritten) break;
	    if(check_event(dma_buf, offset) < 0) {
                printf("ERROR: bad event\n");
                break;
            }
            offset += eventLength / 4;
        }
//        printf("lastlastWritten = 0x%08X, offset = 0x%08X, lastWritten = 0x%08X\n", lastlastWritten, offset, lastWritten);
        if(offset > dma_buf_nwords) offset -= dma_buf_nwords;
        lastWritten = offset;

        // get midas buffer
        uint32_t* pdata = nullptr;
        int rb_status = rb_get_wp(rbh, (void**)&pdata, 0);
        if ( rb_status != DB_SUCCESS ) {
            printf("ERROR: rb_get_wp -> rb_status != DB_SUCCESS\n");
            lastlastWritten = lastWritten;
            continue;
        }

        // number of words written to midas buffer
        uint32_t wlen = 0;

        // copy midas buffer and adjust bank data_size to multiple of 8 bytes
        for(int src_i = 0, dst_i = 0; src_i < lastWritten;) {
            int nwords = copy_event(pdata + dst_i, dma_buf + src_i);
            src_i += 4 + dma_buf[src_i + 3] / 4;
            dst_i += nwords;
            wlen = dst_i;
        }
        lastlastWritten = lastWritten;

        // copy data to midas and increment wp of the midas buffer
        if(lastWritten < lastlastWritten) {
            // partial copy when wrap around
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

        // update midas buffer
        rb_status = rb_increment_wp(rbh, wlen * 4); // in byte length
        if ( rb_status != DB_SUCCESS ) {
            printf("ERROR: rb_increment_wp -> rb_status != DB_SUCCESS\n");
        }

        cur_status = update_equipment_status(rb_status, cur_status, equipment);
    }

    // tell framework that we finished
    signal_readout_thread_active(0, FALSE);

    return 0;
}
