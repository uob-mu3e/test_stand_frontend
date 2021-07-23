#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <unistd.h>

#include "midas.h"
#include "odbxx.h"
#include "msystem.h"
#include "mcstd.h"
#include "experim.h"
#include "switching_constants.h"
#include "link_constants.h"
#include "util.h"

#include <fcntl.h>
#include <sys/mman.h>

#include <sstream>
#include <fstream>

#include "mudaq_device.h"
#include "mfe.h"
#include "history.h"

using namespace std;
using midas::odb;

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "Stream Frontend";

/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

/* Overwrite equipment struct in ODB from values in code*/
BOOL equipment_common_overwrite = FALSE;

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
INT max_event_size = dma_buf_size; //TODO: how to define this?

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


void setup_odb();
void setup_watches();

INT init_mudaq();

uint64_t get_link_active_from_odb(odb o);
void link_active_settings_changed(odb);
void stream_settings_changed(odb);
/*-- Equipment list ------------------------------------------------*/

EQUIPMENT equipment[] = {

   {"Stream",                /* equipment name */
    {1, 0,                   /* event ID, trigger mask */
     "SYSTEM",               /* event buffer */
     EQ_USER,                /* equipment type */
     0,                      /* event source crate 0, all stations */
     "MIDAS",                /* format */
     TRUE,                   /* enabled */
     RO_RUNNING  | RO_STOPPED | RO_ODB,             /* read while running and stopped but not at transitions and update ODB */
     1000,                    /* poll for 1s */
     0,                      /* stop run after this event limit */
     0,                      /* number of sub events */
     0,                      /* don't log history */
     "", "", "",},
     NULL,                    /* readout routine */
    },
    {"Stream Logger",                /* equipment name */
    {11, 0,                   /* event ID, trigger mask */
     "SYSTEM",               /* event buffer */
     EQ_PERIODIC,                /* equipment type */
     0,                      /* event source crate 0, all stations */
     "MIDAS",                /* format */
     FALSE,                   /* enabled */
     RO_ALWAYS  | RO_ODB,             /* read only when running */
     1000,                    /* poll for 1s */
     0,                      /* stop run after this event limit */
     0,                      /* number of sub events */
     1,                      /* log history every event */
     "", "", "",},
     read_stream_event,                    /* readout routine */
    },

   {""}
};

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
    // TODO: for debuging
    //odb::set_debug(true);

    set_equipment_status(equipment[0].name, "Initializing...", "var(--myellow)");

    // setup odb and watches
    setup_odb();
    setup_watches();

    // init dma and mudaq device
    INT status = init_mudaq();
    if (status != SUCCESS) return FE_ERR_DRIVER;

    usleep(5000);

    // create ring buffer for readout thread
    create_event_rb(0);

    // create readout thread
    ss_thread_create(read_stream_thread, NULL);

    set_equipment_status(equipment[0].name, "Ready for running", "var(--mgreen)");

    //Set our transition sequence. The default is 500.
    cm_set_transition_sequence(TR_START, 300);

    //Set our transition sequence. The default is 500. Setting it
    // to 700 means we are called AFTER most other clients.
    cm_set_transition_sequence(TR_STOP, 700);

    return SUCCESS;
}

void stream_settings_changed(odb o)
{
    std::string name = o.get_name();

    cm_msg(MINFO, "stream_settings_changed", "Stream stettings changed");

    if (name == "Datagen Divider") {
        uint32_t divider = o;
        cm_msg(MINFO, "stream_settings_changed", "Set Divider to %d", divider);
        mup->write_register(DATAGENERATOR_DIVIDER_REGISTER_W, divider);
    }

    if (name == "Datagen Enable") {
        cm_msg(MINFO, "stream_settings_changed", "Set Disable Datagen to %s", o ? "y" : "n");
        if (o) {
            uint32_t current_state = mup->read_register_rw(SWB_READOUT_STATE_REGISTER_W);
            current_state |= (1 << 0);
            mup->write_register(SWB_READOUT_STATE_REGISTER_W, current_state);
        } else {
            uint32_t current_state = mup->read_register_rw(SWB_READOUT_STATE_REGISTER_W);
            current_state &= ~(1 << 0);
            mup->write_register(SWB_READOUT_STATE_REGISTER_W, current_state);
        }
    }

    if (name == "use_merger") {
        cm_msg(MINFO, "stream_settings_changed", "Set Disable Merger to %s", o ? "y" : "n");
        if (o) {
            uint32_t current_state = mup->read_register_rw(SWB_READOUT_STATE_REGISTER_W);
            current_state |= (1 << 2);
            mup->write_register(SWB_READOUT_STATE_REGISTER_W, current_state);
        } else {
            uint32_t current_state = mup->read_register_rw(SWB_READOUT_STATE_REGISTER_W);
            current_state &= ~(1 << 2);
            mup->write_register(SWB_READOUT_STATE_REGISTER_W, current_state);
        }
    }

    if (name == "mask_n_scifi") {
        uint32_t mask = o;
        char buffer [255];
        sprintf(buffer, "Set Mask Links Scifi to " PRINTF_BINARY_PATTERN_INT32, PRINTF_BYTE_TO_BINARY_INT32((long long int) mask));
        cm_msg(MINFO, "stream_settings_changed", buffer);
        mup->write_register(SWB_LINK_MASK_SCIFI_REGISTER_W, mask);
    }

    if (name == "mask_n_pixel") {
        uint32_t mask = o;
        char buffer [255];
        sprintf(buffer, "Set Mask Links Pixel to " PRINTF_BINARY_PATTERN_INT32, PRINTF_BYTE_TO_BINARY_INT32((long long int) mask));
        cm_msg(MINFO, "stream_settings_changed", buffer);
        mup->write_register(SWB_LINK_MASK_PIXEL_REGISTER_W, mask);
    }
}

void link_active_settings_changed(odb o){

    /* get link active from odb */
    uint64_t link_active_from_odb = 0;
    //printf("Data link active: 0x");
    int idx=0;
    for(int link : o) {
        int offset = 0;//MAX_LINKS_PER_SWITCHINGBOARD* switch_id;
        if(link & FEBLINKMASK::DataOn)
            //a standard FEB link (SC and data) is considered enabled if RX and TX are.
            //a secondary FEB link (only data) is enabled if RX is.
            //Here we are concerned only with run transitions and slow control, the farm frontend may define this differently.
            link_active_from_odb += (1 << idx);
        //printf("%u",(frontend_board_active_odb[offset + link] & FEBLINKMASK::DataOn?1:0));
        idx ++;
    }
    //printf("\n");
    //mup->write_register(DATA_LINK_MASK_REGISTER_HIGH_W, enablebits >> 32); TODO make 64 bits
    mup->write_register(DATA_LINK_MASK_REGISTER_W, link_active_from_odb & 0xFFFFFFFF);

}

// ODB Setup //////////////////////////////
void setup_odb(){

    // TODO: use me for the default values
    //odb cur_links_odb("/Equipment/Links/Settings/LinkMask");
    //std::bitset<64> cur_link_active_from_odb = get_link_active_from_odb(cur_links_odb);

    // Map /equipment/Stream/Settings
    odb stream_settings = {
        {"Datagen Divider", 1000},     // int
        {"Datagen Enable", false},     // bool
        {"mask_n_scifi", 0x0},         // int
        {"mask_n_pixel", 0x0},         // int
        {"use_merger", false},         // int
        {"dma_buf_nwords", int(dma_buf_nwords)},
        {"dma_buf_size", int(dma_buf_size)}
    };

    stream_settings.connect("/Equipment/Stream/Settings");

    // add custom page to ODB
    odb custom("/Custom");
    custom["Farm&"] = "farm.html";
    
    // add error cnts to ODB
    //odb error_settings = {
    //    {"DC FIFO ALMOST FUll", 0},
    //    {"DC LINK FIFO FULL", 0},
    //    {"TAG FIFO FULL", 0},
    //    {"MIDAS EVENT RAM FULL", 0},
    //    {"STREAM FIFO FULL", 0},
    //    {"DMA HALFFULL", 0},
    //    {"SKIP EVENT LINK FIFO", 0},
    //    {"SKIP EVENT DMA RAM", 0},
    //    {"IDLE NOT EVENT HEADER", 0},
    //};
    //error_settings.connect("/Equipment/Stream Logger/Variables", true);
    
    // Define history panels
    //hs_define_panel("Stream Logger", "MIDAS Bank Builder", {"Stream Logger:DC FIFO ALMOST FUll",
    //                                                        "Stream Logger:DC LINK FIFO FULL",
    //                                                        "Stream Logger:TAG FIFO FULL",
    //                                                        "Stream Logger:MIDAS EVENT RAM FULL",
    //                                                        "Stream Logger:STREAM FIFO FULL",
    //                                                        "Stream Logger:DMA HALFFULL",
    //                                                        "Stream Logger:SKIP EVENT LINK FIFO",
    //                                                        "Stream Logger:SKIP EVENT DMA RAM",
    //                                                        "Stream Logger:IDLE NOT EVENT HEADER"});
}

void setup_watches(){

    // datagenerator changed settings
    odb stream_settings("/Equipment/Stream/Settings");
    stream_settings.watch(stream_settings_changed);

    // link mask changed settings
    odb links("/Equipment/Links/Settings/LinkMask");
    links.watch(link_active_settings_changed);

}

// INIT MUDAQ //////////////////////////////
INT init_mudaq(){
    
/*    cudaError_t cuda_error = cudaMallocHost( (void**)&dma_buf, dma_buf_size );
    if(cuda_error != cudaSuccess){
        cm_msg(MERROR, "frontend_init" , "Allocation failed, aborting!");
        return FE_ERR_DRIVER;
    }*/
    
    int fd = open("/dev/mudaq0_dmabuf", O_RDWR);
    if(fd < 0) {
        printf("fd = %d\n", fd);
        return FE_ERR_DRIVER;
    }
    dma_buf = (uint32_t*)mmap(nullptr, MUDAQ_DMABUF_DATA_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(dma_buf == MAP_FAILED) {
        cm_msg(MERROR, "frontend_init" , "mmap failed: dmabuf = %x\n", dma_buf);
        return FE_ERR_DRIVER;
    }
    
    // initialize to zero
    for (int i = 0; i < dma_buf_nwords ; i++) {
        (dma_buf)[i] = 0;
    }
    
    // open mudaq
    mup = new mudaq::DmaMudaqDevice("/dev/mudaq0");
    if ( !mup->open() ) {
        cout << "Could not open device " << endl;
        cm_msg(MERROR, "frontend_init" , "Could not open device");
        return FE_ERR_DRIVER;
    }

    // check mudaq
    if ( !mup->is_ok() )
        return FE_ERR_DRIVER;
    else {
        cm_msg(MINFO, "frontend_init" , "Mudaq device is ok");
    }
    
    // set fpga write pointers
    lastlastWritten = 0;
    lastRunWritten = mup->last_written_addr();

    // switch off and reset DMA for now
    mup->disable();
    usleep(2000);

    // switch off the data generator (just in case ..)
    mup->write_register(DATAGENERATOR_REGISTER_W, 0x0);
    usleep(2000);

    // DMA_CONTROL_W
    mup->write_register(0x5,0x0);

    //set data link enable
    odb link;
    link.connect("/Equipment/Links/Settings/LinkMask");
    link_active_settings_changed(link);

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
//   cudaFreeHost((void *)dma_buf);
   
   return SUCCESS;
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
   reset_reg = SET_RESET_BIT_EVENT_COUNTER(reset_reg);
   reset_reg = SET_RESET_BIT_DATAGEN(reset_reg);
   reset_reg = SET_RESET_BIT_DATA_PATH(reset_reg);
   mu.write_register_wait(RESET_REGISTER_W, reset_reg, 100);

   // empty dma buffer
   for (int i = 0; i < dma_buf_nwords ; i++) {
      (dma_buf)[i] = 0;
   }

   // Enable register on FPGA for continous readout and enable dma
   mu.enable_continous_readout(0);
   usleep(10);
   mu.write_register_wait(RESET_REGISTER_W, 0x0, 100);

   // Set up data generator: enable only if set in ODB
   uint32_t reg=mu.read_register_rw(DATAGENERATOR_REGISTER_W);
   odb stream_settings;
   stream_settings.connect("/Equipment/Stream/Settings");

   if(stream_settings["Datagen Enable"]) {
        int divider = stream_settings["Datagen Divider"];
        cm_msg(MINFO,"farm_fe", "Use datagenerator with divider register %d", divider);
        if(stream_settings["use_merger"]) {
            // readout merger
            mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x5);
        } else {
            // readout stream
            mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x3);
            mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, divider);
        }
   } else if(stream_settings["use_merger"]) {
        // readout links with merger
        mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x44);
   } else {
        cm_msg(MINFO,"farm_fe", "Use link data");
        // readout link
        mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x42);
   }

   mu.write_register(SWB_LINK_MASK_PIXEL_REGISTER_W, stream_settings["mask_n_pixel"]);
   mu.write_register(SWB_LINK_MASK_SCIFI_REGISTER_W, stream_settings["mask_n_scifi"]);
  
   // reset lastlastwritten
   lastlastWritten = 0;
   lastRunWritten = mu.last_written_addr();//lastWritten;

   // Note: link masks are already set during fe_init and via ODB callback

   set_equipment_status(equipment[0].name, "Running", "var(--mgreen)");
   
   return SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT run_number, char *error)
{

   mudaq::DmaMudaqDevice & mu = *mup;
   cm_msg(MINFO,"farm_fe","Waiting for buffers to empty");
   uint16_t timeout_cnt = 0;
   while(! mu.read_register_ro(BUFFER_STATUS_REGISTER_R) & 1<<0/* TODO right bit */ &&
         timeout_cnt++ < 50) {
      //printf("Waiting for buffers to empty %d/50\n", timeout_cnt);
      timeout_cnt++;
      usleep(1000);
   };

   if(timeout_cnt>=50) {
      //cm_msg(MERROR,"farm_fe","Buffers on Switching Board not empty at end of run");
      cm_msg(MINFO,"farm_fe","Buffers on Switching Board not empty at end of run");
      set_equipment_status(equipment[0].name, "Buffers not empty", "var(--mred)");
      // TODO: at the moment we dont care
      //return CM_TRANSITION_CANCELED;
   }else{
      printf("Buffers all empty\n");
   }

   //Finish DMA while waiting for last requested data to be finished
   cm_msg(MINFO, "farm_fe", "Waiting for DMA to finish");
   usleep(1000); // Wait for DMA to finish
   timeout_cnt = 0;
   
   //wait for requested data
   //TODO: in readout th poll on run end reg from febs
   //write variable and check this one here and then disable readout th
   //also check if readout th is disabled by midas at run end
   while ( (mu.read_register_ro(0x1C) & 1) == 0 && timeout_cnt < 100 ) {
       timeout_cnt++;
       usleep(1000);
   };
        
   if(timeout_cnt>=100) {
        //cm_msg(MERROR, "farm_fe", "DMA did not finish");
        cm_msg(MINFO, "farm_fe", "DMA did not finish");
        set_equipment_status(equipment[0].name, "DMA did not finish", "var(--mred)");
        // TODO: at the moment we dont care
        //return CM_TRANSITION_CANCELED;
   }else{
      cm_msg(MINFO, "farm_fe", "DMA is finished\n");
   }

    // disable dma
    mu.disable();
    // stop readout
    mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0);
    mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x0);
    mu.write_register(SWB_LINK_MASK_PIXEL_REGISTER_W, 0x0);
    mu.write_register(SWB_LINK_MASK_SCIFI_REGISTER_W, 0x0);
    mu.write_register(SWB_READOUT_LINK_REGISTER_W, 0x0);
    mu.write_register(GET_N_DMA_WORDS_REGISTER_W, 0x0);
    // reset data path
    mu.write_register(RESET_REGISTER_W, 0x0 | (1<<21));

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
   return SUCCESS;
}

/*-- Interrupt configuration ---------------------------------------*/

INT interrupt_configure(INT cmd, INT source, POINTER_T adr)
{
   return SUCCESS;
}

/*-- Event readout -------------------------------------------------*/

INT read_stream_event(char *pevent, INT off)
{
    
   // get mudaq 
   mudaq::DmaMudaqDevice & mu = *mup;
 
   // get odb for errors
   odb error_cnt("/Equipment/Stream Logger/Variables");

   // create bank, pdata, stream buffer is name
   bk_init(pevent);
   DWORD *pdata;
   bk_create(pevent, "STBU", TID_DWORD, (void **)&pdata);
    
   // TODO: save value to variable before and dont call function all the time
   // get error regs and write to odb
   error_cnt["DC FIFO ALMOST FUll"] = mu.read_register_ro(0x1D);
   error_cnt["TAG FIFO FULL"] =  mu.read_register_ro(0x1E);
   error_cnt["MIDAS EVENT RAM FULL"] = mu.read_register_ro(0x1F);
   error_cnt["STREAM FIFO FULL"] = mu.read_register_ro(0x20);
   error_cnt["DMA HALFFULL"] = mu.read_register_ro(0x21);
   error_cnt["DC LINK FIFO FULL"] = mu.read_register_ro(0x22);
   error_cnt["SKIP EVENT LINK FIFO"] = mu.read_register_ro(0x23);
   error_cnt["SKIP EVENT DMA RAM"] =  mu.read_register_ro(0x24);
   error_cnt["IDLE NOT EVENT HEADER"] =  mu.read_register_ro(0x25);

   *pdata++ = mu.read_register_ro(0x1D);
   *pdata++ = mu.read_register_ro(0x1E);
   *pdata++ = mu.read_register_ro(0x1F);
   *pdata++ = mu.read_register_ro(0x20);
   *pdata++ = mu.read_register_ro(0x21);
   *pdata++ = mu.read_register_ro(0x22);
   *pdata++ = mu.read_register_ro(0x23);
   *pdata++ = mu.read_register_ro(0x24);
   *pdata++ = mu.read_register_ro(0x25);

   bk_close(pevent, pdata);
 
   return bk_size(pevent);
  
}

// check if the event is good
template < typename T >
uint32_t check_event(T* buffer, uint32_t idx, uint32_t* pdata) {
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
    if ( bh->flags != 0x31 ) {
        printf("Error: Wrong flags 0x%08X\n", bh->flags);
        return -1;
    }
    
    uint32_t eventDataSize = eh->data_size; // bytes

    //printf("EventDataSize: %8.8x\n", eventDataSize);
    //printf("Header Buffer: %8.8x\n", buffer[idx]);
    //printf("Data: %8.8x\n", buffer[idx+4+eventDataSize/4-1]);

    if ( !(buffer[idx+4+eventDataSize/4-1] == 0xAFFEAFFE or buffer[idx+4+eventDataSize/4-1] == 0xFC00009C or buffer[idx+4+eventDataSize/4-1] == 0xFC00019C) ) {
      printf("Data: %8.8x\n", buffer[idx+4+eventDataSize/4-2]);
      return -1;
    }

    uint32_t dma_buf[4+eventDataSize/4];
    for ( int i = 0; i<4+eventDataSize/4; i++ ) {
      dma_buf[i] = buffer[idx + i];
      //printf("%8.8x %8.8x\n", i, buffer[idx + i]);
    }


    copy_n(&dma_buf[0], sizeof(dma_buf)/4, pdata);

    return sizeof(dma_buf);
}

/*-- Event readout -------------------------------------------------*/

INT read_stream_thread(void *param) {

    // get mudaq
    mudaq::DmaMudaqDevice & mu = *mup;

    uint32_t reset_reg = 0;
    reset_reg = SET_RESET_BIT_EVENT_COUNTER(reset_reg);
    reset_reg = SET_RESET_BIT_DATAGEN(reset_reg);
    reset_reg = SET_RESET_BIT_DATA_PATH(reset_reg);

    // tell framework that we are alive
    signal_readout_thread_active(0, TRUE);

    // obtain ring buffer for inter-thread data exchange
    int rbh = get_event_rbh(0);

    uint32_t max_requested_words = dma_buf_nwords/2;
    
    while (is_readout_thread_enabled()) {

        // don't readout events if we are not running
        if (run_state != STATE_RUNNING) {
            ss_sleep(100);
            continue;
        }

        // get midas buffer
        uint32_t* pdata = nullptr;
        int rb_status = rb_get_wp(rbh, (void**)&pdata, 0);
        if ( rb_status != DB_SUCCESS ) {
            printf("ERROR: rb_get_wp -> rb_status != DB_SUCCESS\n");
            continue;
        }

        // change readout state to switch between pixel and scifi
        uint32_t current_readout_register = mu.read_register_rw(SWB_READOUT_STATE_REGISTER_W);
        uint32_t current_pixel_mask_n = mu.read_register_rw(SWB_LINK_MASK_PIXEL_REGISTER_W);
        uint32_t current_scifi_mask_n = mu.read_register_rw(SWB_LINK_MASK_SCIFI_REGISTER_W);

//        if ( current_pixel_mask_n != 0 && current_scifi_mask_n != 0 ) {
//            current_readout_register ^= 1UL << 7;
//            mu.write_register(SWB_READOUT_STATE_REGISTER_W, current_readout_register);
//            //cout << "1state: " << hex << mu.read_register_rw(SWB_READOUT_STATE_REGISTER_W) << endl;
//        } else if ( current_pixel_mask_n != 0 && current_scifi_mask_n == 0 ) {
            current_readout_register |= (1 << 7);
            mu.write_register(SWB_READOUT_STATE_REGISTER_W, current_readout_register);
            //cout << "2state: " << hex << mu.read_register_rw(SWB_READOUT_STATE_REGISTER_W) << endl;
//        } else if ( current_pixel_mask_n == 0 && current_scifi_mask_n != 0 ) {
//            current_readout_register &= ~(1 << 7);
//            mu.write_register(SWB_READOUT_STATE_REGISTER_W, current_readout_register);
//            //cout << "3state: " << hex << mu.read_register_rw(SWB_READOUT_STATE_REGISTER_W) << endl;
//        } else {
//            //cout << "4state: " << endl;
//            continue;
//        }

        // disable dma
        mu.disable();
        // start dma
        mu.enable_continous_readout(0);

        // wait for requested data
        // request to read dma_buffer_size/2 (count in blocks of 256 bits)
        mu.write_register(GET_N_DMA_WORDS_REGISTER_W, max_requested_words / (256/32));
        //cout << "request " << max_requested_words << endl;

        // reset data path
        mu.write_register(RESET_REGISTER_W, reset_reg);
        usleep(10);
        mu.write_register(RESET_REGISTER_W, 0x0);

        uint32_t cnt_loop = 0;
        while ( (mu.read_register_ro(0x1C) & 1) == 0 ) {
            if (cnt_loop == 100000) break;
            cnt_loop++;
//             check mask for timeout
            usleep(100);
//            if ( mu.read_register_rw(SWB_LINK_MASK_PIXEL_REGISTER_W) == 0 && mu.read_register_rw(SWB_LINK_MASK_SCIFI_REGISTER_W) == 0 ) {
//                break;
//            } else if ( mu.read_register_rw(SWB_LINK_MASK_PIXEL_REGISTER_W) != 0 && mu.read_register_rw(SWB_LINK_MASK_SCIFI_REGISTER_W) != 0 ) {
//                continue;
//            } else if ( mu.read_register_rw(SWB_LINK_MASK_PIXEL_REGISTER_W) != 0 ) {
//                current_readout_register = mu.read_register_rw(SWB_READOUT_STATE_REGISTER_W);
//                current_readout_register |= (1 << 7);
//                mu.write_register(SWB_READOUT_STATE_REGISTER_W, current_readout_register);
//            } else if ( mu.read_register_rw(SWB_LINK_MASK_SCIFI_REGISTER_W) != 0 ) {
//                current_readout_register = mu.read_register_rw(SWB_READOUT_STATE_REGISTER_W);
//                current_readout_register &= ~(1 << 7);
//                mu.write_register(SWB_READOUT_STATE_REGISTER_W, current_readout_register);
//           }
        }

        uint32_t words_written = mu.read_register_ro(0x32);

        // disable dma
        mu.disable();
        // stop readout
        mu.write_register(SWB_READOUT_LINK_REGISTER_W, 0x0);
        mu.write_register(GET_N_DMA_WORDS_REGISTER_W, 0x0);
        // reset all
        mu.write_register(RESET_REGISTER_W, reset_reg);

        // check mask
        if ( mu.read_register_rw(SWB_LINK_MASK_PIXEL_REGISTER_W) == 0 && mu.read_register_rw(SWB_LINK_MASK_SCIFI_REGISTER_W) == 0 ) continue;

        // check cnt loop
        if (cnt_loop == 100000) continue;
        
        // and get lastWritten / endofevent
        // NOTE (24.06.2021): for the moment we dont really care for the endofevent
        // since we only use the DMA write 4kB at the end on the farm firmware
        lastlastWritten = 0;
        uint32_t lastWritten = mu.last_written_addr();
        //uint32_t endofevent = mu.last_endofevent_addr();

        // walk events to find end of last event
        uint32_t offset = 0;
        uint32_t cnt = 0;
        while(true) {
            int rb_status = rb_get_wp(rbh, (void**)&pdata, 10);
            if ( rb_status != DB_SUCCESS ) {
                printf("ERROR: rb_get_wp -> rb_status != DB_SUCCESS\n");
                printf("Events written %d\n", cnt);
                continue;
            }

            // check enough space for header
            if(offset + 4 > lastWritten) break;
            uint32_t eventLength = 16 + dma_buf[(offset + 3) % dma_buf_nwords];
            // check if length is to big (not needed at the moment but we still check it)
            if(eventLength > max_requested_words * 4) {
                printf("ERROR: (eventLength = 0x%08X) > max_event_size\n", eventLength);
                printf("Events written %d\n", cnt);
                break;
            }

            // check enough space for data
            if(offset + eventLength / 4 > lastWritten) break;
            uint32_t size_dma_buf = check_event(dma_buf, offset, pdata);
            //printf("data2: %8.8x offset: %8.8x lastwritten: %8.8x sizeEvent: %d\n", dma_buf[offset], offset, lastWritten, size_dma_buf);
            if ( size_dma_buf == -1 ) {
                printf("size_dma_buf == -1\n");
                printf("Events written %d\n", cnt);
                break;
            }

            offset += eventLength / 4;
            
            if ( offset > lastWritten/2 ) {
                printf("Offset to big\n");
                printf("Events written %d\n", cnt);
                break;
            }
            if ( dma_buf[offset] != 0x00000001 ) {
                printf("dma_buf[offset] != 0x00000001\n");
                printf("Events written %d\n", cnt);
                break;
            }
            
            cnt++;
            pdata+=size_dma_buf;
            rb_increment_wp(rbh, size_dma_buf); // in byte length

        }
    }

    // tell framework that we finished
    signal_readout_thread_active(0, FALSE);

    return 0;
}

uint64_t get_link_active_from_odb(odb o){

   /* get link active from odb */
   uint64_t link_active_from_odb = 0;
   for(uint32_t link = 0; link < MAX_LINKS_PER_SWITCHINGBOARD; link++) {
      // TODO: for Int run we only have on switch -> offset is zero
      int offset = 0;//MAX_LINKS_PER_SWITCHINGBOARD * switch_id;
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