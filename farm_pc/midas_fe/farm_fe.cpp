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
uint32_t *dma_buf_copy = (uint32_t *) malloc(dma_buf_size);
uint32_t dma_buf_nwords = dma_buf_size/sizeof(uint32_t);
uint32_t cnt_loop = 0;
uint32_t reset_regs = 0;
uint32_t readout_state_regs = 0;


/* maximum event size produced by this frontend */
INT max_event_size = dma_buf_size; // we fix this for now to 32MB

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 4 * max_event_size;

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

    // set reset registers
    reset_regs = SET_RESET_BIT_DATA_PATH(reset_regs);
    reset_regs = SET_RESET_BIT_DATAGEN(reset_regs);
    reset_regs = SET_RESET_BIT_SWB_TIME_MERGER(reset_regs);
    reset_regs = SET_RESET_BIT_SWB_STREAM_MERGER(reset_regs);

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
    uint32_t current_state;

    cm_msg(MINFO, "stream_settings_changed", "Stream stettings changed");

    if (name == "Datagen Divider") {
        uint32_t divider = o;
        cm_msg(MINFO, "stream_settings_changed", "Set Divider to %d", divider);
        mup->write_register(DATAGENERATOR_DIVIDER_REGISTER_W, divider);
    }

    if (name == "Datagen Enable") {
        cm_msg(MINFO, "stream_settings_changed", "Set Disable Datagen to %s", o ? "y" : "n");
        if (o) {
            current_state = mup->read_register_rw(SWB_READOUT_STATE_REGISTER_W);
            current_state |= (1 << 0);
            mup->write_register(SWB_READOUT_STATE_REGISTER_W, current_state);
        } else {
            current_state = mup->read_register_rw(SWB_READOUT_STATE_REGISTER_W);
            current_state &= ~(1 << 0);
            mup->write_register(SWB_READOUT_STATE_REGISTER_W, current_state);
        }
    }

    if (name == "use_merger") {
        cm_msg(MINFO, "stream_settings_changed", "Set Disable Merger to %s", o ? "y" : "n");
        if (o) {
            current_state = mup->read_register_rw(SWB_READOUT_STATE_REGISTER_W);
            current_state |= (1 << 2);
            mup->write_register(SWB_READOUT_STATE_REGISTER_W, current_state);
        } else {
            current_state = mup->read_register_rw(SWB_READOUT_STATE_REGISTER_W);
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
        {"use_scifi", false},          // bool
        {"use_pixel_ds", false},       // bool
        {"use_pixel_us", false},       // bool
        {"use_merger", false},         // bool
        {"dma_buf_nwords", int(dma_buf_nwords)},
        {"dma_buf_size", int(dma_buf_size)}
    };

    stream_settings.connect("/Equipment/Stream/Settings");

}

void setup_watches(){

    // datagenerator changed settings
    odb stream_settings("/Equipment/Stream/Settings");
    stream_settings.watch(stream_settings_changed);

    // link mask changed settings
    //odb links("/Equipment/Links/Settings/LinkMask");
    //links.watch(link_active_settings_changed);

}

// INIT MUDAQ //////////////////////////////
INT init_mudaq(){
       
    int fd = open("/dev/mudaq0_dmabuf", O_RDWR);
    if(fd < 0) {
        printf("fd = %d\n", fd);
        return FE_ERR_DRIVER;
    }
    dma_buf = (uint32_t*)mmap(nullptr, MUDAQ_DMABUF_DATA_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(dma_buf == MAP_FAILED) {
        cm_msg(MERROR, "frontend_init" , "mmap failed: dmabuf = %x\n", MAP_FAILED);
        return FE_ERR_DRIVER;
    }

    // initialize to zero
    for (uint32_t i = 0; i < dma_buf_nwords ; i++) {
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

    // switch off and reset DMA for now
    mup->disable();
    usleep(2000);

    // switch off the data generator (just in case ..)
    mup->write_register(DATAGENERATOR_REGISTER_W, 0x0);
    usleep(2000);

    // set DMA_CONTROL_W
    mup->write_register(DMA_CONTROL_W, 0x0);

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
   
   return SUCCESS;
}


/*-- Begin of Run --------------------------------------------------*/

INT begin_of_run(INT run_number, char *error)
{ 
    set_equipment_status(equipment[0].name, "Starting run", "var(--myellow)");

    mudaq::DmaMudaqDevice & mu = *mup;

    // set all in reset
    mu.write_register_wait(RESET_REGISTER_W, reset_regs, 100);

    // empty dma buffer
    for (uint32_t i = 0; i < dma_buf_nwords ; i++) {
        (dma_buf)[i] = 0;
    }

    // setup readout registers
    odb stream_settings;
    stream_settings.connect("/Equipment/Stream/Settings");

    readout_state_regs = 0;

    if(stream_settings["Datagen Enable"]) {
        // setup data generator
        cm_msg(MINFO,"farm_fe", "Use datagenerator with divider register %d", stream_settings["Datagen Divider"]);
        mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, stream_settings["Datagen Divider"]);
        readout_state_regs = SET_USE_BIT_GEN_LINK(readout_state_regs);
    }
    if(stream_settings["use_merger"]) {
        // readout merger
        cm_msg(MINFO,"farm_fe", "Use Time Merger");
        readout_state_regs = SET_USE_BIT_MERGER(readout_state_regs);
    } else {
        // readout stream
        cm_msg(MINFO,"farm_fe", "Use Stream Merger");
        readout_state_regs = SET_USE_BIT_STREAM(readout_state_regs);
    }

    if ( stream_settings["use_scifi"] ) {
        readout_state_regs = SET_USE_BIT_SCIFI(readout_state_regs);
    } else if ( stream_settings["use_pixel_ds"] ) {
        readout_state_regs = SET_USE_BIT_PIXEL_DS(readout_state_regs);
    } else if ( stream_settings["use_pixel_us"] ) {
        readout_state_regs = SET_USE_BIT_PIXEL_US(readout_state_regs);
    }

    // write readout register
    mu.write_register(SWB_READOUT_STATE_REGISTER_W, readout_state_regs);

    // link masks 
    // Note: link masks are already set via ODB watch
    mu.write_register(SWB_LINK_MASK_PIXEL_REGISTER_W, stream_settings["mask_n_pixel"]);
    mu.write_register(SWB_LINK_MASK_SCIFI_REGISTER_W, stream_settings["mask_n_scifi"]);

    // release reset
    mu.write_register_wait(RESET_REGISTER_W, 0x0, 100);

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
    mu.write_register(RESET_REGISTER_W, reset_regs);
    mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0);
    mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x0);
    mu.write_register(SWB_READOUT_LINK_REGISTER_W, 0x0);
    mu.write_register(GET_N_DMA_WORDS_REGISTER_W, 0x0);

    set_equipment_status(equipment[0].name, "Ready for running", "var(--mgreen)");

    return SUCCESS;
}

/*-- Pause Run -----------------------------------------------------*/

INT pause_run(INT run_number, char *error)
{
   mudaq::DmaMudaqDevice & mu = *mup;

   // disable DMA
   mu.disable();

   set_equipment_status(equipment[0].name, "Paused", "var(--myellow)");

   return SUCCESS;
}

/*-- Resume Run ----------------------------------------------------*/

INT resume_run(INT run_number, char *error)
{
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

INT read_stream_thread(void *param) {

    // get mudaq
    mudaq::DmaMudaqDevice & mu = *mup;

    // tell framework that we are alive
    signal_readout_thread_active(0, TRUE);

    // obtain ring buffer for inter-thread data exchange
    int rbh = get_event_rbh(0);
    int status;

    // max number of requested words
    uint32_t max_requested_words = dma_buf_nwords / 2;
    
    // get midas buffer
    uint32_t *pdata;
    
    // DMA buffer stuff
    uint32_t size_dma_buf;
    
    // actuall readout loop
    while (is_readout_thread_enabled()) {

        // don't readout events if we are not running
        if (!readout_enabled()) {
            // do not produce events when run is stopped
            ss_sleep(10);// don't eat all CPU
            continue;
        }

        // obtain buffer space with 10 ms timeout
        status = rb_get_wp(rbh, (void **) &pdata, 10);
        
        // just try again if buffer has no space
        if (status == DB_TIMEOUT) {
            printf("WARNING: DB_TIMEOUT\n");
            ss_sleep(10);// don't eat all CPU
            continue;
        }

        // stop if there is an error in the ODB
        if ( status != DB_SUCCESS ) {
            printf("ERROR: rb_get_wp -> rb_status != DB_SUCCESS\n");
            break;
        }

        // request to read dma_buffer_size / 2 (count in blocks of 256 bits)
        mu.write_register(GET_N_DMA_WORDS_REGISTER_W, max_requested_words / (256/32));

        // start dma
        mu.enable_continous_readout(0);
        // wait for requested data
        cnt_loop = 0;
        while ( (mu.read_register_ro(EVENT_BUILD_STATUS_REGISTER_R) & 0x1) == 0x0 ) { 
            if ( cnt_loop > 1000 ) break;
            cnt_loop++; 
            ss_sleep(10); 
        }

        // disable dma
        mu.disable();

        // get written words from FPGA in bytes
        size_dma_buf = mu.last_endofevent_addr() * 256 / 8;

        // copy data
        memcpy(pdata, const_cast<uint32_t*>(dma_buf), size_dma_buf);

        // increment write pointer of ring buffer
        rb_increment_wp(rbh, size_dma_buf); // in byte length         

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
