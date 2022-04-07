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

//#include "gpu_variables.h"
#include <condition_variable>
#include <thread>
#include <chrono>

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

//uint32_t *A, *B;          // Host variables
uint32_t *B;
uint32_t *d_A, *d_B;      // Device variables

// Event variables
uint32_t *h_evt;
uint32_t *d_evt;
// Hit variables
uint32_t *h_hit;
uint32_t *d_hit;
// SubHeader Overflow variables
uint32_t *h_subovr;
uint32_t *d_subovr;
// bankdatasize variables
uint32_t *h_bnkd;
uint32_t *d_bnkd;
// gpu check flag
uint32_t *h_gpucheck;
uint32_t *d_gpucheck;

// fpga counter
uint32_t fpga_counter = 0;

std::mutex cv_m;
std::condition_variable cv;
bool gpu_done;
uint32_t gpu_checkflag = 0;
uint32_t gpu_data_flag = 0;

uint32_t evc 	= 0;
uint32_t hits 	= 0;
uint32_t ovrflw = 0;
uint32_t rem 	= 0;

// device variable to store the counts
__device__ uint32_t Evecnt;
__device__ uint32_t Hitcnt;
__device__ uint32_t Overflowcnt;
__device__ uint32_t bank_data;    

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
//INT gpu_counter(uint32_t ngputhread, uint32_t endofevt);

void setup_odb();
void setup_watches();

// device function to get counted values
__device__ uint32_t get_eventcount() { return Evecnt; }
__device__ uint32_t get_hitcount() { return Hitcnt; }
__device__ uint32_t get_subheaderoverflow() { return Overflowcnt; }
__device__ uint32_t get_bankdata() { return bank_data; } 

__global__ void gpu_counters(uint32_t *dest, uint32_t *src, uint32_t Endofevent, uint32_t* evt, uint32_t* hit, uint32_t* sub_ovr, uint32_t* bnkd, size_t n, uint32_t *gpu_flag);

void gpu_counter(uint32_t dma_bufnwords, uint32_t end_of_event);
void data_to_gpu(volatile uint32_t* dma_buf_gpu, uint32_t numf_thr, uint32_t end_ofevent);
void gpu_check(uint32_t gpu_check_flag, uint32_t gpudata_flag);

void Set_EventCount(uint32_t ec) { evc = ec; }
void Set_Hits(uint32_t h) { hits = h; }
void Set_SubHeaderOvrflw(uint32_t ov) { ovrflw = ov; }
void Set_Reminders(uint32_t rm) { rem = rm; }

uint32_t Get_EventCount() { return evc; }
uint32_t Get_Hits() { return hits; }
uint32_t Get_SubHeaderOvrflw() { return ovrflw; }
uint32_t Get_Reminders() { return rem; }

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
    {"GPU0",                /* equipment name */
    {11, 0,                   /* event ID, trigger mask */
     "SYSTEM",               /* event buffer */
     EQ_PERIODIC,                /* equipment type */
     0,                      /* event source crate 0, all stations */
     "MIDAS",                /* format */
     TRUE,                   /* enabled */
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
        {"use_merger", false},         // int
        {"dma_buf_nwords", int(dma_buf_nwords)},
        {"dma_buf_size", int(dma_buf_size)}
    };

    stream_settings.connect("/Equipment/Stream/Settings");

    odb gpu_variables = {
    	{"GPU0", std::array<uint32_t, 5>()}
    };

    gpu_variables.connect("/Equipment/GPU0/Variables");

}

void setup_watches(){

    // datagenerator changed settings
    odb stream_settings("/Equipment/Stream/Settings");
    stream_settings.watch(stream_settings_changed);

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
    cm_msg(MINFO,"farm_fe", "WARNING: For now just use US Pixel data");
    readout_state_regs = SET_USE_BIT_PIXEL_DS(readout_state_regs);
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

INT read_stream_event(char *pevent, INT off)
{
    
   // get mudaq 
   mudaq::DmaMudaqDevice & mu = *mup;
 
   // get odb for errors
   odb error_cnt("/Equipment/Stream Logger/Variables");

   // create bank, pdata, stream buffer is name
   bk_init(pevent);
   DWORD *pdata;
   bk_create(pevent, "GPU0", TID_DWORD, (void **)&pdata);
    
   // check gpu counters
   std::thread gpu_checkthread(gpu_check, gpu_checkflag, gpu_data_flag);
   gpu_checkthread.join();

   uint32_t eventcounts = Get_EventCount();
   uint32_t hits        = Get_Hits();
   uint32_t subovrflows = Get_SubHeaderOvrflw();
   uint32_t reminders   = Get_Reminders();
                                                                          
   // get FPGA counter
   mup->write_register(SWB_COUNTER_REGISTER_W, 4);
   fpga_counter = mup->read_register_ro(SWB_COUNTER_REGISTER_R);
                                                                          
   cout << "Event Count GPU" << "\t" << eventcounts << endl;
   cout << "Event Count FPGA" << "\t" << fpga_counter << endl;
   cout << "Hit Count GPU" << "\t" << hits << endl;
   cout << "SubHeader Overflow Count GPU" << "\t" << subovrflows << endl;
   cout << "Reminder Count GPU" << "\t" << reminders << endl;

   *pdata++ = eventcounts;
   *pdata++ = fpga_counter;
   *pdata++ = hits;
   *pdata++ = subovrflows;
   *pdata++ = reminders;

   bk_close(pevent, pdata);
 
   return bk_size(pevent);
  
}

/*------------------------GPU counters block-----------------------------*/

__device__ void Counter(uint32_t i, uint32_t *db, uint32_t endofevent) {

    uint32_t hpos = 0;
    uint32_t hitcnt = 0;
    uint32_t sbovcnt = 0;

    uint32_t eventlength = db[i+3]/4+3;
    uint32_t datasize = db[i+8]/4;
    int h;
    int sh;
    if (db[i] == 0x00000001 and db[i+eventlength+1] == 0x00000001) {

        for (int j = 0; j < datasize; j++) {
            hpos = i+13+j;

            for (int b = (32-6); b < 32; ++b) {
                if ( (db[hpos] >> b) & 1) h = 1;
                else {
                    h = 0;
                    break;
                }
            }

        if (h != 1) {
            if (db[hpos] == 0xAFFEAFFE or db[hpos] == 0xFC00009C) break;
            hitcnt++;
        }
        else {
            for (int ovr = 0; ovr < (32-16); ++ovr) {
                if ( (db[hpos] >> ovr) & 1){
                    sh = 1;
                    break;
                }
                else sh = 0;
            }
            if (sh) sbovcnt++;
        }
    }
    atomicAdd(&Evecnt, 1);
    atomicAdd(&Hitcnt, hitcnt);
    atomicAdd(&Overflowcnt, sbovcnt);
    atomicAdd(&bank_data, datasize);
    }
}

// Kernel definition
__global__ void gpu_counters(uint32_t *dest, uint32_t *src, uint32_t Endofevent, uint32_t* evt, uint32_t* hit, uint32_t* sub_ovr, uint32_t* bnkd, size_t n, uint32_t *gpu_flag)
{
    /*int blockid = (gridDim.x * blockIdx.y) + blockIdx.x;  // Block Id
    int i = (blockid * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x; // Thread Index */

    uint32_t event_counter = 0;
    uint32_t hit_counter = 0;
    uint32_t subhead_ovrflow = 0;
    uint32_t total_bankdata = 0;

    uint32_t thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x; // Thread Index
    uint32_t n_threads_per_grid = blockDim.x * gridDim.x;

    for (int i = thread_idx; i < n; i += n_threads_per_grid) {
        dest[i] = src[i]+1;
    }
   //event_serial = Serial_EventCount(src, Endofevent);
 
   //printf("Thread id: %d, %d", thread_idx, event_counter);   

   if (thread_idx == 0) {
       Counter(thread_idx, src, Endofevent);
   }
    else if (src[thread_idx] == 0x00000001 and (src[thread_idx-1] == 0xAFFEAFFE or src[thread_idx-1] == 0xFC00009C)) {
        Counter(thread_idx, src, Endofevent);
    }
    else {
        event_counter = 0;
    }

    event_counter = get_eventcount();
    hit_counter = get_hitcount();
    subhead_ovrflow = get_subheaderoverflow();
    total_bankdata = get_bankdata();
    //printf("i %d count: %d \n", thread_idx, event_counter);
    uint32_t reminder = total_bankdata - hit_counter;

    *evt = event_counter;
    *hit = hit_counter;
    *sub_ovr = subhead_ovrflow;
    *bnkd = reminder;
    __syncthreads();
    *gpu_flag = 1;
}

void gpu_counter(uint32_t dma_bufnwords, uint32_t end_of_event) {

    int threadsPerBlock = 1024;                 	 // threads   (Max threadsPerBlock)
    int numBlocks = ceil(dma_bufnwords/threadsPerBlock); // # blocks

    //...
    // Kernel invocation with N threads
    gpu_counters<<<numBlocks, threadsPerBlock>>>(d_B, d_A, end_of_event, d_evt, d_hit, d_subovr, d_bnkd, dma_bufnwords, d_gpucheck);
    //...
    // end GPU calc

    cudaMemcpy(h_evt, d_evt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hit, d_hit, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_subovr, d_subovr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bnkd, d_bnkd, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gpucheck, d_gpucheck, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t EventCount = *h_evt;
    uint32_t Hit        = *h_hit;
    uint32_t SubOvr     = *h_subovr;
    uint32_t Reminder   = *h_bnkd;
    gpu_checkflag      	= *h_gpucheck;

    Set_EventCount(EventCount);
    Set_Hits(Hit);
    Set_SubHeaderOvrflw(SubOvr);
    Set_Reminders(Reminder);
}

void data_to_gpu(volatile uint32_t* dma_buf_gpu, uint32_t numf_thr, uint32_t end_ofevent) {

    h_evt = (uint32_t*)malloc(sizeof(uint32_t));
    h_hit = (uint32_t*)malloc(sizeof(uint32_t));
    h_subovr = (uint32_t*)malloc(sizeof(uint32_t));
    h_bnkd = (uint32_t*)malloc(sizeof(uint32_t));
    h_gpucheck = (uint32_t*)malloc(sizeof(uint32_t));

    cudaHostAlloc((void **) &B, numf_thr*sizeof(uint32_t), cudaHostAllocWriteCombined);

    // set size in gpu ram
    cudaMalloc(&d_A, numf_thr*sizeof(uint32_t));
    cudaMalloc(&d_B, numf_thr*sizeof(uint32_t));
    cudaMalloc(&d_evt, sizeof(uint32_t));
    cudaMalloc(&d_hit, sizeof(uint32_t));
    cudaMalloc(&d_subovr, sizeof(uint32_t));
    cudaMalloc(&d_bnkd, sizeof(uint32_t));
    cudaMalloc(&d_gpucheck, sizeof(uint32_t));

    cout << "DMA to GPU on WAIT" << "\t" << gpu_data_flag << endl;
    std::unique_lock<std::mutex> lk(cv_m);
    cv.wait(lk, []{return gpu_done == true;});
    // send data to gpu
    //cudaMalloc(&d_A, numf_thr*sizeof(uint32_t));
    cudaMemcpy(d_A, const_cast<uint32_t*>(dma_buf_gpu), numf_thr, cudaMemcpyHostToDevice);
    gpu_data_flag = 1;
    cout << "DMA to GPU sent" << "\t" << gpu_data_flag << endl;
    gpu_counter(numf_thr, end_ofevent); // gpu code for counters
}

void gpu_check(uint32_t gpu_check_flag, uint32_t gpudata_flag) {

    if (gpu_check_flag == 1) {
	std::lock_guard<std::mutex> lk(cv_m);
	gpu_done = true;
	cv.notify_one();

 	cout << "Gpu flag con1" << "\t" << gpu_check_flag << endl;
    }

    else if (gpudata_flag != 1) {
	cout << "Gpu flag con2" << "\t" << gpu_check_flag << "\t" << gpu_data_flag << endl;
    std::lock_guard<std::mutex> lk(cv_m);
	gpu_done = true;
    cv.notify_one();
    }
	
    else {
	gpu_done = false;
 	cudaMemcpy(d_gpucheck, &gpu_check_flag, sizeof(uint32_t), cudaMemcpyHostToDevice);
	cout << "Gpu flag con3" << "\t" << gpu_check_flag << "\t" << gpu_data_flag << endl;
    }
}

/*------------------------GPU counters block-----------------------------*/

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
        
        uint32_t endofevent = mu.read_register_ro(DMA_CNT_WORDS_REGISTER_R) * 8 - 1;    // gpu endof event

        // disable dma
        mu.disable();
        
        // get written words from FPGA in bytes
        size_dma_buf = mu.last_endofevent_addr() * 256 / 8;
        
        // copy data
        memcpy(pdata, const_cast<uint32_t*>(dma_buf), size_dma_buf);

        std::thread data_to_gputhread(data_to_gpu, dma_buf, dma_buf_nwords*sizeof(uint32_t), endofevent);

        data_to_gputhread.join();

        // increment write pointer of ring buffer
        rb_increment_wp(rbh, size_dma_buf); // in byte length

    	cudaFree((void *)d_A);
    	cudaFree((void *)d_B);
    	cudaFree((void *)d_evt);
    	cudaFree((void *)d_hit);
    	cudaFree((void *)d_subovr);
    	cudaFree((void *)d_bnkd);
    	cudaFree((void *)d_gpucheck);

    	cudaFreeHost((void *)B);
    	cudaFreeHost((void *)h_evt);
    	cudaFreeHost((void *)h_hit);
    	cudaFreeHost((void *)h_subovr);
    	cudaFreeHost((void *)h_bnkd);
    	cudaFreeHost((void *)h_gpucheck);

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
