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

#include <cuda.h>
#include <cuda_runtime_api.h>

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
        int value = o;
        cm_msg(MINFO, "stream_settings_changed", "Set Divider to %d", value);
        mup->write_register(DATAGENERATOR_DIVIDER_REGISTER_W, o);
        // TODO: test me
    }

    if (name == "Datagen Enable") {
        bool value = o;
        cm_msg(MINFO, "stream_settings_changed", "Set Disable to %d", value);
        //this is set once we start the run
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

    // Map /equipment/Stream/Settings
    odb stream_settings = {
        {"Datagen Divider", 1000},     // int
        {"Datagen Enable", false},     // bool
        {"dma_buf_nwords", int(dma_buf_nwords)},
        {"dma_buf_size", int(dma_buf_size)},
    };

    stream_settings.connect("/Equipment/Stream/Settings", true);

    // add custom page to ODB
    odb custom("/Custom");
    custom["Farm&"] = "farm.html";
    
    // add error cnts to ODB
    odb error_settings = {
        {"DC FIFO ALMOST FUll", 0},
        {"DC LINK FIFO FULL", 0},
        {"TAG FIFO FULL", 0},
        {"MIDAS EVENT RAM FULL", 0},
        {"STREAM FIFO FULL", 0},
        {"DMA HALFFULL", 0},
        {"SKIP EVENT LINK FIFO", 0},
        {"SKIP EVENT DMA RAM", 0},
        {"IDLE NOT EVENT HEADER", 0},
    };
    error_settings.connect("/Equipment/Stream Logger/Variables", true);
    
    // Define history panels
    hs_define_panel("Stream Logger", "MIDAS Bank Builder", {"Stream Logger:DC FIFO ALMOST FUll",
                                                            "Stream Logger:DC LINK FIFO FULL",
                                                            "Stream Logger:TAG FIFO FULL",
                                                            "Stream Logger:MIDAS EVENT RAM FULL",
                                                            "Stream Logger:STREAM FIFO FULL",
                                                            "Stream Logger:DMA HALFFULL",
                                                            "Stream Logger:SKIP EVENT LINK FIFO",
                                                            "Stream Logger:SKIP EVENT DMA RAM",
                                                            "Stream Logger:IDLE NOT EVENT HEADER"});
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
    
    cudaError_t cuda_error = cudaMallocHost( (void**)&dma_buf, dma_buf_size );
    if(cuda_error != cudaSuccess){
        cm_msg(MERROR, "frontend_init" , "Allocation failed, aborting!");
        return FE_ERR_DRIVER;
    }
    
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
   cudaFreeHost((void *)dma_buf);
   
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
   reset_reg |= 1<<RESET_BIT_EVENT_COUNTER;
   reset_reg |= 1<<RESET_BIT_DATAGEN;
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
       // TODO: test me
       mu.write_register_wait(DMA_SLOW_DOWN_REGISTER_W, stream_settings["Datagen Divider"], 100);
       reg = SET_DATAGENERATOR_BIT_ENABLE(reg);
   }
   mu.write_register(DATAGENERATOR_REGISTER_W,reg);

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

   // Finish DMA while waiting for last requested data to be finished
   cm_msg(MINFO, "farm_fe", "Waiting for DMA to finish");
   usleep(1000); // Wait for DMA to finish
   timeout_cnt = 0;
   
   // wait for requested data
   // TODO: in readout th poll on run end reg from febs
   // write variable and check this one here and then disable readout th
   // also check if readout th is disabled by midas at run end
   while ( (mu.read_register_ro(0x1C) & 1) == 0 && timeout_cnt < 100 ) {
       timeout_cnt++;
       usleep(1000);
   };
        
   if(timeout_cnt>=100) {
      cm_msg(MERROR, "farm_fe", "DMA did not finish");
      set_equipment_status(equipment[0].name, "Not OK", "var(--mred)");
//      return CM_TRANSITION_CANCELED;
   }else{
      cm_msg(MINFO, "farm_fe", "DMA is finished\n");
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
    if ( bh->flags != 0x11 ) {
        printf("Error: Wrong flags 0x%08X\n", bh->flags);
        return -1;
    }
    
    uint32_t eventDataSize = eh->data_size; // bytes

    printf("EventDataSize: %8.8x\n", eventDataSize);
    printf("Data: %8.8x\n", buffer[idx+eventDataSize/4]);
    uint32_t endFirstBank = 0;
    for ( int i = idx; i<=idx+4+eventDataSize/4; i++ ) {
        // check if 9c is in data range
        if ( buffer[i] == 0x0000009c and i > 10+idx ) {
            endFirstBank = i;
            break;
        }
    }
    printf("endFirstBank: %8.8x %8.8x\n", endFirstBank, buffer[endFirstBank]);
    printf("idx: %8.8x %8.8x\n", idx, buffer[idx]);
    
    
    if ( endFirstBank == 0 ) {
        printf("Error: endFirstBank == 0\n");
        return -1;
    }
    
    int mod2 = 0;
    if ( (endFirstBank-idx+1) % 2 == 0 ) {
        mod2 = 1;
    }
    
    printf("mod2: %8.8x\n", mod2);
    printf("endFirstBank-idx: %8.8x\n", endFirstBank-idx);
    
    uint32_t dma_buf_dummy[endFirstBank+2+mod2-idx];
    
    uint32_t cnt_bank_words = 0;
    uint32_t head_idx = 0;
    for ( int i = 0; i<=endFirstBank+1+mod2-idx; i++ ) {
        if ( i == 3 ) {
            // change event size
            dma_buf_dummy[i] = (endFirstBank+2+mod2-idx)*4-4*4;
        } else if ( i == 4 ) {
            // change banks size
            dma_buf_dummy[i] = (endFirstBank+2+mod2-idx)*4-6*4;
        } else if ( i == 5 ) {
            // change flags
            dma_buf_dummy[i] = 0x31;
        } else if ( i == 6 ) {
            // change bank name
            dma_buf_dummy[i] = 0x58495049;
        } else if ( i == 7 ) {
            // change bank type
            dma_buf_dummy[i] = 0x6;
        } else if ( i == 8 ) {
            // change bank size
            dma_buf_dummy[i] = 0x0;
        } else if ( i == 9 ) {
            // set reserved
            dma_buf_dummy[i] = 0x0;
        } else if ( i > 9 ) {
            cnt_bank_words += 1;
            if ( mod2 == 0 ) {
                dma_buf_dummy[i] = buffer[idx+i-1];
            } else { 
                if ( i == endFirstBank+1+mod2-idx ) {
                    dma_buf_dummy[i] = 0xAFFEAFFE;
                    printf("Write 0xAFFEAFFE\n");
                } else {
                    dma_buf_dummy[i] = buffer[idx+i-1];
                }
            }
        } else {
            dma_buf_dummy[i] = buffer[idx+i];
        }
    }
    // set bank data size
    dma_buf_dummy[8] = (cnt_bank_words)*4;
    
    printf("Bank Size: %8.8x, %8.8x\n", dma_buf_dummy[8], dma_buf_dummy[endFirstBank+1+mod2-idx]);
    
//     for ( int i=0; i<=endFirstBank+1+mod2-idx; i++ ) {
//         printf("%i %8.8x\n", i, dma_buf_dummy[i]);
//     }
    
    // check data size
    printf("Data Size: %8.8x, %8.8x\n", dma_buf_dummy[3], dma_buf_dummy[4+dma_buf_dummy[3]/4]);
    printf("Buffer Size: %8.8x\n", (endFirstBank+1+mod2-idx)*4);
    
    if ( !(dma_buf_dummy[3+dma_buf_dummy[3]/4] == 0x0000009c or dma_buf_dummy[3+dma_buf_dummy[3]/4] == 0xAfFEAFFE) ) {
        printf("!(dma_buf_dummy[3+dma_buf_dummy[3]/4] == 0x0000009c or dma_buf_dummy[3+dma_buf_dummy[3]/4] == 0xAfFEAFFE)\n");
        printf("%8.8x\n",dma_buf_dummy[3+dma_buf_dummy[3]/4]);
        return -1;
    }
    
//     for ( int i = 0; i<=endFirstBank+1+mod2-idx; i++ ) {
//         printf("%d %8.8x %8.8x\n", i, dma_buf_dummy[i], buffer[i+idx]);
//     }
    
    if ( !(dma_buf_dummy[endFirstBank+1+mod2-idx] == 0x0000009c or dma_buf_dummy[endFirstBank+1+mod2-idx] == 0xAFFEAFFE) ) {
        printf("Error: dma_buf_dummy[endFirstBank+1-idx] != 0x0000009c 0x%08X\n", dma_buf_dummy[endFirstBank+1-idx]);
        for ( int i = 0; i<=endFirstBank+1+mod2-idx; i++ ) {
            printf("%8.8x %8.8x\n", dma_buf_dummy[i], buffer[i+idx]);
        }
        return -1;
    }
    
//     for ( int i=0; i<=endFirstBank+1+mod2-idx; i++ ) {
//         printf("%i %8.8x\n", i, dma_buf_dummy[i]);
//     }
    if ( (endFirstBank+2+mod2-idx)%2 != 0 ) {
        printf("Not mod==0");
    }

    copy_n(&dma_buf_dummy[0], sizeof(dma_buf_dummy)/4, pdata);
    
    return sizeof(dma_buf_dummy);
    

    
    // offset bank relative to event data
    uint32_t bankOffset = 8; // bytes
    // iterate through banks
    while(true) {
        BANK32* b = (BANK32*)(&buffer[idx + 4 + bankOffset / 4]);
        printf("bank: name = %4.4s, data_size = %u bytes, offset = %u bytes\n", b->name, b->data_size, bankOffset);
        printf("bank: name = %8.8x, data_size = %8.8x bytes, offset = %8.8x bytes\n", b->name, b->data_size, bankOffset);
        bankOffset += sizeof(BANK32) + b->data_size; // bytes
        printf("bankOffset = %u bytes\n", bankOffset);
        printf("eventDataSize = %u bytes\n", eventDataSize);
        
        if(bankOffset > eventDataSize) { 
            sleep(10);
            printf("Error: bankOffset > eventDataSize\n");
            return -1; 
        }
        if(bankOffset == eventDataSize) break;
        // TODO: uncomment for new bank format from firmware
//        bankOffset += b->data_size % 8;
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
        // TODO: uncomment for new bank format from firmware
//        src_i += b->data_size % 8;
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
    
    while (is_readout_thread_enabled()) {
        // don't readout events if we are not running
        if (run_state != STATE_RUNNING) {
            set_equipment_status(equipment[0].name, "Not running", "var(--myellow)");
            //ss_sleep(100);
            //TODO: signalling from main thread?
            continue;
        }

        set_equipment_status(equipment[0].name, "Running", "var(--mgreen)");
        
        // get midas buffer
        uint32_t* pdata = nullptr;
        int rb_status = rb_get_wp(rbh, (void**)&pdata, 0);
        if ( rb_status != DB_SUCCESS ) {
            printf("ERROR: rb_get_wp -> rb_status != DB_SUCCESS\n");
            continue;
        }

        // start dma
        mu.enable_continous_readout(0);

        // wait for requested data
        
        mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x2);
        // setup stream
        mu.write_register(SWB_LINK_MASK_PIXEL_REGISTER_W, 0x1);
        // readout datagen
        mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x3);
        // readout link
        //mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x42);
        // request to read dma_buffer_size/2 (count in blocks of 256 bits)
        mu.write_register(0xC, max_requested_words / (256/32));
        
        // reset all
        mu.write_register_wait(RESET_REGISTER_W, 0x1, 100);
//         sleep(1);
        mu.write_register_wait(RESET_REGISTER_W, 0x0, 100);
        
        //while ( (mu.read_register_ro(0x1C) & 1) == 0 ) {}

        // disable dma
        mu.disable();
        // stop readout
        mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0);
        mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x0);
        mu.write_register(SWB_LINK_MASK_PIXEL_REGISTER_W, 0x0);
        mu.write_register(SWB_READOUT_LINK_REGISTER_W, 0x0);
        mu.write_register(GET_N_DMA_WORDS_REGISTER_W, 0x0);
        // reset all
        mu.write_register(RESET_REGISTER_W, 0x1);
        
        // and get lastWritten
        lastlastWritten = 0;
        uint32_t lastWritten = mu.last_written_addr();
//         printf("lastWritten = 0x%08X\n", lastWritten);

        // print dma_buf content
//        for ( int i = lastWritten - 0x100; i < lastWritten + 0x100; i++) {
//            if(i % 8 == 0) printf("[0x%08X]", i);
//            printf("  %08X", dma_buf[i]);
//            if(i % 8 == 7) printf("\n");
//        } printf("\n");

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
            printf("data2: %8.8x offset: %8.8x lastwritten: %8.8x sizeEvent: %d\n", dma_buf[offset], offset, lastWritten, size_dma_buf);
            if ( size_dma_buf == -1 ) {
                printf("size_dma_buf == -1\n");
                printf("Events written %d\n", cnt);
                break;
            }
//             if (offset > 0 ){
//                 printf("Break offset > 0\n");
//                 break;
//             }
            
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
        continue;
        
        if(offset > dma_buf_nwords) offset -= dma_buf_nwords;
        lastWritten = offset;
        printf("Offset: %i\n", offset);


        
        printf("AAAAAAAAAAAAA\n");

        // number of words written to midas buffer
        uint32_t wlen = 0;

        // copy midas buffer and adjust bank data_size to multiple of 8 bytes
        for(int src_i = 0, dst_i = 0; src_i < lastWritten;) {
            int nwords = copy_event(pdata + dst_i, dma_buf + src_i);
            src_i += 4 + dma_buf[src_i + 3] / 4;
            dst_i += nwords;
            wlen = dst_i;
        }
        //lastlastWritten = lastWritten;
        printf("WLEN: %i %8.8x %8.8x\n", wlen, dma_buf[0], dma_buf[wlen]);
        
        copy_n(&dma_buf[0], wlen, pdata);
        

        // copy data to midas and increment wp of the midas buffer
        if(lastWritten < lastlastWritten) {
            // partial copy when wrap around
            printf("CCCCCCCCCCC\n");
            copy_n(&dma_buf[lastlastWritten], dma_buf_nwords - lastlastWritten, pdata);
            wlen += dma_buf_nwords - lastlastWritten;
            lastlastWritten = 0;
        }
        if(lastWritten != lastlastWritten) {
            // complete copy
            printf("DDDDDDDDDDD\n");
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
