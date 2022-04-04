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
#include "gpu_variables.h"

using namespace std;
using midas::odb;

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "GPU Frontend";

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

/* GPU Variables */
uint32_t gpu_eventcount = 0;
uint32_t gpu_hits = 0;
uint32_t gpu_subovrflw = 0;
uint32_t gpu_reminder = 0;

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

INT read_stream_event(char *pevent, INT off);
INT read_stream_thread(void *param);

INT poll_event(INT source, INT count, BOOL test);
INT interrupt_configure(INT cmd, INT source, POINTER_T adr);

INT init_mudaq();

/*-- Equipment list ------------------------------------------------*/

EQUIPMENT equipment[] = {

   {"GPU",                /* equipment name */
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
   
    gpu_eventcount = Get_EventCount();
    gpu_hits = Get_Hits();
    gpu_subovrflw = Get_SubHeaderOvrflw();
    gpu_reminder = Get_Reminders();

    return SUCCESS;
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

    

    set_equipment_status(equipment[0].name, "Running", "var(--mgreen)");

return SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT run_number, char *error)
{
    
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
   // read from GPU
 
   return bk_size(pevent);
  
}

/*-- Event readout -------------------------------------------------*/

INT read_stream_thread(void *param) {


    return 0;
}
