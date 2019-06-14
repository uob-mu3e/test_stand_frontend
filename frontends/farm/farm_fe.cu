#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <unistd.h>

#include "midas.h"
#include "mcstd.h"
#include "experim.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <sstream>
#include <fstream>


#include "mudaq_device.h"


using namespace std;


/* make frontend functions callable from the C framework */
#ifdef __cplusplus
extern "C" {
#endif

/*-- Globals -------------------------------------------------------*/

/* The frontend name (client name) as seen by other MIDAS clients   */
const char *frontend_name = "rec_board_frontend";
/* The frontend file name, don't change it */
const char *frontend_file_name = __FILE__;

/* frontend_loop is called periodically if this variable is TRUE    */
BOOL frontend_call_loop = FALSE;

/* a frontend status page is displayed with this frequency in ms */
INT display_period = 3000;

/* maximum event size produced by this frontend */
INT max_event_size = 1000000;

/* maximum event size for fragmented events (EQ_FRAGMENTED) */
INT max_event_size_frag = 5 * 1024 * 1024;

/* buffer size to hold events */
INT event_buffer_size = 100 * 1000000;

/* DMA Buffer and related */
volatile uint32_t *dma_buf;
size_t dma_buf_size = MUDAQ_DMABUF_DATA_LEN;
uint32_t dma_buf_nwords = dma_buf_size/sizeof(uint32_t);
uint32_t laddr;
uint32_t newdata;
uint32_t readindex;
bool moreevents;
bool firstevent;
ofstream myfile;

mudaq::DmaMudaqDevice * mup;
mudaq::DmaMudaqDevice::DataBlock block;


/* DB and related */
HNDLE hDB, hdatagenerator;

/*-- Function declarations -----------------------------------------*/

INT frontend_init();
INT frontend_exit();
INT begin_of_run(INT run_number, char *error);
INT end_of_run(INT run_number, char *error);
INT pause_run(INT run_number, char *error);
INT resume_run(INT run_number, char *error);
INT frontend_loop();

INT read_stream_event(char *pevent, INT off);

INT poll_event(INT source, INT count, BOOL test);
INT interrupt_configure(INT cmd, INT source, POINTER_T adr);

/*-- Equipment list ------------------------------------------------*/

EQUIPMENT equipment[] = {

   {"rec_board_frontend",               /* equipment name */
    {1, 0,                   /* event ID, trigger mask */
     "SYSTEM",               /* event buffer */
     EQ_POLLED,              /* equipment type */
     0,                      /* event source crate 0, all stations */
     "MIDAS",                /* format */
     TRUE,                   /* enabled */
     RO_RUNNING |            /* read only when running */
     RO_ODB,                 /* and update ODB */
     100,                    /* poll for 100ms */
     0,                      /* stop run after this event limit */
     0,                      /* number of sub events */
     0,                      /* don't log history */
     "", "", "",},
    read_stream_event,      /* readout routine */
    },

   {""}
};

#ifdef __cplusplus
}
#endif

/********************************************************************\
              Callback routines for system transitions

  These routines are called whenever a system transition like start/
  stop of a run occurs. The routines are called on the following
  occations:

  frontend_init:  When the frontend program is started. This routine
                  should initialize the hardware.

  frontend_exit:  When the frontend program is shut down. Can be used
                  to releas any locked resources like memory, commu-
                  nications ports etc.

  begin_of_run:   When a new run is started. Clear scalers, open
                  rungates, etc.

  end_of_run:     Called on a request to stop a run. Can send
                  end-of-run event and close run gates.

  pause_run:      When a run is paused. Should disable trigger events.

  resume_run:     When a run is resumed. Should enable trigger events.
\********************************************************************/

/*-- Frontend Init -------------------------------------------------*/

INT frontend_init()
{
    set_equipment_status(equipment[0].name, "Initializing...", "yellow");

    /* Get database */
    cm_get_experiment_database(&hDB, NULL);


    /* Book Setting space */
      STREAM_DATAGENERATOR_STR(stream_datagenerator_str);


      /* Map /equipment/Stream/Datagenerator (structure defined in experim.h) */
      char set_str[255];
      sprintf(set_str, "/Equipment/Stream/Datagenerator");
      int status = db_create_record(hDB, 0, set_str, strcomb(stream_datagenerator_str));
      status = db_find_key (hDB, 0, set_str, &hdatagenerator);
      if (status != DB_SUCCESS){
        cm_msg(MINFO,"frontend_init","Key %s not found", set_str);
        return status;
    }


    // Allocate memory for the DMA buffer - this can fail!
    if(cudaMallocHost( (void**)&dma_buf, dma_buf_size ) != cudaSuccess){
        cout << "Allocation failed, aborting!" << endl;
        cm_msg(MERROR, "frontend_init" , "Allocation failed, aborting!");
        return FE_ERR_DRIVER;
    }

    // initialize to zero
    for (int i = 0; i <  dma_buf_nwords ; i++) {
      (dma_buf)[ i ] = 0;
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

    struct mesg user_message;
    user_message.address = dma_buf;
    user_message.size = dma_buf_size;

    /* map memory to bus addresses for FPGA */
    int ret_val = mup->map_pinned_dma_mem( user_message );

    if ( ret_val < 0 ) {
      cout << "Mapping failed " << endl;
      cm_msg(MERROR, "frontend_init" , "Mapping failed");
      mup->disable();
      mup->close();
      free( (void *)dma_buf );
      delete mup;
      return FE_ERR_DRIVER;
    }
    // switch off and reste DMA for now
    mup->disable();
    // switch off the data generator (just in case..)
    mup->write_register(DATAGENERATOR_REGISTER_W, 0x0);
    usleep(2000);
    mup->write_register(LED_REGISTER_W,0x0);
    usleep(5000);

    set_equipment_status(equipment[0].name, "Ready for running", "green");


   return SUCCESS;
}

/*-- Frontend Exit -------------------------------------------------*/

INT frontend_exit()
{
    if(mup){
        mup->disable();
        mup->close();
        delete mup;
   }

    cout<<"frontend exit called"<<endl;
    free( (void *)dma_buf );
    cout<<"frontend exit done"<<endl;

   return SUCCESS;
}

/*-- Begin of Run --------------------------------------------------*/

INT begin_of_run(INT run_number, char *error)
{ 
    set_equipment_status(equipment[0].name, "Starting run", "yellow");

    mudaq::DmaMudaqDevice & mu = *mup;

    /*Reset last written address used for polling */
    laddr = mu.last_written_addr();
    newdata = 0;
    readindex = 0;
    moreevents = false;
    firstevent = true;

    /* Reset dma part and data generator */
    uint32_t reset_reg =0;
    reset_reg = SET_RESET_BIT_DATAGEN(reset_reg);
    mu.write_register_wait(RESET_REGISTER_W, reset_reg,100);
    mu.write_register_wait(RESET_REGISTER_W, 0x0,100);

    /* Enable register on FPGA for continous readout */
    mu.enable_continous_readout(0);


    /* Get DB info for datagenerator */
    INT status,size;
    STREAM_DATAGENERATOR sdg;  // defined in experim.h


    /* Get current  settings */
    size = sizeof(sdg);
    status = db_get_record(hDB, hdatagenerator, &sdg, &size, 0); // HNDLE hdatagenerator obtained in frontend_init()
    if (status != DB_SUCCESS)
    {
        cm_msg(MERROR, "begin_of_run", "cannot retrieve datagenerotor settings record (size of ps=%d)", size);
        return status;
    }

    /* Set up data generator */
    mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, sdg.divider);

    uint32_t datagen_setup =0;
    if(sdg.enable_pixel)
        datagen_setup  = SET_DATAGENERATOR_BIT_ENABLE_PIXEL(datagen_setup);
    if(sdg.enable_fibre)
        datagen_setup  = SET_DATAGENERATOR_BIT_ENABLE_FIBRE(datagen_setup);
    if(sdg.enable_tile)
        datagen_setup  = SET_DATAGENERATOR_BIT_ENABLE_TILE(datagen_setup);
    datagen_setup  = SET_DATAGENERATOR_NPIXEL_RANGE(datagen_setup, sdg.npixel);
    datagen_setup  = SET_DATAGENERATOR_NFIBRE_RANGE(datagen_setup, sdg.nfibre);
    datagen_setup  = SET_DATAGENERATOR_NTILE_RANGE(datagen_setup, sdg.ntile);
    if(sdg.enable)
        datagen_setup  = SET_DATAGENERATOR_BIT_ENABLE(datagen_setup);

    //usleep(500000);

   // cout << "Starting!" << endl;
    cm_msg(MINFO, "begin_of_run" , "addr 0x%x" , mu.last_written_addr());
   // mu.write_register(DATAGENERATOR_REGISTER_W, datagen_setup);
    mu.write_register(DATAGENERATOR_REGISTER_W, 0xffffffff);// start data generator
    mu.write_register(LED_REGISTER_W,0xffffffff);


    set_equipment_status(equipment[0].name, "Running", "#00FF00");

  //  myfile.open("/home/martin/Desktop/memory_content.txt");
  //  if ( !myfile ) {
  //    cout << "Could not open file " << endl;
  //    return -1;
  //  }

  //  myfile << "begin of run " << run_number << endl;


   return SUCCESS;
}

/*-- End of Run ----------------------------------------------------*/

INT end_of_run(INT run_number, char *error)
{
    mudaq::DmaMudaqDevice & mu = *mup;

    //myfile << "stopping run " << run_number << endl;
    //myfile.close();

   uint32_t datagen_setup = mu.read_register_rw(DATAGENERATOR_REGISTER_W);
   datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
   //mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup,1000);
   mu.write_register(LED_REGISTER_W,0x0);
   mu.write_register(DATAGENERATOR_REGISTER_W, 0x0);
   usleep(100000); // wait for remianing data to be pushed
   mu.disable(); // disable DMA
   set_equipment_status(equipment[0].name, "Ready for running", "green");

   return SUCCESS;
}

/*-- Pause Run -----------------------------------------------------*/

INT pause_run(INT run_number, char *error)
{
    mudaq::DmaMudaqDevice & mu = *mup;

    uint32_t datagen_setup = mu.read_register_rw(DATAGENERATOR_REGISTER_W);
    datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
    mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup,1000);

    set_equipment_status(equipment[0].name, "Paused", "yellow");

   return SUCCESS;
}

/*-- Resume Run ----------------------------------------------------*/

INT resume_run(INT run_number, char *error)
{
    mudaq::DmaMudaqDevice & mu = *mup;

    uint32_t datagen_setup = mu.read_register_rw(DATAGENERATOR_REGISTER_W);
    datagen_setup = SET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
    mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup,1000);

    set_equipment_status(equipment[0].name, "Running", "#00FF00");

   return SUCCESS;
}

/*-- Frontend Loop -------------------------------------------------*/

INT frontend_loop()
{
   /* if frontend_call_loop is true, this routine gets called when
      the frontend is idle or once between every event */
   return SUCCESS;
}

/*------------------------------------------------------------------*/

/********************************************************************\

  Readout routines for different events

\********************************************************************/

/*-- Trigger event routines ----------------------------------------*/

INT poll_event(INT source, INT count, BOOL test)
/* Polling routine for events. Returns TRUE if event
   is available. If test equals TRUE, don't return. The test
   flag is used to time the polling */
{

    if(moreevents && !test)
        return 1;

    mudaq::DmaMudaqDevice & mu = *mup;

    for (int i = 0; i < count; i++) {
        uint32_t addr = mu.last_written_addr();
        //cm_msg(MINFO, "poll_event" , "addr 0x%x - laddr 0x%x" , addr, laddr);
        if((addr != laddr) && !test){
            if(firstevent){
                newdata = addr;
                firstevent = false;
            } else {
                if(addr > laddr)
                    newdata = addr - laddr;
                else
                    newdata = 0x10000 - laddr + addr;
            }
            if(newdata > 0x10000){
                return 0;
            }
            laddr = addr;
            //cm_msg(MINFO, "poll_event" , "Data found: newdata 0x%x, laddr 0x%x" , newdata, laddr);
            return 1;
        }
    }
    return 0;
}

/*-- Interrupt configuration ---------------------------------------*/

INT interrupt_configure(INT cmd, INT source, POINTER_T adr)
{
   switch (cmd) {
   case CMD_INTERRUPT_ENABLE:
      break;
   case CMD_INTERRUPT_DISABLE:
      break;
   case CMD_INTERRUPT_ATTACH:
      break;
   case CMD_INTERRUPT_DETACH:
      break;
   }
   return SUCCESS;
}

/*-- Event readout -------------------------------------------------*/

INT read_stream_event(char *pevent, INT off)
{
    // end of run reached
   // if(dma_buf[(readindex)%dma_buf_nwords] == 0xeeeeeeee){
   //     moreevents = false;
   //     return 0;
   // }

    //if(!moreevents)
    //cm_msg(MINFO, "read_stream_event" , "Newdata 0x%x, readindex: 0x%x, laddr: 0x%x" ,
    //       newdata, readindex, laddr);
    /* init bank structure */
    bk_init(pevent);

    DWORD *pdata;
    uint32_t read = 0;
    //    while(read < newdata){
    // Search for an event start    
    //while(dma_buf[(readindex)%dma_buf_nwords] != 0xaa3eaa3e && read < newdata){
     //   if(!read)
         //cm_msg(MERROR, "read_stream_event" , "This should not happen... 0x%x, readindex: 0x%x" ,
         //       dma_buf[(readindex)%dma_buf_nwords], readindex);
        //read++; readindex++;
    //}
    // We are at the start of an event, of which we know
    // that it is complete in the DMA buffer
    // Three word HEAD bank
    bk_create(pevent, "HEAD", TID_DWORD, (void **)&pdata);
//    for(int i =0; i < 7; i ++){

//        readindex++;
//    }

        for(int i =0; i < 8; i ++){
            //cout<< dma_buf[(readindex)%dma_buf_nwords]<<endl;;

          //  myfile << dma_buf[readindex] << endl;
            *pdata++ = dma_buf[readindex];

            //myfile << dma_buf[(readindex)%dma_buf_nwords] << endl;
            //*pdata++ = dma_buf[(readindex)%dma_buf_nwords];
            readindex++;
        }
bk_close(pevent, pdata);
    //readindex++; read++;

    //cm_msg(MDEBUG, "read_stream_event" , "Looking for pixels, found %x" ,dma_buf[(readindex)%dma_buf_size]);

//    // Next there could be pixel data
//    if(dma_buf[(readindex)%dma_buf_nwords] == 0xdecafbad){
//        bk_create(pevent, "PIXH", TID_DWORD, (void **)&pdata);
//        uint32_t data;
//        while(1){
//         readindex++; read++;
//         data = dma_buf[readindex%dma_buf_nwords];
//         if(data == 0xf005ba11 || data == 0xcafebabe || data == 0xdeadbee7)
//             break;
//         if((data & 0xff000000) != 0xaa000000){
//             cm_msg(MERROR, "read_stream_event" , "Wrong pixel data... 0x%x, readindex: 0x%x" ,
//                    data, readindex);
//             break;
//         }
//         *pdata++ = data;
//        }
//        bk_close(pevent, pdata);
//    }



//   // Next there could be fibre data
//    if(dma_buf[(readindex)%dma_buf_nwords] == 0xf005ba11){
//        bk_create(pevent, "FIBH", TID_DWORD, (void **)&pdata);
//        uint32_t data;
//        while(1){
//         readindex++; read++;
//         data = dma_buf[readindex%dma_buf_nwords];
//         if(data == 0xcafebabe || data == 0xdeadbee7)
//             break;
//         if((data & 0xff000000) != 0xbb000000){
//             cm_msg(MERROR, "read_stream_event" , "Wrong fibre data... 0x%x, readindex: 0x%x" ,
//                    data, readindex);
//             break;
//         }
//         *pdata++ = data;
//        }
//        bk_close(pevent, pdata);
//    }


//    // Next there could be tile data
//    if(dma_buf[(readindex)%dma_buf_nwords] == 0xcafebabe){
//         bk_create(pevent, "TILH", TID_DWORD, (void **)&pdata);
//         uint32_t data;
//         while(1){
//          readindex++; read++;
//          data = dma_buf[readindex%dma_buf_nwords];
//          if(data == 0xdeadbee7)
//              break;
//          if((data & 0xff000000) != 0xcc000000){
//              cm_msg(MERROR, "read_stream_event" , "Wrong tile data... 0x%x, readindex: 0x%x" ,
//                     data, readindex);
//              break;
//          }
//          *pdata++ = data;
//         }
//         bk_close(pevent, pdata);
//     }


//     if(dma_buf[readindex%dma_buf_nwords] != 0xdeadbee7){
//         cm_msg(MERROR, "read_stream_event" , "Missed end of event 0x%x, read 0x%x" ,dma_buf[(readindex)%dma_buf_nwords], read);
//     }
//    //cm_msg(MDEBUG, "read_stream_event" , "End of event: data 0x%x, read 0x%x" , newdata, read);
//     readindex++; read++;
//     newdata -= read;

//     if(read < newdata && newdata < 0x10000)
//         moreevents = true;
//     else
//         moreevents = false;


     //cm_msg(MINFO, "read_stream_event" , "At end: read 0x%x, newdata 0x%x, readindex 0x%x" , read, newdata, readindex);

     //cm_msg(MDEBUG, "read_stream_event" , "Data for next: 0x%x" ,dma_buf[(readindex)%dma_buf_nwords]);

     return bk_size(pevent);
}
