/**
 * open a mudaq device and measure DMA speed
 * use data generator from counter with 250 MHz clock
 *
 * @author      Dorothea vom Bruch <vombruch@physi.uni-heidelberg.de>
 *              adapted from code by Fabian Foerster and Qinhua Huang
 * @date        2015-01-22
 */

#include <iostream>
#include <unistd.h>
#include <chrono>
#include <stdio.h>
#include <sstream>
#include <limits>
#include <fstream>
#include <sys/mman.h>
#include <chrono>
#include <stdlib.h>
#include <cassert>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


#include "mudaq_device.h"

#define DMA_SLOW_DOWN_REGISTER_W		0x06

using namespace std;

int main(int argc, char *argv[])
{

    ofstream myfile;
    myfile.open("memory_content.txt");
    if ( !myfile ) {
      cout << "Could not open file " << endl;
      return -1;
    }

    myfile << "idx" << "\t" << "data" << "\t" << "event_length" << endl;

    system("echo machmalkeins | sudo -S /home/labor/daq/driver/compactify.sh");
    usleep(1000000);
    system("echo machmalkeins | sudo -S /home/labor/daq/driver/compactify.sh");
    usleep(1000000);

    size_t dma_buf_size = MUDAQ_DMABUF_DATA_LEN;
    volatile uint32_t *dma_buf;
    size_t size = MUDAQ_DMABUF_DATA_LEN;
    uint32_t dma_buf_nwords = dma_buf_size/sizeof(uint32_t);

    if(cudaMallocHost( (void**)&dma_buf, size ) != cudaSuccess){
        cout << "Allocation failed!" << endl;
        return -1;
    }

    // initialize to zero
    for (int i = 0; i <  size/sizeof(uint32_t) ; i++) {
      (dma_buf)[ i ] = 0;

    }

    // Host memory
    uint32_t * cpu_mem = (uint32_t *)malloc(size);
    if(!cpu_mem){
        cout << "CPU memory allocation failed" << endl;
        return -1;
    }


    /* Open mudaq device */
    mudaq::DmaMudaqDevice mu("/dev/mudaq0");
    if ( !mu.open() ) {
        cout << "Could not open device " << endl;
        return -1;
    }

    if ( !mu.is_ok() ) return -1;
    cout << "MuDaq is ok" << endl;

    struct mesg user_message;
    user_message.address = dma_buf;
    user_message.size = size;

    /* map memory to bus addresses for FPGA */
    int ret_val = mu.map_pinned_dma_mem( user_message );

    if ( ret_val < 0 ) {
        cout << "Mapping failed " << endl;
        mu.disable();
        mu.close();
        free( (void *)dma_buf );
        return ret_val;
    }

    // Set up data generator
    uint32_t datagen_setup = 0;
    mu.write_register_wait(DMA_SLOW_DOWN_REGISTER_W, 0x3E8, 100);//3E8); // slow down to 64 MBit/s
    datagen_setup = SET_DATAGENERATOR_BIT_ENABLE_PIXEL(datagen_setup);
    mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);

    // reset all
    uint32_t reset_reg = 0;
    reset_reg = SET_RESET_BIT_ALL(reset_reg);
    mu.write_register_wait(RESET_REGISTER_W, reset_reg, 100);

    // Enable register on FPGA for continous readout and enable dma
    uint32_t lastlastWritten = mu.last_written_addr();
    mu.enable_continous_readout(0);
    usleep(10);
    mu.write_register_wait(RESET_REGISTER_W, 0x0, 100);

    mudaq::DmaMudaqDevice::DataBlock block;
    uint32_t newoffset;
    size_t read_words;

    uint32_t event_length = 0;
    uint32_t readindex = 0;
    uint32_t lastWritten = 0;
    int errno;
    uint64_t noData = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for(int i=0; i < 8; i++)
        cout << hex << "0x" <<  dma_buf[i] << " ";
    cout << endl;

    while(true){


        if (readindex > 1000000) break;

        lastWritten = mu.last_written_addr();

//        cout << "lastWritten" << hex << lastWritten << endl;
//        cout << "lastlastWritten" << hex << lastlastWritten << endl;


        if (lastWritten == 0 || lastWritten == lastlastWritten ){
            noData += 1;
            continue;
        }
        if(lastlastWritten != 1){
            for(int i=0; i < 8; i++)
                cout << hex << "0x" <<  dma_buf[i] << " ";
            cout << endl;
        }

        lastlastWritten = 1;

        event_length = dma_buf[(readindex+7)%dma_buf_nwords];
        if (event_length == 0) continue;

        cout <<"length " << event_length << endl;
        // do not overtake dma engine
          if((readindex%dma_buf_nwords) > lastWritten){
              if(dma_buf_nwords - (readindex % dma_buf_nwords) + lastWritten < event_length * 8 + 1){
//                  usleep(10);
                  //cout<<"FE SLOW DOWN 1 index"<< (readindex%dma_buf_nwords) <<" lwr "<<lastWritten<<" eventL:"<<event_length<<" nWords "<<dma_buf_nwords<<endl;
                  continue;
              }
          }else{
              if(lastWritten - (readindex % dma_buf_nwords) < event_length * 8 + 1){
//                  usleep(10);
                  //cout<<"FE SLOW DOWN 2 index"<< (readindex%dma_buf_nwords) <<" lwr "<<lastWritten<<" eventL:"<<event_length<<" nWords "<<dma_buf_nwords<<endl;
                  continue;
              }
          }

//          auto current_time = std::chrono::high_resolution_clock::now();
//          auto time = current_time - start_time;
//          if(std::chrono::duration_cast<std::chrono::microseconds>(time).count() >= 10000)// 3.6e+9)
//              break;



        for (int j = 0 ; j < event_length; j++){
            char dma_buf_str[256];
            sprintf(dma_buf_str, "%08X", dma_buf[(readindex + 6)%dma_buf_nwords]);
            myfile << readindex + 6 << "\t" << dma_buf_str << "\t" << event_length  << endl;
            readindex = readindex + 8;
        }



//        errno = mu.read_block(block, dma_buf);
//        if(errno == mudaq::DmaMudaqDevice::READ_SUCCESS){
//            /* Extract # of words read, set new position in ring buffer */

//            newoffset = block.give_offset();
//            read_words += block.size();

//            auto current_time = std::chrono::high_resolution_clock::now();
//            auto time = current_time - start_time;
//            if(std::chrono::duration_cast<std::chrono::microseconds>(time).count() >= 100000)// 3.6e+9)
//                break;
//        }
//        else if(errno == mudaq::DmaMudaqDevice::READ_NODATA){
//            noData += 1;
//            continue;
//        }
//        else {
//            cout << "DMA Read error " << errno << endl;
//            break;
//        }

//    cout << "No data: " << noData << endl;

    uint64_t lastmemaddr = mu.last_written_addr();

    cout << "lastmemaddr is " << hex << lastmemaddr << endl;

    cout << "Writing file!" << endl;

//    int firstindex = -1;
//    int lastindex = -1;
//    for(uint64_t i = 0; i < lastmemaddr; i++){
//        char dma_buf_str[256];
//        sprintf(dma_buf_str, "%08X", dma_buf[i]);
//        myfile << i << "\t" << dma_buf_str  << endl;
//        if(dma_buf[i] != 0){
//            if(firstindex < 0)
//                firstindex = i;
//        lastindex = i;
//        }
//    }
    }

    // stop generator
    datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
    mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);
    mu.write_register_wait(DMA_SLOW_DOWN_REGISTER_W, 0x3E8, 100);//3E8); // slow down to 64 MBit/s

    mu.disable();
    mu.close();

    myfile.close();
    return 0;
}
