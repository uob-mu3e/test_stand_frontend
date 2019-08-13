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


    string filename;
    uint32_t datagen_setup;
    uint32_t reset_reg;
    uint32_t lastWritten;
    uint32_t lastlastWritten;
    uint32_t noData;
    char dma_buf_str[256];

    uint32_t dma_speed[] = {0xFF, 0xEE, 0xDD, 0xCC};

    for (int idx = 0; idx < sizeof(dma_speed); idx++ ) {

        ofstream myfile;
        filename = "memory_content_" + to_string( idx ) + ".txt";
        myfile.open(filename.c_str());
        if ( !myfile ) {
          cout << "Could not open file " << endl;
          return -1;
        }

        myfile << "idx" << "\t" << "counter" << endl;

        // Set up data generator
        datagen_setup = 0;
        mu.write_register_wait(DMA_SLOW_DOWN_REGISTER_W, dma_speed[idx], 100);//3E8); // slow down to 64 MBit/s
        datagen_setup = SET_DATAGENERATOR_BIT_ENABLE_2(datagen_setup);
        mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);

        // reset all
        reset_reg = 0;
        reset_reg = SET_RESET_BIT_ALL(reset_reg);
        mu.write_register_wait(RESET_REGISTER_W, reset_reg, 100);

        // Enable register on FPGA for continous readout and enable dma
        lastWritten = mu.last_written_addr();
        lastlastWritten = mu.last_written_addr();
        noData = 0;
        mu.enable_continous_readout(0);
        usleep(10);
        mu.write_register_wait(RESET_REGISTER_W, 0x0, 100);

        auto start_time = std::chrono::high_resolution_clock::now();

        while(true){

            if (lastWritten == 0 || lastWritten == lastlastWritten ){
                noData += 1;
                continue;
            }

            lastlastWritten = 1;

            auto current_time = std::chrono::high_resolution_clock::now();
            auto time = current_time - start_time;
            if(std::chrono::duration_cast<std::chrono::microseconds>(time).count() >= 10000)// 3.6e+9)
              break;
        }

        cout << "no data: " << hex << noData << endl;

        // stop generator
        datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
        mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);
        mu.write_register_wait(DMA_SLOW_DOWN_REGISTER_W, 0x0, 100);

        // reset all
        reset_reg = 0;
        reset_reg = SET_RESET_BIT_ALL(reset_reg);
        mu.write_register_wait(RESET_REGISTER_W, reset_reg, 100);

        mu.disable();

        for (int j = 0 ; j < sizeof(dma_buf); j++){
            sprintf(dma_buf_str, "%08X", dma_buf[j]);
            myfile << j << "\t" << dma_buf_str << endl;
        }

        myfile.close();

    }

    mu.close();

    return 0;
}
