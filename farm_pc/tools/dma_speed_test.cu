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

void print_usage(){
    cout << "Usage: " << endl;
    cout << "       dma_speed_test <speed_frac> <use_halffull>" << endl;

}

int main(int argc, char *argv[])
{
    if(argc < 3){
        print_usage();
        return -1;
    }

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
    char dma_buf_str_counter[256];
    char dma_buf_str_halfful[256];
    char dma_buf_str_not_halfful[256];

    mudaq::DmaMudaqDevice::DataBlock block;


    // initialize to zero
    for (int i = 0; i <  size/sizeof(uint32_t) ; i++) {
      dma_buf[i] = 0x0;

    }

    ofstream myfile;
    string file_num(argv[1]);
    if (atoi(argv[2]) == 1) {
        filename = "half_memory_content_" + file_num + ".txt";
    }
    else {
        filename = "memory_content_" + file_num + ".txt";
    }
    myfile.open(filename.c_str());
    if ( !myfile ) {
      cout << "Could not open file " << endl;
      return -1;
    }

    myfile << "idx" << "\t" << "counter" << "\t" << "halfful" << "\t" << "nothalfful" << endl;

    // Set up data generator
    datagen_setup = 0;
    uint32_t frac = atoi(argv[1]);
    mu.write_register_wait(DATAGENERATOR_DIVIDER_REGISTER_W, frac, 100);//3E8); // slow down to 64 MBit/s
    //datagen_setup = SET_DATAGENERATOR_BIT_ENABLE_2(datagen_setup);
    datagen_setup = SET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
    if (atoi(argv[2]) == 1) datagen_setup = ((1<<5)| datagen_setup); // enable dma_half_mode
    mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);

    // reset all
    reset_reg = 0;
    reset_reg = SET_RESET_BIT_ALL(reset_reg);
    mu.write_register_wait(RESET_REGISTER_W, reset_reg, 100);

    // Enable register on FPGA for continous readout and enable dma
    noData = 0;
    mu.enable_continous_readout(0);
    usleep(10);

    // disable reset
    mu.write_register(RESET_REGISTER_W, 0x0);

    //sleep(1);

    //lastWritten = mu.last_written_addr();
    //lastlastWritten = mu.last_written_addr();

    //cout << "lastWritten: " << hex << lastWritten << endl;
    //cout << "dma_buf[lastWritten]: " << hex <<  dma_buf[lastWritten] << endl;
    //cout << "counter: " << dma_buf[0] << endl;
    //cout << "halfful: " << dma_buf[1] << endl;
    //cout << "nothalfful: " << dma_buf[3] << endl;

    while(dma_buf[size/sizeof(uint32_t)-8] <= 0){

       // sleep(1);
       // lastWritten = mu.last_written_addr();
       // cout << "lastWritten: " << hex << lastWritten << endl;
       // cout << "dma_buf[lastWritten]: " << hex << dma_buf[lastWritten] << endl;
       // cout << "dma_buf[size/sizeof(uint32_t)-8]: " << dma_buf[size/sizeof(uint32_t)-8] << endl;

       // cout << "counter: " << dma_buf[8] << endl;
       // cout << "halfful: " << dma_buf[9] << endl;
       // cout << "nothalfful: " << dma_buf[10] << endl;

//            errno = mu.read_block(block, dma_buf);
//            if(errno == mudaq::DmaMudaqDevice::READ_SUCCESS){
//                break;
//            }
//            else if(errno == mudaq::DmaMudaqDevice::READ_NODATA){
//                noData += 1;
//                continue;
//            }
//            else {
//                cout << "DMA Read error " << errno << endl;
//                break;
//            }
    }

    //cout << "noData: " << noData << endl;
    mu.disable();
    // stop generator
    datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
    mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);
    mu.write_register_wait(DMA_SLOW_DOWN_REGISTER_W, 0x0, 100);

    // reset all
    reset_reg = 0;
    reset_reg = SET_RESET_BIT_ALL(reset_reg);
    mu.write_register_wait(RESET_REGISTER_W, reset_reg, 100);

//    mu.disable();

    cout << "write file number: " << file_num << endl;

    for (int j = 0 ; j < size/sizeof(uint32_t)/8; j++){ //
        if (j*8 + 8 == size/sizeof(uint32_t)) continue;
        sprintf(dma_buf_str_counter, "%08X", dma_buf[j*8 + 0]);
        sprintf(dma_buf_str_halfful, "%08X", dma_buf[j*8 + 1] + dma_buf[j*8 + 2]);
        sprintf(dma_buf_str_not_halfful, "%08X", dma_buf[j*8 + 3] + dma_buf[j*8 + 4]);
        myfile << j << "\t" << dma_buf_str_counter << "\t" << dma_buf_str_halfful << "\t" << dma_buf_str_not_halfful << endl;
    }

    myfile.close();


    mu.close();

    return 0;
}
