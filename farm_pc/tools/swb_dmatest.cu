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

#include <fcntl.h>

#include "mudaq_device.h"

using namespace std;

void print_usage(){
    cout << "Usage: " << endl;
    cout << "       dmatest <use_data_gen> <stop_dma> <use_loop>" << endl;
}

int main(int argc, char *argv[])
{

    if(argc < 4){
        print_usage();
        return -1;
    }

    if(atoi(argv[2]) == 1) {
        /* Open mudaq device */
        mudaq::DmaMudaqDevice mu("/dev/mudaq0");
        if ( !mu.open() ) {
            cout << "Could not open device " << endl;
            return -1;
        }

        mu.disable();
        uint32_t datagen_setup = 0;
        datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
        mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);
        mu.write_register_wait(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0, 100);
        mu.write_register_wait(DATA_LINK_MASK_REGISTER_2_W, 0x0, 100);
        mu.write_register_wait(DATA_LINK_MASK_REGISTER_W, 0x0, 100);
        mu.write_register_wait(SWB_READOUT_STATE_REGISTER_W, 0x0, 100);
        mu.close();
        return 0;
    }

    ofstream myfile;
    myfile.open("memory_content.txt");
    if ( !myfile ) {
      cout << "Could not open file " << endl;
      return -1;
    }

    myfile << "idx" << "\t" << "data" << endl;

//    system("echo machmalkeins | sudo -S ../../../common/kerneldriver/compactify.sh");
//    usleep(1000000);
//    system("echo machmalkeins | sudo -S ../../../common/kerneldriver/compactify.sh");
//    usleep(1000000);

    size_t dma_buf_size = MUDAQ_DMABUF_DATA_LEN;
    volatile uint32_t *dma_buf;
    size_t size = MUDAQ_DMABUF_DATA_LEN;
    uint32_t dma_buf_nwords = dma_buf_size/sizeof(uint32_t);

    cudaError_t cuda_error = cudaMallocHost( (void**)&dma_buf, size );
    if(cuda_error != cudaSuccess){
        cout << "Error: " << cudaGetErrorString(cuda_error) << endl;
        cout << "Allocation failed!" << endl;
        return -1;
    }

    int fd = open("/dev/mudaq0_dmabuf", O_RDWR);
    if(fd < 0) {
        printf("fd = %d\n", fd);
        return EXIT_FAILURE;
    }
    dma_buf = (uint32_t*)mmap(nullptr, MUDAQ_DMABUF_DATA_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(dma_buf == MAP_FAILED) {
        printf("mmap failed: dmabuf = %x\n", dma_buf);
        return EXIT_FAILURE;
    }

    // initialize to zero
    for (int i = 0; i <  size/sizeof(uint32_t) ; i++) {
      (dma_buf)[ i ] = 0;
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
    int ret_val = 0;//mu.map_pinned_dma_mem( user_message );

    if ( ret_val < 0 ) {
        cout << "Mapping failed " << endl;
        mu.disable();
        mu.close();
        free( (void *)dma_buf );
        return ret_val;
    }

    // request data to read dma_buffer_size/2 (count in blocks of 256 bits) 
    uint32_t max_requested_words = dma_buf_nwords/64;
    cout << "request " << max_requested_words << endl;
    mu.write_register_wait(0xC, max_requested_words / (256/32), 100);

    // Enable register on FPGA for continous readout and enable dma
    uint32_t lastlastWritten = mu.last_written_addr();
    mu.enable_continous_readout(0);

    // use on link to readout
    if (atoi(argv[1]) == 1) {
        uint32_t USE_GEN_LINK_setup = 0;
        uint32_t USE_LINK_setup = 0;

        //mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x0);
        
        // readout only link 0 without merging
        //USE_GEN_LINK_setup = SET_USE_BIT_GEN_LINK(USE_GEN_LINK_setup);
        //USE_LINK_setup = SET_USE_BIT_LINK(USE_LINK_setup);
        //mu.write_register(SWB_READOUT_LINK_REGISTER_W, 0x0);
        //mu.write_register(SWB_READOUT_STATE_REGISTER_W, USE_LINK_setup);
        
        // setup datagenerator

        mu.write_register(0x13, 0x9);
        mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0);//3E8, 100);//3E8); // slow down to 64 MBit/s
        mu.write_register(SWB_READOUT_STATE_REGISTER_W, USE_GEN_LINK_setup);
    }

    // use stream merger for readout
    if (atoi(argv[1]) == 2) {
        uint32_t USE_GEN_LINK_setup = 0;
        uint32_t USE_STREAM_setup = 0;
        
        // readout only link 0 without merging
        // USE_GEN_LINK_setup = SET_USE_BIT_GEN_LINK(USE_GEN_LINK_setup);
        // USE_STREAM_setup = SET_USE_BIT_STREAM(USE_STREAM_setup);
        // mu.write_register(SWB_READOUT_STATE_REGISTER_W, USE_STREAM_setup);
        
        // setup datagenerator
        mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0);//, 100);//3E8, 100);//3E8); // slow down to 64 MBit/s
        mu.write_register(SWB_READOUT_STATE_REGISTER_W, USE_GEN_LINK_setup);
    }

    // Enable all links (SC)
    mu.write_register_wait(FEB_ENABLE_REGISTER_W, 0xF, 100);
    // Enable all links (DATA)
    //mu.write_register_wait(DATA_LINK_MASK_REGISTER_2_W, 0xF, 100);
    //mu.write_register_wait(DATA_LINK_MASK_REGISTER_W, 0xFFFFFFFF, 100);
    mu.write_register_wait(DATA_LINK_MASK_REGISTER_W, 0xF, 100);
    // Enable only one link
    //mu.write_register_wait(DATA_LINK_MASK_REGISTER_W, 0x1, 100);

    // reset all
    mu.write_register_wait(RESET_REGISTER_W, 0x1, 100);
    sleep(2);
    mu.write_register_wait(RESET_REGISTER_W, 0x0, 100);

    if ( atoi(argv[1]) == 1 ) mu.write_register(0x13, 0x9);
    if ( atoi(argv[1]) == 2 ) mu.write_register(0x13, 0x3);

    mudaq::DmaMudaqDevice::DataBlock block;
    uint32_t newoffset;
    size_t read_words;

    uint32_t event_length = 0;
    uint32_t readindex = 0;
    uint32_t endofevent = 0;
    uint32_t lastendofevent = 0;
    uint32_t lastWritten = 0;
    int errno;
    uint64_t noData = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for(int i=0; i < 8; i++)
        cout << hex << "0x" <<  dma_buf[i] << " ";
    cout << endl;

    if (atoi(argv[3]) == 1) {
        for(int i=0; i < 8; i++)
            cout << hex << "0x" <<  dma_buf[i+8] << " ";
        cout << endl;
        int cnt_loop = 0;
        // wait for requested data
        while ( (mu.read_register_ro(0x1C) & 1) == 0 ) {
            if ( cnt_loop == 1000 ) {
                cnt_loop = 0;
//                cout << mu.read_register_ro(0x1C) << endl;
            }
            cnt_loop = cnt_loop + 1;
        }
//        while(dma_buf[size/2/sizeof(uint32_t)-8] <= 0){

//         if (mu.last_written_addr() == 0) {
//             cout << "last_written" << endl;
//             continue;
//         }
//         if (mu.last_written_addr() == lastlastWritten) {
//             cout << "lastlast_written" << endl;
//             continue;
//         }
//         lastlastWritten = lastWritten;
//         lastWritten = mu.last_written_addr();
// 
//        myfile << "lastWritten" << endl;
//        for (int i = 0; i < 20; i++) {
//        char dma_buf_str[256];
//        sprintf(dma_buf_str, "%08X", dma_buf[lastWritten+i-20]);
//        myfile << lastWritten + i - 20 << "\t" << dma_buf_str << endl;
//        }
// 
//        myfile << "endofevent" << endl;
//         lastendofevent = endofevent;
//         endofevent = mu.last_endofevent_addr(); // now begin of event :)
// 
//         if ((endofevent+1)*8 > lastlastWritten) {
//             cout << "endofevent" << endl;
//             continue;
//         }
//         if ((dma_buf[(endofevent)*8-1] == 0xAFFEAFFE or dma_buf[(endofevent)*8-1] == 0x0000009c) && dma_buf[(endofevent)*8] == 0x1){
//             cout << hex << (endofevent+1)*8 << " " << lastWritten << " " << dma_buf[(endofevent+1)*8] << endl;
//         };
//        for (int i = 0; i < 20; i++) {
//        char dma_buf_str[256];
//        sprintf(dma_buf_str, "%08X", dma_buf[endofevent+i-20]);
//        myfile << endofevent + i - 20 << "\t" << dma_buf_str << endl;
//        }
  //      }
    }
    
    if ( atoi(argv[3]) != 1) {
        for ( int i = 0; i < 3; i++ ) {
            cout << "sleep " << i << "/3 s" << endl;
            sleep(i);
        }
    }

    cout << "start to write file" << endl;


//        if (readindex > 1000000) break;

//        lastWritten = mu.last_written_addr();

////        cout << "lastWritten" << hex << lastWritten << endl;
////        cout << "lastlastWritten" << hex << lastlastWritten << endl;


//        if (lastWritten == 0 || lastWritten == lastlastWritten ){
//            noData += 1;
//            continue;
//        }
//        if(lastlastWritten != 1){
//            for(int i=0; i < 8; i++)
//                cout << hex << "0x" <<  dma_buf[i] << " ";
//            cout << endl;
//        }
//        lastlastWritten = 1;

//        event_length = dma_buf[(readindex+7)%dma_buf_nwords];
//        if (event_length == 0) continue;

////        cout <<"length " << event_length << endl;
//        // do not overtake dma engine
//          if((readindex%dma_buf_nwords) > lastWritten){
//              if(dma_buf_nwords - (readindex % dma_buf_nwords) + lastWritten < event_length * 8 + 1){
////                  usleep(10);
//                  //cout<<"FE SLOW DOWN 1 index"<< (readindex%dma_buf_nwords) <<" lwr "<<lastWritten<<" eventL:"<<event_length<<" nWords "<<dma_buf_nwords<<endl;
//                  continue;
//              }
//          }else{
//              if(lastWritten - (readindex % dma_buf_nwords) < event_length * 8 + 1){
////                  usleep(10);
//                  //cout<<"FE SLOW DOWN 2 index"<< (readindex%dma_buf_nwords) <<" lwr "<<lastWritten<<" eventL:"<<event_length<<" nWords "<<dma_buf_nwords<<endl;
//                  continue;
//              }
//          }

////          auto current_time = std::chrono::high_resolution_clock::now();
////          auto time = current_time - start_time;
////          if(std::chrono::duration_cast<std::chrono::microseconds>(time).count() >= 10000)// 3.6e+9)
////              break;







////        errno = mu.read_block(block, dma_buf);
////        if(errno == mudaq::DmaMudaqDevice::READ_SUCCESS){
////            /* Extract # of words read, set new position in ring buffer */

////            newoffset = block.give_offset();
////            read_words += block.size();

////            auto current_time = std::chrono::high_resolution_clock::now();
////            auto time = current_time - start_time;
////            if(std::chrono::duration_cast<std::chrono::microseconds>(time).count() >= 100000)// 3.6e+9)
////                break;
////        }
////        else if(errno == mudaq::DmaMudaqDevice::READ_NODATA){
////            noData += 1;
////            continue;
////        }
////        else {
////            cout << "DMA Read error " << errno << endl;
////            break;
////        }

////    cout << "No data: " << noData << endl;

//    uint64_t lastmemaddr = mu.last_written_addr();

////    cout << "lastmemaddr is " << hex << lastmemaddr << endl;

////    cout << "Writing file!" << endl;

////    int firstindex = -1;
////    int lastindex = -1;
////    for(uint64_t i = 0; i < lastmemaddr; i++){
////        char dma_buf_str[256];
////        sprintf(dma_buf_str, "%08X", dma_buf[i]);
////        myfile << i << "\t" << dma_buf_str  << endl;
////        if(dma_buf[i] != 0){
////            if(firstindex < 0)
////                firstindex = i;
////        lastindex = i;
////        }
////    }
//    }

    mu.disable();
    // stop generator
    if (atoi(argv[1]) == 1) {
        uint32_t datagen_setup = 0;
        datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
        mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);
        mu.write_register_wait(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0, 100);
    }

    // reset all
    mu.write_register_wait(RESET_REGISTER_W, 0x1, 100);

    char dma_buf_str[256];
    for (int j = 0 ; j < size/sizeof(uint32_t); j++){
        if(j % (1024*1024) == 0) printf("j = %d\n", j);
        sprintf(dma_buf_str, "%08X", dma_buf[j]);
        myfile << j << "\t" << dma_buf_str << endl;
    }

    // stop generator
//    datagen_setup = UNSET_DATAGENERATOR_BIT_ENABLE(datagen_setup);
//    mu.write_register_wait(DATAGENERATOR_REGISTER_W, datagen_setup, 100);
//    mu.write_register_wait(DMA_SLOW_DOWN_REGISTER_W, 0x3E8, 100);//3E8); // slow down to 64 MBit/s

    mu.close();

    myfile.close();
    return 0;
}
