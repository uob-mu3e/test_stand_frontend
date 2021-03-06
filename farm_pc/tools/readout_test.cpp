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

#include <fcntl.h>

#include "mudaq_device.h"
#include "a10_counters.h"

using namespace std;

uint32_t read_counters(mudaq::DmaMudaqDevice & mu, uint32_t write_value, uint8_t link, uint8_t detector, uint8_t type)
{
    // SWB_COUNTER_ADDR_RANGE 7 downto 0
    // SWB_LINK_RANGE 15 downto 8 Link addrs for link specific counters
    // SWB_DETECTOR_RANGE 17 dowto 16 detector for readout, 00->PIXEL US, 01->PIXEL DS, 10->SCIFI
    // SWB_COUNTER_TYPE 18 counter type 0=link, 1=datapath
    mu.write_register(SWB_COUNTER_REGISTER_W, write_value | (link << 8) | (detector << 16) | (type << 18));
    return mu.read_register_ro(SWB_COUNTER_REGISTER_R);
}

uint32_t cntBits(uint32_t n)
{
    uint32_t count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

void print_usage() {
    cout << "Usage: " << endl;
    cout << "       dmatest <readout mode> <stop dma> <readout words> <link mask> <use pixel>" << endl;
    cout << " readout mode: 0 = use stream merger to readout links" << endl;
    cout << " readout mode: 2 = use stream merger to readout datagen" << endl;
    cout << " readout mode: 3 = use time merger to readout datagen" << endl;
    cout << " readout mode: 4 = use time merger to readout links" << endl;
    cout << " stop DMA: 0 = no effect" << endl;
    cout << " stop DMA: 1 = reset FPGA and stop DMA" << endl;
    cout << " readout words: 0 = readout half of DMA buffer" << endl;
    cout << " readout words: 1 = dump DMA readout with time stop" << endl;
    cout << " link mask: 0xFFFF mask links (one is use this link)" << endl;
    cout << " use pixel: 0 if pixel data, 1 if scifi data" << endl;
}

int main(int argc, char *argv[]) {
    if(argc < 6) {
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
        mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0);
        mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x0);
        mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0);
        mu.write_register(SWB_LINK_MASK_PIXEL_REGISTER_W, 0x0);
        mu.write_register(SWB_READOUT_LINK_REGISTER_W, 0x0);
        mu.write_register(GET_N_DMA_WORDS_REGISTER_W, 0x0);
        mu.close();
        return 0;
    }


    size_t dma_buf_size = MUDAQ_DMABUF_DATA_LEN;
    volatile uint32_t *dma_buf;
    size_t size = MUDAQ_DMABUF_DATA_LEN;
    uint32_t dma_buf_nwords = dma_buf_size/sizeof(uint32_t);

/*    cudaError_t cuda_error = cudaMallocHost( (void**)&dma_buf, size );
    if(cuda_error != cudaSuccess){
        cout << "Error: " << cudaGetErrorString(cuda_error) << endl;
        cout << "Allocation failed!" << endl;
        return -1;
    }*/

    int fd = open("/dev/mudaq0_dmabuf", O_RDWR);
    if(fd < 0) {
        printf("fd = %d\n", fd);
        return EXIT_FAILURE;
    }
    dma_buf = (uint32_t*)mmap(nullptr, MUDAQ_DMABUF_DATA_LEN, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if(dma_buf == MAP_FAILED) {
        printf("mmap failed: dmabuf = MAP_FAILED\n");
        return EXIT_FAILURE;
    }

    // initialize to zero
    for(size_t i = 0; i < size/sizeof(uint32_t); i++) {
        dma_buf[i] = 0;
    }

    /* Open mudaq device */
    mudaq::DmaMudaqDevice mu("/dev/mudaq0");
    if ( !mu.open() ) {
        cout << "Could not open device " << endl;
        return -1;
    }

    if ( !mu.is_ok() ) return -1;
    cout << "MuDaq is ok" << endl;

    /* map memory to bus addresses for FPGA */
    int ret_val = 0;
    if ( ret_val < 0 ) {
        cout << "Mapping failed " << endl;
        mu.disable();
        mu.close();
        free( (void *)dma_buf );
        return ret_val;
    }

    // reset all
    uint32_t reset_regs = 0;
    reset_regs = SET_RESET_BIT_DATA_PATH(reset_regs);
    reset_regs = SET_RESET_BIT_DATAGEN(reset_regs);
    reset_regs = SET_RESET_BIT_SWB_STREAM_MERGER(reset_regs);
    reset_regs = SET_RESET_BIT_SWB_TIME_MERGER(reset_regs);
    cout << "Reset Regs: " << hex << reset_regs << endl;
    mu.write_register(RESET_REGISTER_W, reset_regs);

    // request data to read dma_buffer_size/2 (count in blocks of 256 bits)
    uint32_t max_requested_words = dma_buf_nwords/2;
    cout << "request " << max_requested_words << endl;
    mu.write_register(GET_N_DMA_WORDS_REGISTER_W, max_requested_words / (256/32));

    // setup datagen
    mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x2);

    uint32_t mask_n_add;
    if (atoi(argv[5]) == 1) mask_n_add = SWB_LINK_MASK_SCIFI_REGISTER_W;
    if (atoi(argv[5]) == 0) mask_n_add = SWB_LINK_MASK_PIXEL_REGISTER_W;
    uint32_t set_pixel;
    if (atoi(argv[5]) == 1) set_pixel = 0;
    if (atoi(argv[5]) == 0) set_pixel = 1;

    uint32_t readout_state_regs = 0;
    // use stream merger to readout links
    if ( atoi(argv[1]) == 0 ) mu.write_register(mask_n_add, strtol(argv[4], NULL, 16));
    if ( atoi(argv[1]) == 0 ) mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x42 | (set_pixel << 7));
    // use stream merger to readout datagen
    if ( atoi(argv[1]) == 2 ) mu.write_register(mask_n_add, strtol(argv[4], NULL, 16));
    if ( atoi(argv[1]) == 2 ) mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x3 | (set_pixel << 7));
    // use time merger to readout datagen
    if ( atoi(argv[1]) == 3 ) {
        mu.write_register(mask_n_add, strtol(argv[4], NULL, 16));
        readout_state_regs = SET_USE_BIT_GEN_LINK(readout_state_regs);
        readout_state_regs = SET_USE_BIT_MERGER(readout_state_regs);
        readout_state_regs = SET_USE_BIT_PIXEL_US(readout_state_regs);
        //readout_state_regs = SET_USE_BIT_TEST_ERROR(readout_state_regs);
        mu.write_register(SWB_READOUT_STATE_REGISTER_W, readout_state_regs); 
    }
    // use time merger to readout links
    if ( atoi(argv[1]) == 4 ) mu.write_register(mask_n_add, strtol(argv[4], NULL, 16));
    if ( atoi(argv[1]) == 4 ) mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x44| (set_pixel << 7));

    usleep(10);
    mu.write_register(RESET_REGISTER_W, 0x0);

    int readoutCounter =
    while (1) {
        
        printf("  [1] => trigger a readout \n");
        printf("  [2] => readout counters \n");
        printf("  [q] => return \n");
        cout << "Select entry ...";
        cin >> cmd;
        switch(cmd) {
        case '1':
            // start dma
            mu.enable_continous_readout(0);
            // wait for requested data
            while ( (mu.read_register_ro(EVENT_BUILD_STATUS_REGISTER_R) & 1) == 0 ) {}
            // stop dma
            mu.disable();
            break;
        case '2':
            cout << "DataPath counters" << endl;
            cout << "SWB_STREAM_FIFO_FULL_CNT: 0x" << hex << read_counters(mu, SWB_STREAM_FIFO_FULL_CNT, 0, 0, 1) << endl;
            cout << "SWB_BANK_BUILDER_IDLE_NOT_HEADER_CNT: 0x" << hex << read_counters(mu, SWB_BANK_BUILDER_IDLE_NOT_HEADER_CNT, 0, 0, 1) << endl;
            cout << "SWB_BANK_BUILDER_SKIP_EVENT_CNT: 0x" << hex << read_counters(mu, SWB_BANK_BUILDER_SKIP_EVENT_CNT, 0, 0, 1) << endl;
            cout << "SWB_BANK_BUILDER_EVENT_CNT: 0x" << hex << read_counters(mu, SWB_BANK_BUILDER_EVENT_CNT, 0, 0, 1) << endl;
            cout << "SWB_BANK_BUILDER_TAG_FIFO_FULL_CNT: 0x" << hex << read_counters(mu, SWB_BANK_BUILDER_TAG_FIFO_FULL_CNT, 0, 0, 1) << endl;
            cout << "SWB_EVENTS_TO_FARM_CNT: 0x" << hex << read_counters(mu, SWB_EVENTS_TO_FARM_CNT, 0, 0, 1) << endl;

            cout << "Link counters" << endl;
            for ( uint32_t i = 0; i < cntBits(strtol(argv[4], NULL, 16)); i++ ) {
                cout << "SWB_LINK_FIFO_ALMOST_FULL_CNT: 0x" << hex << read_counters(mu, SWB_LINK_FIFO_ALMOST_FULL_CNT, i, 0, 1) << endl;
                cout << "SWB_LINK_FIFO_FULL_CNT: 0x" << hex << read_counters(mu, SWB_LINK_FIFO_FULL_CNT, i, 0, 1) << endl;
                cout << "SWB_SKIP_EVENT_CNT: 0x" << hex << read_counters(mu, SWB_SKIP_EVENT_CNT, i, 0, 1) << endl;
                cout << "SWB_EVENT_CNT: 0x" << hex << read_counters(mu, SWB_EVENT_CNT, i, 0, 1) << endl;
                cout << "SWB_SUB_HEADER_CNT: 0x" << hex << read_counters(mu, SWB_SUB_HEADER_CNT, i, 0, 1) << endl;
            }
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    cout << "start to write file" << endl;

    // stop dma
    mu.disable();
    // stop readout
    mu.write_register(RESET_REGISTER_W, reset_regs);
    mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0);
    mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x0);
    mu.write_register(SWB_LINK_MASK_PIXEL_REGISTER_W, 0x0);
    mu.write_register(SWB_READOUT_LINK_REGISTER_W, 0x0);
    mu.write_register(GET_N_DMA_WORDS_REGISTER_W, 0x0);

    // output data
    auto fout = fopen("memory_content.txt", "w");
    for(size_t j = 0; j < size/sizeof(uint32_t); j++) {
        fprintf(fout, "%ld\t%08X\n", j, dma_buf[j]);
    }
    fclose(fout);

    mu.close();

    return 0;
}
