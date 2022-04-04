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
#include "timer.cpp"

#include "midas.h"
#include "mfe.h"
#include "msystem.h"

#include <bits/stdc++.h>

#include "gpu_variables.h"

#include <fcntl.h>

#include "mudaq_device.h"

using namespace std;
using namespace std::chrono;

void print_usage(){
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

/*uint32_t * memcpy_v(uint32_t *dest, volatile uint32_t *src, size_t n)
{
    for (int i = 0; i < n; i++) {
    dest[i] = src[i];
    }

    return dest;
}*/

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// device variable to store the counts
__device__ uint32_t Evecnt;
__device__ uint32_t Hitcnt;
__device__ uint32_t Overflowcnt;
__device__ uint32_t bank_data;

// device function to get counted values
__device__ uint32_t Get_EventCount() { return Evecnt; }
__device__ uint32_t Get_HitCount() { return Hitcnt; }
__device__ uint32_t Get_SubHeaderOverFlow() { return Overflowcnt; }
__device__ uint32_t Get_BankData() { return bank_data; }

// device function for Counters in the GPU
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
__global__ void gpu_counters(uint32_t *dest, uint32_t *src, uint32_t Endofevent, uint32_t* evt, uint32_t* hit, uint32_t* sub_ovr, uint32_t* bnkd, size_t n)
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
        dest[i] = src[i] + 1;
    }

   //event_serial = Serial_EventCount(src, Endofevent);
   if (thread_idx == 0) {
       Counter(thread_idx, src, Endofevent);
   }
    else if (src[thread_idx] == 0x00000001 and (src[thread_idx-1] == 0xAFFEAFFE or src[thread_idx-1] == 0xFC00009C)) {
        Counter(thread_idx, src, Endofevent);
    }
    else {
        event_counter = 0;
    }

    event_counter = Get_EventCount();
    hit_counter = Get_HitCount();
    subhead_ovrflow = Get_SubHeaderOverFlow();
    total_bankdata = Get_BankData();
    //printf("i %d count: %d \n", thread_idx, event_counter);
    uint32_t reminder = total_bankdata - hit_counter;

    *evt = event_counter;
    *hit = hit_counter;
    *sub_ovr = subhead_ovrflow;
    *bnkd = reminder;
}

int main(int argc, char *argv[])
{

    if(argc < 6){
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

    ofstream myfile;
    myfile.open("memory_content.txt");
    if ( !myfile ) {
      cout << "Could not open file " << endl;
      return -1;
    }

    myfile << "idx" << "\t" << "data" << endl;

    ofstream gpufile;
    gpufile.open("gpu_content.txt");
    if ( !gpufile ) {
      cout << "Could not open gpufile " << endl;
      return -1;
    }

    gpufile << "idx" << "\t" << "data" << endl;

    const size_t dma_buf_size = MUDAQ_DMABUF_DATA_LEN;
    volatile uint32_t *dma_buf;
    const size_t size = MUDAQ_DMABUF_DATA_LEN;
    const uint32_t dma_buf_nwords = dma_buf_size/sizeof(uint32_t);

    cout << "DMA WORDS " << dma_buf_nwords << endl;
    cout << "MUDAQ_DMABUF_DATA_LEN " << MUDAQ_DMABUF_DATA_LEN << endl;

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

    /* map memory to bus addresses for FPGA */
    int ret_val = 0;
    if ( ret_val < 0 ) {
        cout << "Mapping failed " << endl;
        mu.disable();
        mu.close();
        free( (void *)dma_buf );
        return ret_val;
    }

    // request data to read dma_buffer_size/2 (count in blocks of 256 bits) 
    uint32_t max_requested_words = dma_buf_nwords/2;
    cout << "request " << max_requested_words << endl;
    mu.write_register(GET_N_DMA_WORDS_REGISTER_W, max_requested_words / (256/32));

    // Enable register on FPGA for continous readout and enable dma
    mu.enable_continous_readout(0);
    
    // setup datagen
    mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x2);
    
    uint32_t mask_n_add;
    if (atoi(argv[5]) == 1) mask_n_add = SWB_LINK_MASK_SCIFI_REGISTER_W;
    if (atoi(argv[5]) == 0) mask_n_add = SWB_LINK_MASK_PIXEL_REGISTER_W;
    uint32_t set_pixel;
    if (atoi(argv[5]) == 1) set_pixel = 0;
    if (atoi(argv[5]) == 0) set_pixel = 1;

    // use stream merger to readout links
    if ( atoi(argv[1]) == 0 ) mu.write_register(mask_n_add, strtol(argv[4], NULL, 16));
    if ( atoi(argv[1]) == 0 ) mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x42 | (set_pixel << 7));
    // use stream merger to readout datagen
    if ( atoi(argv[1]) == 2 ) mu.write_register(mask_n_add, strtol(argv[4], NULL, 16));
    if ( atoi(argv[1]) == 2 ) mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x3 | (set_pixel << 7));
    // use time merger to readout datagen
    if ( atoi(argv[1]) == 3 ) mu.write_register(mask_n_add, strtol(argv[4], NULL, 16));
    if ( atoi(argv[1]) == 3 ) mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x5 | (set_pixel << 7));  
    // use time merger to readout links
    if ( atoi(argv[1]) == 4 ) mu.write_register(mask_n_add, strtol(argv[4], NULL, 16));
    if ( atoi(argv[1]) == 4 ) mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x44| (set_pixel << 7));
    

    // test rw
    auto start_rw = high_resolution_clock::now();
    for (int i = 0; i < 64*1024; i++) {
        mu.write_memory_rw(i, 0xA);
    }
    auto stop_rw = high_resolution_clock::now();

    auto duration_rw = duration_cast<microseconds>(stop_rw - start_rw);
    cout << "Time - 64*1024: " << duration_rw.count() << " " << "micro sec" <<"\n";

    // reset all
    uint32_t reset_regs = 0;
    reset_regs = SET_RESET_BIT_DATA_PATH(reset_regs);
	reset_regs = SET_RESET_BIT_DATAGEN(reset_regs);
    cout << "Reset Regs: " << /*hex <<*/ reset_regs << endl;

	usleep(10);

    uint32_t *A, *B;          // Host variables
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

    // set size in cpu ram

    /*A = (uint32_t*)malloc(size*sizeof(uint32_t));
    B = (uint32_t*)malloc(size*sizeof(uint32_t));*/

    cudaHostAlloc((void **) &A, dma_buf_nwords*sizeof(uint32_t), cudaHostAllocWriteCombined);
    cudaHostAlloc((void **) &B, dma_buf_nwords*sizeof(uint32_t), cudaHostAllocWriteCombined);

    //cudaHostAlloc((void **) &h_evt, sizeof(uint32_t), cudaHostAllocWriteCombined);
    h_evt = (uint32_t*)malloc(sizeof(uint32_t));
    h_hit = (uint32_t*)malloc(sizeof(uint32_t));
    h_subovr = (uint32_t*)malloc(sizeof(uint32_t));
    h_bnkd = (uint32_t*)malloc(sizeof(uint32_t));

    // set size in gpu ram
    cudaMalloc(&d_A, dma_buf_nwords*sizeof(uint32_t));
    cudaMalloc(&d_B, dma_buf_nwords*sizeof(uint32_t));

    cudaMalloc(&d_evt, sizeof(uint32_t));
    cudaMalloc(&d_hit, sizeof(uint32_t));
    cudaMalloc(&d_subovr, sizeof(uint32_t));
    cudaMalloc(&d_bnkd, sizeof(uint32_t));

    //int test_loop = 100;

    Timer* t1 = new Timer();
    Timer* t2 = new Timer();
    Timer* t3 = new Timer();
    Timer* t4 = new Timer();

    // start fpga to ram
    //for (int i = 0; i < test_loop; i++) {
    t1->start("FPGA_to_RAM");
    mu.write_register(RESET_REGISTER_W, reset_regs);
    mu.write_register(RESET_REGISTER_W, 0x0);
    while ( (mu.read_register_ro(EVENT_BUILD_STATUS_REGISTER_R) & 1) == 0 ) {}
    t1->stop("FPGA_to_RAM", 0);
    // end fpga to ram

//         for(int i=0; i < 8; i++)
//             cout << hex << "0x" <<  dma_buf[i] << " ";
//         cout << endl;

    uint32_t endofevent = mu.read_register_ro(DMA_CNT_WORDS_REGISTER_R) * 8 - 1;    // replaced dma_buf_nwords
    uint32_t dmabufnwords = mu.read_register_ro(DMA_CNT_WORDS_REGISTER_R) * 8 - 1;


    // stop dma
    mu.disable();
    // stop readout
    mu.write_register(DATAGENERATOR_DIVIDER_REGISTER_W, 0x0);
    mu.write_register(SWB_READOUT_STATE_REGISTER_W, 0x0);
    mu.write_register(SWB_LINK_MASK_PIXEL_REGISTER_W, 0x0);
    mu.write_register(SWB_READOUT_LINK_REGISTER_W, 0x0);
    mu.write_register(GET_N_DMA_WORDS_REGISTER_W, 0x0);
    mu.write_register(RESET_REGISTER_W, reset_regs);

    //     if (atoi(argv[3]) == 1) {
    //         for(int i=0; i < 8; i++)
    //             cout << hex << "0x" <<  dma_buf[i+8] << " ";
    //         cout << endl;
    //         int cnt_loop = 0;
    //         wait for requested data
    //
    //             if ( cnt_loop == 1000 ) {
    //                 cnt_loop = 0;
    //             }
    //             cnt_loop = cnt_loop + 1;
    //         }
    //     }
    //     end fpga to ram
    //     if ( atoi(argv[3]) != 1) {
    //         for ( int i = 0; i < 3; i++ ) {
    //             cout << "sleep " << i << "/3 s" << endl;
    //             sleep(i);
    //         }
    //     }


    // start RAM to GPU copy
    t2->start("RAM_to_GPU");
    cudaMemcpy(d_A, const_cast<uint32_t*>(dma_buf), dma_buf_nwords*sizeof(uint32_t), cudaMemcpyHostToDevice);
    // end RAM to GPU copy
    t2->stop("RAM_to_GPU", 0);

    // start GPU calc
    /*  dim3 threadsPerBlock(32, 32);   // block dimensions
    dim3 numBlocks(100, 20);        // grid dimensions   */

    int threadsPerBlock = 1024;                 // threads   (Max threadsPerBlock)
    int numBlocks = ceil(dma_buf_nwords/threadsPerBlock); // # blocks

    t3->start("GPU calc");
    //...
    // Kernel invocation with N threads
    gpu_counters<<<numBlocks, threadsPerBlock>>>(d_B, d_A, endofevent, d_evt, d_hit, d_subovr, d_bnkd, dma_buf_nwords);  // <<<Blocks [(B+(N-1))/N], No. of threads per Block>>>
    //...
    // end GPU calc
    t3->stop("GPU calc", 0);

    cudaMemcpy(h_evt, d_evt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_hit, d_hit, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_subovr, d_subovr, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bnkd, d_bnkd, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    uint32_t EventCount = *h_evt;
    uint32_t Hit        = *h_hit;
    uint32_t SubOvr     = *h_subovr;
    uint32_t Reminder   = *h_bnkd; 

    cout << "Event Count GPU" << "\t" << EventCount << endl;
    cout << "Hit Count GPU" << "\t" << Hit << endl;
    cout << "SubHeader Overflow Count GPU" << "\t" << SubOvr << endl;
    cout << "Reminder Count GPU" << "\t" << Reminder << endl;

    Set_EventCount(EventCount);
    Set_Hits(Hit);
    Set_SubHeaderOvrflw(SubOvr);
    Set_Reminders(Reminder);
	
    // start GPU to RAM copy
    t4->start("GPU_to_RAM");
    cudaMemcpy(B, d_B, dma_buf_nwords*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    // end GPU to RAM copy
    t4->stop("GPU_to_RAM", 0);
   // }

    /*ofstream hitfile;
    hitfile.open("Bank_Data.txt");
    if ( !hitfile ) {
      cout << "Could not open file " << endl;
      return -1;
    }*/

    //Counter Loop in the CPU
    uint32_t cnter = 0;
    uint32_t evlen = 0;
    uint32_t tothits = 0;
    uint32_t totsbovr = 0;
    uint32_t totbnkd = 0;

    int i = 0;
    while (i < endofevent) {

        evlen = dma_buf[i+3]/4+3;

        if (dma_buf[i] == 0x00000001 and dma_buf[i+evlen+1] == 0x00000001) cnter++;

        uint32_t datasize = dma_buf[i+8]/4;
        uint32_t hitcnt = 0;
        uint32_t sbovcnt = 0;
        uint32_t bnkd = 0;
        int hpos = 0;

        for (int j = 0; j < datasize; j++) {

            hpos = i+13+j;
            int sh;
            int so;

            for (int sb = (32-6); sb < 32; ++sb) {
                if ( (dma_buf[hpos] >> sb) & 1) so = 1;
                else {
                    so = 0;
                    break;
                }
            }

            if (so) {

                for (int ovr = 0; ovr < (32-16); ++ovr) {
                    if ( (dma_buf[hpos] >> ovr) & 1){
                        sh = 1;
                        break;
                    }
                    else sh = 0;
                }
                if (sh) sbovcnt++;
            }
            else {
                if (dma_buf[hpos] == 0xAFFEAFFE or dma_buf[hpos] == 0xFC00009C) break;
                hitcnt++;
            }

            bnkd = datasize;

            /*bitset<32> val(dma_buf[hpos]);
            std::stringstream sstream;
            sstream << std::hex << dma_buf[hpos];
            std::string result = sstream.str();*/

            //hitfile << hpos << "\t" << "dma_buf in bitset:" << val << "\t" << "Totbnkd: " << totbnkd << "\t" << "Hit: " << hitcnt << "dma_buf in hex:" << result << "\n";

            //"Hit: " << spechitcnt << "\t" << "TotHit: " << totspechits << "dma_buf in hex:" << result << "\n";
        }
        tothits += hitcnt;
        totsbovr += sbovcnt;
        totbnkd += bnkd;
        i += evlen+1;
    }

    //hitfile.close();
    cout << "Event Count CPU" << "\t" << cnter << endl;
    cout << "Hit Count CPU" << "\t" << tothits << endl;
    cout << "SubHeader Overflow Count CPU" << "\t" << totsbovr << endl;
    cout << "Reminder Count CPU" << "\t" << (totbnkd-tothits) << endl;

    // Test for the RAM of GPU copy
    uint32_t err_counter = 0;
    for (int i = 0; i < dma_buf_nwords/sizeof(uint32_t); i++) {
    if(!(B[i] == dma_buf[i]+1)) {
    err_counter++;
    }
    }

    const double err_rate = (err_counter / static_cast<double>(dma_buf_nwords)) * 100;
    cout << "error(%) = " << err_rate << "\n";

    uint64_t mean_fr = t1->average("FPGA_to_RAM");
    uint64_t mean_rg = t2->average("RAM_to_GPU");
    uint64_t mean_gc = t3->average("GPU calc");
    uint64_t mean_gr = t4->average("GPU_to_RAM");

    delete(t1);
    delete(t2);
    delete(t3);
    delete(t4);

    cout << "-------------------------------- Mean Time ----------------------------------" <<"\n";
    cout << " " <<"\n";

    cout << "FPGA to RAM: " << mean_fr << " " << "micro sec" <<"\n";
    cout << "RAM to GPU: " << mean_rg << " " << "micro sec" <<"\n";
    cout << "GPU calc: " << mean_gc << " " << "micro sec" <<"\n";
    cout << "GPU to RAM: " << mean_gr << " " << "micro sec" <<"\n";

    cout << " " <<"\n";
    cout << "----------------------------------------------------------------------------" <<"\n";


    cout << "start to write file" << endl;
    
	// write data

    char gpu_str[256];
    for (int i = 0; i < size/sizeof(uint32_t); i++) {
    if(i % (1024*1024) == 0) printf("i = %d\n", i);
    sprintf(gpu_str, "%08X", B[i]);
    gpufile << i << "\t" << gpu_str << endl;
    }

    gpufile.close();

    cudaFree((void *)d_A);
    cudaFree((void *)d_B);
    cudaFree((void *)d_evt);

    cudaFreeHost((void *)A);
    cudaFreeHost((void *)B);
    cudaFreeHost((void *)h_evt);

    /*free((void *)A);
    free((void *)B);*/

    char dma_buf_str[256];
    for (int j = 0 ; j < size/sizeof(uint32_t); j++)    {
    if(j % (1024*1024) == 0) printf("j = %d\n", j);
    sprintf(dma_buf_str, "%08X", dma_buf[j]);
    myfile << j << "\t" << dma_buf_str << endl;
    }

    mu.close();

    myfile.close();
    return 0;
}
