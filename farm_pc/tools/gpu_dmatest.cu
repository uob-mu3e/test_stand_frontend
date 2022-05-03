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

#include <fcntl.h>

#include "mudaq_device.h"
#include "a10_counters.h"

#include <condition_variable>
#include <thread>
#include <chrono>

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

uint32_t read_counters(mudaq::DmaMudaqDevice & mu, uint32_t write_value, uint8_t link, uint8_t detector, uint8_t type, uint8_t treeLayer)
{
    // write_value: counter one wants to read
    // link:        addrs for link specific counters
    // detector:    for readout, 0=PIXEL US, 1=PIXEL DS, 2=SCIFI
    // type:        0=link, 1=datapath, 2=tree
    // layer:       layer of the tree 0, 1 or 2

    // counter range for each sub detector
    // 0 to 7:
    //      e_stream_fifo full
    //      e_debug_stream_fifo almost full
    //      bank_builder_idle_not_header
    //      bank_builder_skip_event_dma
    //      bank_builder_event_dma
    //      bank_builder_tag_fifo_full
    //      events send to the farm
    //      e_debug_time_merger_fifo almost full
    // 8 to 3 * (1 + 2 + 4):
    //      tree layer0: 8 to 3 * 4
    //      tree layer1: 8 + 3 * 4 to 3 * 4 + 3 * 2
    //      tree layer2: 8 + 3 * 4 + 3 * 2 to 3 * 4 + 3 * 2 + 3 * 1
    //          layerN link output: # HEADER, SHEADER, HIT
    // 8 + 3 * (1 + 2 + 4) to 8 + 3 * (1 + 2 + 4) + NLINKS * 5:
    //      fifo almost_full
    //      fifo wrfull
    //      # of skip event
    //      # of events
    //      # of sub header

    // link counters
    if ( type == 0 ) {
        write_value += SWB_DATAPATH_CNT + SWB_TREE_CNT * (SWB_LAYER0_OUT_CNT + SWB_LAYER1_OUT_CNT + SWB_LAYER2_OUT_CNT) + link * SWB_LINK_CNT;
    // tree counters
    } else if ( type == 2 ) {
        uint32_t treeLinkOffset[3] = { 0, 4, 6 };
        write_value += SWB_DATAPATH_CNT + SWB_TREE_CNT * (treeLinkOffset[treeLayer] + link);
        //printf("write_value %d, link %d, treeLinkOffset[treeLayer] %d\n", write_value, link, treeLinkOffset[treeLayer]);
    }

    // readout detector
    uint32_t nLinks[2] = {5, 5};
    for ( int i = 0; i < detector; i++ ) {
        //      offset: 8 for general counters   tree offset       link offset
        write_value += (8                      + 3 * (1 + 2 + 4) + nLinks[i] * 5);
    }

    mu.write_register(SWB_COUNTER_REGISTER_W, write_value);
    return mu.read_register_ro(SWB_COUNTER_REGISTER_R);
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

std::mutex cv_m;
std::condition_variable cv;
bool gpu_done;
uint32_t gpu_checkflag = 0;
uint32_t gpu_data_flag = 0;

// device variable to store the counts
__device__ uint32_t Evecnt;
__device__ uint32_t Hitcnt;
__device__ uint32_t Overflowcnt;
__device__ uint32_t bank_data;

// device function to get counted values
__device__ uint32_t get_eventcount() { return Evecnt; }
__device__ uint32_t get_hitcount() { return Hitcnt; }
__device__ uint32_t get_subheaderoverflow() { return Overflowcnt; }
__device__ uint32_t get_bankdata() { return bank_data; }
                              
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

uint32_t evc 	= 0;
uint32_t hits 	= 0;
uint32_t ovrflw = 0;
uint32_t rem 	= 0;

void Set_EventCount(uint32_t ec) { evc = ec; }
void Set_Hits(uint32_t h) { hits = h; }
void Set_SubHeaderOvrflw(uint32_t ov) { ovrflw = ov; }
void Set_Reminders(uint32_t rm) { rem = rm; }

uint32_t Get_EventCount() { return evc; }
uint32_t Get_Hits() { return hits; }
uint32_t Get_SubHeaderOvrflw() { return ovrflw; }
uint32_t Get_Reminders() { return rem; }

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

    uint32_t readout_state_regs = 0;
    if (atoi(argv[5]) == 0) readout_state_regs = SET_USE_BIT_PIXEL_DS(readout_state_regs);
    if (atoi(argv[5]) == 1) readout_state_regs = SET_USE_BIT_PIXEL_US(readout_state_regs);
    if (atoi(argv[5]) == 2) readout_state_regs = SET_USE_BIT_SCIFI(readout_state_regs);
    uint32_t detector = 0;
    if (atoi(argv[5]) == 0) detector = 0;
    if (atoi(argv[5]) == 1) detector = 1;
    if (atoi(argv[5]) == 2) detector = 2;
    // set mask bits
    mu.write_register(mask_n_add, strtol(argv[4], NULL, 16));
    // use stream merger
    if ( atoi(argv[1]) == 0 or atoi(argv[1]) == 2 ) readout_state_regs = SET_USE_BIT_STREAM(readout_state_regs);
    // use datagen
    if ( atoi(argv[1]) == 2 or atoi(argv[1]) == 3 ) readout_state_regs = SET_USE_BIT_GEN_LINK(readout_state_regs);
    // use time merger
    if ( atoi(argv[1]) == 4 or atoi(argv[1]) == 3 ) readout_state_regs = SET_USE_BIT_MERGER(readout_state_regs);
    // write regs
    mu.write_register(SWB_READOUT_STATE_REGISTER_W, readout_state_regs);

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

    //int test_loop = 100;

    // start fpga to ram
    //for (int t = 0; t < test_loop; t++) {
    //t1->start("FPGA_to_RAM");
    mu.write_register(RESET_REGISTER_W, reset_regs);
    mu.write_register(RESET_REGISTER_W, 0x0);
    while ( (mu.read_register_ro(EVENT_BUILD_STATUS_REGISTER_R) & 1) == 0 ) {}
    //t1->stop("FPGA_to_RAM", 0);
    // end fpga to ram

//         for(int i=0; i < 8; i++)
//             cout << hex << "0x" <<  dma_buf[i] << " ";
//         cout << endl;

    uint32_t endofevent = mu.read_register_ro(DMA_CNT_WORDS_REGISTER_R) * 8 - 1;    // replaced dma_buf_nwords
    uint32_t dmabufnwords = mu.read_register_ro(DMA_CNT_WORDS_REGISTER_R) * 8 - 1;

    cout << "SWB_BANK_BUILDER_EVENT_CNT: 0x" << read_counters(mu, SWB_BANK_BUILDER_EVENT_CNT, 0, detector, 1, 0) << endl;

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

   /* h_evt = (uint32_t*)malloc(sizeof(uint32_t));
    h_hit = (uint32_t*)malloc(sizeof(uint32_t));
    h_subovr = (uint32_t*)malloc(sizeof(uint32_t));
    h_bnkd = (uint32_t*)malloc(sizeof(uint32_t));
    h_gpucheck = (uint32_t*)malloc(sizeof(uint32_t));*/  

    //cudaHostAlloc((void **) &B, dma_buf_nwords*sizeof(uint32_t), cudaHostAllocWriteCombined);

    // set size in gpu ram
    /*cudaMalloc(&d_A, dma_buf_nwords*sizeof(uint32_t));
    cudaMalloc(&d_B, dma_buf_nwords*sizeof(uint32_t));
    cudaMalloc(&d_evt, sizeof(uint32_t));
    cudaMalloc(&d_hit, sizeof(uint32_t));
    cudaMalloc(&d_subovr, sizeof(uint32_t));
    cudaMalloc(&d_bnkd, sizeof(uint32_t));
    cudaMalloc(&d_gpucheck, sizeof(uint32_t));*/

    std::thread gpu_checkthread(gpu_check, gpu_checkflag, gpu_data_flag);
    std::thread data_to_gputhread(data_to_gpu, dma_buf, dma_buf_nwords*sizeof(uint32_t), endofevent);

    gpu_checkthread.join();
    data_to_gputhread.join();

    uint32_t eventcounts = Get_EventCount();
    uint32_t hits        = Get_Hits();
    uint32_t subovrflows = Get_SubHeaderOvrflw();
    uint32_t reminders   = Get_Reminders();

    cout << "Event Count GPU" << "\t" << eventcounts << endl;
    cout << "Hit Count GPU" << "\t" << hits << endl;
    cout << "SubHeader Overflow Count GPU" << "\t" << subovrflows << endl;
    cout << "Reminder Count GPU" << "\t" << reminders << endl;
    
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
    //}
    cout << "Check Error %" << endl;

    // Test for the RAM of GPU copy
    uint32_t err_counter = 0;
    for (int i = 0; i < dma_buf_nwords/sizeof(uint32_t); i++) {
    if(!(B[i] == dma_buf[i]+1)) {
    err_counter++;
    //cout << "i" << "\t" << i << "\n";
    }
    //cout << "j" << "\t" << i << "\n";
    }

    const double err_rate = (err_counter / static_cast<double>(dma_buf_nwords)) * 100;
    cout << "error(%) = " << err_rate << "\n";

    cout << "Out of Error %" << endl;

   /* uint64_t mean_fr = t1->average("FPGA_to_RAM");
    uint64_t mean_rg = t2->average("RAM_to_GPU");
    uint64_t mean_gc = t3->average("GPU calc");
    uint64_t mean_gr = t4->average("GPU_to_RAM");

    delete(t1);
    delete(t2);
    delete(t3);
    delete(t4);*/

    /*cout << "-------------------------------- Mean Time ----------------------------------" <<"\n";
    cout << " " <<"\n";

    cout << "FPGA to RAM: " << mean_fr << " " << "micro sec" <<"\n";
    cout << "RAM to GPU: " << mean_rg << " " << "micro sec" <<"\n";
    cout << "GPU calc: " << mean_gc << " " << "micro sec" <<"\n";
    cout << "GPU to RAM: " << mean_gr << " " << "micro sec" <<"\n";

    cout << " " <<"\n";
    cout << "----------------------------------------------------------------------------" <<"\n";*/

    cout << "start to write file" << endl;

    // write data
    char gpu_str[256];
    for (int i = 0; i < size/sizeof(uint32_t); i++) {
    if(i % (1024*1024) == 0) printf("i = %d\n", i);
    sprintf(gpu_str, "%08X", B[i]);
    gpufile << i << "\t" << gpu_str << endl;
    }

    gpufile.close();

    char dma_buf_str[256];
    for (int j = 0 ; j < size/sizeof(uint32_t); j++)    {
    if(j % (1024*1024) == 0) printf("j = %d\n", j);
    sprintf(dma_buf_str, "%08X", dma_buf[j]);
    myfile << j << "\t" << dma_buf_str << endl;
    }

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

    mu.close();

    myfile.close();
    return 0;
}
