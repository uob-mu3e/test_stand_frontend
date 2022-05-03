#pragma once

#include <iostream>
#include <condition_variable>
#include <thread>
#include <chrono>

using namespace std;

{
public:
    /* GPU host and device variables */
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
        // gpu check flag
        uint32_t *h_gpucheck;
        uint32_t *d_gpucheck;

        uint32_t gp_ch = 0;
        
        uint32_t EventCount = 0;
        uint32_t Hit        = 0;
        uint32_t SubOvr     = 0;
        uint32_t Reminder   = 0;
        
        // set size in cpu ram
        /*A = (uint32_t*)malloc(size*sizeof(uint32_t));
        B = (uint32_t*)malloc(size*sizeof(uint32_t));*/

        //cudaHostAlloc((void **) &h_evt, sizeof(uint32_t), cudaHostAllocWriteCombined);
    
        void Set_EventCount(uint32_t ec) { evc = ec; }
        void Set_Hits(uint32_t h) { hits = h; }
        void Set_SubHeaderOvrflw(uint32_t ov) { ovrflw = ov; }
        void Set_Reminders(uint32_t rm) { rem = rm; }

        uint32_t Get_EventCount() { return evc; }
        uint32_t Get_Hits() { return evc; }
        uint32_t Get_SubHeaderOvrflw() { return evc; }
        uint32_t Get_Reminders() { return evc; }
        
        uint32_t evc;
        uint32_t hits;
        uint32_t ovrflw;
        uint32_t rem;

        std::atomic<bool> gpu_done(false);
        std::mutex m;
        std::condition_variable v;
};
