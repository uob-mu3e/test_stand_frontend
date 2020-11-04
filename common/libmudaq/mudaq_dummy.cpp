//
// Created by Marius Köppel on 09.06.20.
//

#include <atomic>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <time.h>
#include <cstdint>

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fstream>

#include "mudaq_device.h"
#include "mudaq_dummy.h"
#include "utils.hpp"
#include "midas.h"

#define PAGEMAP_LENGTH 8 // each page table entry has 64 bits = 8 bytes
#define PAGE_SHIFT 12    // on x86_64: 4 kB pages, so shifts of 12 bits

using namespace std;

namespace dummy_mudaq {

    dummy_mudaq::DummyMudaqDevice::DummyMudaqDevice(const std::string& path) :
            MudaqDevice(path)
    {
        //
    }

    bool DummyMudaqDevice::is_ok() {
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: is_ok()");
        return true;
    }

    bool DummyMudaqDevice::open() {
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: open()");
        return true;
    }

    void DummyMudaqDevice::close() {
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: close()");
    }

    void DummyMudaqDevice::FEBsc_resetMain(){
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: FEBsc_resetMain()");
    }
    
    void DummyMudaqDevice::FEBsc_resetSecondary(){
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: FEBsc_resetSecondary()");
    }

    int DummyMudaqDevice::FEBsc_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr, bool request_reply, bool retryOnError){
        cm_msg(MINFO, "Dummy MudaqDevice", "Dummy mudaq: FEBsc_write()");
        return SUCCESS;
    }

    int DummyMudaqDevice::FEBsc_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr, bool request_reply){
        cm_msg(MINFO, "Dummy MudaqDevice", "Dummy mudaq: FEBsc_write()");
        return SUCCESS;
    }

    int DummyMudaqDevice::FEBsc_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr){
        cm_msg(MINFO, "Dummy MudaqDevice", "Dummy mudaq: FEBsc_write()");
        return SUCCESS;
    }

    int DummyMudaqDevice::FEBsc_read(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr, bool request_reply, bool retryOnError){
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: FEBsc_read()");
        return length;
    }

    int DummyMudaqDevice::FEBsc_read(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr){
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: FEBsc_read()");
        return length;
    }

    void DummyMudaqDevice::write_register(unsigned idx, uint32_t value){
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: write_register()");
        if(idx > 63){
            cout << "Invalid register address " << idx << endl;
            exit (EXIT_FAILURE);
        }
        _regs_rw[idx] = value;
    }
    
    uint32_t DummyMudaqDevice::read_register_ro(unsigned idx) const {
       if(idx > 63){
           cout << "Invalid register address " << idx << endl;
           exit (EXIT_FAILURE);
       }
       return _regs_ro[idx];
    }
    
    uint32_t DummyMudaqDevice::read_memory_ro(unsigned idx) const {
        if(idx > 64*1024){
           cout << "Invalid memory address " << idx << endl;
           exit (EXIT_FAILURE);
       }
       return _mem_ro[idx & MUDAQ_MEM_RO_MASK];
    }
    
    // DMA dummy mudaq

    dummy_mudaq::DummyDmaMudaqDevice::DummyDmaMudaqDevice(const std::string& path) :
            DummyMudaqDevice(path),
            _dmabuf_ctrl(nullptr),
            _last_end_of_buffer(0)
    {
        //
    }

    bool DummyDmaMudaqDevice::open() {
        cm_msg(MINFO, "Dummy DmaMudaqDevice" , "Open dummy dma mudaq");
        return true;
    }

    void DummyDmaMudaqDevice::close() {
        cm_msg(MINFO, "Dummy DmaMudaqDevice" , "Close dummy dma mudaq");
    }


}

