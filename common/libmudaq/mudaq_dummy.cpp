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
#include <thread>
#include <chrono>

#include "mudaq_device.h"
#include "mudaq_dummy.h"
#include "utils.hpp"
#include "midas.h"

#define PAGEMAP_LENGTH 8 // each page table entry has 64 bits = 8 bytes
#define PAGE_SHIFT 12    // on x86_64: 4 kB pages, so shifts of 12 bits

using namespace std;

namespace mudaq {

    DummyMudaqDevice::DummyMudaqDevice(const std::string& path) :
        MudaqDevice(path)
    {
        _last_read_address = 0;
    }

    bool DummyMudaqDevice::is_ok() const {
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: is_ok()");
        return true;
    }
    
    bool DummyMudaqDevice::operator!() const {
        return false;
    }

    bool DummyMudaqDevice::open() {
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: open()");
        return true;
    }

    void DummyMudaqDevice::close() {
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: close()");
    }


    int DummyMudaqDevice::FEBsc_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr){
        cm_msg(MINFO, "Dummy MudaqDevice", "Dummy mudaq: FEBsc_write()");
        for(uint16_t i = 0; i < length; i++){
            scmems[FPGA_ID][startaddr + i] = data[i];
        }
        return SUCCESS;
    }



    int DummyMudaqDevice::FEBsc_read(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr){
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: FEBsc_read()");
        for(uint16_t i = 0; i < length; i++){
           data[i] = scmems[FPGA_ID][startaddr + i] ;
        }
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

    void DummyMudaqDevice::write_register_wait(unsigned idx, uint32_t value, unsigned wait_ns){
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq: write_register()");
        if(idx > 63){
            cout << "Invalid register address " << idx << endl;
            exit (EXIT_FAILURE);
        }
        _regs_rw[idx] = value;
        std::this_thread::sleep_for(std::chrono::nanoseconds(wait_ns));

    }

    void DummyMudaqDevice::toggle_register(unsigned idx, uint32_t value, unsigned wait_ns)
    {
        uint32_t old_value = read_register_rw(idx);
        write_register_wait(idx, value, wait_ns);
        write_register(idx, old_value);
    }


    
    uint32_t DummyMudaqDevice::read_register_rw(unsigned idx) const {
       if(idx > 63){
           cout << "Invalid register address " << idx << endl;
           exit (EXIT_FAILURE);
       }
       return _regs_rw[idx];
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

    uint32_t DummyMudaqDevice::read_memory_rw(unsigned idx) const {
        if(idx > 64*1024){
           cout << "Invalid memory address " << idx << endl;
           exit (EXIT_FAILURE);
       }
        return _mem_rw[idx & MUDAQ_MEM_RO_MASK];
    }

    void DummyMudaqDevice::write_memory_rw(unsigned idx, uint32_t value)
    {
        if(idx > 64*1024){
            cout << "Invalid memory address " << idx << endl;
            exit (EXIT_FAILURE);
        }
        else {
            _mem_rw[idx & MUDAQ_MEM_RW_MASK] = value;
        }
    }
    
    // DMA dummy mudaq

   DummyDmaMudaqDevice::DummyDmaMudaqDevice(const std::string& path) :
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

