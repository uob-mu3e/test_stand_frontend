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

    bool DummyMudaqDevice::is_ok() const {
        cm_msg(MINFO, "Dummy MudaqDevice" , "Dummy mudaq is ok");
        return true;
    }

    bool DummyMudaqDevice::open() {
        cm_msg(MINFO, "Dummy MudaqDevice" , "Open dummy mudaq");
        return true;
    }

    void DummyMudaqDevice::close() {
        cm_msg(MINFO, "Dummy MudaqDevice" , "Close dummy mudaq");
    }

    void DummyMudaqDevice::FEBsc_resetMaster(){
        cm_msg(MINFO, "Dummy MudaqDevice" , "Resetting slow control master");
    }
    void DummyMudaqDevice::FEBsc_resetSlave(){
        cm_msg(MINFO, "Dummy MudaqDevice" , "Resetting slow control slave");
    }

    int DummyMudaqDevice::FEBsc_read(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr, bool request_reply, bool retryOnError){
        return length;
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

