//
// Created by Marius Köppel on 09.06.20.
//

#ifndef MU3E_ONLINE_TEST_MUDAQ_H
#define MU3E_ONLINE_TEST_MUDAQ_H

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <map>

#include <sys/ioctl.h>

#include <boost/dynamic_bitset.hpp>

#include "mudaq_circular_buffer.hpp"
#include "../include/mudaq_device_constants.h"
#include "../include/switching_constants.h"
#include "utils.hpp"
#include "time.h"
#include <stdlib.h>
#include <stdio.h>
#include "../kerneldriver/mudaq.h"
#include "mudaq_device.h"

using namespace std;
namespace mudaq {

    class DummyMudaqDevice : public mudaq::MudaqDevice {
        public:
            // a device can exist only once. forbid copying and assignment
            DummyMudaqDevice() = delete;
            DummyMudaqDevice(const DummyMudaqDevice &) = delete;
            DummyMudaqDevice & operator=(const DummyMudaqDevice &) = delete;

            DummyMudaqDevice(const std::string& path);
            virtual ~DummyMudaqDevice() { close(); }

            virtual bool is_ok();
            virtual bool open();
            virtual void close();
            virtual bool operator!() const;

            virtual void write_register(unsigned idx, uint32_t value);
            virtual void write_register_wait(unsigned idx, uint32_t value, unsigned wait_ns);
            virtual void toggle_register(unsigned idx, uint32_t value, unsigned wait_ns);
            virtual uint32_t read_register_rw(unsigned idx) const;
            virtual uint32_t read_register_ro(unsigned idx) const;
            virtual uint32_t read_memory_ro(unsigned idx) const;
            virtual uint32_t read_memory_rw(unsigned idx) const;
            virtual void write_memory_rw(unsigned idx, uint32_t value);

            virtual int FEBsc_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr);
            virtual int FEBsc_read(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr);
            
            
            
        private:
            const std::string _path;
            uint32_t _regs_rw[64] = {};
            uint32_t _regs_ro[64] = {};
            uint32_t _mem_ro[64*1024] = {};
            uint32_t _mem_rw[64*1024] = {};

            map<uint32_t, array<uint32_t,256>> scmems;

            uint16_t  _last_read_address; // for reading command of slow control
            
            friend std::ostream& operator<<(std::ostream&, const DummyMudaqDevice&);
    };


    class DummyDmaMudaqDevice : public DummyMudaqDevice {
        public:
            virtual bool open();
            virtual void close();
            void disable();

            // a device can exist only once. forbid copying and assignment
            DummyDmaMudaqDevice() = delete;
            DummyDmaMudaqDevice(const DummyDmaMudaqDevice &) = delete;
            DummyDmaMudaqDevice & operator=(const DummyDmaMudaqDevice &) = delete;

            DummyDmaMudaqDevice(const std::string & path);

        private:
            volatile uint32_t * _dmabuf_ctrl;

            uint32_t _last_interrupt;   // updated in read_block() function
            unsigned _last_end_of_buffer;
    };

}

#endif //MU3E_ONLINE_TEST_MUDAQ_H
