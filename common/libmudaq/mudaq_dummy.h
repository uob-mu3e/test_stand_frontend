//
// Created by Marius Köppel on 09.06.20.
//

#ifndef MU3E_ONLINE_TEST_MUDAQ_H
#define MU3E_ONLINE_TEST_MUDAQ_H

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <sys/ioctl.h>		/* ioctl */

#include <boost/dynamic_bitset.hpp>

#include "mudaq_circular_buffer.hpp"
#include "../include/mudaq_device_constants.h"
#include "../include/switching_constants.h" //K.B. changed from "mudaq_registers.h"
#include "utils.hpp"
#include "time.h"
#include <stdlib.h>
#include <stdio.h>
/*??????*/
#include "../kerneldriver/mudaq.h"
#include "mudaq_device.h"

using namespace std;

namespace dummy_mudaq {

    class DummyMudaqDevice : public mudaq::MudaqDevice {
        public:
            virtual bool is_ok();
            virtual bool open();
            virtual void close();

            // a device can exist only once. forbid copying and assignment
            DummyMudaqDevice() = delete;
            DummyMudaqDevice(const DummyMudaqDevice &) = delete;
            DummyMudaqDevice(const MudaqDevice &) = delete;
            DummyMudaqDevice & operator=(const MudaqDevice &) = delete;
            DummyMudaqDevice & operator=(const DummyMudaqDevice &) = delete;

            DummyMudaqDevice(const std::string& path);
            virtual ~DummyMudaqDevice() { close(); }

            void FEBsc_resetMain();
            void FEBsc_resetSecondary();
            int FEBsc_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr, bool request_reply, bool retryOnError);
            int FEBsc_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr, bool request_reply);
            int FEBsc_write(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr);
            int FEBsc_read(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr, bool request_reply, bool retryOnError);
            int FEBsc_read(uint32_t FPGA_ID, uint32_t* data, uint16_t length, uint32_t startaddr);
            void write_register(unsigned idx, uint32_t value);
            uint32_t read_register_ro(unsigned idx) const;
            uint32_t read_memory_ro(unsigned idx) const;
        protected:
            int _fd;

        private:
            const std::string       _path;
            uint32_t _regs_rw[64] = {};
            uint32_t _regs_ro[64] = {};
            uint32_t _mem_ro[64*1024] = {};
            uint32_t _mem_rw[64*1024] = {};

            uint16_t                _last_read_address; // for reading command of slow control

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
