#include "clockboard_a10.h"

#include <iostream>
#include <unistd.h>
#include <chrono>
#include <stdio.h>
#include <sstream>
#include <limits>
#include <fstream>
#include <sys/mman.h>

#include <cassert>


#include "../../common/include/switching_constants.h"
#include "mudaq_device.h"

    clockboard_a10::clockboard_a10(std::string addr, int port):clockboard(addr,port),connected(false){
         /* Open mudaq device */
        mudaq::DmaMudaqDevice mu("/dev/mudaq0");
        if ( !mu.open() ) {
            cout << "Could not open mudaq device " << endl;
            connected = false;
            return;
        }
        if ( !mu.is_ok() ) { 
            connected = false;
            return;
        }
        connected = true;
        cout << "MuDaq is ok" << endl;
    };

    bool clockboard_a10::isConnected(){return connected;};

    int clockboard_a10::init_clockboard(uint16_t clkinvert = 0x0A00, uint16_t rstinvert= 0x0008, uint16_t clkdisable = 0x0AA, uint16_t rstdisable = 0xAA0){

    }

    // Write "reset" commands
    int clockboard_a10::write_command(uint8_t command, uint32_t payload =0, bool has_payload = false){

    };

    int clockboard_a10::write_command(const char * name, uint32_t payload =0, uint16_t address =0){

    };