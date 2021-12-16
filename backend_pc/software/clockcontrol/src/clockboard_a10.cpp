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

using std::cout;
using std::endl;


    clockboard_a10::clockboard_a10(std::string addr, int port):clockboard(addr,port),connected(false), mu("/dev/mudaq0") {

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

    int clockboard_a10::init_clockboard(uint16_t clkinvert, uint16_t rstinvert, uint16_t clkdisable, uint16_t rstdisable){
        // reset all (is this safe??)
        uint32_t reset_regs = 0;
        reset_regs = SET_RESET_BIT_ALL(reset_regs);
        mu.write_register(RESET_REGISTER_W, reset_regs);
        mu.write_register(RESET_REGISTER_W, 0x0);

        // Set clocks and reset again  
        mu.write_register(CLK_LINK_0_REGISTER_W, 0xFFF00000);
        mu.write_register(CLK_LINK_1_REGISTER_W, 0xFFF00000);
        mu.write_register(CLK_LINK_2_REGISTER_W, 0xFFF00000);
        mu.write_register(CLK_LINK_3_REGISTER_W, 0xFFF00000);
        mu.write_register(CLK_LINK_REST_REGISTER_W, 0xFFFFFFFFF);
        mu.write_register(RESET_REGISTER_W, reset_regs);
        mu.write_register(RESET_REGISTER_W, 0x0);

        return 1;
    }

    // Write "reset" commands
    int clockboard_a10::write_command(uint8_t command, uint32_t payload, bool has_payload){
        if(has_payload)
            mu.write_register(RESET_LINK_RUN_NUMBER_REGISTER_W, payload);

        mu.write_register(RESET_LINK_CTL_REGISTER_W, 0xF0000000 | command);
        mu.write_register(RESET_LINK_CTL_REGISTER_W, 0x0);

        return 0;
    }

    int clockboard_a10::write_command(const char * name, uint32_t payload, uint16_t address){
        auto it = reset_protocol.commands.find(name);
        if(it != reset_protocol.commands.end()){
            if(address==0){
                return write_command(it->second.command, payload, it->second.has_payload);
             }else{
                 cout << "Addressed commands not yet implemented for A10" << endl;
                 return -1;
             }
            return 0;
        }
    
        cout << "Unknown command " << name << endl;
        return -1;
    }