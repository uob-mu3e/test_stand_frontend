/**
 * test slow control Arria 10 to FEB
 * 
 *
 * @author      Marius Koeppel <mkoeppel@uni-mainz.de>
 *
 * @date        2021-12-13
 */

#include <iostream>
#include <unistd.h>
#include <chrono>
#include <stdio.h>
#include <sstream>
#include <limits>
#include <fstream>
#include <math.h>
#include <sys/mman.h>

#include <cassert>


#include "../../switching_pc/slowcontrol/FEBSlowcontrolInterface.h"
#include "../../common/include/switching_constants.h"
#include "mudaq_device.h"

using namespace std;

void print_usage() {
    cout << "Usage: " << endl;
    cout << "       test_slowcontrol <FPGA Addr>" << endl;
}

int main(int argc, char *argv[])
{
    if(argc < 2) {
        print_usage();
        return -1;
    }


    
    /* Open mudaq device */
    mudaq::DmaMudaqDevice mu("/dev/mudaq0");
    if ( !mu.open() ) {
      cout << "Could not open device " << endl;
      return -1;
    }
    if ( !mu.is_ok() ) return -1;
    cout << "MuDaq is ok" << endl;
    
    cout << "Reset Link Status Reg " << RESET_LINK_STATUS_REGISTER_R << endl;
    
    // reset all
    uint32_t reset_regs = 0;
    reset_regs = SET_RESET_BIT_ALL(reset_regs);
    mu.write_register(RESET_REGISTER_W, reset_regs);
    mu.write_register(RESET_REGISTER_W, 0x0);

    // test write
    const vector<uint32_t> data = {0,1,5};
    uint32_t startaddr = 0;
    uint32_t FPGA_ID = atoi(argv[1]);
    uint32_t packet_type = PACKET_TYPE_SC_WRITE;

    if(startaddr >= pow(2,16)){
        cout << "Address out of range: " << std::hex << startaddr << endl;
    }

    if(!data.size()){
        cout << "Length zero" << endl;
    }

    if(!(mu.read_register_ro(SC_MAIN_STATUS_REGISTER_R)&0x1)){ // FPGA is busy, should not be here...
       cout << "FPGA busy" << endl;
    }

    // two most significant bits are 0
    mu.write_memory_rw(0, PACKET_TYPE_SC << 26 | packet_type << 24 | ((uint16_t)(1UL << FPGA_ID)) << 8 | 0xBC);
    mu.write_memory_rw(1, startaddr);
    mu.write_memory_rw(2, data.size());

    for (uint32_t i = 0; i < data.size(); i++) {
        mu.write_memory_rw(3 + i, data[i]);
    }
    mu.write_memory_rw(3 + data.size(), 0x0000009c);

    // SC_MAIN_LENGTH_REGISTER_W starts from 1
    // length for SC Main does not include preamble and trailer, thats why it is 2+length
    mu.write_register(SC_MAIN_LENGTH_REGISTER_W, 2 + data.size());
    mu.write_register(SC_MAIN_ENABLE_REGISTER_W, 0x0);
    mu.toggle_register(SC_MAIN_ENABLE_REGISTER_W, 0x1,100);
    // firmware regs SC_MAIN_ENABLE_REGISTER_W so that it only starts on a 0->1 transition

    // check if SC Main is done
    uint32_t count = 0;
    while(count < 10){
        if ( mu.read_register_ro(SC_MAIN_STATUS_REGISTER_R) & 0x1 ) break;
        count++;
    }
    uint32_t fpga_rmem_addr=(mu.read_register_ro(MEM_WRITEADDR_LOW_REGISTER_R)+1) & 0xffff;
    cout << "Last written SC Sec Low: " << std::hex << fpga_rmem_addr << endl;
    fpga_rmem_addr=(mu.read_register_ro(MEM_WRITEADDR_HIGH_REGISTER_R)+1) & 0xffff;
    cout << "Last written SC Sec High: " << std::hex << fpga_rmem_addr << endl;
    
}
