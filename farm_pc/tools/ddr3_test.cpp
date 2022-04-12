/**
 * test DDR3 moduls on Arria 10
 * 
 *
 * @author      Marius Koeppel <mkoeppel@uni-mainz.de>
 *
 * @date        2021-01-19
 */

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

using namespace std;


int main()
{
    
    /* Open mudaq device */
    mudaq::DmaMudaqDevice mu("/dev/mudaq0");
    if ( !mu.open() ) {
      cout << "Could not open device " << endl;
      return -1;
    }
    if ( !mu.is_ok() ) return -1;
    cout << "MuDaq is ok" << endl;
    
    // reset all
    cout << "reset all" << endl;
    uint32_t reset_reg = 0;
    reset_reg = SET_RESET_BIT_ALL(reset_reg);
    mu.write_register_wait(RESET_REGISTER_W, reset_reg, 100);
    
    // get DDR Status
    cout << hex << "DDR Status: 0x" << mu.read_register_ro(DDR_STATUS_R) << endl;
    cout << hex << "DDR Error: 0x" << mu.read_register_ro(DDR_ERR_R) << endl;
    
//     constant DDR_STATUS_R									: integer := 16#27#;
//     constant DDR_BIT_CAL_SUCCESS							: integer := 0;
//     constant DDR_BIT_CAL_FAIL								: integer := 1;
//     constant DDR_BIT_RESET_N								: integer := 2;
//     constant DDR_BIT_READY									: integer := 3;
//     constant DDR_BIT_TEST_WRITING						    : integer := 4;
//     constant DDR_BIT_TEST_READING						    : integer := 5;
//     constant DDR_BIT_TEST_DONE							    : integer := 6;
    
    // DDR_BIT_ENABLE_A
    cout << "Enable DDR BIT A" << endl;
    mu.write_register(DDR_CONTROL_W, 0x1);
    cout << hex << "DDR Status: 0x" << mu.read_register_ro(DDR_STATUS_R) << endl;
    
//     constant DDR_CONTROL_W									: integer := 16#20#;
//     constant DDR_BIT_ENABLE_A								: integer := 0;
//     constant DDR_BIT_COUNTERTEST_A						    : integer := 1;
//     subtype  DDR_COUNTERSEL_RANGE_A						is integer range 15 downto 14;		
//     constant DDR_BIT_ENABLE_B								: integer := 16;
//     constant DDR_BIT_COUNTERTEST_B						    : integer := 17;
//     subtype  DDR_COUNTERSEL_RANGE_B						is integer range 31 downto 30;
    
}
