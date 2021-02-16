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
    
    // get DDR3 Status
    cout << hex << "DDR3 Status: 0x" << mu.read_register_ro(DDR3_STATUS_R) << endl;
    cout << hex << "DDR3 Error: 0x" << mu.read_register_ro(DDR3_ERR_R) << endl;
    
//     constant DDR3_STATUS_R									: integer := 16#27#;
//     constant DDR3_BIT_CAL_SUCCESS							: integer := 0;
//     constant DDR3_BIT_CAL_FAIL								: integer := 1;
//     constant DDR3_BIT_RESET_N								: integer := 2;
//     constant DDR3_BIT_READY									: integer := 3;
//     constant DDR3_BIT_TEST_WRITING						    : integer := 4;
//     constant DDR3_BIT_TEST_READING						    : integer := 5;
//     constant DDR3_BIT_TEST_DONE							    : integer := 6;
    
    // DDR3_BIT_ENABLE_A
    cout << "Enable DDR3 BIT A" << endl;
    mu.write_register(DDR3_CONTROL_W, 0x1);
    cout << hex << "DDR3 Status: 0x" << mu.read_register_ro(DDR3_STATUS_R) << endl;
    
//     constant DDR3_CONTROL_W									: integer := 16#20#;
//     constant DDR3_BIT_ENABLE_A								: integer := 0;
//     constant DDR3_BIT_COUNTERTEST_A						    : integer := 1;
//     subtype  DDR3_COUNTERSEL_RANGE_A						is integer range 15 downto 14;		
//     constant DDR3_BIT_ENABLE_B								: integer := 16;
//     constant DDR3_BIT_COUNTERTEST_B						    : integer := 17;
//     subtype  DDR3_COUNTERSEL_RANGE_B						is integer range 31 downto 30;
    
}
