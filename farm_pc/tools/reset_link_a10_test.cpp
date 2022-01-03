/**
 * test reset link on Arria 10
 * 
 *
 * @author      Marius Koeppel <mkoeppel@uni-mainz.de>
 *
 * @date        2021-12-03
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
    
    cout << "Reset Link Status Reg " << RESET_LINK_STATUS_REGISTER_R << endl;
    
    // reset all
    uint32_t reset_regs = 0;
    reset_regs = SET_RESET_BIT_ALL(reset_regs);
    mu.write_register(RESET_REGISTER_W, reset_regs);
    mu.write_register(RESET_REGISTER_W, 0x0);
    
    // get Reset Link Status
    mu.write_register(RESET_LINK_RUN_NUMBER_REGISTER_W, 0xAAAAAAAA);
    char cmd;
    
    while (1) {
        printf(" Run RESET_LINK_CTL_REGISTER_W: 0x%02X\n", mu.read_register_rw(RESET_LINK_CTL_REGISTER_W));
        printf(" Reset Link Status: 0x%02X\n", mu.read_register_ro(RESET_LINK_STATUS_REGISTER_R));
        printf(" Run Number: 0x%02X\n", mu.read_register_rw(RESET_LINK_RUN_NUMBER_REGISTER_W));
        printf("  [1] => run_prep\n");
        printf("  [2] => sync\n");
        printf("  [3] => start run\n");
        printf("  [4] => end run\n");
        printf("  [5] => abort run\n");
        printf("  [6] => enable clk phase test 0 \n");
        printf("  [7] => enable clk phase test 1 \n");

        cout << "Select entry ...";
        cin >> cmd;
        switch(cmd) {
        case '1':
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0xF0000001);
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0x0);
            break;
        case '2':
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0xF0000002);
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0x0);
            break;
        case '3':
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0xF0000004);
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0x0);
            break;
        case '4':
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0xF0000008);
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0x0);
            break;
        case '5':
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0xF0000010);
            mu.write_register(RESET_LINK_CTL_REGISTER_W, 0x0);
            break;
        case '6':
            mu.write_register(CLK_LINK_0_REGISTER_W, 0xFFF00000);
            mu.write_register(CLK_LINK_1_REGISTER_W, 0x000FFFFF);
            mu.write_register(CLK_LINK_2_REGISTER_W, 0xFFF00000);
            mu.write_register(CLK_LINK_3_REGISTER_W, 0xFFF00000);
            mu.write_register(CLK_LINK_REST_REGISTER_W, 0xFFFFF00FF);
            mu.write_register(RESET_REGISTER_W, reset_regs);
            mu.write_register(RESET_REGISTER_W, 0x0);
            break;
        case '7':
            mu.write_register(CLK_LINK_0_REGISTER_W, 0xFFF00000);
            mu.write_register(CLK_LINK_1_REGISTER_W, 0xFFF00000);
            mu.write_register(CLK_LINK_2_REGISTER_W, 0xFFF00000);
            mu.write_register(CLK_LINK_3_REGISTER_W, 0xFFF00000);
            mu.write_register(CLK_LINK_REST_REGISTER_W, 0xFFFFFFFFF);
            mu.write_register(RESET_REGISTER_W, reset_regs);
            mu.write_register(RESET_REGISTER_W, 0x0);
            break;    
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
