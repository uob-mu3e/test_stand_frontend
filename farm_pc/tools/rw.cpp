/**
 * open a mudaq device and measure DMA speed
 * use data generator from counter with 250 MHz clock
 *
 * @author      Dorothea vom Bruch <vombruch@physi.uni-heidelberg.de>
 *              adapted from code by Fabian Foerster and Qinhua Huang
 * @date        2015-01-22
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




#include "mudaq_device.h"

using namespace std;

void print_usage(){
    cout << "Usage: " << endl;
    cout << "       rw rr <addr> for reading read register at addr" << endl;
    cout << "       rw rm <addr> for reading read memory at addr" << endl;
    cout << "       rw wr <addr> for reading write register at addr" << endl;
    cout << "       rw wm <addr> for reading write memory at addr" << endl;
    cout << "       rw wwr <addr> <value> for writing write register at addr" << endl;
    cout << "       rw wwm <addr> <value> for writing write memory at addr" << endl;

}

constexpr unsigned int str2int(const char* str, int h = 0)
{
    return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
}

int main(int argc, char * argv[])
{
    if(argc < 3){
        print_usage();
        return -1;
    }

    char * command = argv[1];

    char * addrs = argv[2];

    unsigned int addr  = strtol( addrs, 0,0);

    /* Open mudaq device */
    mudaq::DmaMudaqDevice mu("/dev/mudaq0");
    if ( !mu.open() ) {
      cout << "Could not open device " << endl;
      return -1;
    }
    if ( !mu.is_ok() ) return -1;


//    /* Reset dma part and data generator */
//    uint32_t reset_reg =0;

//    //reset_reg = SET_RESET_BIT_DATAGEN(reset_reg);

//    cout << mu.read_register_ro(RESET_REGISTER_W) << endl;
//    cout << mu.read_register_ro(RESET_REGISTER_W) << endl;

//    mu.write_register_wait(RESET_REGISTER_W, reset_reg,100);

//    cout << mu.read_register_ro(RESET_REGISTER_W) << endl;

//    //cout << mu.read_register_ro(EVENTCOUNTER64_REGISTER_R) << endl;

//    mu.write_register_wait(RESET_REGISTER_W, 0x0,100);

//    cout << mu.read_register_ro(RESET_REGISTER_W) << endl;

//    /* Blink with LEDs */
//    mu.write_register_wait(LED_REGISTER_W,0xffffffff, 100000000);
//    mu.write_register_wait(LED_REGISTER_W,0x0, 100000000);




    unsigned int value;

    switch (str2int(command))
    {
      case str2int("rr"):
//        if(addr > 63){
//            cout << "Invalid register address " << addr << endl;
//            break;
//        }
        cout << hex << "0x" << mu.read_register_ro(addr) << endl;
        break;
      case str2int("rm"):
//        if(addr > 64*1024){
//            cout << "Invalid memory address " << addr << endl;
//            break;
//        }
        cout << hex << "0x" << mu.read_memory_ro(addr) << endl;
        break;
      case str2int("wr"):
//        if(addr > 63){
//            cout << "Invalid register address " << addr << endl;
//            break;
//        }
        cout << hex << "0x" << mu.read_register_rw(addr) << endl;
       break;
      case str2int("wm"):
//        if(addr > 64*1024-1){
//            cout << "Invalid memory address " << addr << endl;
//            break;
//        }
        cout << hex << "0x" << mu.read_memory_rw(addr) << endl;
       break;
      case str2int("wwr"):
        if(argc < 4){
            cout << "Too few arguments" << endl;
            break;
        }
//        if(addr > 63){
//            cout << "Invalid register address " << addr << endl;
//            break;
//        }
        value  = strtol(argv[3], 0,0);
        mu.write_register(addr, value);
        cout << hex << "0x" << mu.read_register_rw(addr) << endl;
       break;
      case str2int("wwm"):
        if(argc < 4){
            cout << "Too few arguments" << endl;
            break;
        }
        if(addr > 64*1024-1){
            cout << "Invalid memory address " << addr << endl;
            break;
        }
        value  = strtol(argv[3], 0,0);
        mu.write_memory_rw(addr, value);
        cout << hex << "0x" << mu.read_memory_rw(addr) << endl;
       break;
       default:
        cout << "Unknown command" << endl;
        print_usage();
        break;
    }

  mu.close();

  return 0;
}
