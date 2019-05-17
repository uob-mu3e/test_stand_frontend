#include "clockboard.h"

#include <iostream>

using std::cout;
using std::endl;

clockboard::clockboard(const char *addr, int port):bus(addr, port)
{
    if(!bus.isConnected())
        cout << "Connection failed" << endl;

}

int clockboard::init_12c()
{
    if(!isConnected())
        return -1;

    cout << "Going to write " << endl;
    bus.write(ADDR_I2C_PS_LO,0x35);  // Clock prescale low byte
    bus.write(ADDR_I2C_PS_HI,0x0);   // Clock prescale high byte
    bus.write(ADDR_I2C_CTRL,0x80);   // Enable I2C core
}

int clockboard::read_i2c(uint8_t dev_addr, uint8_t & data)
{

    if(!isConnected())
        return -1;

    bus.write(ADDR_I2C_DATA, (dev_addr << 1)|0x1); // Set slave address and read bit
    bus.write(ADDR_I2C_CMD_STAT, 0x90);            // Start I2C transmission

    cout << "I2C started " << endl;

    uint32_t tip =1;
    uint32_t reg =0;
    while(tip){ // check the TIP bit indicating transaction in progress
        reg = bus.read(ADDR_I2C_CMD_STAT);
        tip = reg & 0x2;
        cout << "TIP: " << tip << endl;
    }

    if(reg & 0x80) return 0; // Wrong address, no ACK

    cout << "Now reading" << endl;

    bus.write(ADDR_I2C_CMD_STAT, 0x28);  // Read command plus ACK

    tip = 1;
    while(tip){ // check the TIP bit indicating transaction in progress
        reg = bus.read(ADDR_I2C_CMD_STAT);
        tip = reg & 0x2;
        cout << "TIP: " << tip << endl;
    }

    reg = bus.read(ADDR_I2C_DATA);
    data = reg;

     cout << "Now stopping" << endl;

    bus.write(ADDR_I2C_CMD_STAT, 0x40); // Stop command

    uint32_t busy = 1;
    while(busy){ // check the I2C busy flag
        reg = bus.read(ADDR_I2C_CMD_STAT);
        busy = reg & 0x40;

    }


    return 1;
}



