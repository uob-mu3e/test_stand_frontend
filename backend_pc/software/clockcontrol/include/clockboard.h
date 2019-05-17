#ifndef CLOCKBOARD_H
#define CLOCKBOARD_H

#include "ipbus.h"

class clockboard
{
public:
    clockboard(const char * addr, int port);
    bool isConnected(){return bus.isConnected();}

    int init_12c();
    int read_i2c(uint8_t dev_addr, uint8_t &data);


protected:
    ipbus bus;

    const uint32_t ADDR_FIFO_REG_OUT        = 0x0;
    const uint32_t ADDR_FIFO_REG_CHARISK    = 0x2;
    const uint32_t ADDR_FIFO_REG_IN         = 0x4;
    const uint32_t ADDR_CTRL_REG            = 0x6;
    const uint32_t BIT_CTRL_RESET           = 0;
    const uint32_t BIT_CTRL_CALIBRATE       = 1;
    const uint32_t MASK_CTRL_PARTITION      = 0x07F8;
    const uint32_t MASK_CTRL_PARTITION_ADDR = 0x3800;
    const uint32_t MASK_CTRL_CLK_CTRL       = 0xC000;
    const uint32_t MASK_CTRL_FIREFLY_CTRL   = 0xF0000;
    const uint32_t ADDR_DATA_CALIBRATED     = 0x7;
    const uint32_t ADDR_I2C_PS_LO           = 0x8;
    const uint32_t ADDR_I2C_PS_HI           = 0x9;
    const uint32_t ADDR_I2C_CTRL            = 0xA;
    const uint32_t ADDR_I2C_DATA            = 0xB;
    const uint32_t ADDR_I2C_CMD_STAT        = 0xC;

};

#endif // CLOCKBOARD_H
