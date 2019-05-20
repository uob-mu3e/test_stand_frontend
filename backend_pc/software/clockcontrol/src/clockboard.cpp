#include "clockboard.h"

#include <iostream>

#include "SI5345_REVD_REG_CONFIG.h"

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

    return 0;
}

int clockboard::read_i2c(uint8_t dev_addr, uint8_t & data)
{

    if(!setSlave(dev_addr))
        return 0;

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_READPLUSNACK);  // Read command plus ACK

    checkTIP();

    uint32_t reg = bus.read(ADDR_I2C_DATA);
    data = reg;

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_STOP); // Stop command

    checkBUSY();

    return 1;
}

int clockboard::read_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t &data)
{

    if(!setSlave(dev_addr))
        return 0;

    bus.write(ADDR_I2C_DATA, reg_addr);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

    checkTIP();

    bus.write(ADDR_I2C_DATA, (dev_addr << 1)|I2C_BIT_WRITE); // Set slave address and read bit
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_START);            // Start I2C transmission

    checkTIP();

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_READPLUSNACK);  // Read command plus ACK

    checkTIP();

    uint32_t reg = bus.read(ADDR_I2C_DATA);
    data = reg;

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_STOP); // Stop command

    checkBUSY();

    return 1;
}

int clockboard::read_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[])
{
    if(!setSlave(dev_addr))
        return 0;

    bus.write(ADDR_I2C_DATA, reg_addr);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

    checkTIP();

    bus.write(ADDR_I2C_DATA, (dev_addr << 1)|I2C_BIT_WRITE); // Set slave address and read bit
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_START);            // Start I2C transmission

    checkTIP();

    uint32_t reg;

    for(uint8_t byte_count =0; byte_count < byte_num -1; byte_count++){
        bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_READ);
        checkTIP();
        reg = bus.read(ADDR_I2C_DATA);
        data[byte_count] = reg;
    }

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_READPLUSNACK);
    checkTIP();
    reg = bus.read(ADDR_I2C_DATA);
    data[byte_num -1] = reg;

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_STOP); // Stop command
    checkBUSY();

    return 1;

}

int clockboard::write_i2c(uint8_t dev_addr, uint8_t data)
{
    if(!setSlave(dev_addr,false))
        return 0;

    bus.write(ADDR_I2C_DATA, data);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

    checkTIP();

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_STOP);

    checkBUSY();

    return 1;
}

int clockboard::write_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t data)
{
    if(!setSlave(dev_addr,false))
        return 0;

    bus.write(ADDR_I2C_DATA, reg_addr);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

    checkTIP();

    bus.write(ADDR_I2C_DATA, data);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

    checkTIP();

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_STOP);

    checkBUSY();

    return 1;

}

int clockboard::write_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[])
{
    if(!setSlave(dev_addr,false))
        return 0;

    bus.write(ADDR_I2C_DATA, reg_addr);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

    checkTIP();

    for(uint8_t byte_count =0; byte_count < byte_num -1; byte_count++){
        bus.write(ADDR_I2C_DATA, data[byte_count]);
        bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

        checkTIP();
    }

    bus.write(ADDR_I2C_DATA, data[byte_num -1]);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE|I2C_CMD_STOP);

    checkBUSY();

    return 1;

}

int clockboard::load_SI3545_reg_map(uint8_t dev_addr)
{
    uint8_t current_page_num    = 0;
    uint8_t reg_page_num        = 0;
    uint8_t reg_addr            = 0;

    for(int i =0; i < SI5345_REVD_REG_CONFIG_NUM_REGS; i++){
        reg_page_num = ((0x0f00&si5345_revd_registers[i].address)>>8);
        reg_addr     =  (0x00ff&si5345_revd_registers[i].address);

        if (current_page_num!=reg_page_num){ //set reg 0x0001 to the new page number
            if (write_i2c_reg(dev_addr, 0x01, reg_page_num)) {
                read_i2c_reg(dev_addr, 0x01, current_page_num);
            }
        }
        write_i2c_reg(dev_addr, reg_addr, (uint8_t)si5345_revd_registers[i].value);
    }
    return 1;
}

uint16_t clockboard::read_disabled_tx_channels()
{
    uint8_t data;
    uint16_t data_holder;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_HI_ADDR, data);
    data_holder = data;
    data_holder = data_holder << 8;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_LO_ADDR, data);
    data_holder = (data_holder & 0x0f00)|data;
    return data_holder;
}

int clockboard::disable_tx_channels(uint16_t channels)
{
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_HI_ADDR, (uint8_t)((channels>>8)&0x0f));
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_LO_ADDR, (uint8_t)(channels&0xff));
    return 1;
}

uint16_t clockboard::read_inverted_tx_channels()
{
    uint8_t data;
    uint16_t data_holder;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_HI_ADDR, data);
    data_holder = data;
    data_holder = data_holder << 8;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_LO_ADDR, data);
    data_holder = (data_holder & 0x0f00)|data;
    return data_holder;
}

int clockboard::invert_tx_channels(uint16_t channels)
{
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_HI_ADDR, (uint8_t)((channels>>8)&0x0f));
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_LO_ADDR, (uint8_t)(channels&0xff));
    return 1;
}

int clockboard::set_rx_amplitude(uint8_t amplitude)
{
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_0_1_ADDR, amplitude);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_2_3_ADDR, amplitude);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_4_5_ADDR, amplitude);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_6_7_ADDR, amplitude);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_8_9_ADDR, amplitude);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_A_B_ADDR, amplitude);
    return 1;
}

int clockboard::set_rx_emphasis(uint8_t emphasis)
{
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_0_1_ADDR, emphasis);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_2_3_ADDR, emphasis);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_4_5_ADDR, emphasis);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_6_7_ADDR, emphasis);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_8_9_ADDR, emphasis);
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_A_B_ADDR, emphasis);
    return 1;
}

vector<uint8_t> clockboard::read_rx_amplitude()
{
    uint8_t data;
    vector<uint8_t> res;
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_0_1_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_2_3_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_4_5_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_6_7_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_8_9_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_AMP_A_B_ADDR, data);
    res.push_back(data);

    return res;
}

vector<uint8_t> clockboard::read_rx_emphasis()
{
    uint8_t data;
    vector<uint8_t> res;
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_0_1_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_2_3_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_4_5_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_6_7_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_8_9_ADDR, data);
    res.push_back(data);
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_RX_EMP_A_B_ADDR, data);
    res.push_back(data);

    return res;
}



uint32_t clockboard::checkTIP()
{
    uint32_t tip =1;
    uint32_t reg =0;
    while(tip){ // check the TIP bit indicating transaction in progress
        reg = bus.read(ADDR_I2C_CMD_STAT);
        tip = reg & I2C_BIT_TIP;
    }
    return reg;
}

uint32_t clockboard::checkBUSY()
{
    uint32_t busy = 1;
    uint32_t reg =0;
    while(busy){ // check the I2C busy flag
        reg = bus.read(ADDR_I2C_CMD_STAT);
        busy = reg & I2C_BIT_BUSY;

    }
    return reg;
}

int clockboard::setSlave(uint8_t dev_addr, bool write_bit)
{
    if(!isConnected())
        return -1;

    if(write_bit)
        bus.write(ADDR_I2C_DATA, (dev_addr << 1)|I2C_BIT_WRITE); // Set slave address and read bit
    else
        bus.write(ADDR_I2C_DATA, (dev_addr << 1));

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_START);            // Start I2C transmission


    uint32_t reg = checkTIP();

    if(reg & I2C_BIT_NOACK) return 0; // Wrong address, no ACK

    return 1;

}



