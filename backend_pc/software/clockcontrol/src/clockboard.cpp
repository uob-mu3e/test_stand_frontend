#include "clockboard.h"

#include "reset_protocol.h"

#include <iostream>

#include "SI5345_REVD_REG_CONFIG.h"

using std::cout;
using std::endl;
using std::hex;

clockboard::clockboard(std::string addr, int port):bus(addr, port),recording(false),currentdaughter(-1),currentbus(-1)
{
    if(!bus.isConnected())
        cout << "Connection failed" << endl;

}

clockboard::clockboard():bus("0.0.0.0",0),recording(false),currentdaughter(-1),currentbus(-1)
{
}


int clockboard::init_clockboard(uint16_t clkinvert, uint16_t rstinvert, uint16_t clkdisable, uint16_t rstdisable)
{
    if(!bus.isConnected())
        return -1;

    init_i2c();
    // Turn on Si chip output
    bus.readModifyWriteBits(ADDR_CTRL_REG,~MASK_CTRL_CLK_CTRL,BIT_CTRL_CLK_CTRL_SI_OE) ;

    // Turn on Reset output
    bus.readModifyWriteBits(ADDR_CTRL_REG,~MASK_CTRL_PARTITION,MASK_CTRL_PARTITION) ;

    disable_all_daughter_12c();

    // set inverted channels on the reset firefly
    invert_tx_rst_channels(rstinvert);
    disable_tx_clk_channels(clkdisable);

    // set inverted channels on the clock firefly
    invert_tx_clk_channels(clkinvert);
    disable_tx_rst_channels(rstdisable);
    return 1;
}

int clockboard::map_daughter_fibre(uint8_t daughter_num, uint16_t fibre_num)
{
  uint16_t inverted_channel = (fibre_num&0x0fff);
  uint8_t fibre_type = (fibre_num&0x8000)>>15; //0 clk 1 rst
  uint8_t fibre_polarity = (fibre_num&0x1000)>>12; //0 non 1 inv
  uint8_t daughter_polarity = 0;

  if ((daughter_num==DAUGHTER_0) || (daughter_num==DAUGHTER_1)  || (daughter_num==DAUGHTER_6) || (daughter_num==DAUGHTER_7)) daughter_polarity = NON_INVERTED;
  else if ((daughter_num==DAUGHTER_2)  || (daughter_num==DAUGHTER_3) || (daughter_num==DAUGHTER_4) || (daughter_num==DAUGHTER_5)) {daughter_polarity = INVERTED;}

  if (daughter_polarity != fibre_polarity) {
    if (fibre_type == CLK_FIBRE) {
        invert_tx_clk_channels(inverted_channel);
    } else if (fibre_type == RST_FIBRE) {
        invert_tx_rst_channels(inverted_channel);
    }
    std::cout << "inverted tx channel" << std::endl;

    return 1;
  }

  return 1;
}

int clockboard::write_command(uint8_t command, uint32_t payload, bool has_payload)
{

    vector<uint32_t> senddata;
    senddata.push_back(reverse_bytes(0xbcbcbc00 + command));
    if(has_payload)
            senddata.push_back(reverse_bytes(payload));
    bus.write(ADDR_FIFO_REG_OUT,senddata,true);

    senddata.clear();
    senddata.push_back(0x7);
    if(has_payload)
        senddata.push_back(0x0);
    bus.write(ADDR_FIFO_REG_CHARISK, senddata, true);

    return 0;
}

int clockboard::write_command(std::string name, uint32_t payload, uint16_t address)
{

    auto it = reset_protocol.commands.find(name);
    if(it != reset_protocol.commands.end()){
        if(address==0){
            return write_command(it->second.command, payload, it->second.has_payload);
        }else{
            // addressed command
            vector<uint32_t> senddata;
            bool has_payload = it->second.has_payload;

            senddata.push_back(reverse_bytes(reset_protocol.commands.find("Address")->second.command*0x1000000 + address*0x100 + it->second.command));
            if(has_payload) senddata.push_back(reverse_bytes(payload));
            bus.write(ADDR_FIFO_REG_OUT,senddata,true);

            senddata.clear();
            senddata.push_back(0x0);
            if(has_payload) senddata.push_back(0x0);
            bus.write(ADDR_FIFO_REG_CHARISK, senddata, true);

            return 0;
        }
    }
    cout << "Unknown command " << name << endl;
    return -1;
}


int clockboard::init_i2c()
{

    bus.write(ADDR_I2C_PS_LO,0x35);  // Clock prescale low byte
    bus.write(ADDR_I2C_PS_HI,0x0);   // Clock prescale high byte
    bus.write(ADDR_I2C_CTRL,0x80);   // Enable I2C core
    return 0;
}

int clockboard::read_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t &data){
    if(recording)
        ofile << "ReadReg " << std::hex<< (uint32_t)dev_addr << " " << (uint32_t)reg_addr << endl;

    if(!FASTI2C)
        return read_i2c_reg_allbus(dev_addr, reg_addr, data);
    else
        return read_i2c_reg_fpga(dev_addr, reg_addr, data);
}

int clockboard::read_i2c_reg_allbus(uint8_t dev_addr, uint8_t reg_addr, uint8_t &data)
{
    if(!setSlave(dev_addr, false)){
        return 0;
    }

    bus.write(ADDR_I2C_DATA, reg_addr);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

    checkTIP();

    if(!setSlave(dev_addr,true)){
        cout << "Set Slave failed" << endl;
        return 0;
    }

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_READPLUSNACK);
    checkTIP();

    uint32_t reg = bus.read(ADDR_I2C_DATA);
    data = reg;

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_STOP); // Stop command

    checkBUSY();

    return 1;
}

int clockboard::read_i2c_reg_fpga(uint8_t dev_addr, uint8_t reg_addr, uint8_t &data)
{
    uint32_t addr = ADDR_I2C_FPGA + (dev_addr << 25) + (1<<24) + (reg_addr << 16);
    data = bus.read(addr);
    return 1;
}

int clockboard::read_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[]){
    if(recording)
        ofile << "ReadRegN " << std::hex<< (uint32_t)dev_addr << " " << (uint32_t)reg_addr << " " << (uint32_t)byte_num << endl;


    if(!FASTI2C)
       return read_i2c_reg_allbus(dev_addr, reg_addr, byte_num, data);
    else
       return read_i2c_reg_fpga(dev_addr, reg_addr, byte_num, data);
}

int clockboard::read_i2c_reg_allbus(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[])
{

    if(!setSlave(dev_addr, false)){
        return 0;
    }

    bus.write(ADDR_I2C_DATA, reg_addr);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

    checkTIP();

    if(!setSlave(dev_addr, true)){
        return 0;
    }

    checkTIP();

    uint32_t reg;

    for(int16_t byte_count =byte_num -1; byte_count >= 0 ; byte_count--){
        bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_READ);
        checkTIP();
        reg = bus.read(ADDR_I2C_DATA);
        data[static_cast<uint8_t>(byte_count)] = reg;
    }

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_READPLUSNACK);
    checkTIP();
    reg = bus.read(ADDR_I2C_DATA);
    data[byte_num -1] = reg;

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_STOP); // Stop command
    checkBUSY();

    return 1;

}

int clockboard::read_i2c_reg_fpga(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[])
{
    assert(byte_num <= 4 && byte_num > 0);
    uint32_t addr = ADDR_I2C_FPGA + (dev_addr << 25) + (1<<24) + (reg_addr << 16) + ((byte_num-1)<< 14);
    uint32_t alldata = bus.read(addr);
    data[0] = alldata & 0xFF;
    data[1] = (alldata & 0xFF00)>>8;
    data[2] = (alldata & 0xFF0000)>>16;
    data[3] = (alldata & 0xFF000000)>>24;
    return 1;
}

int clockboard::write_i2c(uint8_t dev_addr, uint8_t data){
    if(recording)
        ofile << "Write " << std::hex<< (uint32_t)dev_addr << " " << (uint32_t)data << endl;


    if(!FASTI2C)
        return write_i2c_allbus(dev_addr, data);
    else
        return write_i2c_fpga(dev_addr, data);
}


int clockboard::write_i2c_allbus(uint8_t dev_addr, uint8_t data)
{

    if(!setSlave(dev_addr,false)){
        return 0;
     }

    bus.write(ADDR_I2C_DATA, data);
    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_WRITE);

    checkTIP();

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_STOP);

    checkBUSY();

    return 1;
}

int clockboard::write_i2c_fpga(uint8_t dev_addr, uint8_t data)
{
    uint32_t addr = ADDR_I2C_FPGA + (dev_addr << 25) + (0<<24);
    bus.write(addr, data);
    return 1;
}

int clockboard::write_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t data){

    if(recording)
        ofile << "WriteReg " << std::hex<< (uint32_t)dev_addr << " " << (uint32_t)reg_addr << " "  << (uint32_t)data << endl;

    if(!FASTI2C)
        return write_i2c_reg_allbus(dev_addr, reg_addr, data);
    else
        return write_i2c_reg_fpga(dev_addr, reg_addr, data);
}


int clockboard::write_i2c_reg_allbus(uint8_t dev_addr, uint8_t reg_addr, uint8_t data)
{
    if(!setSlave(dev_addr,false)){
        return 0;
    }

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

int clockboard::write_i2c_reg_fpga(uint8_t dev_addr, uint8_t reg_addr, uint8_t data)
{
    uint32_t addr = ADDR_I2C_FPGA + (dev_addr << 25) + (1<<24) + (reg_addr<<16);
    bus.write(addr, data);
    return 1;
}

int clockboard::write_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[]){
    if(recording)
        ofile << "WriteRegN " << std::hex << (uint32_t)dev_addr << " " << (uint32_t)reg_addr
              << " "  << (uint32_t)byte_num << " " << (uint32_t)data[3]<<(uint32_t)data[2]<<(uint32_t)data[1]<<(uint32_t)data[0] << endl;



    return write_i2c_reg_allbus(dev_addr, reg_addr, byte_num, data);

}

int clockboard::write_i2c_reg_allbus(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[])
{
    if(!setSlave(dev_addr,false)){
        return 0;
    }

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

int clockboard::load_SI3545_reg_map()
{
    uint8_t current_page_num    = 0;
    uint8_t reg_page_num        = 0;
    uint8_t reg_addr            = 0;

    for(int i =0; i < SI5345_REVD_REG_CONFIG_NUM_REGS; i++){
        reg_page_num = ((0x0f00&si5345_revd_registers[i].address)>>8);
        reg_addr     =  (0x00ff&si5345_revd_registers[i].address);

        if (current_page_num!=reg_page_num){ //set reg 0x0001 to the new page number
            if (write_i2c_reg(SI_I2C_ADDR, 0x01, reg_page_num)) {
                read_i2c_reg(SI_I2C_ADDR, 0x01, current_page_num);
            }
        }
        write_i2c_reg(SI_I2C_ADDR, reg_addr, (uint8_t)si5345_revd_registers[i].value);
    }
    return 1;
}

bool clockboard::firefly_present(uint8_t daughter, uint8_t index)
{
    enable_daughter_12c(daughter,FIREFLY_SEL[index]);
    if(!setSlave(FIREFLY_TX_ADDR,false)){
       return false;
     }
    return true;
}

uint16_t clockboard::read_disabled_tx_clk_channels()
{
    enable_daughter_12c(8,0);
    uint8_t data;
    uint16_t data_holder;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_HI_ADDR, data);
    data_holder = data;
    data_holder = data_holder << 8;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_LO_ADDR, data);
    data_holder = (data_holder & 0x0f00)|data;
    return data_holder;
}

int clockboard::disable_tx_clk_channels(uint16_t channels)
{
    enable_daughter_12c(8,0);
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_HI_ADDR, (uint8_t)((channels>>8)&0x0f));
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_LO_ADDR, (uint8_t)(channels&0xff));
    return 1;
}

uint16_t clockboard::read_inverted_tx_clk_channels()
{    
    enable_daughter_12c(8,0);
    uint8_t data;
    uint16_t data_holder;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_HI_ADDR, data);
    data_holder = data;
    data_holder = data_holder << 8;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_LO_ADDR, data);
    data_holder = (data_holder & 0x0f00)|data;
    return data_holder;
}

int clockboard::invert_tx_clk_channels(uint16_t channels)
{   
    enable_daughter_12c(8,0);
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_HI_ADDR, (uint8_t)((channels>>8)&0x0f));
    uint8_t dat;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_HI_ADDR,dat);

    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_LO_ADDR, (uint8_t)(channels&0xff));
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_LO_ADDR,dat);
    return 1;
}


uint16_t clockboard::read_disabled_tx_rst_channels()
{
    enable_daughter_12c(9,0);
    uint8_t data;
    uint16_t data_holder;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_HI_ADDR, data);
    data_holder = data;
    data_holder = data_holder << 8;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_LO_ADDR, data);
    data_holder = (data_holder & 0x0f00)|data;
    return data_holder;
}

int clockboard::disable_tx_rst_channels(uint16_t channels)
{
    enable_daughter_12c(9,0);
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_HI_ADDR, (uint8_t)((channels>>8)&0x0f));
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_LO_ADDR, (uint8_t)(channels&0xff));
    return 1;
}

uint16_t clockboard::read_inverted_tx_rst_channels()
{
    enable_daughter_12c(9,0);
    uint8_t data;
    uint16_t data_holder;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_HI_ADDR, data);
    data_holder = data;
    data_holder = data_holder << 8;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_LO_ADDR, data);
    data_holder = (data_holder & 0x0f00)|data;
    return data_holder;
}

int clockboard::invert_tx_rst_channels(uint16_t channels)
{
    enable_daughter_12c(9,0);
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_HI_ADDR, (uint8_t)((channels>>8)&0x0f));
    uint8_t dat;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_HI_ADDR,dat);

    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_LO_ADDR, (uint8_t)(channels&0xff));
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_INVERT_LO_ADDR,dat);
    return 1;
}


int clockboard::disable_rx_channels(uint16_t channelmask)
{
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_DISABLE_HI_ADDR, (uint8_t)((channelmask>>8)&0x0f));
    write_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_DISABLE_LO_ADDR, (uint8_t)(channelmask&0xff));
    return 1;
}

uint16_t clockboard::read_disabled_rx_channels()
{
    uint8_t data;
    uint16_t data_holder;
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_DISABLE_HI_ADDR, data);
    data_holder = data;
    data_holder = data_holder << 8;
    read_i2c_reg(FIREFLY_RX_ADDR, FIREFLY_DISABLE_LO_ADDR, data);
    data_holder = (data_holder & 0x0f00)|data;
    return data_holder;
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

vector<uint8_t> clockboard::read_rx_amplitude(){
	//KB: Unused function, implementation added to link properly
	return vector<uint8_t>();
}

float clockboard::read_rx_firefly_temp()
{
    uint8_t data;
    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_TEMP_REG,data);
    return data * FIREFLY_TEMP_CONVERSION;
}

float clockboard::read_rx_firefly_voltage()
{

    uint8_t data;
    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_VOLTAGE_LO_REG,data);
    uint16_t voltage = data;
    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_VOLTAGE_HI_REG,data);
    voltage += ((uint16_t)data << 8);
    return (float)voltage / 10.0;
    // voltage in mV
}

uint16_t clockboard::read_rx_firefly_los()
{
    uint16_t los;
    uint8_t data;

    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_RX_LOS_LO_REG,data);
    los = data;
    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_RX_LOS_HI_REG,data);
    los += ((uint16_t)data << 8);

    return los;
}

uint16_t clockboard::read_rx_firefly_alarms()
{
    uint16_t alarm;
    uint8_t data;

    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_RX_TEMP_ALARM_REG,data);
    alarm = data;
    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_RX_VCC_ALARM_REG,data);
    alarm += ((uint16_t)data << 8);

    return alarm;
}

uint16_t clockboard::read_tx_clk_firefly_lf()
{
    enable_daughter_12c(8,0);
    uint16_t lf;
    uint8_t data;

    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_LF_LO_REG,data);
    lf = data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_LF_HI_REG,data);
    lf += ((uint16_t)data << 8);
    return lf;
}

uint16_t clockboard::read_tx_clk_firefly_alarms()
{
    enable_daughter_12c(8,0);
    uint16_t alarm;
    uint8_t data;

    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_TEMP_ALARM_REG,data);
    alarm = data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_VCC_ALARM_REG,data);
    alarm += ((uint16_t)data << 8);
    bus.readModifyWriteBits(ADDR_CTRL_REG,~MASK_CTRL_FIREFLY_CTRL, 0x0);
    return alarm;
}

uint16_t clockboard::read_tx_rst_firefly_lf()
{
    enable_daughter_12c(9,0);
    uint16_t lf;
    uint8_t data;

    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_LF_LO_REG,data);
    lf = data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_LF_HI_REG,data);
    lf += ((uint16_t)data << 8);
    return lf;
}

uint16_t clockboard::read_tx_rst_firefly_alarms()
{
    enable_daughter_12c(9,0);
    uint16_t alarm;
    uint8_t data;

    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_TEMP_ALARM_REG,data);
    alarm = data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_VCC_ALARM_REG,data);
    alarm += ((uint16_t)data << 8);
    return alarm;
}



float clockboard::read_tx_clk_firefly_temp()
{
    enable_daughter_12c(8,0);
    uint8_t data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TEMP_REG,data);
    return data * FIREFLY_TEMP_CONVERSION;
}

float clockboard::read_tx_rst_firefly_temp()
{
    enable_daughter_12c(9,0);
    uint8_t data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TEMP_REG,data);
    return data * FIREFLY_TEMP_CONVERSION;
}

float clockboard::read_tx_clk_firefly_voltage()
{
    enable_daughter_12c(8,0);
    uint8_t data;
    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_VOLTAGE_LO_REG,data);
    uint16_t voltage = data;
    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_VOLTAGE_HI_REG,data);
    voltage += ((uint16_t)data << 8);
    return (float)voltage / 10.0;
    // voltage in mV
}

float clockboard::read_tx_rst_firefly_voltage()
{
    enable_daughter_12c(8,0);
    uint8_t data;
    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_VOLTAGE_LO_REG,data);
    uint16_t voltage = data;
    read_i2c_reg(FIREFLY_RX_ADDR,FIREFLY_VOLTAGE_HI_REG,data);
    voltage += ((uint16_t)data << 8);
    return (float)voltage / 10.0;
    // voltage in mV
}


float clockboard::read_tx_firefly_temp(uint8_t daughter, uint8_t index)
{
    enable_daughter_12c(daughter,FIREFLY_SEL[index]);
    uint8_t data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TEMP_REG,data);
    return data * FIREFLY_TEMP_CONVERSION;
}

uint16_t clockboard::read_tx_firefly_lf(uint8_t daughter, uint8_t index)
{
    enable_daughter_12c(daughter,FIREFLY_SEL[index]);
    uint16_t lf;
    uint8_t data;

    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_LF_LO_REG,data);
    lf = data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_LF_HI_REG,data);
    lf += ((uint16_t)data << 8);
    return lf;
}

uint16_t clockboard::read_tx_firefly_alarms(uint8_t daughter, uint8_t index)
{
    enable_daughter_12c(daughter,FIREFLY_SEL[index]);
    uint16_t alarm;
    uint8_t data;

    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_TEMP_ALARM_REG,data);
    alarm = data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_TX_VCC_ALARM_REG,data);
    alarm += ((uint16_t)data << 8);
    return alarm;
}


float clockboard::read_tx_firefly_voltage(uint8_t daughter, uint8_t index)
{
    enable_daughter_12c(daughter,FIREFLY_SEL[index]);
    uint8_t data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_VOLTAGE_LO_REG,data);
    uint16_t voltage = data;
    read_i2c_reg(FIREFLY_TX_ADDR,FIREFLY_VOLTAGE_HI_REG,data);
    voltage += ((uint16_t)data << 8);
    return (float)voltage / 10.0;
    // voltage in mV
}

int clockboard::disable_tx_channels(uint8_t daughter, uint8_t firefly, uint16_t channelmask)
{
    // This part is not yet working as hoped for...
    enable_daughter_12c(daughter,FIREFLY_SEL[firefly]);
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_HI_ADDR, (uint8_t)((channelmask>>8)&0x0f));
    write_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_LO_ADDR, (uint8_t)(channelmask&0xff));

    uint8_t data;
    read_i2c_reg(FIREFLY_TX_ADDR, FIREFLY_DISABLE_HI_ADDR,data);
    return 0;
}



bool clockboard::daughter_present(uint8_t daughter)
{
    if(!setSlave(DAUGHTERS[daughter],false)){
       return false;
     }

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_STOP); // Stop command
    checkBUSY();

    return true;
}

uint8_t clockboard::daughters_present()
{
    uint8_t result = 0;
    for(uint8_t d =0; d < 8; d++){
        result += ((uint8_t)daughter_present(d))<<d;
    }
    return result;
}

int clockboard::enable_daughter_12c(int daughter, uint8_t i2c_bus_num)
{
    if(currentdaughter == daughter && (daughter == -1 || daughter > 7 || currentbus == i2c_bus_num))
        return 1;

    if(currentdaughter >= 0 && currentdaughter < 8)
        write_i2c(DAUGHTERS[currentdaughter], 0x0);

    if(currentdaughter > 7)
        bus.readModifyWriteBits(ADDR_CTRL_REG,~MASK_CTRL_FIREFLY_CTRL, 0x0);

    if(daughter >= 0 && daughter < 8){
        write_i2c(DAUGHTERS[daughter], i2c_bus_num);
        currentbus = i2c_bus_num;
    } else {
        currentbus = -1;
    }

    if(daughter == 8)
        bus.readModifyWriteBits(ADDR_CTRL_REG,~MASK_CTRL_FIREFLY_CTRL, BIT_FIREFLY_CLOCK_SEL);
    if(daughter == 9)
        bus.readModifyWriteBits(ADDR_CTRL_REG,~MASK_CTRL_FIREFLY_CTRL, BIT_FIREFLY_RESET_SEL);

    currentdaughter = daughter;
    return 1;
}

int clockboard::disable_daughter_12c()
{

    if(currentdaughter >= 0 && currentdaughter < 8)
        write_i2c(DAUGHTERS[currentdaughter], 0x0);

    if(currentdaughter > 7)
        bus.readModifyWriteBits(ADDR_CTRL_REG,~MASK_CTRL_FIREFLY_CTRL, 0x0);

    currentdaughter = -1;

    return 1;
}

int clockboard::disable_all_daughter_12c()
{

    for(int i=0; i<8 ; i++)
        write_i2c(DAUGHTERS[i], 0x0);

    bus.readModifyWriteBits(ADDR_CTRL_REG,~MASK_CTRL_FIREFLY_CTRL, 0x0);

    currentdaughter = -1;

    return 1;
}

float clockboard::read_daughter_board_current(uint8_t daughter)
{
    uint8_t data[2];
    enable_daughter_12c(daughter,I2C_MUX_POWER_ADDR);
    if(!read_i2c_reg(I2C_DAUGHTER_CURRENT_ADDR,I2C_SHUNT_VOLTAGE_REG_ADDR,2,data))
        return -1;
    //The factor of 2 comes from the 5mOhm shunt resistor
    // Current is now in mA
    float current = (((data[1] << 8)&0xFF00)|(data[0]&0xFF))*2.0;
    return current;
}

float clockboard::read_mother_board_current()
{
    uint8_t data[2];
    if(!read_i2c_reg(I2C_MOTHER_CURRENT_ADDR,I2C_SHUNT_VOLTAGE_REG_ADDR,2,data))
        return -1;
    //The factor of 2 comes from the 5mOhm shunt resistor
    // Current is now in mA
    float current = (((data[1] << 8)&0xFF00)|(data[0]&0xFF))*2.0;
    return current;
}

float clockboard::read_daughter_board_voltage(uint8_t daughter)
{
    uint8_t data[2];
    enable_daughter_12c(daughter,I2C_MUX_POWER_ADDR);
    if(!read_i2c_reg(I2C_DAUGHTER_CURRENT_ADDR,I2C_BUS_VOLTAGE_REG_ADDR,2,data))
        return -1;
    // 1 = 4mV - *4 gives voltage in mV
    float current = ((data[1] << 5)|(data[0]>>3))*4.0;
    return current;
}

float clockboard::read_mother_board_voltage()
{
    uint8_t data[2];
    if(!read_i2c_reg(I2C_MOTHER_CURRENT_ADDR,I2C_BUS_VOLTAGE_REG_ADDR,2,data))
        return -1;
    // 1 = 4mV - *4 gives voltage in mV
    float current = ((data[1] << 5)|(data[0]>>3))*4.0;
    return current;
}

float clockboard::read_fan_current()
{

    uint32_t reg;
    reg = bus.read(ADDR_DATA_CALIBRATED);
    reg = reg >> 16;
    float current = (1000/89.95)*(((int)reg)-2058);
    return current;
}

int clockboard::configure_daughter_current_monitor(uint8_t daughter, uint16_t config)
{
    uint8_t data[2];
    data[0]=(config>>8);
    data[1]=(config&0xff);
    enable_daughter_12c(daughter,I2C_MUX_POWER_ADDR);
    write_i2c_reg(I2C_DAUGHTER_CURRENT_ADDR,
                  I2C_CURRENT_MONITOR_CONFIG_REG_ADDR, 2, data);
    return 1;
}

int clockboard::configure_mother_current_monitor(uint16_t config)
{
    uint8_t data[2];
    data[0]=(config>>8);
    data[1]=(config&0xff);
    write_i2c_reg(I2C_MOTHER_CURRENT_ADDR,
                  I2C_CURRENT_MONITOR_CONFIG_REG_ADDR, 2, data);
    return 1;
}

void clockboard::start_recording()
{
    ofile.open("clocki2crecord.txt");
    recording = true;
}

void clockboard::stop_recording()
{
    recording = false;
    ofile.close();
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

int clockboard::setSlave(uint8_t dev_addr, bool read_bit)
{
    if(!isConnected())
        return -1;

    if(read_bit)
        bus.write(ADDR_I2C_DATA, (dev_addr << 1)|I2C_BIT_READ); // Set slave address and read bit
    else
        bus.write(ADDR_I2C_DATA, (dev_addr << 1));

    bus.write(ADDR_I2C_CMD_STAT, I2C_CMD_START);            // Start I2C transmission


    uint32_t reg = checkTIP();

    if(reg & I2C_BIT_NOACK) return 0; // Wrong address, no ACK

    return 1;

}

uint32_t clockboard::reverse_bytes(uint32_t bytes)
{
    uint32_t aux = 0;
    uint8_t byte;
    int i;

    for(i = 0; i < 32; i+=8)
    {
        byte = (bytes >> i) & 0xff;
        aux |= byte << (32 - 8 - i);
    }
    return aux;
}

