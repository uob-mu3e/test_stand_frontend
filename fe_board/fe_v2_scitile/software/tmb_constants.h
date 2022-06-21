#ifndef TMB_CONSTANT_H__
#define TMB_CONSTANT_H__

#define  N_CHIP 13
    
struct i2c_reg_t {
    alt_u8 slave;
    alt_u8 addr;
    alt_u8 data;
};

const i2c_reg_t TMB_init_regs[20] = { //TODO add the monitor init
    {0x38,0x01,0x0C^0x20},
    {0x38,0x03,0x00},
    {0x38,0x01,0x0D^0x20},
    {0xff,0x00,0x00},
    {0x38,0x01,0x0F^0x20},
    {0xff,0x00,0x00},
    {0x39,0x01,0x3C},
    {0x39,0x03,0x00},
    {0x3a,0x01,0x3C},
    {0x3a,0x03,0x00},
    {0x3b,0x01,0x3C},
    {0x3b,0x03,0x00},
    {0x3c,0x01,0x3C},
    {0x3c,0x03,0x00},
    {0x3d,0x01,0x3C},
    {0x3d,0x03,0x00},
    {0x3f,0x01,0x3C},
    {0x3f,0x03,0x00},
    {0x44,0x01,0x00}, // mux gpio
    {0x44,0x02,0x1C}  // mux gpio
};

/**
 * Power down cycle:
 * - Power down each ASIC (both 1.8V supplies at the same time)
 * - Power down 3.3V supplies
 */
const i2c_reg_t TMB_powerdown_regs[17] = {
    {0x3f,0x01,0x3C},
    {0xff,0x00,0x00},
    {0x3e,0x01,0x3C},
    {0xff,0x00,0x00},
    {0x3d,0x01,0x3C},
    {0xff,0x00,0x00},
    {0x3c,0x01,0x3C},
    {0xff,0x00,0x00},
    {0x3b,0x01,0x3C},
    {0xff,0x00,0x00},
    {0x3a,0x01,0x3C},
    {0xff,0x00,0x00},
    {0x39,0x01,0x3C},
    {0xff,0x00,0x00},
    {0x38,0x01,0x0D},
    {0xff,0x00,0x00},
    {0x38,0x01,0x0C}
};

const alt_u8 GPIO_out_reg[2] = {0x02,0x03};

const int VCCA18_first3_index[3] = {13,14,0};
const int VCCD18_first3_index[3] = {1,6,5};
const int VCCA18_4_index[4] = {16,17,3,1};
const int VCCD18_4_index[4] = {15,11,10,2};

const int SPI_first3_index[3] = {10,3,12};
const int SPI_4_index[4] = {12,6,14,0};

//====monitor related=====
const int I2C_mux_index[4] = {3,0,1,2}; // this is the I2C mux fanout 
const int I2C_bus_index[7] = {3,2,4,1,3,2,1};

const int I2C_tmp_bus_over23[4] = {6,6,1,5}; //subbus used for Temp 24,25,26,27
const alt_u8 I2C_tmp_addr_over23[4] = {0x48,0x49,0x4d,0x4e}; //adresses for Temp 24,25,26,27

// address
const alt_u8 addr_tmp[4] = {0x48,0x49,0x4a,0x4b};
//7-bit I2C addr; addr==0xff: skip this mon\\TODO add the TMB I2C address //this is the address of all the power monitor 13(VCC18) + 2(VCC33)//
const alt_u8 addr_pow_mon[3]={0x28,0x29,0x4f};//TMB - reordering needed
//const alt_u8 addr_MUX[4]={0x40,0x41,0x42,0x43}; //TMB schematic
const alt_u8 addr_MUX[2]={0x45,0x44}; //TMB #2 actual configuration + fix un U27
const alt_u8 addr_GPIO[4]={0x21,0x22,0x23,0x24};


// TMP117 related reg address 
const alt_u8 reg_temp_result = 0x00;
const alt_u8 reg_temp_config = 0x01;
const alt_u8 reg_temp_deviceID[2] = {0x0f, 0x07}; //should the readback should be 0x117
const alt_u16 temp_deviceID_result[2] = {0x0117, 0x0190};

//{{{//current monitor related register address and command [PAC1720]
//=================configuration register========
//select function register
const alt_u8 reg_sel = 0x00; //TODO jsut enable both channel in all components
const alt_u8 config_sel_mask[2]={0xfc,0xe7};
//vsource sampling and averaging
const alt_u8 reg_vsource_config = 0x0a;
const alt_u8 cmd_vsource_config = 0xff; //ch1 and ch2: 20ms sampling time and averaging by 8
//vsense sampling and averaging
const alt_u8 reg_vsense_sampling[2] = {0x0b,0x0c};
const alt_u8 cmd_vsense_sampling[2] = {0x53,0x53};//80ms;averaging 8; 10mV 
//alt_u8 cmd_vsense_sampling[2] = {0x5c,0x5c};//80ms;averaging 8; 10mV 

//===================results register=========
//high limit status register address
const alt_u8 reg_HL= {0x04};    //high limit
const alt_u8 reg_LL= {0x05};    //low limit
//data register of PAC1720 [high byte]
const alt_u8 reg_vsource[2]={0x11,0x13};
const alt_u8 reg_vsense[2]={0x0d,0x0f};
//}}}

#endif
