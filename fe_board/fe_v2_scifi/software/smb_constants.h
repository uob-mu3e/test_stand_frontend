#ifndef SMB_CONSTANT_H__
#define SMB_CONSTANT_H__

#define  N_CHIP 13
    
struct i2c_reg_t {
    alt_u8 slave;
    alt_u8 addr;
    alt_u8 data;
};

const i2c_reg_t SMB_init_regs[18] = { //TODO add the monitor init
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
    {0x3f,0x03,0x00}
};

/**
 * Power down cycle:
 * - Power down each ASIC (both 1.8V supplies at the same time)
 * - Power down 3.3V supplies
 */
const i2c_reg_t SMB_powerdown_regs[17] = {
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


//====monitor related=====
const int I2C_mux_index[4] = {3,0,1,2};         // this is the I2C mux fanout 


// address
const alt_u8 addr_tmp[2] = {0x48,0x49};
//7-bit I2C addr; addr==0xff: skip this mon\\TODO add the SMB I2C address //this is the address of all the power monitor 13(VCC18) + 2(VCC33)//
const alt_u8 addr_pow_mon[15]={0x4c,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x4d,0x18};//MALIBU
const alt_u8 addr_MUX[4]={0x40,0x41,0x42,0x43};
const alt_u8 addr_GPIO[7]={0x39,0x3a,0x3b,0x3c,0x3d,0x3e,0x3f};


// TMP117 related reg address 
const alt_u8 reg_temp_result = 0x00;
const alt_u8 reg_temp_config = 0x01;
const alt_u8 reg_temp_deviceID= 0x0f; //should the readback should be 0x117

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
