#ifndef TMB_CONSTANT_H__
#define TMB_CONSTANT_H__

//#define STIC3_CONFIG_LEN_BITS 4657
//#define STIC3_CONFIG_LEN_BYTES 583
#define  N_CHIP 13

struct i2c_reg_t {
    alt_u8 slave;
    alt_u8 addr;
    alt_u8 data;
};

const i2c_reg_t malibu_init_regs[18] = { //TODO add the monitor init
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
const i2c_reg_t malibu_powerdown_regs[17] = {
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
//7-bit I2C addr; addr==0xff: skip this mon\\TODO add the TMB I2C address //this is the address of all the power monitor 13(VCC18) + 2(VCC33)//
const alt_u8 addr_pow_mon[15]={0x4c,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0x2e,0x18};//MALIBU
const alt_u8 addr_MUX[4]={0x40,0x41,0x42,0x43};
const alt_u8 addr_GPIO[7]={0x39,0x3a,0x3b,0x3c,0x3d,0x3e,0x3f};

//GPIO related 
alt_u8 D_bit;   //GPIO bit to control 1.8D of ASIC 
alt_u8 A_bit;   //GPIO bit to control 1.8A of ASIC
alt_u8 CS_bit;  //GPIO bit to select spi talk to which chip


// TMP117 related reg address 
alt_u8 reg_temp_result = 0x00;
alt_u8 reg_temp_config = 0x01;
alt_u8 reg_temp_deviceID= 0x0f; //should the readback should be 0x117

//{{{//current monitor related register address and command [PAC1720]
//=================configuration register========
//select function register
alt_u8 reg_sel = 0x00; //TODO jsut enable both channel in all components
alt_u8 config_sel_mask[2]={0xfc,0xe7};
//vsource sampling and averaging
alt_u8 reg_vsource_config = 0x0a;
alt_u8 cmd_vsource_config = 0xff; //ch1 and ch2: 20ms sampling time and averaging by 8
//vsense sampling and averaging
alt_u8 reg_vsense_sampling[2] = {0x0b,0x0c};
alt_u8 cmd_vsense_sampling[2] = {0x53,0x53};//80ms;averaging 8; 10mV 
//alt_u8 cmd_vsense_sampling[2] = {0x5c,0x5c};//80ms;averaging 8; 10mV 

//===================results register=========
//high limit status register address
alt_u8 reg_HL= {0x04};    //high limit
alt_u8 reg_LL= {0x05};    //low limit
//data register of PAC1720 [high byte]
alt_u8 reg_vsource[2]={0x11,0x13};
alt_u8 reg_vsense[2]={0x0d,0x0f};
//}}}


//read constant register
void read_ProductID(alt_u8 addr);
void read_ManufID(alt_u8 addr);
void read_Revision(alt_u8 addr);

//TODO set the limit and read the alart line

//for terminal
void test_menu();

#endif
