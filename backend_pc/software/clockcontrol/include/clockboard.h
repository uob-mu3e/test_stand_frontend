#ifndef CLOCKBOARD_H
#define CLOCKBOARD_H

#include "ipbus.h"
#include "reset_protocol.h"

class clockboard
{
public:
    clockboard(const char * addr, int port);
    bool isConnected(){return bus.isConnected();}

    int init_clockboard(uint16_t clkinvert = 0x0A00, uint16_t rstinvert= 0x0008, uint16_t clkdisable = 0x0AA, uint16_t rstdisable = 0xAA0);
    int map_daughter_fibre(uint8_t daughter_num, uint16_t fibre_num);

    // Write "reset" commands
    int write_command(uint8_t command, uint32_t payload =0, bool has_payload = false);
    int write_command(const char * name, uint32_t payload =0, uint16_t address =0);

    // I2C interface
    int init_i2c();
    int read_i2c(uint8_t dev_addr, uint8_t &data);
    int read_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t &data);
    int read_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[]);

    int write_i2c(uint8_t dev_addr, uint8_t data);
    int write_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t data);
    int write_i2c_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t byte_num, uint8_t data[]);

    // SI3545 programming - note that this uses the register map
    // generated with the clock builder tool
    int load_SI3545_reg_map();

    // Firefly interface
    bool firefly_present(uint8_t daughter, uint8_t index);

    uint16_t read_disabled_tx_clk_channels();
    int disable_tx_clk_channels(uint16_t channels);
    uint16_t read_inverted_tx_clk_channels();
    int invert_tx_clk_channels(uint16_t channels);

    uint16_t read_disabled_tx_rst_channels();
    int disable_tx_rst_channels(uint16_t channels);
    uint16_t read_inverted_tx_rst_channels();
    int invert_tx_rst_channels(uint16_t channels);

    int disable_rx_channels(uint16_t channelmask);
    uint16_t read_disabled_rx_channels();

    int set_rx_amplitude(uint8_t amplitude);
    int set_rx_emphasis(uint8_t emphasis);

    vector<uint8_t> read_rx_amplitude();
    vector<uint8_t> read_rx_emphasis();

    float read_rx_firefly_temp();
    float read_rx_firefly_voltage();
    uint16_t read_rx_firefly_los();
    uint16_t read_rx_firefly_alarms();


    float read_tx_clk_firefly_temp();
    float read_tx_rst_firefly_temp();
    float read_tx_clk_firefly_voltage();
    float read_tx_rst_firefly_voltage();
    uint16_t read_tx_clk_firefly_lf();
    uint16_t read_tx_clk_firefly_alarms();
    uint16_t read_tx_rst_firefly_lf();
    uint16_t read_tx_rst_firefly_alarms();

    float read_tx_firefly_temp(uint8_t daughter, uint8_t index);
    float read_tx_firefly_voltage(uint8_t daughter, uint8_t index);

    int disable_tx_channels(uint8_t daughter, uint8_t firefly, uint16_t channelmask);

    uint16_t read_tx_firefly_lf(uint8_t daughter, uint8_t index);
    uint16_t read_tx_firefly_alarms(uint8_t daughter, uint8_t index);

    // Mother and daughter card monitoring
    bool daughter_present(uint8_t daughter);
    uint8_t daughters_present();

    int enable_daughter_12c(uint8_t dev_addr, uint8_t i2c_bus_num);
    int disable_daughter_12c(uint8_t dev_addr);

    float read_daughter_board_current(uint8_t daughter);
    float read_mother_board_current();

    float read_daughter_board_voltage(uint8_t daughter);
    float read_mother_board_voltage();
    
    float read_fan_current();

    int configure_daughter_current_monitor(uint8_t daughter, uint16_t config);
    int configure_mother_current_monitor(uint16_t config);

    reset reset_protocol;
protected:
    ipbus bus;

    //I2C helpers
    uint32_t checkTIP();
    uint32_t checkBUSY();
    int setSlave(uint8_t dev_addr, bool read_bit=true);

    const uint32_t ADDR_FIFO_REG_OUT        = 0x0;
    const uint32_t ADDR_FIFO_REG_CHARISK    = 0x2;
    const uint32_t ADDR_FIFO_REG_IN         = 0x4;
    const uint32_t ADDR_CTRL_REG            = 0x6;
    const uint32_t BIT_CTRL_RESET           = 0;
    const uint32_t BIT_CTRL_CALIBRATE       = 1;
    const uint32_t MASK_CTRL_PARTITION      = 0x07F8;
    const uint32_t MASK_CTRL_PARTITION_ADDR = 0x3800;
    const uint32_t MASK_CTRL_CLK_CTRL       = 0xC000;
    const uint32_t BIT_CTRL_CLK_CTRL_SI_OE  = 0x4000;
    const uint32_t BIT_CTRL_CLK_CTRL_SI_RST = 0x8000;
    const uint32_t MASK_CTRL_FIREFLY_CTRL   = 0xF0000;
    const uint32_t MASK_FAN_CURRENT         = 0xFFFF0000;
    const uint32_t BIT_FIREFLY_RESET_RST    = 0x10000;
    const uint32_t BIT_FIREFLY_RESET_SEL    = 0x20000;
    const uint32_t BIT_FIREFLY_CLOCK_RST    = 0x40000;
    const uint32_t BIT_FIREFLY_CLOCK_SEL    = 0x80000;
    const uint32_t ADDR_DATA_CALIBRATED     = 0x7;
    const uint32_t ADDR_I2C_PS_LO           = 0x8;
    const uint32_t ADDR_I2C_PS_HI           = 0x9;
    const uint32_t ADDR_I2C_CTRL            = 0xA;
    const uint32_t ADDR_I2C_DATA            = 0xB;
    const uint32_t ADDR_I2C_CMD_STAT        = 0xC;

    const uint32_t I2C_BIT_READ             = 0x1;
    const uint32_t I2C_BIT_TIP              = 0x2;
    const uint32_t I2C_BIT_NOACK            = 0x80;
    const uint32_t I2C_BIT_BUSY             = 0x40;

    const uint32_t I2C_CMD_START            = 0x90;
    const uint32_t I2C_CMD_READ             = 0x20;
    const uint32_t I2C_CMD_READPLUSNACK     = 0x28;
    const uint32_t I2C_CMD_STOP             = 0x40;
    const uint32_t I2C_CMD_WRITE            = 0x10;

    // Firefly addresses on I2c
    const uint8_t FIREFLY_TX_ADDR           = 0x50;
    const uint8_t FIREFLY_RX_ADDR           = 0x54;

    // Firefly buses
    const uint8_t FIREFLY_0                 = 0x07;
    const uint8_t FIREFLY_2                 = 0x05;
    const uint8_t FIREFLY_1                 = 0x06;

    const uint8_t FIREFLY_SEL[3]             = {FIREFLY_0, FIREFLY_1, FIREFLY_2};

    // Firefly register map
    // RX specific
    const uint8_t FIREFLY_RX_LOS_LO_REG     = 0x08;
    const uint8_t FIREFLY_RX_LOS_HI_REG     = 0x07;
    const uint8_t FIREFLY_RX_TEMP_ALARM_REG = 0x10;
    const uint8_t FIREFLY_RX_VCC_ALARM_REG  = 0x12;

    //TX Specific
    const uint8_t FIREFLY_TX_LF_LO_REG      = 0x0A;
    const uint8_t FIREFLY_TX_LF_HI_REG      = 0x09;

    const uint8_t FIREFLY_TX_TEMP_ALARM_REG = 0x11;
    const uint8_t FIREFLY_TX_VCC_ALARM_REG  = 0x12;




    const uint8_t FIREFLY_TEMP_REG          = 0x16;
    const uint8_t FIREFLY_VOLTAGE_LO_REG    = 0x1B;
    const uint8_t FIREFLY_VOLTAGE_HI_REG    = 0x1A;

    const uint8_t FIREFLY_DISABLE_HI_ADDR   = 0x34;
    const uint8_t FIREFLY_DISABLE_LO_ADDR   = 0x35;

    const uint8_t FIREFLY_INVERT_HI_ADDR   = 0x3A;
    const uint8_t FIREFLY_INVERT_LO_ADDR   = 0x3B;

    const uint8_t FIREFLY_RX_AMP_0_1_ADDR  = 0x43;
    const uint8_t FIREFLY_RX_AMP_2_3_ADDR  = 0x42;
    const uint8_t FIREFLY_RX_AMP_4_5_ADDR  = 0x41;
    const uint8_t FIREFLY_RX_AMP_6_7_ADDR  = 0x40;
    const uint8_t FIREFLY_RX_AMP_8_9_ADDR  = 0x3F;
    const uint8_t FIREFLY_RX_AMP_A_B_ADDR  = 0x3E;

    const uint8_t FIREFLY_RX_EMP_0_1_ADDR  = 0x49;
    const uint8_t FIREFLY_RX_EMP_2_3_ADDR  = 0x48;
    const uint8_t FIREFLY_RX_EMP_4_5_ADDR  = 0x47;
    const uint8_t FIREFLY_RX_EMP_6_7_ADDR  = 0x46;
    const uint8_t FIREFLY_RX_EMP_8_9_ADDR  = 0x45;
    const uint8_t FIREFLY_RX_EMP_A_B_ADDR  = 0x44;

    // conversion from firefly register to temperature - to be checked.
    const float FIREFLY_TEMP_CONVERSION = 1.0;

    const uint8_t I2C_MUX_POWER_ADDR       = 0x4;
    const uint8_t I2C_DAUGHTER_CURRENT_ADDR= 0x40;
    const uint8_t I2C_MOTHER_CURRENT_ADDR  = 0x45;
    const uint8_t I2C_CURRENT_MONITOR_CONFIG_REG_ADDR = 0x0;
    const uint8_t I2C_SHUNT_VOLTAGE_REG_ADDR = 0x1;
    const uint8_t I2C_BUS_VOLTAGE_REG_ADDR = 0x2;

    const uint8_t SI_I2C_ADDR              = 0x68;

    const uint8_t INVERTED      = 0x1;
    const uint8_t NON_INVERTED  = 0x0;

    const uint8_t CLK_FIBRE     = 0x0;
    const uint8_t RST_FIBRE     = 0x1;

    // Daughter addresses
    const uint8_t DAUGHTER_0 =  0x70;
    const uint8_t DAUGHTER_1 =  0x74;
    const uint8_t DAUGHTER_2 =  0x71;
    const uint8_t DAUGHTER_3 =  0x75;
    const uint8_t DAUGHTER_4 =  0x72;
    const uint8_t DAUGHTER_5 =  0x76;
    const uint8_t DAUGHTER_6 =  0x73;
    const uint8_t DAUGHTER_7 =  0x77;

    const vector<uint8_t> DAUGHTERS{ DAUGHTER_0,
                                     DAUGHTER_1,
                                     DAUGHTER_2,
                                     DAUGHTER_3,
                                     DAUGHTER_4,
                                     DAUGHTER_5,
                                     DAUGHTER_6,
                                     DAUGHTER_7};

    uint32_t reverse_bytes(uint32_t bytes);
};

#endif // CLOCKBOARD_H
