#ifndef MALIBU_BASIC_CMD_H_
#define MALIBU_BASIC_CMD_H_

#include "../include/base.h"
#include "../include/i2c.h"
#include "../../../fe/software/app_src/sc.h"
//#include "../../../fe/software/app_src/sc_ram.h"

#include "TMB_constants.h"

struct malibu_t {
    i2c_t& i2c;
    sc_t& sc;
    //TODO: add spi in parameters
    malibu_t(i2c_t& _i2c, sc_t& _sc):i2c(_i2c),sc(_sc){};

    //=========================
    // higher level functions
    void        power_TMB(bool enable=true);
    void        power_ASIC(int asic, bool enable=true);
    
    //void    GPIO_sel(){};
    void        power_VCC18A(int asic, bool enable=true);
    void        power_VCC18D(int asic, bool enable=true);
    void        SPI_sel(int asic, bool enable=true);
    int         chip_configure(int asic, const alt_u8* bitpattern);
    void        read_CEC(int asic){};//TODO
    
    void        init_current_monitor();
    void        init_tmp_monitor(){}; 
    void        read_tmp_all();
    void        read_power_all();
    void        print_tmp_all();
    void        print_power_all();
   
    //=========================
    //lower level functions
    void    I2C_mux_sel(int id);
    void    read_tmp(int id);//id 0 to 13
    alt_u16 read_vsense(int id, int ch);
    alt_u16 read_vsource(int id,int ch); // call I2C_mux_sel inside
    void    read_pow_limit(int id);// purpose: to check which measuremen is out of limit
    //TODO add function to check alert line and handling of interrupt - polled with some timer, fw interrupt?

    //periperal access for communication with TMB    
    //I2C R/W related functions
    void        i2c_write_regs(const i2c_reg_t* regs, int n);
    void        i2c_write_u32(volatile alt_u32* data_u32, int n); 
    i2c_reg_t   u32_to_i2c_reg(alt_u32 data_u32);
    alt_u8      I2C_read(alt_u8 slave, alt_u8 addr);
    alt_u16     I2C_read_16(alt_u8 slave, alt_u8 addr);
    void        I2C_write(alt_u8 slave, alt_u8 addr, alt_u8 data);
    
    //SPI write function
    static alt_u8   spi_write(alt_u32 slave, alt_u8 w);


    alt_u16 data_all_tmp[32];//TODO this should be the point to register addr in sc_ram
    alt_u16 data_all_power[64];//TODO this should be the point to register addr in sc_ram
    alt_u8  data_all_powerStat[16];//TODO this should be the point to register addr in sc_ram

    //TODO: what is this checking exactly?
    bool read_ProductID(alt_u8 addr){
        return (I2C_read(addr,0xfd)==0x57 ? true :false); 
    }
    bool read_ManufID(alt_u8 addr){
        return (I2C_read(addr,0xfe)==0x5d ? true :false); 
    }
    bool read_Revision(alt_u8 addr){
        return (I2C_read(addr,0xff)==0x81 ? true :false); 
    }
    bool read_tmp_deviceID(int id, int i_side=0);

    //=========================
    //Menu functions for command line use
    void menu_TMB_monitors();
    void menu_TMB_main();
    //Slow control callback
    alt_u16 sc_callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n);

};
#endif
