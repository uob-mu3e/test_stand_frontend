#ifndef TMB_MODULE_H_
#define TMB_MODULE_H_
/* FEB
#include <sys/alt_alarm.h>
*/
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include "include/i2c.h"

//forward declarations
struct sc_t;

#include "tmb_constants.h"

struct TMB_t{
    i2c_t& i2c;
    sc_t& sc;
    //TODO: add spi in parameters
    TMB_t(i2c_t& _i2c, sc_t& _sc):i2c(_i2c),sc(_sc){};

    //=========================
    // higher level functions
    void        init_TMB(bool enable=true);
    void        power_ASIC(int asic, bool enable=true);

    //control the ASIC power domains of the TMB
    void        power_VCC18A(int asic, bool enable=true);
    void        power_VCC18D(int asic, bool enable=true);
    void	power_ASIC_all(bool enable);
    //control the pulse injection (pll_test) distribution tree of the TMB - output enable signal
    void	setInject(bool enable);

    //ASIC configuration
    void        SPI_sel(int asic, bool enable=true);
    //write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
    int spi_write_pattern(alt_u32 spi_slave, const alt_u8* bitpattern);
    //write and verify pattern twice, toggle i2c lines via i2c
    alt_u16     configure_asic(alt_u32 asic, const alt_u8* bitpattern);
    //print out a given pattern for debugging
    void        print_config(const alt_u8* bitpattern);

 
    void        read_CEC(int asic){asic++;};//TODO
    //monitoring 
    void        init_current_monitor();
    void        init_tmp_monitor(){}; 
    void        read_tmp_all();
    void        read_power_all();
    void        print_tmp_all();
    void        print_power_all();
   
    //=========================
    //lower level functions
    void    I2C_mux_sel(int gid);
    void    I2C_bus_sel(int id);
    alt_u8  get_temperature_address(int id);
    alt_u16 read_temperature_sensor(int id);//id 0 to 25
    alt_u16 switch_readout(alt_u16 tmp);
    alt_u16 read_vsense(int id, int ch);
    alt_u16 read_vsource(int id,int ch); // call I2C_mux_sel inside
    void    read_pow_limit(int id);// purpose: to check which measurement is out of limit
    //TODO add function to check alert line and handling of interrupt - polled with some timer, fw interrupt?

    //periperal access for communication with TMB    
    //I2C R/W related functions - KB: needed?
    void        i2c_write_regs(const i2c_reg_t* regs, int n);
    //void        i2c_write_u32(volatile alt_u32* data_u32, int n); //KB - needed?
    //i2c_reg_t   u32_to_i2c_reg(alt_u32 data_u32); //KB - needed?
    alt_u8      I2C_read(alt_u8 slave, alt_u8 addr);
    alt_u16     I2C_read_16(alt_u8 slave, alt_u8 addr);
    void        I2C_write(alt_u8 slave, alt_u8 addr, alt_u8 data);
    

    alt_u16 data_all_tmp[32];//TODO this should be the point to register addr in sc_ram
    alt_u16 data_all_power[64];//TODO this should be the point to register addr in sc_ram
    alt_u8*  data_all_powerStat;//[16];//TODO this should be the point to register addr in sc_ram



    //Read back PAC1720 ProductID to check response
    bool check_power_monitor(int id, int ch);
    void check_power_monitor_all();

    //Read back the TMP117 deviceid to check if the sensor responds
    bool check_temperature_sensor(int id); //z from 0 to 25;
    void check_temperature_sensor_all();

    //=========================
    //Menu functions for command line use
    void menu_TMB_monitors();
    void menu_TMB_debug();
    void menu_TMB_main();
    void menu_TMB_ASIC();
    //Slow control callback
    alt_u16 sc_callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n);

};
#endif
