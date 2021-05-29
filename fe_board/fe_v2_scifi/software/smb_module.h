#ifndef SMB_MODULE_H_
#define SMB_MODULE_H_

#include <sys/alt_alarm.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#include "include/i2c.h"

//forward declarations
struct sc_t;

#include "smb_constants.h"

struct SMB_t {
    sc_t& sc;
    //TODO: add spi in parameters
    SMB_t(sc_t& _sc):sc(_sc){};
    int n_MODULES=1; // TODO: Should this be a variable?

    //=========================
    //ASIC configuration
    void        SPI_sel(int asic, bool enable=true);
    //write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
    int spi_write_pattern(alt_u32 spi_slave, const alt_u8* bitpattern);
    int spi_write_pattern_nb(alt_u32 spi_slave, alt_u16 nBytes, alt_u8 byteValue); 
    //write and verify pattern twice, toggle i2c lines via i2c
    alt_u16     configure_asic(alt_u32 asic, const alt_u8* bitpattern);
    alt_u16     configure_asic_nb(alt_u32 asic, alt_u16 nBytes, alt_u8 byteValue);
    //print out a given pattern for debugging
    void        print_config(const alt_u8* bitpattern);
 
    void        read_CEC(int asic){asic++;};//TODO
    //monitoring 
    void        read_tmp_all();
    void        print_tmp_all();

    //Copied from FEB1 including comment
    //Reset skew configuration
    //shadow storage of reset skew configuration,
    //we do not have this in a register
    uint8_t resetskew_count[4];
    void RSTSKWctrl_Clear();
    void RSTSKWctrl_Set(uint8_t channel, uint8_t value);
   
    //=========================
    //lower level functions
    void    read_temperature_sensor(int z, int phi);//id 0 to 13


    alt_u16* data_all_tmp;//[32];//TODO this should be the point to register addr in sc_ram


    //Read back the TMP117 deviceid to check if the sensor responds
    bool check_temperature_sensor(int z, int phi); //z from 0 to 12; phi from 0 to 1

    //=========================
    //Menu functions for command line use
    void menu_SMB_monitors();
    void menu_SMB_debug();
    void menu_SMB_main();
    void menu_reset();
    void menu_counters();
    void menu_reg_dummyctrl();
    void menu_reg_datapathctrl();
    void menu_reg_resetskew();
    alt_u16 reset_counters();
    alt_u16 store_counters(volatile alt_u32* data);
    //Slow control callback
    alt_u16 sc_callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n);
    alt_u16 sc_callback_nb(alt_u16 cmd,  alt_u16 nBytes, alt_u8 byteValue);

};
#endif
