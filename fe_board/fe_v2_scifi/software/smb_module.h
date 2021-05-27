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
    void        init_tmp_monitor(){}; 
    void        read_tmp_all();
    void        print_tmp_all();
   
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
    //Slow control callback
    alt_u16 sc_callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n);
    alt_u16 sc_callback_nb(alt_u16 cmd,  alt_u16 nBytes, alt_u8 byteValue);

};
#endif
