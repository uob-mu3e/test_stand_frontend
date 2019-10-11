#ifndef SCIFI_MODULE_H_
#define SCIFI_MODULE_H_
#include <system.h>

#ifndef ALT_CPU_DCACHE_BYPASS_MASK
    #define ALT_CPU_DCACHE_BYPASS_MASK 0
#endif

#include <sys/alt_alarm.h>

#include <altera_avalon_i2c.h>
#include <altera_avalon_spi.h>
#include <altera_avalon_pio_regs.h>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

//forward declarations
struct sc_t;

//declaration of interface to scifi module: hardware access, menu, slow control handler
struct scifi_module_t {
    const uint32_t MUTRIG1_CONFIG_LEN_BYTES=295;
    const uint32_t MUTRIG1_CONFIG_LEN_BITS =2358;
    const uint8_t  n_ASICS=15;
    //write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
    int spi_write_pattern(alt_u32 asic, const alt_u8* bitpattern);
    int configure_asic(alt_u32 asic, const alt_u8* bitpattern);

    void powerup() {
        printf("[scifi] powerup: not implemented\n");
    }

    void powerdown() {
        printf("[scifi] powerdown: not implemented\n");
    }



    void menu(sc_t* sc);
    void callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n);


};


#endif
