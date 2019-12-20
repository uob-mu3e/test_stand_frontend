#ifndef mupix_H_
#define mupix_H_
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
struct mupix_t {
    sc_t* sc;
    mupix_t(sc_t* sc_): sc(sc_){};
    
    const uint32_t MUPIX8_LEN32 = 94;
    const uint32_t MUPIX_CONFIG_LEN_BYTES=MUPIX8_LEN32*4;
    const uint32_t MUPIX_CONFIG_LEN_BITS =MUPIX8_LEN32*4*8;
    const uint32_t MUPIXBOARD_LEN32 = 2;

    const uint8_t  n_ASICS=1;

    //write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
    alt_u16 set_chip_dacs(alt_u32 asic, volatile alt_u32* config);
    alt_u16 set_board_dacs(alt_u32 asic, volatile alt_u32* config);


    void powerup() {
        printf("[scifi] powerup: not implemented\n");
    }

    void powerdown() {
        printf("[scifi] powerdown: not implemented\n");
    }


    void menu();
    alt_u16 callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n);


};


#endif
