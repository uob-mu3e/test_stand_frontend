#ifndef mupix_H_
#define mupix_H_

#include "default_mupix_dacs.h"

#include "sc_mupix.h"

#include "../../../../common/include/feb.h"
using namespace mu3e::daq::feb;

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
    alt_u16 set_chip_dacs(alt_u32 asic, volatile alt_u32* bitpattern) {
        printf("[mupix] configure asic(%u)\n", asic);

        sc->ram->data[0xFF8D] = 0x005e0000 + (17 << asic); // 4 Sensors
        for(int i = 0; i < MUPIX8_LEN32; i++) {
            sc->ram->data[0xFF8D] = bitpattern[i];
            //printf("0x%08x\n",bitpattern[i]);
        }
        sc->ram->data[0xFF8E] = 0x00100001;
        sc->ram->data[0xFF95] = 0;

        return FEB_REPLY_SUCCESS;
    }

    alt_u16 set_board_dacs(alt_u32 asic, volatile alt_u32* bitpattern) {
        printf("[mupix] configure board(%u)\n", asic);

        for(unsigned int i = 0; i < MUPIXBOARD_LEN32; i++) {
            sc->ram->data[0xFF83+i] = bitpattern[i];
            //printf("0x%08x\n",bitpattern[i]);
        }
        sc->ram->data[0xFF8C] = 0x1;
        sc->ram->data[0xFF8C] = 0x0;

        return FEB_REPLY_SUCCESS;
    }

    void powerup() {
        printf("[scifi] powerup: not implemented\n");
    }

    void powerdown() {
        printf("[scifi] powerdown: not implemented\n");
    }

    void menu() {
        auto& regs = sc->ram->regs.scifi;
        while(1) {
            printf("  [b] => set default board DACs (All)\n");
            printf("  [0] => set default chip A DACs\n");
            printf("  [1] => set default chip B DACs\n");
            printf("  [2] => set default chip C DACs\n");
            printf("  [3] => set default chip E DACs\n");
	    printf("  [4] => lvds links\n");
            printf("  [q] => exit\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case '0':
                set_chip_dacs(0, default_mupix_dacs);
                break;
            case '1':
                set_chip_dacs(1, default_mupix_dacs);
                break;
            case '2':
                set_chip_dacs(2, default_mupix_dacs);
                break;
            case '3':
                set_chip_dacs(3, default_mupix_dacs);
                break;
            case '4':
	        menu_lvds(sc->ram);		
                break;
            case 'b':
                set_board_dacs(0, default_board_dacs);
                break;
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
        }
    }

    alt_u16 callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
//        auto& regs = ram->regs.scifi;
        alt_u16 status=FEB_REPLY_SUCCESS;
        switch(cmd){
        case CMD_PING:
            printf("-ping-\n");
            break;
        case CMD_MUPIX_CHIP_CFG:
            status=set_chip_dacs(data[0], &(data[1]));
            return status;
        case CMD_MUPIX_BOARD_CFG:
            status=set_board_dacs(data[0], &(data[1]));

/*
        if(sc->ram->regs.scifi.ctrl.dummy&1){
              //when configured as dummy do the spi transaction,
              //but always return success to switching board
	      if(status!=FEB_REPLY_SUCCESS) printf("[WARNING] Using configuration dummy\n");
              status=FEB_REPLY_SUCCESS;

           }*/
            return status;
        default:
            return FEB_REPLY_ERROR;
        }

        return 0;
    }

};

#endif
