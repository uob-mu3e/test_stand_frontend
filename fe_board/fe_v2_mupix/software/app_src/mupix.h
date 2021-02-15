#ifndef mupix_H_
#define mupix_H_

#include "default_mupix_dacs.h"

#include "sc_mupix.h"

//declaration of interface to scifi module: hardware access, menu, slow control handler
struct mupix_t {
    sc_t* sc;
    mupix_t(sc_t* sc_): sc(sc_){};
    
    const uint32_t MUPIX8_LEN32 = 94;
    const uint32_t MUPIX_CONFIG_LEN_BYTES=MUPIX8_LEN32*4;
    const uint32_t MUPIX_CONFIG_LEN_BITS =MUPIX8_LEN32*4*8;
    const uint32_t MUPIXBOARD_LEN32 = 2;

    const uint8_t  n_ASICS=1;

    void test_mupix_write() {
        printf("running mupix test write function ..\n");
        printf("Chip mask     was set to : 0x%08x\n", sc->ram->data[0xFF48]);
        printf("SPI slow down was set to : 0x%08x  (do not use 0 !)\n", sc->ram->data[0xFF47]);
        
        // example: writing to BIAS shift reg
        
        // clear config fifos
        sc->ram->data[0xFF40]=0x00000FC0;
        sc->ram->data[0xFF40]=0x00000000;
        
        // invert 29 bit shift reg order ? (no sure if i took the correct one in firmware) --> set bit 0 to 1
        // invert csn ? --> set bit 1 to 1
        sc->ram->data[0xFF49]=0x00000002;
        
        // write data for the  complete BIAS reg into FEB storage

        sc->ram->data[0xFF41]=0x2A000A03;
        sc->ram->data[0xFF41]=0xFA3F002F;
        sc->ram->data[0xFF41]=0x1E041041;
        sc->ram->data[0xFF41]=0x041E9A51;
        sc->ram->data[0xFF41]=0x40280000;
        sc->ram->data[0xFF41]=0x1400C20A;
        sc->ram->data[0xFF41]=0x028A0000;
        
        // enable signal for BIAS reg
        sc->ram->data[0xFF40]=1;
        sc->ram->data[0xFF40]=0;
        usleep(0.1); // dont need this, but i have not simulated simultanious write at the moment
        
        //write conf defaults
        sc->ram->data[0xFF42]=0x001F0002;
        sc->ram->data[0xFF42]=0x08380000;
        sc->ram->data[0xFF42]=0xFC05F000;
        sc->ram->data[0xFF40]=2;
        sc->ram->data[0xFF40]=0;
        usleep(0.1);
        
        // write vdac defaults
        sc->ram->data[0xFF43]=0x00720000;
        sc->ram->data[0xFF43]=0x4C000047;
        sc->ram->data[0xFF43]=0x00000000;
        sc->ram->data[0xFF40]=4;
        sc->ram->data[0xFF40]=0;
        usleep(0.1);
        
        // zero the rest
        sc->ram->data[0xFF44]=0x00000000;
        sc->ram->data[0xFF40]=8;
        sc->ram->data[0xFF40]=0;
        usleep(0.1);
        sc->ram->data[0xFF45]=0x00000000;
        sc->ram->data[0xFF40]=16;
        sc->ram->data[0xFF40]=0;
        usleep(0.1);
        sc->ram->data[0xFF46]=0x00000000;
        sc->ram->data[0xFF40]=32;
        sc->ram->data[0xFF40]=0;
        usleep(0.1);
        
        //sc->ram->data[0xFF40]=63;
        //sc->ram->data[0xFF40]=0;
        return;
    }

    
    //write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
    alt_u16 set_chip_dacs(alt_u32 asic, volatile alt_u32* bitpattern) {
        return FEB_REPLY_SUCCESS;
    }

    alt_u16 set_board_dacs(alt_u32 asic, volatile alt_u32* bitpattern) {
        return FEB_REPLY_SUCCESS;
    }

    void powerup() {
        printf("[scifi] powerup: not implemented\n");
    }

    void powerdown() {
        printf("[scifi] powerdown: not implemented\n");
    }
    
    void read_counters() {
        printf("[mupix] trigger read counters\n");
    }

    void menu() {
        auto& regs = sc->ram->regs.scifi;
        alt_u32 value = 0x0;
        char str[2] = {0};
        
        while(1) {
            printf("  [t] => test mupix DAB (All)\n");
            printf("  [1] => set mupix config mask\n");
            printf("  [2] => set spi clk slow down reg\n");
            printf("  [q] => exit\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 't':
                test_mupix_write();
                break;
            case '1':
                value = 0x0;
                printf("Enter Chip Mask in hex: ");

                for(int i = 0; i<8; i++){
                    printf("mask: 0x%08x\n", value);
                    str[0] = wait_key();
                    value = value*16+strtol(str,NULL,16);
                }

                printf("setting mask to 0x%08x\n",value);
                sc->ram->data[0xFF48]=value;
                break;
            case '2':
                value = 0x0;
                printf("Enter value in hex:(clk period will be something like 12.8ns * this value)");

                for(int i = 0; i<8; i++){
                    printf("value: 0x%08x\n", value);
                    str[0] = wait_key();
                    value = value*16+strtol(str,NULL,16);
                }

                printf("setting spi slow down to 0x%08x\n",value);
                sc->ram->data[0xFF47]=value;
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
        case 0x0101: //power up (not implemented in current FEB)
            break;
        case 0x0102: //power down (not implemented in current FEB)
            break;
        case 0x0105: //read counters
            read_counters();
            break;
        case 0xfffe:
            printf("-ping-\n");
            break;
        case 0xffff:
            break;
        case 0x0110:
            status=set_chip_dacs(data[0], &(data[1]));
            return status;
        case 0x0120:
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
