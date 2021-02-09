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
        
        // example: writing to BIAS shift reg
        
        // set spi clk slow down (spi clk period will be something like 12.8ns * this value, not sure what we need/can do here)
        sc->ram->data[0xFF47]=0x0000002;
        
        // to which mupix chips do you NOT want to write this (bit mask, 0 = write to all mupix)
        sc->ram->data[0xFF48]=0;
        
        // write data for the  complete BIAS reg into FEB storage
        sc->ram->data[0xFF41]=0xD1AFB54D;
        sc->ram->data[0xFF41]=0xAB75183F;
        sc->ram->data[0xFF41]=0x12345678;
        sc->ram->data[0xFF41]=0xD1AFB54D;
        sc->ram->data[0xFF41]=0xAB75183F;
        sc->ram->data[0xFF41]=0x12345678;
        sc->ram->data[0xFF41]=0xD1AFB54D;
        
        // enable signal for BIAS reg
        sc->ram->data[0xFF40]=1;
        sc->ram->data[0xFF40]=0;
        // now you should see stuff happening on the Pins
        return;
    }

    
    //write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
    alt_u16 set_chip_dacs(alt_u32 asic, volatile alt_u32* bitpattern) {
        printf("[mupix] configure asic(%u)\n", asic);

        sc->ram->data[0xFF8D] = 0x005e0000 + (17 << asic); // 4 Sensors
        for(int i = 0; i < MUPIX8_LEN32; i++) {
            sc->ram->data[0xFF8D] = bitpattern[i];
            //printf("0x%08x\n",bitpattern[i]);
        }
        sc->ram->data[0xFF8E] = 0x00100001;
        sc->ram->data[0xFF95] = 0; //TODO: check if we really want to do this here (reset of chip dac fifo, why do we need this, is the fifo content already written to the chip here ???)

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
    
    void read_counters() {
        printf("[mupix] trigger read counters\n");
    }

    void menu() {
        auto& regs = sc->ram->regs.scifi;
        alt_u32 lvdsMask = 0x0;
        char str[2] = {0};
        
        while(1) {
            printf("  [t] => test mupix DAB (All)\n");
            /*
            printf("  [b] => set default board DACs (All)\n");
            printf("  [0] => set default chip A DACs\n");
            printf("  [1] => set default chip B DACs\n");
            printf("  [2] => set default chip C DACs\n");
            printf("  [3] => set default chip E DACs\n");
            printf("  [4] => lvds links\n");
            printf("  [5] => print lvds mask\n");
            printf("  [6] => write lvds mask\n");
            printf("  [7] => print lvds dvalid\n");
            printf("  [8] => disable/enable run prep ack\n");
            */
            printf("  [q] => exit\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 't':
                test_mupix_write();
                break;
            /*case '0':
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
            case '5':
                printf("lvds mask: 0x%08x\n",sc->ram->data[0xFF96]);
                break;
            case '6':
                lvdsMask = 0x0;
                printf("Enter lvdsMask in hex: ");

                for(int i = 0; i<8; i++){
                    printf("mask: 0x%08x\n", lvdsMask);
                    str[0] = wait_key();
                    lvdsMask = lvdsMask*16+strtol(str,NULL,16);
                }

                printf("setting mask to 0x%08x\n", lvdsMask);
                sc->ram->data[0xFF96] = lvdsMask;
                break;
            case '7':
                printf("lvds dvalid: 0x%08x\n",sc->ram->data[0xFF97]);
                break;
            case '8':
                sc->ram->data[0xFF98]=(sc->ram->data[0xFF98]+1)%2;
                printf("run prep ack (1=always ack): 0x%08x\n",sc->ram->data[0xFF98]);
                break;
            case 'b':
                set_board_dacs(0, default_board_dacs);
                break;
            */
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
