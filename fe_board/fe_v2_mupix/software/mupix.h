#ifndef mupix_H_
#define mupix_H_

#include "include/mupix_registers.h"

//declaration of interface to scifi module: hardware access, menu, slow control handler
struct mupix_t {
    sc_t* sc;
    mupix_t(sc_t* sc_): sc(sc_){};
    
    const uint32_t MUPIX8_LEN32 = 94;
    const uint32_t MUPIX_CONFIG_LEN_BYTES=MUPIX8_LEN32*4;
    const uint32_t MUPIX_CONFIG_LEN_BITS =MUPIX8_LEN32*4*8;
    const uint32_t MUPIXBOARD_LEN32 = 2;

    const uint8_t  n_ASICS=1;

    void mupix_write_all_off(){
        
        sc->ram->data[MP_CTRL_SLOW_DOWN_REGISTER_W]=0x0000000F; // set spi slow down
        sc->ram->data[MP_CTRL_RESET_REGISTER_W]=0x00000001;
        
        for(int i = 0; i<12; i++){
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x2A000A03;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0xFA3F002F;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x1E041041;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x041E9A51;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x40280000;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x1400C20A;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x0280001F;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x00020038;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x0000FC09;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0xF0001C80;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x00148000;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x11802E00;
        }
    }

    void test_tdacs() {

        printf("test tdacs both");

        for(int i = 0; i<24; i++) {
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0xFFFFFFFF;
        }
        //for(int i = 30; i<40; i++) {
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
        //}
        for(int i = 35; i<128; i++) {
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0xFFFFFFFF;
        }

    }

    void test_tdacs2() {

        printf("test tdacs mask");

        for(int i = 0; i<128; i++) {
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0x0;
		}

    }

    void test_tdacs3() {

        printf("test tdacs no mask");

        for(int i = 0; i<128; i++) {
            sc->ram->data[MP_CTRL_TDAC_START_REGISTER_W]=0xFFFFFFFF;
		}

    }

    void test_tdac_pattern() {
        printf("test tdac pattern");
        sc->ram->data[MP_CTRL_RUN_TEST_REGISTER_W]=0x1;
    }
    
    void test_write_all(bool maskPixel) {
        sc->ram->data[MP_CTRL_SPI_ENABLE_REGISTER_W]=0x00000001;
        sc->ram->data[MP_CTRL_RESET_REGISTER_W]=0x00000001;
        printf("test write all");
        for(int i = 0; i<12; i++){
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x2A000A03;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0xFA3F0025;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x1E041041;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x041E5951;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x40280000;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x1400C20A;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x028A001F;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x00000038;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x0000FC09;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0xF0001C80;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x00148000;
            sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x11802E00;
        }
        
    }

    void test_write_one(int i) {
        printf("test write %i", i);
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x2A000A03;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0xFA3F0025;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x1E041041;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x041E5951;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x40280000;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x1400C20A;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x028A001F;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x00000038;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x0000FC09;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0xF0001C80;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x00148000;
        sc->ram->data[MP_CTRL_COMBINED_START_REGISTER_W+i]=0x11802E00;
    }

    void menu_lvds() {
        alt_u32 value = 0x0;
        while (1) {
            char cmd;
            if(read(uart, &cmd, 1) > 0) switch(cmd) {
            case '?':
                wait_key();
                break;
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
            
            printf("pll_lock should always be '1', rx_state 0: wait for dpa_lock 1: alignment 2:ok, disp_err is only counting in rx_state 2\n");
            printf("order is CON2 ModuleA chip1 ABC, chip2 ABC, .. ModuleB chip1 ABC .. CON3..\n");
            for(int i=0; i<37; i++){
                value = sc->ram->data[MP_LVDS_STATUS_START_REGISTER_W+i];
                printf("chip%i, Link%i ready: %01x  rx_state: %01x  pll_lock: %01x  disp_err: %01x\n ",i/3,i,value>>31,(value>>29) & 0x3,(value>>28) & 0x1,value & 0x0FFFFFFF);
            }
            printf("----------------------------\n");
            usleep(200000);
        }
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
        //auto& regs = sc->ram->regs.scifi;
        alt_u32 value = 0x0;
        alt_u32 value2 = 0x0;
        char str[2] = {0};
        
        while(1) {
            printf("  [a] => write all OFF\n");
            printf("  [t] => test tdacs\n");
            printf("  [0] => configure chip Number N\n");
            printf("  [1] => set mupix config mask\n");
            printf("  [2] => set spi clk slow down reg\n");
            printf("  [3] => print lvds status\n");
            if((sc->ram->data[MP_LVDS_INVERT_REGISTER_W]) & 1U){
                printf("  [4] => do not invert lvds in\n");
            }else{
                printf("  [4] => invert lvds in\n");
            }
            printf("  [5] => set lvds mask\n");
            printf("  [6] => test write all\n");
            printf("  [7] => write sorter delay\n");
            printf("  [q] => exit\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 'r':
                sc->ram->data[MP_CTRL_RESET_REGISTER_W]=0x00000001;
                break;
            case 'a':
                mupix_write_all_off();
                break;
	        case 'b':
                test_tdacs();
		        break;
	        case 'm':
                test_tdacs2();
		        break;
            case 'n':
                test_tdacs3();
		        break;
            case 'p':
                test_tdac_pattern();
		        break;
            case '0':
                value = 0x0;
                printf("Enter Chip to configure in hex: ");

                str[0] = wait_key();
                value = strtol(str,NULL,16);
                printf("configuring chip: 0x%08x\n", value);
                if(value<12) {
                    test_write_one(value);
                } else {
                    printf("chip 0x%08x does not exist on any FEB\n", value);
                }

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
                //sc->ram->data[MP_CTRL_CHIP_MASK_REGISTER_W]=value;
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
                sc->ram->data[MP_CTRL_SLOW_DOWN_REGISTER_W]=value;
                break;
            case '3':
                menu_lvds();
                break;
            case '4':
                sc->ram->data[MP_LVDS_INVERT_REGISTER_W] ^= 1UL;
                break;
            case '5':
                value = 0x0;
                value2 = 0x0;
                printf("Enter mask in hex: (36 bit number)\n");
                printf("mask: 0x%01x%08x\n",value2, value);
                str[0] = wait_key();
                value2 = value2*16+strtol(str,NULL,16);
                for(int i = 0; i<8; i++){
                    printf("mask: 0x%01x%08x\n",value2, value);
                    str[0] = wait_key();
                    value = value*16+strtol(str,NULL,16);
                }

                printf("setting lvds mask to 0x%01x%08x\n",value2, value);
                sc->ram->data[MP_LVDS_LINK_MASK_REGISTER_W]=value;
                sc->ram->data[MP_LVDS_LINK_MASK2_REGISTER_W]=value2;
                break;
            case '6':
                test_write_all(false);
                break;
            case '7':
                sc->ram->data[MP_SORTER_DELAY_REGISTER_W]=10;
                break;
            case '9': 
		        sc->ram->data[MP_SORTER_DELAY_REGISTER_W]=0x5FC;
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
