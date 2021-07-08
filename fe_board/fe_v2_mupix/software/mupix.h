#ifndef mupix_H_
#define mupix_H_

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
        
        sc->ram->data[0xFF47]=0x0000000F; // set spi slow down
        sc->ram->data[0xFF40]=0x00000FC0;// clear fifos
        sc->ram->data[0xFF40]=0x00000000;
        sc->ram->data[0xFF49]=0x00000003;
        sc->ram->data[0xFF48]=0x00000000; // config mask write to all
        
        sc->ram->data[0xFF4A]=0x2A000A03;
        sc->ram->data[0xFF4A]=0xFA3F002F;
        sc->ram->data[0xFF4A]=0x1E041041;
        sc->ram->data[0xFF4A]=0x041E9A51;
        sc->ram->data[0xFF4A]=0x40280000;
        sc->ram->data[0xFF4A]=0x1400C20A;
        sc->ram->data[0xFF4A]=0x0280001F;
        sc->ram->data[0xFF4A]=0x00020038;
        sc->ram->data[0xFF4A]=0x0000FC09;
        sc->ram->data[0xFF4A]=0xF0001C80;
        sc->ram->data[0xFF4A]=0x00148000;
        sc->ram->data[0xFF4A]=0x11802E00;
        for(int i = 0; i<85; i++){
        sc->ram->data[0xFF4A]=0x00000000;}
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
            for(int i=0; i<64; i++){
                value = sc->ram->data[0xFF66];
                if (i>36) continue;
                printf("%i ready: %01x  rx_state: %01x  pll_lock: %01x  disp_err: %01x\n ",i,value>>31,(value>>29) & 0x3,(value>>28) & 0x1,value & 0x0FFFFFFF);
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
            printf("  [0] => write mupix default conf (All)\n");
            printf("  [1] => set mupix config mask\n");
            printf("  [2] => set spi clk slow down reg\n");
            printf("  [3] => print lvds status\n");
            if((sc->ram->data[0xFF90]) & 1U){
                printf("  [4] => do not invert lvds in\n");
            }else{
                printf("  [4] => invert lvds in\n");
            }
            printf("  [5] => set lvds mask\n");
            printf("  [q] => exit\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 'a':
                mupix_write_all_off();
                break;
            case '0':
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
            case '3':
                menu_lvds();
                break;
            case '4':
                sc->ram->data[0xFF90] ^= 1UL;
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
                sc->ram->data[0xFF61]=value;
                sc->ram->data[0xFF62]=value2;
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
