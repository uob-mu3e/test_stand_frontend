#include "mupix.h"
#include "default_mupix_dacs.h"
char wait_key(useconds_t us = 100000);


#include "../../../fe/software/app_src/sc.h"

//Standard slow control patterns for mutrig1
//#include "builtin_config/No_TDC_Power.h"
//#include "builtin_config/ALL_OFF.h"
//#include "builtin_config/PLL_TEST.h"
//#include "builtin_config/PRBS_single.h"

#include <ctype.h>

//configure ASIC
alt_u16 mupix_t::set_chip_dacs(alt_u32 asic, volatile alt_u32* bitpattern) {
    printf("[mupix] configure asic(%u)\n", asic);

    sc->ram->data[0xFF8D] = 0x005e0003;
    for(int i = 0; i < MUPIX8_LEN32; i++) {
        sc->ram->data[0xFF8D] = bitpattern[i];
        }
   sc->ram->data[0xFF8E] = 0x00100001;
   sc->ram->data[0xFF95] = 0;
 
    return FEB_REPLY_SUCCESS;
}

//configure Board
alt_u16 mupix_t::set_board_dacs(alt_u32 asic, volatile alt_u32* bitpattern) {
    printf("[mupix] configure board(%u)\n", asic);

    for(unsigned int i = 0; i < MUPIXBOARD_LEN32; i++) {
        sc->ram->data[0xFF83+i] = bitpattern[i];
        }
   sc->ram->data[0xFF8C] = 0x1;
   sc->ram->data[0xFF8C] = 0x0;
 
    return FEB_REPLY_SUCCESS;
}

extern int uart;
void mupix_t::menu(){

    auto& regs = sc->ram->regs.scifi;
    while(1) {
        printf("  [0] => reset asic\n");
        printf("  [1] => reset datapath\n");
        printf("  [2 || o] => configure all off\n");
        printf("  [t] => configure pll test\n");
        printf("  [p] => configure prbs single hit\n");
        printf("  [3] => data\n");
        printf("  [4] => get datapath status\n");
        printf("  [5] => get slow control registers\n");
	printf("  [6] => dummy generator settings\n");
	printf("  [7] => datapath settings\n");
	printf("  [8] => reset skew settings\n");
	printf("  [d] => set default DACs\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            regs.ctrl.reset = 1;
            usleep(100);
            regs.ctrl.reset = 0;
            break;
        case '1':
            regs.ctrl.reset = 2;
            usleep(100);
            regs.ctrl.reset = 0;
            break;
	case '2':
	case 'o':
            printf("[scifi] configuring all off\n");
            for(int i=0;i<n_ASICS;i++)
                //set_chip_dacs(i,mutrig_config_ALL_OFF);
            break;
        case 't':
            printf("[scifi] configuring pll test\n");
            for(int i=0;i<n_ASICS;i++)
                //set_chip_dacs(i,mutrig_config_plltest);
            break;
        case 'p':
            break;
        case '3':
            printf("TODO...\n");
            break;
        case '4':
            printf("Datapath status registers: press 'q' to end\n");
	    while(1){
                printf("buffer_full / frame_desync / rx_pll_lock : 0x%03X ", regs.mon.status);
                printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\r", regs.mon.rx_dpa_lock, regs.mon.rx_ready);
		if (read(uart,&cmd, 1) > 0){
		   printf("--\n");
		   if(cmd=='q') break;
		}
                usleep(200000);
	    };
            break;
        case '5':
            printf("dummyctrl_reg:    0x%08X\n", regs.ctrl.dummy);
            printf("    :cfgdummy_en  0x%X\n", (regs.ctrl.dummy>>0)&1);
            printf("    :datagen_en   0x%X\n", (regs.ctrl.dummy>>1)&1);
            printf("    :datagen_fast 0x%X\n", (regs.ctrl.dummy>>2)&1);
            printf("    :datagen_cnt  0x%X\n", (regs.ctrl.dummy>>3)&0x3ff);

            printf("dpctrl_reg:       0x%08X\n", regs.ctrl.dp);
            printf("    :mask         0b");
            for(int i=15;i>=0;i--) printf("%d", (regs.ctrl.dp>>i)&1);
            printf("\n");

            printf("    :dec_disable  0x%X\n", (regs.ctrl.dp>>31)&1);
            printf("    :rx_wait_all  0x%X\n", (regs.ctrl.dp>>30)&1);
            printf("subdet_reset_reg: 0x%08X\n", regs.ctrl.reset);
            break;
        case '6':
	        printf("Not implemented\n");
            //menu_reg_dummyctrl();
            break;
        case '7':
 	        printf("Not implemented\n");
	    //menu_reg_datapathctrl();
            break;
        case '8':
 	        printf("Not implemented\n");
	    //menu_reg_resetskew();
            break;
        case 'b':
            printf("span w=                   =%x\n",AVM_SC_SPAN/4);
            printf("ram                       =%x (%x)\n",&(sc->ram)			,((uint32_t)sc->ram           - (uint32_t)(sc->ram))/4);
            printf("ram->regs                 =%x (%x)\n",&(sc->ram->regs)		,((uint32_t)&(sc->ram->regs)  - (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi           =%x (%x)\n",&(regs)			,((uint32_t)&(regs)	      - (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi.ctrl.dummy=%x (%x)\n",&(regs.ctrl.dummy)		,((uint32_t)&(regs.ctrl.dummy)- (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi.ctrl.dp   =%x (%x)\n",&(regs.ctrl.dp)		,((uint32_t)&(regs.ctrl.dp)   - (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi.ctrl.reset=%x (%x)\n",&(regs.ctrl.reset)		,((uint32_t)&(regs.ctrl.reset)- (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi.ctrl.resetdelay=%x (%x)\n",&(regs.ctrl.resetdelay)		,((uint32_t)&(regs.ctrl.resetdelay)- (uint32_t)(sc->ram))/4);
                break;
        case 'd':
            set_chip_dacs(0, default_mupix_dacs);
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}



alt_u16 mupix_t::callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
//    auto& regs = ram->regs.scifi;
    alt_u16 status=FEB_REPLY_SUCCESS; 
    switch(cmd){
    case 0x0101: //power up (not implemented in current FEB)
        break;
    case 0x0102: //power down (not implemented in current FEB)
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
