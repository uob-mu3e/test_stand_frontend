#include "scifi_module.h"

char wait_key(useconds_t us = 100000);


#include "../../../fe/software/app_src/sc.h"

//Standard slow control patterns for mutrig1
#include "builtin_config/No_TDC_Power.h"
#include "builtin_config/ALL_OFF.h"
#include <ctype.h>

//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int scifi_module_t::spi_write_pattern(alt_u32 asic, const alt_u8* bitpattern) {
	int status=0;
	uint16_t rx_pre=0xff00;
        //printf("tx | rx\n");
	for(uint16_t nb=MUTRIG1_CONFIG_LEN_BYTES-1; nb>=0; nb--){
		//do spi transaction, one byte at a time
                alt_u8 rx = 0xCC;
                alt_u8 tx = bitpattern[nb];
		
                alt_avalon_spi_command(SPI_BASE, asic, 1, &tx, 0, &rx, nb==0?0:ALT_AVALON_SPI_COMMAND_MERGE);
                rx = IORD_8DIRECT(SPI_BASE, 0);
                //printf("%02X %02x\n",tx,rx);

		//pattern is not in full units of bytes, so shift back while receiving to check the correct configuration state
		unsigned char rx_check= (rx_pre | rx ) >> (8-MUTRIG1_CONFIG_LEN_BITS%8);
		if(nb==MUTRIG1_CONFIG_LEN_BYTES-1){
			rx_check &= 0xff>>(8-MUTRIG1_CONFIG_LEN_BITS%8);
		};

		if(rx_check!=bitpattern[nb]){
//			printf("Error in byte %d: received %2.2x expected %2.2x\n",nb,rx_check,bitpattern[nb]);
			status=-1;
		}
		rx_pre=rx<<8;
	}
		
	return status;
}

//configure ASIC
int scifi_module_t::configure_asic(alt_u32 asic, const alt_u8* bitpattern) {
    printf("[scifi] configure asic(%u)\n", asic);

    int ret;
    ret = spi_write_pattern(asic, bitpattern);
    ret = spi_write_pattern(asic, bitpattern);

    if(ret != 0) {
        printf("[scifi] Configuration error\n");
        return -1;
    }

    return 0;
}


void scifi_module_t::menu(sc_t* sc){

    auto& regs = sc->ram->regs.scifi;

    while(1) {
        printf("  [0] => reset asic\n");
        printf("  [1] => reset datapath\n");
        printf("  [2] => configure all off\n");
        printf("  [3] => data\n");
        printf("  [4] => get datapath status\n");
        printf("  [5] => get slow control registers\n");
	printf("  [6] => dummy generator settings");
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
            printf("[scifi] configuring all off\n");
            for(int i=0;i<n_ASICS;i++)
                configure_asic(i,mutrig_config_ALL_OFF);
            break;
            break;
        case '3':
            printf("TODO...\n");
            break;
        case '4':
            printf("rx_pll_lock / frame_desync / buffer_full : 0x%03X\n", regs.mon.status);
            printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\n", regs.mon.rx_dpa_lock, regs.mon.rx_ready);
            break;
        case '5':
            printf("dummyctrl_reg:    0x%08X\n", regs.ctrl.dummy);
            printf("    :datagen_en   0x%X\n", (regs.ctrl.dummy>>1)&1);
            printf("    :datagen_fast 0x%X\n", (regs.ctrl.dummy>>2)&1);
            printf("    :datagen_cnt  0x%X\n", (regs.ctrl.dummy>>3)&0x3ff);

            printf("dpctrl_reg:       0x%08X\n", regs.ctrl.dp);
            printf("    :mask         0b");
            for(int i=16;i>0;i--) printf("%d", (regs.ctrl.dp>>i)&1);
            printf("\n");

            printf("    :prbs_dec     0x%X\n", (regs.ctrl.dp>>31)&1);
            printf("subdet_reset_reg: 0x%08X\n", regs.ctrl.reset);
            break;
        case '6':
	    menu_reg_dummyctrl(sc);
            break;
        case '7':
	    menu_reg_datapathctrl(sc);
            break;

        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
void scifi_module_t::menu_reg_dummyctrl(sc_t* sc){
    auto& regs = sc->ram->regs.scifi;
    auto reg = regs.ctrl.dummy;

    while(1) {
        printf("  [0] => %s dummy\n",(reg&2) == 0?"enable":"disable");
        printf("  [1] => %s fast hit mode\n",(reg&4) == 0?"enable":"disable");
        printf("  [+] => increase count (currently %u)\n",(reg>>3&0x3fff));
        printf("  [-] => decrease count\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
	uint32_t val;
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            regs.ctrl.dummy = regs.ctrl.dummy ^ (1<<1);
            break;
        case '1':
            regs.ctrl.dummy = regs.ctrl.dummy ^ (1<<2);
            break;
        case '+':
	    val=(reg>>3&0x3fff)+1;
	    regs.ctrl.dummy = (regs.ctrl.dummy & 0x07) | (0x3fff&(val <<3));
            break;
        case '-':
	    val=(reg>>3&0x3fff)-1;
	    regs.ctrl.dummy = (regs.ctrl.dummy & 0x07) | (0x3fff&(val <<3));
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}


void scifi_module_t::menu_reg_datapathctrl(sc_t* sc){
    auto& regs = sc->ram->regs.scifi;
    auto reg = regs.ctrl.dp;

    while(1) {
        printf("  [0] => %s prbs decoder\n",(reg&(1<<31)) == 0?"enable":"disable");
	for(alt_u8 i=0;i<16;i++){
            printf("  [%1x] => %s ASIC %u\n",i,(reg&(1<<i)) == 0?"  mask":"unmask",i);
	}
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            regs.ctrl.dp = regs.ctrl.dp ^ (1<<31);
            break;
        case 'q':
            return;
	default:
	    if(isdigit(cmd))
	        regs.ctrl.dp = regs.ctrl.dp ^ (1<<(cmd-'0'));
	    else
	        if(isxdigit(cmd))
	            regs.ctrl.dp = regs.ctrl.dp ^ (1<<(tolower(cmd)-'a'));
	    else
		printf("invalid command: '%c'\n", cmd);
            break;
        }
    }
}


void scifi_module_t::callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
//    auto& regs = ram->regs.scifi;
    switch(cmd){
    case 0x0101: //power up (not implemented in current FEB)
        break;
    case 0x0102: //power down (not implemented in current FEB)
        break;
    case 0x0103: //configure all off
	printf("[scifi] configuring all off\n");
        for(int i=0;i<n_ASICS;i++)
            configure_asic(i,mutrig_config_ALL_OFF);
	    //TODO: write some reply to RAM
        break;
    case 0xfffe:
	printf("-ping-\n");
        break;
    case 0xffff:
        break;
    default:
        if((cmd&0xfff0) ==0x0110){ //configure ASIC
		uint8_t chip=cmd&0x000f;
		configure_asic(chip,(alt_u8*) data);
	        //TODO: write some reply to RAM
        }
    }
}
