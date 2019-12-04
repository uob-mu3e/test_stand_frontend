#include "scifi_module.h"

char wait_key(useconds_t us = 100000);


#include "../../../fe/software/app_src/sc.h"

//Standard slow control patterns for mutrig1
#include "builtin_config/No_TDC_Power.h"
#include "builtin_config/ALL_OFF.h"
#include "builtin_config/PLL_TEST.h"
#include "builtin_config/PRBS_single.h"

#include <ctype.h>
extern sc_t sc;

//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int scifi_module_t::spi_write_pattern(alt_u32 asic, const alt_u8* bitpattern) {
	int status=0;
	uint16_t rx_pre=0xff00;
//        printf("tx | rx\n");
	uint16_t nb=MUTRIG1_CONFIG_LEN_BYTES;
       	do{
		nb--;
		//do spi transaction, one byte at a time
                alt_u8 rx = 0xCC;
                alt_u8 tx = bitpattern[nb];

                alt_avalon_spi_command(SPI_BASE, asic, 1, &tx, 0, &rx, nb==0?0:ALT_AVALON_SPI_COMMAND_MERGE);
                rx = IORD_8DIRECT(SPI_BASE, 0);
//                printf("%02X %02x\n",tx,rx);
//                printf("%02X ",tx);

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
	}while(nb>0);
//        printf("\n");
	return status;
}
void scifi_module_t::print_config(const alt_u8* bitpattern) {
	uint16_t nb=MUTRIG1_CONFIG_LEN_BYTES;
	do{
		nb--;
                printf("%02X ",bitpattern[nb]);
	}while(nb>0);
}


//configure ASIC
alt_u16 scifi_module_t::configure_asic(alt_u32 asic, const alt_u8* bitpattern) {
    printf("[scifi] configure asic(%u)\n", asic);

    int ret;
    ret = spi_write_pattern(asic, bitpattern);
    ret = spi_write_pattern(asic, bitpattern);

    if(ret != 0) {
        printf("[scifi] Configuration error\n");
        return FEB_REPLY_ERROR;
    }

    return FEB_REPLY_SUCCESS;
}

extern int uart;
void scifi_module_t::menu(){

    auto& regs = sc->ram->regs.scifi;
    while(1) {
        printf("  [0] => reset asic\n");
        printf("  [1] => reset datapath\n");
        printf("  [2 || o] => configure all off\n");
        printf("  [t] => configure pll test\n");
        printf("  [p] => configure prbs single hit\n");
	printf("\n");
	printf("  [3] => counters\n");
	printf("  [4] => get datapath status\n");
        printf("  [5] => get slow control registers\n");
	printf("\n");
	printf("  [6] => dummy generator settings\n");
	printf("  [7] => datapath settings\n");
	printf("  [8] => reset skew settings\n");
	printf("\n");
	printf("  [d] => show offsets\n");
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
                configure_asic(i,mutrig_config_ALL_OFF);
            break;
        case 't':
            printf("[scifi] configuring pll test\n");
            for(int i=0;i<n_ASICS;i++)
                configure_asic(i,mutrig_config_plltest);
            break;
	case 'l':
            printf("[scifi] last configured:\n");
            print_config((alt_u8*)(&sc->ram->data[1]));
            break;
        case 'p':
            printf("[scifi] configuring prbs signle hit\n");
            for(int i=0;i<n_ASICS;i++)
		configure_asic(i,config_PRBS_single);
            break;
        case '3':
            menu_counters();
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
	    menu_reg_dummyctrl();
            break;
        case '7':
	    menu_reg_datapathctrl();
            break;
        case '8':
	    menu_reg_resetskew();
            break;
        case 'd':
            printf("span w=                   =%x\n",AVM_SC_SPAN/4);
            printf("ram                       =%x (%x)\n",&(sc->ram)			,((uint32_t)sc->ram           - (uint32_t)(sc->ram))/4);
            printf("ram->regs                 =%x (%x)\n",&(sc->ram->regs)		,((uint32_t)&(sc->ram->regs)  - (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi           =%x (%x)\n",&(regs)			,((uint32_t)&(regs)	      - (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi.ctrl.dummy=%x (%x)\n",&(regs.ctrl.dummy)		,((uint32_t)&(regs.ctrl.dummy)- (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi.ctrl.dp   =%x (%x)\n",&(regs.ctrl.dp)		,((uint32_t)&(regs.ctrl.dp)   - (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi.ctrl.reset=%x (%x)\n",&(regs.ctrl.reset)		,((uint32_t)&(regs.ctrl.reset)- (uint32_t)(sc->ram))/4);
            printf("ram->regs.scifi.ctrl.resetdelay=%x (%x)\n",&(regs.ctrl.resetdelay)		,((uint32_t)&(regs.ctrl.resetdelay)- (uint32_t)(sc->ram))/4);
	    break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
void scifi_module_t::menu_reg_dummyctrl(){
    auto& regs = sc->ram->regs.scifi;

    while(1) {
        auto reg = regs.ctrl.dummy;
	//printf("Dummy reg now: %16.16x / %16.16x\n",regs.ctrl.dummy, reg);
        printf("  [0] => %s config dummy\n",(reg&1) == 0?"enable":"disable");
        printf("  [1] => %s data dummy\n",(reg&2) == 0?"enable":"disable");
        printf("  [2] => %s fast hit mode\n",(reg&4) == 0?"enable":"disable");
        printf("  [+] => increase count (currently %u)\n",(reg>>3&0x3fff));
        printf("  [-] => decrease count\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
	uint32_t val;
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            regs.ctrl.dummy = regs.ctrl.dummy ^ (1<<0);
            break;
        case '1':
            regs.ctrl.dummy = regs.ctrl.dummy ^ (1<<1);
            break;
        case '2':
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


void scifi_module_t::menu_reg_datapathctrl(){
    auto& regs = sc->ram->regs.scifi;

    while(1) {
        auto reg = regs.ctrl.dp;
        printf("  [p] => %s prbs decoder\n",(reg&(1<<31)) == 0?"enable":"disable");
        printf("  [w] => %s wait for all RX ready\n",(reg&(1<<30)) == 0?"enable":"disable");
        printf("  [s] => %s wait sticky\n",(reg&(1<<29)) == 0?"set":"unset");
	for(alt_u8 i=0;i<16;i++){
            printf("  [%1x] => %s ASIC %u\n",i,(reg&(1<<i)) == 0?"  mask":"unmask",i);
	}
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
	printf("Key= '%s'\n",cmd);
        switch(cmd) {
        case 'p':
            regs.ctrl.dp = regs.ctrl.dp ^ (1<<31);
            break;
        case 'w':
            regs.ctrl.dp = regs.ctrl.dp ^ (1<<30);
            break;
        case 's':
            regs.ctrl.dp = regs.ctrl.dp ^ (1<<29);
            break;
        case 'q':
            return;
	default:
	    if(isdigit(cmd)){
		uint8_t key=(cmd-'0');
	        regs.ctrl.dp = regs.ctrl.dp ^ (1<<key);
	    }else if(isxdigit(cmd)){
		    uint8_t key=(tolower(cmd)-'a')+0x0a;
	            regs.ctrl.dp = regs.ctrl.dp ^ (1<<key);
	    }else
		printf("invalid command: '%c'\n", cmd);
            break;
        }
    }
}



void scifi_module_t::RSTSKWctrl_Clear(){
    auto& reg = sc->ram->regs.scifi.ctrl.resetdelay;
    //reset pll to zero phase counters
    reg = 0x8000;
    for(int i=0; i<4;i++)
       resetskew_count[i]=0;
    reg = 0x0000;
}

void scifi_module_t::RSTSKWctrl_Set(uint8_t channel, uint8_t value){
    if(channel>3) return;
    if(value>7) return;
    auto& regs = sc->ram->regs.scifi;
    uint32_t val=regs.ctrl.resetdelay & 0xffc0;
    //printf("PLL_phaseadjust #%u: ",channel);
    while(value!=resetskew_count[channel]){
        val |= (channel+2)<<2;
        if(value>resetskew_count[channel]){ //increment counter
            val |= 2;
	    //printf("+");
	    resetskew_count[channel]++;
	}else{
            val |= 1;
	    //printf("-");
	    resetskew_count[channel]--;
	}
        regs.ctrl.resetdelay = val;
    }
    //printf("\n");
    regs.ctrl.resetdelay= val & 0xffc0;
}

void scifi_module_t::menu_reg_resetskew(){
    auto& regs = sc->ram->regs.scifi;
    int selected=0;
    while(1) {
        auto reg = regs.ctrl.resetdelay;
	printf("Reset delay reg now: %16.16x\n",reg);
        printf("  [0..3] => Select line N (currently %d)\n",selected);

        printf("  [p] => swap phase bit (currently %d)\n",(reg>>(6 +selected)&0x1));
        printf("  [d] => swap delay bit (currently %d)\n",(reg>>(10+selected)&0x1));
        printf("  [+] => increase count (currently %d)\n",resetskew_count[selected]);
        printf("  [-] => increase count (currently %d)\n",resetskew_count[selected]);
        printf("  [r] => reset phase configuration\n");
	
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            break;
        case '1':
            selected=1;
            break;
        case '2':
            selected=2;
            break;
        case '3':
            selected=3;
            break;
	case 'r':
	    RSTSKWctrl_Clear();
	    break;
        case '+':
	    RSTSKWctrl_Set(selected, resetskew_count[selected]+1);
            break;
        case '-':
	    RSTSKWctrl_Set(selected, resetskew_count[selected]-1);
            break;
        case 'd':
            regs.ctrl.resetdelay = regs.ctrl.resetdelay^(1<<(10+selected));
            break;
        case 'p':
            regs.ctrl.resetdelay = regs.ctrl.resetdelay^(1<<( 6+selected));
            break;
        case 'q':
            return;
        default:
	    if(isdigit(cmd) && (cmd - '0') < 4)
		selected=(cmd-'0');
	    else
		printf("invalid command: '%c'\n", cmd);
        }
    }
}

void scifi_module_t::menu_counters(){
    auto& regs = sc->ram->regs.scifi;
    char cmd;
    printf("Counters: press 'q' to end / 'r' to reset\n");
    while(1){
	for(char selected=0;selected<4; selected++){
		regs.counters.ctrl = selected<<3;
		switch(selected){
			case 0: printf("Events/Time  [8ns] "); break;
			case 1: printf("Errors/Frame       "); break;
			case 2: printf("PRBS: Errors/Words "); break;
			case 3: printf("LVDS: Errors/Words "); break;
		}
		for(int i=0;i<4;i++){
			regs.counters.ctrl = (regs.counters.ctrl & 0x18) + i;
			float frag=regs.counters.nom*1.e6/regs.counters.denom;
			printf("| %10u / %18lu |", regs.counters.nom, regs.counters.denom);
		}
		printf("\n");
	}
	printf("\n");

	if (read(uart,&cmd, 1) > 0){
	   printf("--\n");
	   if(cmd=='q') return;
	   if(cmd=='r'){
		regs.counters.ctrl = regs.counters.ctrl | 1<<15;
	   	printf("-- reset\n");
		regs.counters.ctrl = regs.counters.ctrl ^ 1<<15;
	   };
	 }
        usleep(200000);
    };

}

alt_u16 scifi_module_t::callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
//    auto& regs = ram->regs.scifi;
    alt_u16 status=FEB_REPLY_SUCCESS; 
    switch(cmd){
    case 0x0101: //power up (not implemented in current FEB)
        break;
    case 0x0102: //power down (not implemented in current FEB)
        break;
    case 0x0103: //configure all off
	printf("[scifi] configuring all off\n");
        for(int i=0;i<n_ASICS;i++)
           if(configure_asic(i,mutrig_config_ALL_OFF)==FEB_REPLY_ERROR)
              status=FEB_REPLY_ERROR;
	return status;
        break;
    case 0x0104: //configure reset skew phases
	//data[0..3]=phases
	printf("[scifi] configuring reset skews\n");
        for(int i=0;i<4;i++)
	    RSTSKWctrl_Set(i,data[i]);
	return 0;
    case 0xfffe:
	printf("-ping-\n");
        break;
    case 0xffff:
        break;
    default:
        if((cmd&0xfff0) ==0x0110){ //configure ASIC
	   uint8_t chip=data[0];
           status=configure_asic(chip,(alt_u8*) &(data[1]));
           if(sc->ram->regs.scifi.ctrl.dummy&1){
              //when configured as dummy do the spi transaction,
              //but always return success to switching board
	      if(status!=FEB_REPLY_SUCCESS) printf("[WARNING] Using configuration dummy\n");
              status=FEB_REPLY_SUCCESS;
           }
	   return status;
        }else{//unknown command
           return FEB_REPLY_ERROR;
	}
    }

    return 0;
}
