#include "smb_module.h"
#include "smb_constants.h"
#include "builtin_config/mutrig2_config.h"
#include <ctype.h>

//from base.h
char wait_key(useconds_t us = 100000);


#include "../../fe/software/sc.h"
#include "../../../common/include/feb.h"
#include <altera_avalon_spi.h>
#include "altera_avalon_spi_regs.h"
#include "altera_avalon_spi.h"
#include "include/scifi_registers.h"

//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int SMB_t::spi_write_pattern(alt_u32 spi_slave, const alt_u8* bitpattern) {
    char tx_string[681];
    char rx_string[681];
    int result_i=0;
	int status=0;
	uint16_t rx_pre=0xff00;
    uint16_t nb=MUTRIG_CONFIG_LEN_BYTES;
    do{
        nb--;
//do spi transaction, one byte at a time
        alt_u8 rx = 0xCC;
        alt_u8 tx = bitpattern[nb];

        alt_avalon_spi_command(SPI_BASE, spi_slave, 1, &tx, 0, &rx, nb==0?0:ALT_AVALON_SPI_COMMAND_MERGE);
        rx = IORD_8DIRECT(SPI_BASE, 0);

        //printf("tx:%2.2X rx:%2.2x nb:%d\n",tx,rx,nb);
        //Lchar result_hex[3];
        //Lchar tx_hex[3];
        //Lsprintf(result_hex,"%2.2X",rx);
        //Lsprintf(tx_hex,"%2.2X",tx);
        //Lrx_string[result_i] = result_hex[0];
        //Ltx_string[result_i] = tx_hex[0];
        //Lresult_i++;
        //Lrx_string[result_i] = result_hex[1];
        //Ltx_string[result_i] = tx_hex[1];
        //Lresult_i++;

        //pattern is not in full units of bytes, so shift back while receiving to check the correct configuration state
        unsigned char rx_check= (rx_pre | rx ) >> (8-MUTRIG_CONFIG_LEN_BITS%8);
        if(nb==MUTRIG_CONFIG_LEN_BYTES-1){
            rx_check &= 0xff>>(8-MUTRIG_CONFIG_LEN_BITS%8);
        };

        if(rx_check!=bitpattern[nb]){
            //Lprintf("Error in byte %d: received %2.2x expected %2.2x\n",nb,rx_check,bitpattern[nb]);
            status=-1;
        }
        rx_pre=rx<<8;
    }while(nb>0);
    printf("\n");
    rx_string[680]=0;
    tx_string[680]=0;
    //Lprintf("TX = %s\n", tx_string);
    //Lprintf("RX = %s\n", rx_string);
    return status;
}

//=======
//	uint16_t nb=MUTRIG_CONFIG_LEN_BYTES;
//       	do{
//		nb--;
//		//do spi transaction, one byte at a time
//                alt_u8 rx = 0xCC;
//                alt_u8 tx = bitpattern[nb];
//
//                alt_avalon_spi_command(SPI_BASE, spi_slave, 1, &tx, 0, &rx, nb==0?0:ALT_AVALON_SPI_COMMAND_MERGE);
//                rx = IORD_8DIRECT(SPI_BASE, 0);
////                printf("%02X %02x\n",tx,rx);
////                printf("%02X ",tx);
//
//		//pattern is not in full units of bytes, so shift back while receiving to check the correct configuration state
//		unsigned char rx_check= (rx_pre | rx ) >> (8-MUTRIG_CONFIG_LEN_BITS%8);
//		if(nb==MUTRIG_CONFIG_LEN_BYTES-1){
//			rx_check &= 0xff>>(8-MUTRIG_CONFIG_LEN_BITS%8);
//		};
//
//		if(rx_check!=bitpattern[nb]){
////			printf("Error in byte %d: received %2.2x expected %2.2x\n",nb,rx_check,bitpattern[nb]);
//			status=-1;
//		}
//		rx_pre=rx<<8;
//	}while(nb>0);
////                printf("\n");
//	return status;
//>>>>>>> origin/SciFi_ASIC_cfg



void SMB_t::print_config(const alt_u8* bitpattern) {
	uint16_t nb=MUTRIG_CONFIG_LEN_BYTES;
	do{
		nb--;
                printf("%02X ",bitpattern[nb]);
	}while(nb>0);
}


//configure ASIC
alt_u16 SMB_t::configure_asic(alt_u32 asic, const alt_u8* bitpattern) {
    printf("[SMB] chip_configure(%u)\n", asic);

    int ret;
    ret = spi_write_pattern(asic, bitpattern);
    usleep(0);
    ret = spi_write_pattern(asic, bitpattern);

    if(ret != 0) {
        printf("[scifi] Configuration error\n");
        return FEB_REPLY_ERROR;
    }

    return FEB_REPLY_SUCCESS;
}



//#include "../../../../common/include/feb.h"
using namespace mu3e::daq::feb;
//TODO: add list&document in specbook
//TODO: update functions
alt_u16 SMB_t::sc_callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
    if((cmd & 0xFFF0) == CMD_MUTRIG_ASIC_CFG) {
        int asic = cmd & 0x000F;
        //Lprintf("configuring ASIC %d\n",asic);
        configure_asic(asic, (alt_u8*)data);
    }
    else {
        printf("[sc_callback] unknown command\n");
    }
    return 0;
}

extern int uart;
void SMB_t::menu_SMB_main() {
    volatile sc_ram_t* ram = (sc_ram_t*) AVM_SC_BASE;
    ram->data[0xFF4D] = 0x00;

    while(1) {
        //        TODO: Define menu
        printf("  [0] => Write ALL_OFF config to all ASICs\n");
        printf("  [1] => Write PRBS_single config to all ASICs\n");
        printf("  [2] => Write PLL test config to all ASICs\n");
        printf("  [3] => Write no TDC power config to all ASICs\n");
        printf("  [8] => data\n");
        printf("  [9] => monitor test\n");
        printf("  [a] => counters\n");
        printf("  [s] => get slow control registers\n");
        printf("  [d] => get datapath status\n");
        printf("  [f] => dummy generator settings\n");
        //printf("  [7] => datapath settings\n");
        printf("  [r] => reset things\n");
        //printf(" [b] test led off");
        //printf("[n] test led on");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        printf("%c\n", cmd);
        switch(cmd) {
        case '0':
            for(alt_u8 asic = 0; asic < 8; asic++)
                sc_callback(0x0110 | asic, (alt_u32*) config_ALL_OFF, 0);
            // disbale injext pin
//            sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] & ~(1<<31);
            break;
        case '1':
            for(alt_u8 asic = 0; asic < 8; asic++)
                sc_callback(0x0110 | asic, (alt_u32*) config_PRBS_single, 0);
            // disbale injext pin
  //          sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] & ~(1<<31);
            break;
        case '2':
            for(alt_u8 asic = 0; asic < 8; asic++)
                sc_callback(0x0110 | asic, (alt_u32*) config_plltest, 0);
            // enable inject pin
            sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] | (1<<31);
            break;
        case '3':
            for(alt_u8 asic = 0; asic < 8; asic++)
                sc_callback(0x0110 | asic, (alt_u32*) no_tdc_power, 0);
            // disbale injext pin
    //        sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] & ~(1<<31);
            break;
        case '8':
            printf("buffer_full / frame_desync / rx_pll_lock : 0x%03X\n", sc.ram->data[0xFF00|SCIFI_MON_STATUS_REGISTER_R]);
            printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\n", 
                    sc.ram->data[0xFF00|SCIFI_MON_RX_DPA_LOCK_REGISTER_R], sc.ram->data[0xFF00|SCIFI_MON_RX_READY_REGISTER_R]);
            break;
        case '9':
            menu_SMB_monitors();
            break;
        case 'a':
            menu_counters();
            break;
        //case 'b':
        //    sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W]=0xFFFFFFFF;
        //    break;
        //case 'n':
        //    sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W]=0x00000000;
        //    break;
        case 's': //get slowcontrol registers
            //printf("dummyctrl_reg:    0x%08X\n", regs.ctrl.dummy);
            printf("dummyctrl_reg:    0x%08X\n", sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W]);
            //sc.ram->data[0xFF00|addr];
            printf("    :cfgdummy_en  0x%X\n", (sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W]>>0)&1);
            printf("    :datagen_en   0x%X\n", (sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W]>>1)&1);
            printf("    :datagen_fast 0x%X\n", (sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W]>>2)&1);
            printf("    :datagen_cnt  0x%X\n", (sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W]>>3)&0x3ff);

            printf("dpctrl_reg:       0x%08X\n", sc.ram->data[0xFF00|SCIFI_CTRL_DP_REGISTER_W]);
            printf("    :mask         0b");
            for(int i=15;i>=0;i--) 
                printf("%d", (sc.ram->data[0xFF00|SCIFI_CTRL_DP_REGISTER_W]>>i)&1);
            printf("\n");

            printf("    :dec_disable  0x%X\n", (sc.ram->data[0xFF00|SCIFI_CTRL_DP_REGISTER_W]>>31)&1);
            printf("    :rx_wait_all  0x%X\n", (sc.ram->data[0xFF00|SCIFI_CTRL_DP_REGISTER_W]>>30)&1);
            printf("subdet_reset_reg: 0x%08X\n", sc.ram->data[0xFF00|SCIFI_CTRL_RESET_REGISTER_W]);
            break;
        case 'd': //get datapath status
            printf("Datapath status registers: press 'q' to end\n");
            while(1){
                printf("buffer_full / frame_desync / rx_pll_lock : 0x%03X ", sc.ram->data[0xFF00|SCIFI_MON_STATUS_REGISTER_R]);
                printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\r", 
                        sc.ram->data[0xFF00|SCIFI_MON_RX_DPA_LOCK_REGISTER_R], sc.ram->data[0xFF00|SCIFI_MON_RX_READY_REGISTER_R]);
                if (read(uart,&cmd, 1) > 0){
                    printf("--\n");
                    if(cmd=='q') break;
                }
                usleep(200000);
            };
            break;
        case 'f':
            menu_reg_dummyctrl();
            break;
        //case 'p':
        //    break;
        case 'r':
            menu_reset();
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

void SMB_t::menu_SMB_monitors() {

    while(1) {
        printf("  [0] => read temperature\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            printf("Nice try, but that does not work yet\n");
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
void SMB_t::menu_reg_dummyctrl(){
    while(1) {
        auto reg = sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W];
	//printf("Dummy reg now: %16.16x / %16.16x\n",sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W], reg);
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
            sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] ^ (1<<0);
            break;
        case '1':
            sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] ^ (1<<1);
            break;
        case '2':
            sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] ^ (1<<2);
            break;
        case '+':
	    val=(reg>>3&0x3fff)+1;
	    sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] = (sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] & 0x07) | (0x3fff&(val <<3));
            break;
        case '-':
	    val=(reg>>3&0x3fff)-1;
	    sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] = (sc.ram->data[0xFF00|SCIFI_CTRL_DUMMY_REGISTER_W] & 0x07) | (0x3fff&(val <<3));
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
void SMB_t::menu_reset() {
    while(1) {
        printf("  [1] => reset asic\n");
        printf("  [2] => reset datapath\n");
        printf("  [3] => reset lvds_rx\n");
        printf("  [4] => reset skew settings\n");


        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            sc.ram->data[0xFF00|SCIFI_CTRL_RESET_REGISTER_W] = 1;
            usleep(50000);
            sc.ram->data[0xFF00|SCIFI_CTRL_RESET_REGISTER_W] = 0;
            break;
        case '2':
            sc.ram->data[0xFF00|SCIFI_CTRL_RESET_REGISTER_W] = 2;
            usleep(50000);
            sc.ram->data[0xFF00|SCIFI_CTRL_RESET_REGISTER_W] = 0;
            break;
        case '3':
            sc.ram->data[0xFF00|SCIFI_CTRL_RESET_REGISTER_W] = 4;
            usleep(50000);
            sc.ram->data[0xFF00|SCIFI_CTRL_RESET_REGISTER_W] = 0;
            break;
        case '4':
            menu_reg_resetskew();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

void SMB_t::RSTSKWctrl_Clear(){
    auto& reg = sc.ram->regs.SMB.ctrl.resetdelay;
    //reset pll to zero phase counters
    sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W] = 0x8000;
    for(int i=0; i<4;i++)
       resetskew_count[i]=0;
    sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W] = 0x0000;
}

void SMB_t::RSTSKWctrl_Set(uint8_t channel, uint8_t value){
    if(channel>3) return;
    if(value>7) return;
    uint32_t val = sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W] & 0xffc0;
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
        sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W] = val;
    }
    //printf("\n");
    sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W]= val & 0xffc0;
}
void SMB_t::menu_reg_resetskew(){
    int selected=0;
    while(1) {
        auto reg = sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W];
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
            sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W]^(1<<(10+selected));
            break;
        case 'p':
            sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CTRL_RESETDELAY_REGISTER_W]^(1<<( 6+selected));
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

void SMB_t::menu_counters(){
    char cmd;
    printf("Counters: press 'q' to end / 'r' to reset\n");
    while(1){
	for(char selected=0;selected<5; selected++){
		sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] = selected&0x7;
		switch(selected){
			case 0: printf("Events/Time  [8ns] "); break;
			case 1: printf("Errors/Frame       "); break;
			case 2: printf("PRBS: Errors/Words "); break;
			case 3: printf("LVDS: Errors/Words "); break;
			case 4: printf("SYNCLOSS: Count/-- "); break;
		}
		for(int i=0;i<4;i++){
			sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] = (sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] & 0x7) + (i<<3);
            // FIXME: denominator probably won't work
			printf("| %10u / %18lu |", sc.ram->data[0xFF00|SCIFI_CNT_NOM_REGISTER_REGISTER_R], 
                    (alt_u64) sc.ram->data[0xFF00|SCIFI_CNT_DENOM_LOWER_REGISTER_R]);
		}
		printf("\n");
	}
	printf("\n");

	if (read(uart,&cmd, 1) > 0){
	   printf("--\n");
	   if(cmd=='q') return;
	   if(cmd=='r'){
                reset_counters();
	   	printf("-- reset\n");
	   };
	 }
        usleep(200000);
    };

}

alt_u16 SMB_t::reset_counters(){
    sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] | 1<<15;
    sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] = sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] ^ 1<<15;
	return 0;
}
//write counter values of all channels to memory address *data and following. Return number of asic channels written.
alt_u16 SMB_t::store_counters(volatile alt_u32* data){
	for(uint8_t i=0;i<4*n_MODULES;i++){
		for(uint8_t selected=0;selected<5; selected++){
            sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W] = (selected&0x7) + (i<<3);
			*data = sc.ram->data[0xFF00|SCIFI_CNT_NOM_REGISTER_REGISTER_R];
			printf("%u: %8.8x\n", sc.ram->data[0xFF00|SCIFI_CNT_CTRL_REGISTER_W], *data);
			data++;
            //FIXME: Check the order of upper/lower, compare to code below
            *data = sc.ram->data[0xFF00|SCIFI_CNT_DENOM_UPPER_REGISTER_R];
			printf("%u: %8.8x\n", sc.ram->data[0xFF00|SCIFI_CNT_DENOM_UPPER_REGISTER_R],*data);
            data++;
            *data = sc.ram->data[0xFF00|SCIFI_CNT_DENOM_LOWER_REGISTER_R];
			printf("%u: %8.8x\n", sc.ram->data[0xFF00|SCIFI_CNT_DENOM_LOWER_REGISTER_R],*data);
            data++;
			//*data=(sc.ram->regs.SMB.counters.denom>>32)&0xffffffff;
			//printf("%u: %8.8x\n", sc.ram->regs.SMB.counters.ctrl,*data);
			//data++;
			//*data=(sc.ram->regs.SMB.counters.denom    )&0xffffffff;
			//printf("%u: %8.8x\n", sc.ram->regs.SMB.counters.ctrl,*data);
			//data++;
		}
	}
	return 4*n_MODULES; //return number of asic channels written so we can parse correctly later
}
