

#include "smb_module.h"
#include "smb_constants.h"
#include "builtin_config/mutrig1_config.h"

//from base.h
char wait_key(useconds_t us = 100000);


#include "../../fe/software/sc.h"
#include "../../../common/include/feb.h"
#include <altera_avalon_spi.h>



void SMB_t::SPI_sel(int asic, bool enable){
    alt_u8 CS_bit = 1 << (2 + asic%2*4);//TODO check this for SMB
    if(enable){
    }else{
    }
}


void SMB_t::read_temperature_sensor(int z, int phi){
}


bool SMB_t::check_temperature_sensor(int z, int phi){
    return false;
}

void SMB_t::read_tmp_all(){
}


void SMB_t::print_tmp_all(){
//    for(int id = 0; id<N_CHIP; id++){
//        for(int i_side=0; i_side<2; i_side++){
//            printf("TMP[%d][%d]:\t 0x%04X\n",id,i_side,data_all_tmp[id*2+i_side]);
//        }
//    }
}


//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int SMB_t::spi_write_pattern(alt_u32 spi_slave, const alt_u8* bitpattern) {
	int status=0;
	uint16_t rx_pre=0xff00;
//        printf("tx | rx\n");
	uint16_t nb=MUTRIG_CONFIG_LEN_BYTES;
       	do{
		nb--;
		//do spi transaction, one byte at a time
                alt_u8 rx = 0xCC;
                alt_u8 tx = bitpattern[nb];

                alt_avalon_spi_command(SPI_BASE, spi_slave, 1, &tx, 0, &rx, nb==0?0:ALT_AVALON_SPI_COMMAND_MERGE);
                rx = IORD_8DIRECT(SPI_BASE, 0);
                printf("%02X %02x\n",tx,rx);
                printf("%02X ",tx);

		//pattern is not in full units of bytes, so shift back while receiving to check the correct configuration state
		unsigned char rx_check= (rx_pre | rx ) >> (8-MUTRIG_CONFIG_LEN_BITS%8);
		if(nb==MUTRIG_CONFIG_LEN_BYTES-1){
			rx_check &= 0xff>>(8-MUTRIG_CONFIG_LEN_BITS%8);
		};

		if(rx_check!=bitpattern[nb]){
			printf("Error in byte %d: received %2.2x expected %2.2x\n",nb,rx_check,bitpattern[nb]);
			status=-1;
		}
		rx_pre=rx<<8;
	}while(nb>0);
                printf("\n");
	return status;
}



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
    SPI_sel(asic, true);
    ret = spi_write_pattern(0, bitpattern);
    SPI_sel(asic, false);
    usleep(0);
    SPI_sel(asic, true);
    ret = spi_write_pattern(0, bitpattern);
    SPI_sel(asic, false);

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
        printf("configuring ASIC\n");
        int asic = cmd & 0x000F;
        configure_asic(asic, (alt_u8*)data);
    }
    else {
        printf("[sc_callback] unknown command\n");
    }
    return 0;
}

void SMB_t::menu_SMB_main() {
    auto& regs = sc.ram->regs.SMB;

    while(1) {
        //        TODO: Define menu
//        printf("  [0] => reset\n");
        printf("  [0] => Write ALL_OFF config ASIC\n");
        printf("  [1] => powerup MALIBU\n");
        printf("  [2] => powerdown MALIBU\n");
        printf("  [3] => powerup ASIC 0\n");
//        printf("  [4] => stic3_config_PLL_TEST_ch0to6_noGenIDLE\n");
//        printf("  [5] => data\n");
        printf("  [6] => monitor test\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            sc_callback(0x0110, (alt_u32*) mutrig_config_ALL_OFF, 0);
            sc_callback(0x0110, (alt_u32*) mutrig_config_ALL_OFF, 1);
            sc_callback(0x0110, (alt_u32*) mutrig_config_ALL_OFF, 2);
            sc_callback(0x0110, (alt_u32*) mutrig_config_ALL_OFF, 3);
            break;
        case '1':
            break;
        case '2':
            break;
        case '3':
            break;
        case '4':
//            chip_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
            break;
        case '5':
            printf("buffer_full / frame_desync / rx_pll_lock : 0x%03X\n", regs.mon.status);
            printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\n", regs.mon.rx_dpa_lock, regs.mon.rx_ready);
            break;
        case '6':
            menu_SMB_monitors();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
void SMB_t::menu_SMB_debug() {

    while(1) {
//        printf("  [0] => check power monitors\n");
//        printf("  [1] => check temperature sensors\n");
//        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            for(int i=0;i<13;i++){
                auto ret=0;//check_power_monitor(i);
//                printf("Power monitor #%d: %d\n",i,ret);
            }
            break;
        case '1':
            for(int i=0;i<13;i++){
                for(int phi=0;phi<2;phi++){
                    auto ret=check_temperature_sensor(i,0);
//                    printf("Sensor %d.%c: %d\n",i,phi?'L':'R',ret);
                }
            }
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
void SMB_t::menu_SMB_monitors() {

    while(1) {
        printf("  [0] => read power\n");
        printf("  [1] => read temperature\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            break;
        case '1':
            read_tmp_all();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
