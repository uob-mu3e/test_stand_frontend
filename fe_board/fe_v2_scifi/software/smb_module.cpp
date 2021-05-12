

#include "smb_module.h"
#include "smb_constants.h"
#include "builtin_config/mutrig2_config.h"
#include "builtin_config/mutrig2/FF.h"

//from base.h
char wait_key(useconds_t us = 100000);


#include "../../fe/software/sc.h"
#include "../../../common/include/feb.h"
#include <altera_avalon_spi.h>
#include "altera_avalon_spi_regs.h"
#include "altera_avalon_spi.h"



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
    char tx_string[681];
    char rx_string[681];
    int result_i=0;
	int status=0;
	uint16_t rx_pre=0xff00;
//        printf("tx | rx\n");
    uint16_t nb=MUTRIG_CONFIG_LEN_BYTES;
    //uint16_t nb=500;
    //printf("MUTRIG_CONFIG_LEN_BYTES=%d\n",MUTRIG_CONFIG_LEN_BYTES);
    //do{
    //    nb--;
    //    alt_u8 rx = 0xCC;
    //    alt_u8 tx = 0x00;

    //    alt_avalon_spi_command(SPI_BASE, spi_slave, 1, &tx, 0, &rx, nb==0?0:ALT_AVALON_SPI_COMMAND_MERGE);
    //    rx = IORD_8DIRECT(SPI_BASE, 0);
    //    printf("tx:%2.2X rx:%2.2x nb:%d\n",tx,rx,nb);
    //}while(nb>0);

    //usleep(1000000);

    //nb=MUTRIG_CONFIG_LEN_BYTES;
    do{
        nb--;
//do spi transaction, one byte at a time
        alt_u8 rx = 0xCC;
        alt_u8 tx = bitpattern[nb];

        //TEST:
        //tx = nb & 0xFF;

        alt_avalon_spi_command(SPI_BASE, spi_slave, 1, &tx, 0, &rx, nb==0?0:ALT_AVALON_SPI_COMMAND_MERGE);
        rx = IORD_8DIRECT(SPI_BASE, 0);

        printf("tx:%2.2X rx:%2.2x nb:%d\n",tx,rx,nb);
        //printf("%02X ",tx);
        char result_hex[3];
        char tx_hex[3];
        sprintf(result_hex,"%2.2X",rx);
        sprintf(tx_hex,"%2.2X",tx);
        rx_string[result_i] = result_hex[0];
        tx_string[result_i] = tx_hex[0];
        result_i++;
        rx_string[result_i] = result_hex[1];
        tx_string[result_i] = tx_hex[1];
        result_i++;

        //pattern is not in full units of bytes, so shift back while receiving to check the correct configuration state
        unsigned char rx_check= (rx_pre | rx ) >> (8-MUTRIG_CONFIG_LEN_BITS%8);
        //int shift = 8-MUTRIG_CONFIG_LEN_BITS%8;
        //printf("rx_check: %3.2x\n",rx_check);
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
    rx_string[680]=0;
    tx_string[680]=0;
    printf("TX = %s\n", tx_string);
    printf("RX = %s\n", rx_string);
    return status;
}

int SMB_t::spi_write_pattern_nb(alt_u32 spi_slave, alt_u16 nBytes, alt_u8 byteValue) {
    char tx_string[681];
    char rx_string[681];
    int result_i=0;
	int status=0;
	uint16_t rx_pre=0xff00;
    uint16_t nb=nBytes;
    do{
        nb--;
//do spi transaction, one byte at a time
        alt_u8 rx = 0xCC;
        alt_u8 tx = byteValue;

        alt_avalon_spi_command(SPI_BASE, spi_slave, 1, &tx, 0, &rx, nb==0?0:ALT_AVALON_SPI_COMMAND_MERGE);
        rx = IORD_8DIRECT(SPI_BASE, 0);

        char result_hex[3];
        char tx_hex[3];
        sprintf(result_hex,"%2.2X",rx);
        sprintf(tx_hex,"%2.2X",tx);
        rx_string[result_i] = result_hex[0];
        tx_string[result_i] = tx_hex[0];
        result_i++;
        rx_string[result_i] = result_hex[1];
        tx_string[result_i] = tx_hex[1];
        result_i++;

    }while(nb>0);
    printf("\n");
    rx_string[result_i]=0;
    tx_string[result_i]=0;
    printf("TX = %s\n", tx_string);
    printf("RX = %s\n", rx_string);
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
    ret = spi_write_pattern(asic, bitpattern);
    usleep(5e5);
    ret = spi_write_pattern(asic, bitpattern);

    if(ret != 0) {
        printf("[scifi] Configuration error\n");
        return FEB_REPLY_ERROR;
    }

    return FEB_REPLY_SUCCESS;
}

alt_u16 SMB_t::configure_asic_nb(alt_u32 asic, alt_u16 nBytes, alt_u8 byteValue) {
    printf("[SMB] chip_configure(%u)\n", asic);
    if (nBytes>340) {
        printf("nbytes must be <= 340");
        return FEB_REPLY_ERROR;
    }

    int ret;
    ret = spi_write_pattern_nb(asic, nBytes, byteValue);

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

alt_u16 SMB_t::sc_callback_nb(alt_u16 cmd, alt_u16 nBytes, alt_u8 byteValue) {
    if((cmd & 0xFFF0) == CMD_MUTRIG_ASIC_CFG) {
        printf("configuring ASIC\n");
        int asic = cmd & 0x000F;
        configure_asic_nb(asic, nBytes, byteValue);
    }
    else {
        printf("[sc_callback] unknown command\n");
    }
    return 0;
}

void SMB_t::menu_SMB_main() {
    auto& regs = sc.ram->regs.SMB;
    volatile sc_ram_t* ram = (sc_ram_t*) AVM_SC_BASE;
    ram->data[0xFF4C] = 0x00;

    while(1) {
        //        TODO: Define menu
//        printf("  [0] => reset\n");
        printf("  [0] => Write ALL_OFF config ASIC\n");
        printf("  [1] => Config loop\n");
        printf("  [2] => Send 0xAC 0xAB\n");
        printf("  [3] => send all 0\n");
        printf("  [4] => Send Test config\n");
        printf("  [5] => data\n");
        printf("  [6] => monitor test\n");

        printf("  [u] => invert clk\n");
        printf("  [j] => uninvert clk\n");

        printf("  [i] => invert miso\n");
        printf("  [k] => uninvert miso\n");

        printf("  [o] => invert mosi\n");
        printf("  [l] => uninvert mosi\n");

        printf("  [c] => chip select\n");
        printf("  [s] => control reg SSO\n");
        printf("  [h] => Write ALL_OFF_high_end config ASIC\n");

        printf("  [t] => more test patterns\n");
        printf("  [b] => bytes\n");


        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        printf("%c\n", cmd);
        switch(cmd) {
        case '0':
            sc_callback(0x0110, (alt_u32*) config_ALL_OFF, 0);
            sc_callback(0x0111, (alt_u32*) config_ALL_OFF, 0);
            break;

        case '1':
            //sc_callback(0x0110, (alt_u32*) config_ALL_OFF, 0);
            //sc_callback(0x0111, (alt_u32*) config_ALL_OFF, 0);
            while(1) {
                //printf("114\n");
                //usleep(5e5);
                //sc_callback(0x0114, (alt_u32*) config_ALL_OFF, 0);
                printf("110\n");
                usleep(2e6);
                //sc_callback(0x0110, (alt_u32*) test_config, 0);
                printf("111\n");
                usleep(2e6);
                //sc_callback(0x0111, (alt_u32*) test_config, 0);
                //printf("112\n");
                //usleep(5e5);
                //sc_callback(0x0112, (alt_u32*) config_ALL_OFF, 0);
                //printf("113\n");
                //usleep(5e5);
                //sc_callback(0x0113, (alt_u32*) config_ALL_OFF, 0);
            }
            //sc_callback(0x0113, (alt_u32*) config_ALL_OFF, 0);
            break;
        case 'u':
            ram->data[0xFF4C] |= 0x1;
            break;
        case 'j':
            ram->data[0xFF4C] &= 0x6;
            break;
        case 'i':
            ram->data[0xFF4C] |= 0x2;
            break;
        case 'k':
            ram->data[0xFF4C] &= 0x5;
            break;
        case 'o':
            ram->data[0xFF4C] |= 0x4;
            break;
        case 'l':
            ram->data[0xFF4C] &= 0x3;
            break;
        case '2':
            //sc_callback(0x0111, (alt_u32*) acab, 0);
            break;
        case '3':
            //sc_callback(0x0111, (alt_u32*) zeroes, 0);
            break;
        case '4':
            //sc_callback(0x0111, (alt_u32*) test_config, 0);
            break;
        case '5':
            printf("buffer_full / frame_desync / rx_pll_lock : 0x%03X\n", regs.mon.status);
            printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\n", regs.mon.rx_dpa_lock, regs.mon.rx_ready);
            break;
        case '6':
            menu_SMB_monitors();
            break;
        case 'c':
            {
            char chipselect = wait_key();
            IOWR(SPI_BASE, 5, chipselect - '0');
            printf("chip select: %04x\n", chipselect - '0');
            break;
            }
        case 's':
            {
                char sso_status = wait_key();
                switch(sso_status) {
                    case '0':
                        IOWR(SPI_BASE, 3, IORD(SPI_BASE, 3) & 0xfbff);
                        break;
                    case '1':
                        IOWR(SPI_BASE, 3, IORD(SPI_BASE, 3) | 0x400);
                        break;
                    default:
                        printf("=> nothing done");
                }
                break;
            }
        case 'h':
            //sc_callback(0x0111, (alt_u32*) config_ALL_OFF_high_end, 0);
            break;
        case 't':
            {
                printf("0=beginning, 1=end, 2=middle\n");
                char pattern = wait_key();
                switch(pattern) {
                    case '0':
                        sc_callback(0x0111, (alt_u32*) beginning, 0);
                        break;
                    case '1':
                        sc_callback(0x0111, (alt_u32*) end, 0);
                        break;
                    case '2':
                        sc_callback(0x0111, (alt_u32*) middle, 0);
                        break;
                    case 'l':
                        {
                            for(int i=0;i<5;i++) {
                                sc_callback(0x0110, (alt_u32*) config_ALL_OFF, 0);
                            }
                            usleep(1000000);
                            sc_callback_nb(0x0110, 340, 167);
                            usleep(1000000);
                            sc_callback_nb(0x0110, 340, 171);
                            usleep(1000000);
                            sc_callback_nb(0x0110, 340, 165);

                        }
                    default:
                        printf("=> nothing done");
                }
                break;
            }
        case 'b':
            {
                while(1) {
                    char numBytesDecimal[5];
                    int numBytesDecimal_p=0;
                    alt_u16 nBytes = 0;
                    printf("number of bytes (1-340+enter, f=340, h=170, s=50, q to quit):");
                    char key = wait_key();

                    if (key == 'q') break;
                    if (key == 'f') nBytes=340;
                    if (key == 'h') nBytes=170;
                    if (key == 's') nBytes=100;
                    if (key == 'l') nBytes=50;
                    while (key-'0' >=0 && key-'0'<= 9 && numBytesDecimal_p<3) {
                        numBytesDecimal[numBytesDecimal_p++] = key;
                        key = wait_key();
                    }
                    numBytesDecimal[numBytesDecimal_p] = 0;
                    if (nBytes < 1)
                        nBytes = strtoul(numBytesDecimal, 0 , 10);

                    if (nBytes < 1) {
                        printf("=> nBytes < 1, abort.\n");
                        break;
                    }

                    printf("\nbyte value (hex, e.g. 00, 01, .. FF):");
                    char byteValueHex[3];
                    byteValueHex[0] = wait_key();
                    byteValueHex[1] = wait_key();
                    byteValueHex[2] = 0;
                    alt_u16 byteValue = strtoul(byteValueHex, 0, 16);
                    printf("\nwrite %d times byte %d\n", nBytes, byteValue);
                    sc_callback_nb(0x0110, nBytes, byteValue);
                }
            }
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
