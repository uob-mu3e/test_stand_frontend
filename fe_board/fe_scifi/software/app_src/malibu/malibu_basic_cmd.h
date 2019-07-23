#ifndef MALIBU_BASIC_CMD_H_
#define MALIBU_BASIC_CMD_H_

#include "ALL_OFF.h"
typedef alt_u8 uint8_t;
typedef alt_u16 uint16_t;
typedef alt_u32 uint32_t;
#include "alt_types.h"

#include "altera_avalon_spi_regs.h"
#include "altera_avalon_spi.h"

int SPI_write_pattern2(uint32_t slaveAddr,uint32_t* bitpattern){
	uint32_t rx_word=0xaaaa;
	uint8_t tx_data, rx_data;
	for(int n=0;n<4*74;n++){
		//write new data
		tx_data= 0xff & (bitpattern[n/4]>>(8*(3-n%4)));
		//printf("At %d: new tx_word[%d]=%8.8x --> data=%2.2x\n",n,n/4,bitpattern[n/4],tx_data);
                alt_avalon_spi_command(SPI_BASE, slaveAddr, 1, &tx_data, 0, NULL, (n==4*74-1)?0:ALT_AVALON_SPI_COMMAND_MERGE);
                rx_data= IORD_8DIRECT(SPI_BASE, 0);
		rx_word=(rx_word<<8) | rx_data;
		//printf("Done, rx data=%2.2x --> word[%d]=%8.8x\n",rx_data,n/4,rx_word);
		if(n%4==3) bitpattern[n/4]=rx_word;
	}
	return 0;
}

//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int SPI_write_pattern_old(uint32_t slaveAddr,const unsigned char* bitpattern){
	int status=0;
	uint16_t rx_pre=0xff00;
	for(int nb=STIC3_CONFIG_LEN_BYTES-1; nb>=0; nb--){
                uint8_t rx=0;
                alt_avalon_spi_command(SPI_BASE, slaveAddr, 1, &bitpattern[nb], 0, &rx, 0);
                rx = IORD_8DIRECT(SPI_BASE, 0);

		//pattern is not in full units of bytes, so shift back while receiving to check the correct configuration state
		unsigned char rx_check= (rx_pre | rx ) >> (8-STIC3_CONFIG_LEN_BITS%8);
		if(nb==STIC3_CONFIG_LEN_BYTES-1){
			rx_check &= 0xff>>(8-STIC3_CONFIG_LEN_BITS%8);
		};

		if(rx_check!=bitpattern[nb]){
//			printf("Error in byte %d: received %2.2x expected %2.2x\n",nb,rx_check,bitpattern[nb]);
			status=-1;
		}
		rx_pre=rx<<8;
	}
	return status;
}

//configure a specific ASIC returns 0 if configuration is correct, -1 otherwise.
int SPI_configure(uint32_t slaveAddr, const unsigned char* bitpattern){
	//configure SPI. Note: pattern is not in full bytes, so validation gets a bit more complicated. Shifting out all bytes, and need to realign after.
	//This is to be done still
	int ret;
	ret=SPI_write_pattern_old(slaveAddr,bitpattern);
	ret=SPI_write_pattern_old(slaveAddr,bitpattern);

	//pull high CS line of the given ASIC
	return ret;
}

#endif
