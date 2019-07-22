#ifndef MALIBU_BASIC_CMD_H_
#define MALIBU_BASIC_CMD_H_

#include "ALL_OFF.h"
typedef alt_u8 uint8_t;
typedef alt_u16 uint16_t;
typedef alt_u32 uint32_t;
//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int SPI_write_pattern(uint32_t slaveAddr,const unsigned char* bitpattern){
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
	ret=SPI_write_pattern(slaveAddr,bitpattern);
	ret=SPI_write_pattern(slaveAddr,bitpattern);

	//pull high CS line of the given ASIC
	return ret;
}

#endif
