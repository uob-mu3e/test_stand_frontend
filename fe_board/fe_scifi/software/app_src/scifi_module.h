#ifndef SCIFI_MODULE_H_
#define SCIFI_MODULE_H_

struct scifi_module_t {
    const uint32_t MUTRIG1_CONFIG_LEN_BYTES=10;
    const uint32_t MUTRIG1_CONFIG_LEN_BITS =80;
    const uint8_t  n_ASICS=4;
    //write single byte over spi
    static alt_u8 spi_write(alt_u32 slave, alt_u8 w);
    //write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
    int spi_write_pattern(alt_u32 slave, const alt_u8* bitpattern);

    void powerup() {
        printf("[scifi] powerup: not implemented\n");
    }

    void powerdown() {
        printf("[scifi] powerdown: not implemented\n");
    }

    int configure_asic(int asic, const alt_u8* bitpattern);
};

//Slow control pattern for stic3, pattern length and alloff configuration
//#include "ALL_OFF.h"
//#include "PLL_TEST_ch0to6_noGenIDLE.h"


    //write single byte over spi
alt_u8 scifi_module_t::spi_write(alt_u32 slave, alt_u8 w) {
        alt_u8 r = 0xCC;
//        printf("spi_write: 0x%02X\n", w);
        alt_avalon_spi_command(SPI_BASE, slave, 1, &w, 0, &r, 0);
        r = IORD_8DIRECT(SPI_BASE, 0);
//        printf("spi_read: 0x%02X\n", r);
        return r;
    }


//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int scifi_module_t::spi_write_pattern(alt_u32 spi_slave, const alt_u8* bitpattern) {
	int status=0;
	uint16_t rx_pre=0xff00;
	for(int nb=MUTRIG1_CONFIG_LEN_BYTES-1; nb>=0; nb--){
		uint8_t rx = scifi_module_t::spi_write(spi_slave, bitpattern[nb]);
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
int scifi_module_t::configure_asic(int asic, const alt_u8* bitpattern) {
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

#endif
