/*
Malibu board init procedure:
- Set clock and test inputs
- Enable 3.3V supplies
- Disable 1.8 supplies for all ASICs 
- Set CS lines for all ASICs
-* Power monitor chips not initialized
-* I2C multiplexers not initialized

GPIO[0]: Init
        All 3.3V power off
        SEL_sysCLK->CK_FPGA0 (Mainboard)
        SEL_pllCLK->CK_SI1 (Mainboard)
        SEL_pllTEST->PLL_TEST (Mainboard)
        PLLtest disabled
        Configure GPIO, all output
Write Reg1 00001100 = 0x0C
Write Reg3 0x00

GPIO[1..7]: Init ASICs
        All CS high, All 1.8V power off
        All output
Write Reg1 11001100 = 0xCC
Write Reg3 0x00
*/

struct data_t { uint8_t dev; uint8_t reg; uint8_t data; };
data_t malibu_init_regs[]={
	{0x38,0x01,0x0C^0x38},
	{0x38,0x03,0x00},
	{0x38,0x01,0x0D^0x38},
	{0xff,0x00,0x00},
	{0x38,0x01,0x0F^0x38},
	{0xff,0x00,0x00},
	{0x39,0x01,0x3C},
	{0x39,0x03,0x00},
	{0x3a,0x01,0x3C},
	{0x3a,0x03,0x00},
	{0x3b,0x01,0x3C},
	{0x3b,0x03,0x00},
	{0x3c,0x01,0x3C},
	{0x3c,0x03,0x00},
	{0x3d,0x01,0x3C},
	{0x3d,0x03,0x00},
	{0x3f,0x01,0x3C},
	{0x3f,0x03,0x00}
};

/*
Power down cycle for malibu board
- Power down each ASIC (both 1.8V supplies at the same time)
- Power down 3.3V supplies
*/
data_t malibu_powerdown_regs[]={
	{0x3f,0x01,0x3C},
	{0xff,0x00,0x00},
	{0x3e,0x01,0x3C},
	{0xff,0x00,0x00},
	{0x3d,0x01,0x3C},
	{0xff,0x00,0x00},
	{0x3c,0x01,0x3C},
	{0xff,0x00,0x00},
	{0x3b,0x01,0x3C},
	{0xff,0x00,0x00},
	{0x3a,0x01,0x3C},
	{0xff,0x00,0x00},
	{0x39,0x01,0x3C},
	{0xff,0x00,0x00},
	{0x38,0x01,0x0D},
	{0xff,0x00,0x00},
	{0x38,0x01,0x0C}
};

void Malibu_Powerup() {
    printf("MALIBU Power up\n");

    for(int i = 0; i < sizeof(malibu_init_regs) / sizeof(malibu_init_regs[0]); i++) {
        auto& v = malibu_init_regs[i];
        if(v.dev == 0xFF) {
            printf("  sleep\n");
            usleep(5000);
            continue;
        }
        I2C_write(v.dev, v.reg, v.data);
    }

    printf("DONE\n");
}

void Malibu_Powerdown() {
    printf("MALIBU Power down\n");

    for(int i = 0; i < sizeof(malibu_powerdown_regs) / sizeof(malibu_powerdown_regs[0]); i++) {
        auto& v = malibu_powerdown_regs[i];
        if(v.dev == 0xFF) {
            printf("  sleep\n");
            usleep(5000);
            continue;
        }
        I2C_write(v.dev, v.reg, v.data);
    }

    printf("DONE\n");
}

//Slow control pattern for stic3, pattern length and alloff configuration
#include "ALL_OFF.h"
#include "PLL_TEST_ch0to6_noGenIDLE.h"


//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int SPI_write_pattern(const unsigned char* bitpattern){
	int status=0;
	uint16_t rx_pre=0xff00;
	for(int nb=STIC3_CONFIG_LEN_BYTES-1; nb>=0; nb--){
		unsigned char rx=spi_write(bitpattern[nb]);
		//pattern is not in full units of bytes, so shift back while receiving to check the correct configuration state
		unsigned char rx_check= (rx_pre | rx ) >> (8-STIC3_CONFIG_LEN_BITS%8);
		if(nb==STIC3_CONFIG_LEN_BYTES-1){
			rx_check &= 0xff>>(8-STIC3_CONFIG_LEN_BITS%8);
		};

		if(rx_check!=stic3_config_ALL_OFF[nb]){
//			printf("Error in byte %d: received %2.2x expected %2.2x\n",nb,rx_check,bitpattern[nb]);
			status=-1;
		}
		rx_pre=rx<<8;
	}
	return status;
}

//configure a specific ASIC returns 0 if configuration is correct, -1 otherwise.
int SPI_configure(unsigned char n, const unsigned char* bitpattern){
	//pull low CS line of the given ASIC
	char gpio_value=I2C_read(0x39+n/2,0x01);
	gpio_value ^= 1<<(2+n%2*4);
	I2C_write(0x39+n/2,0x01,gpio_value);

	//configure SPI. Note: pattern is not in full bytes, so validation gets a bit more complicated. Shifting out all bytes, and need to realign after.
	//This is to be done still
	int ret;
	ret=SPI_write_pattern(bitpattern);
	ret=SPI_write_pattern(bitpattern);

	//pull high CS line of the given ASIC
	gpio_value ^= 1<<(2+n%2*4);
	I2C_write(0x39+n/2,0x01,gpio_value);
	return ret;
}


/*
Power up cycle for single ASIC
- Power digital 1.8V domain
- Configure ALL_OFF pattern two times
- Validate read back configuration
- If correct: Power up 1.8V analog domain
- else power down 1.8V digital domain
*/


int PowerUpASIC(unsigned char n){
	printf("Powering up ASIC %u\n",n);
	char gpio_value=I2C_read(0x39+n/2,0x01);

	//enable 1.8V digital
	gpio_value |= 1<<(1+n%2*4);
	I2C_write(0x39+n/2,0x01,gpio_value);
	int ret;
	ret=SPI_configure(n,stic3_config_ALL_OFF);
	ret=SPI_configure(n,stic3_config_ALL_OFF);
	if(ret!=0){ //configuration error, switch off again
		printf("Configuration mismatch, powering off again\n");
		gpio_value ^= 1<<(1+n%2*4);
		I2C_write(0x39+n/2,0x01,gpio_value);
		return -1;	
	}
	//enable 1.8V analog
	gpio_value |= 1<<(0+n%2*4);
	I2C_write(0x39+n/2,0x01,gpio_value);
	printf("... Done\n");
	return 0;
}
