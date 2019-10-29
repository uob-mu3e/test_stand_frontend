#ifndef MALIBU_BASIC_CMD_H_
#define MALIBU_BASIC_CMD_H_

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

#include "../../../../../common/firmware/include/i2c.h"

struct malibu_t {

    i2c_t i2c;

    alt_u8 I2C_read(alt_u8 slave, alt_u8 addr) {
        alt_u8 data = i2c.get(slave, addr);
        printf("i2c_read: 0x%02X[0x%02X] is 0x%02X\n", slave, addr, data);
        return data;
    }

    void I2C_write(alt_u8 slave, alt_u8 addr, alt_u8 data) {
        printf("i2c_write: 0x%02X[0x%02X] <= 0x%02X\n", slave, addr, data);
        i2c.set(slave, addr, data);
    }

    struct i2c_reg_t {
        alt_u8 slave;
        alt_u8 addr;
        alt_u8 data;
    };

    static
    alt_u8 spi_write(alt_u32 slave, alt_u8 w) {
        alt_u8 r = 0xCC;
//        printf("spi_write: 0x%02X\n", w);
        alt_avalon_spi_command(SPI_BASE, slave, 1, &w, 0, &r, 0);
        r = IORD_8DIRECT(SPI_BASE, 0);
//        printf("spi_read: 0x%02X\n", r);
        return r;
    }

    i2c_reg_t malibu_init_regs[18] = {
	{0x38,0x01,0x0C^0x20},
	{0x38,0x03,0x00},
	{0x38,0x01,0x0D^0x20},
	{0xff,0x00,0x00},
	{0x38,0x01,0x0F^0x20},
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

    /**
     * Power down cycle:
     * - Power down each ASIC (both 1.8V supplies at the same time)
     * - Power down 3.3V supplies
    */
    i2c_reg_t malibu_powerdown_regs[17] = {
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
	//send u32 to I2C
    void i2c_write_u32(alt_u32* data_u32, int n) {
		for(int i = 0; i < n; i++) {
			 i2c_write_regs(&(u32_to_i2c_reg(data_u32[i])),1);
		}
	}

    void i2c_write_regs(const i2c_reg_t* regs, int n) {
        for(int i = 0; i < n; i++) {
            auto& reg = regs[i];
            if(reg.slave == 0xFF) {
                usleep(1000);
                continue;
            }
            I2C_write(reg.slave, reg.addr, reg.data);
        }
    }

    void powerup() {
        printf("[malibu] powerup\n");
        i2c_write_regs(malibu_init_regs, sizeof(malibu_init_regs) / sizeof(malibu_init_regs[0]));
        printf("[malibu] powerup DONE\n");
    }

    void powerdown() {
        printf("[malibu] powerdown\n");
        i2c_write_regs(malibu_powerdown_regs, sizeof(malibu_powerdown_regs) / sizeof(malibu_powerdown_regs[0]));
        printf("[malibu] powerdown DONE\n");
    }

    int stic_configure(int asic, const alt_u8* bitpattern);

	i2c_reg_t u32_to_i2c_reg(alt_u32 data_u32){ // save the data to i2c_reg_t struct
		i2c_reg_t i2c_reg = {data_u32 & 0xFF0000, data_u32 & 0xFF00, data_u32 & 0xFF};
		return i2c_reg;
	}

};

//Slow control pattern for stic3, pattern length and alloff configuration
#include "ALL_OFF.h"
#include "PLL_TEST_ch0to6_noGenIDLE.h"

//write slow control pattern over SPI, returns 0 if readback value matches written, otherwise -1. Does not include CSn line switching.
int SPI_write_pattern(uint32_t spi_slave, const alt_u8* bitpattern) {
	int status=0;
	uint16_t rx_pre=0xff00;
	for(int nb=STIC3_CONFIG_LEN_BYTES-1; nb>=0; nb--){
		uint8_t rx = malibu_t::spi_write(spi_slave, bitpattern[nb]);
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
int SPI_configure(uint32_t slaveAddr, const unsigned char* bitpattern) {
	//configure SPI. Note: pattern is not in full bytes, so validation gets a bit more complicated. Shifting out all bytes, and need to realign after.
	//This is to be done still
	int ret;
	ret=SPI_write_pattern(slaveAddr, bitpattern);
	ret=SPI_write_pattern(slaveAddr, bitpattern);

	return ret;
}

/**
 * Configure ASIC
 *
 * - powerup digital 1.8V
 * - configure pattern
 * - configure and validate
 * - if not ok, then powerdown digital and exit
 * - powerup analog 1.8V
 */
int malibu_t::stic_configure(int asic, const alt_u8* bitpattern) {
    printf("[malibu] stic_configure(%u)\n", asic);

    alt_u8 i2c_slave = 0x39 + asic/2;
    alt_u8 A_bit = 1 << (0 + asic%2*4);
    alt_u8 D_bit = 1 << (1 + asic%2*4);
    alt_u8 spi_slave = 1;
    alt_u8 CS_bit = 1 << (2 + asic%2*4);

    // enable 1.8V digital
    I2C_write(i2c_slave, 0x01, I2C_read(i2c_slave, 0x01) | D_bit);
    int ret;

    I2C_write(i2c_slave, 0x01, I2C_read(i2c_slave, 0x01) & ~CS_bit);
    ret = SPI_write_pattern(spi_slave, bitpattern);
    I2C_write(i2c_slave, 0x01, I2C_read(i2c_slave, 0x01) | CS_bit);

    I2C_write(i2c_slave, 0x01, I2C_read(i2c_slave, 0x01) & ~CS_bit);
    ret = SPI_write_pattern(spi_slave, bitpattern);
    I2C_write(i2c_slave, 0x01, I2C_read(i2c_slave, 0x01) | CS_bit);

    if(ret != 0) {
        printf("Configuration error, powering off again\n");
        I2C_write(i2c_slave, 0x01, I2C_read(i2c_slave, 0x01) & ~D_bit);
        return -1;
    }

    I2C_write(i2c_slave, 0x01, I2C_read(i2c_slave, 0x01) | A_bit);

    printf("[malibu] stic_configure DONE\n");
    return 0;
}
#endif
