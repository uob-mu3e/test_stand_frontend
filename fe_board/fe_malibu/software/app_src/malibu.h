
#include "../include/i2c.h"
i2c_t i2c;

alt_u8 I2C_read(alt_u8 dev, alt_u8 reg) {
    alt_u8 data = i2c.get(dev, reg);
    printf("i2c_read: 0x%02X[0x%02X] is 0x%02X\n", dev, reg, data);
    return data;
}

void I2C_write(alt_u8 dev, alt_u8 reg, alt_u8 data) {
    printf("i2c_write: 0x%02X[0x%02X] <= 0x%02X\n", dev, reg, data);
    i2c.set(dev, reg, data);
}

alt_u8 spi_write(alt_u8 w) {
    alt_u8 r = 0xCC;
//    printf("spi_write: 0x%02X\n", w);
    alt_avalon_spi_command(SPI_BASE, 1, 1, &w, 0, &r, 0);
    r = IORD_8DIRECT(SPI_BASE, 0);
//    printf("spi_read: 0x%02X\n", r);
    return r;
}

typedef alt_u8 uint8_t;
typedef alt_u16 uint16_t;

#include "malibu/malibu_basic_cmd.h"

void menu_malibu() {
    while(1) {
        printf("  [0] => reset\n");
        printf("  [1] => powerup MALIBU\n");
        printf("  [2] => powerdown MALIBU\n");
        printf("  [3] => powerup ASIC\n");
        printf("  [4] => stic3_config_PLL_TEST_ch0to6_noGenIDLE\n");
        printf("  [5] => data\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
//            IOWR_ALTERA_AVALON_PIO_SET_BITS(PIO_BASE, 0x00010000);
//            usleep(100);
//            IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0x00010000);
            break;
        case '1':
            Malibu_Powerup();
            break;
        case '2':
            Malibu_Powerdown();
            break;
        case '3':
            PowerUpASIC(0);
            break;
        case '4':
            SPI_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
            break;
        case 's':
            printf("buffer_full/frame_desync/rx_pll_lock: 0x%03X\n", ((volatile alt_u32*)AVM_TEST_BASE)[0x8]);
            printf("rx_dpa_lock: 0x%08X\n", ((volatile alt_u32*)AVM_TEST_BASE)[0x9]);
            printf("rx_ready: 0x%08X\n", ((volatile alt_u32*)AVM_TEST_BASE)[0xA]);
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
