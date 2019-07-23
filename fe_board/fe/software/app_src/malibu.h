
#include "malibu/malibu_basic_cmd.h"
malibu_t malibu;

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
            malibu.powerup();
            break;
        case '2':
            malibu.powerdown();
            break;
        case '3':
            malibu.PowerUpASIC(0);
            break;
        case '4':
            malibu.SPI_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
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
