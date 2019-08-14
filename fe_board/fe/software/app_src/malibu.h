
#include "malibu/malibu_basic_cmd.h"
malibu_t malibu;

void menu_malibu() {
    volatile alt_u32* avm = (alt_u32*)AVM_BASE;

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
        case '1':
            malibu.powerup();
            break;
        case '2':
            malibu.powerdown();
            break;
        case '3':
            malibu.stic_configure(0, stic3_config_ALL_OFF);
            break;
        case '4':
            malibu.stic_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
            break;
        case 's':
            printf("buffer_full/frame_desync/rx_pll_lock: 0x%03X\n", avm[0x8]);
            printf("rx_dpa_lock: 0x%08X\n", avm[0x9]);
            printf("rx_ready: 0x%08X\n", avm[0xA]);
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
