
#include "malibu/malibu_basic_cmd.h"
malibu_t malibu;

void menu_malibu() {
    auto& regs = sc.ram->regs.malibu;

    while(1) {
        printf("  [0] => reset\n");
        printf("  [1] => powerup MALIBU\n");
        printf("  [2] => powerdown MALIBU\n");
        printf("  [3] => powerup ASIC\n");
        printf("  [4] => stic3_config_PLL_TEST_ch0to6_noGenIDLE\n");
        printf("  [5] => data\n");
        printf("  [6] => monitor test\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            malibu.power_TMB(true);
            break;
        case '2':
            malibu.power_TMB(false);
            break;
        case '3':
            malibu.power_ASIC(0);
            break;
        case '4':
            malibu.chip_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
            break;
        case '5':
            printf("buffer_full / frame_desync / rx_pll_lock : 0x%03X\n", regs.mon.status);
            printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\n", regs.mon.rx_dpa_lock, regs.mon.rx_ready);
            break;
        case '6':
            malibu.monitor_test_menu();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
