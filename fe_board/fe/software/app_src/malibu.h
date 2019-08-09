
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
            printf("fifo_status / frame_desync : 0x%02X / 0x%02X\n", sc->regs.malibu.fifo.status, sc->regs.malibu.frame.desync);
            printf("rx_pll_lock / rx_dpa_lock / rx_ready : 0x%01X / 0x%04X / 0x%04X\n", sc->regs.malibu.rx.pll_lock, sc->regs.malibu.rx.dpa_lock, sc->regs.malibu.rx.ready);
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
