
#include "../../../fe/software/app_src/malibu/malibu_basic_cmd.h"

void menu_scifi() {
    auto& regs = sc.ram->regs.scifi;

    while(1) {
        printf("  [0] => reset asic\n");
        printf("  [1] => reset datapath\n");
        printf("  [2] => configure all off\n");
        printf("  [3] => data\n");
        printf("  [4] => get datapath status\n");
        printf("  [5] => get slow control registers\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            regs.ctrl.reset = 1;
            usleep(100);
            regs.ctrl.reset = 0;
            break;
        case '1':
            regs.ctrl.reset = 2;
            usleep(100);
            regs.ctrl.reset = 0;
            break;
        case '2':
            SPI_configure(0, stic3_config_ALL_OFF);
            break;
        case '3':
            printf("TODO...\n");
            break;
        case '4':
            printf("rx_pll_lock / frame_desync / buffer_full : 0x%03X\n", regs.mon.status);
            printf("rx_dpa_lock / rx_ready : 0x%04X / 0x%04X\n", regs.mon.rx_dpa_lock, regs.mon.rx_ready);
            break;
        case '5':
            printf("dummyctrl_reg:    0x%08X\n", regs.ctrl.dummy);
            printf("    :datagen_en   0x%X\n", (regs.ctrl.dummy>>0)&1);
            printf("    :datagen_fast 0x%X\n", (regs.ctrl.dummy>>1)&1);
            printf("    :datagen_cnt  0x%X\n", (regs.ctrl.dummy>>2)&0x3ff);

            printf("dpctrl_reg:       0x%08X\n", regs.ctrl.dp);
            printf("    :mask         0b");
            for(int i=16;i>0;i--) printf("%d", (regs.ctrl.dp>>i)&1);
            printf("\n");

            printf("    :prbs_dec     0x%X\n", (regs.ctrl.dp>>31)&1);
            printf("subdet_reset_reg: 0x%08X\n", regs.ctrl.reset);
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
