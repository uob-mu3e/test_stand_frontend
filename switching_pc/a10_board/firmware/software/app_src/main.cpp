
#include "../include/base.h"

#include "../include/a10/flash.h"
flash_t flash;

#include "../include/a10/fan.h"
fan_t fan;

#include "../include/xcvr.h"
#include "../include/a10/reconfig.h"
reconfig_t reconfig;

#include "../include/si.h"
si_t si { SPI_BASE, 0 }; // spi_slave = 0

#include "in0_125_in1_125_out0_125_out1_125_out2_125_out3_125.h"

void menu_spi_si5345() {
    while (1) {
        printf("SI5345\n");
        printf("  [i] => init\n");
        printf("  [r] => read page 0\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case 'i':
            si.init(si5344_revd_registers, sizeof(si5344_revd_registers) / sizeof(si5344_revd_registers[0]));
            break;
        case 'r' :
            for(alt_u16 address = 0x0000; address < 0x0100; address++) {
                printf("  [0x%02X] = 0x%02X\n", address, si.read(address));
            }
            break;
        case '?':
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

int main() {
    base_init();

    fan.init();

    flash.init();

    while (1) {
        printf("  [1] => flash\n");
        printf("  [2] => xcvr qsfp[A]\n");
        printf("  [3] => xcvr qsfp[B]\n");
        printf("  [4] => xcvr qsfp[C]\n");
        printf("  [5] => xcvr qsfp[D]\n");
        printf("  [8] => fan\n");
        printf("  [0] => spi si chip\n");
        printf("  [r] => reconfig pll\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            flash.menu();
            break;
        case '2':
            menu_xcvr((alt_u32*)((AVM_QSFP_BASE + 0x00000) | ALT_CPU_DCACHE_BYPASS_MASK), 'A');
            break;
        case '3':
            menu_xcvr((alt_u32*)((AVM_QSFP_BASE + 0x10000) | ALT_CPU_DCACHE_BYPASS_MASK), 'B');
            break;
        case '4':
            menu_xcvr((alt_u32*)((AVM_QSFP_BASE + 0x20000) | ALT_CPU_DCACHE_BYPASS_MASK), 'C');
            break;
        case '5':
            menu_xcvr((alt_u32*)((AVM_QSFP_BASE + 0x30000) | ALT_CPU_DCACHE_BYPASS_MASK), 'D');
            break;
        case '8':
            fan.menu();
            break;
        case '0':
            printf("spi:\n");
            menu_spi_si5345();
            break;
        case 'r':
            reconfig.pll(AVM_QSFP_BASE + 0x00000);
            reconfig.pll(AVM_QSFP_BASE + 0x10000);
            reconfig.pll(AVM_QSFP_BASE + 0x20000);
            reconfig.pll(AVM_QSFP_BASE + 0x30000);
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
