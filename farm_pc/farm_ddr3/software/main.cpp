
#include "include/base.h"

#include "include/a10/flash.h"
flash_t flash;

#include "include/a10/fan.h"
fan_t fan(0x01);

#include "include/xcvr.h"
#include "include/a10/reconfig.h"
reconfig_t reconfig;

#include "include/si.h"
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

alt_u32 xcvr0_rx_p[] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
};

alt_u32 xcvr1_rx_p[] = {
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
};

int main() {
    base_init();

    fan.init();

    flash.init();

    while (1) {
        printf("  [1] => xcvr0 - 6.25 GBit/s\n");
        printf("  [2] => xcvr1 - 10 GBit/s\n");
        printf("  [3] => flash\n");
        printf("  [4] => fan\n");
        printf("  [5] => spi si chip\n");
        printf("  [r] => reconfig pll\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1': {
            xcvr_block_t xcvr_block((alt_u32*)(AVM_XCVR0_BASE | ALT_CPU_DCACHE_BYPASS_MASK), AVM_XCVR0_SPAN);
            xcvr_block.rx_p = xcvr0_rx_p;
            xcvr_block.n = 16;
            xcvr_block.menu();
            break;
	}
        case '2': {
            xcvr_block_t xcvr_block((alt_u32*)(AVM_XCVR1_BASE | ALT_CPU_DCACHE_BYPASS_MASK), AVM_XCVR1_SPAN);
            xcvr_block.rx_p = xcvr1_rx_p;
            xcvr_block.n = 16;
            xcvr_block.menu();
            break;
	}
        case '3':
            flash.menu();
            break;
        case '4':
            fan.menu();
            break;
        case '5':
            menu_spi_si5345();
            break;
        case 'r':
//            for (alt_u32 base = AVM_XCVR0_BASE; base < AVM_XCVR0_BASE + AVM_XCVR0_SPAN; base += 0x10000) {
//                if(*(alt_u32*)base == 0xCCCCCCCC) continue;
//                reconfig.pll(base);                                                       
//            }
            for (alt_u32 base = AVM_XCVR1_BASE; base < AVM_XCVR1_BASE + AVM_XCVR1_SPAN; base += 0x10000) {
                if(*(alt_u32*)base == 0xCCCCCCCC) continue;
                reconfig.pll(base);
            }          
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
