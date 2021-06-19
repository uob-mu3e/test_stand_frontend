
#include "include/base.h"

#include "include/xcvr.h"
#include "include/a10/reconfig.h"

#include "include/i2c.h"
#include "include/si534x.h"

i2c_t i2c;

#include "PCIe40.h"

alt_u32 xcvr0_rx_p[] = {
    // default
//     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
//    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    // schematic
    5,  6,  4,  7,  3,  8,  0,  9,  2, 10,  1, 11,
    17, 18, 16, 20, 15, 19, 14, 21, 13, 22, 12, 23
};
alt_u32 xcvr1_rx_p[] = {
    // default
//     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
//    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
    // schematic
    29-24, 30-24, 28-24, 31-24, 27-24, 32-24, 26-24, 33-24, 25-24, 34-24, 24-24, 35-24,
    36-24, 42-24, 37-24, 43-24, 38-24, 44-24, 39-24, 45-24, 40-24, 46-24, 41-24, 47-24
};

int main() {
    base_init();

    PCIe40::init();

    while (1) {
        printf("\n");
        printf("[PCIe40] -------- menu --------\n");

        printf("\n");
        printf("  [0] => ...\n");
        printf("  [1] => XCVR 156\n");
        printf("  [2] => XCVR 250\n");
        printf("  [s] => SFP\n");
        printf("  [c] => clocks\n");
        printf("  [R] => reconfig\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            break;
        case '1': {
            xcvr_block_t xcvr_block((alt_u32*)(AVM_XCVR0_BASE | ALT_CPU_DCACHE_BYPASS_MASK), AVM_XCVR0_SPAN);
            xcvr_block.rx_p = xcvr0_rx_p;
            xcvr_block.n = 24;
            xcvr_block.menu();
            break;
        }
        case '2': {
            xcvr_block_t xcvr_block((alt_u32*)(AVM_XCVR1_BASE | ALT_CPU_DCACHE_BYPASS_MASK), AVM_XCVR1_SPAN);
            xcvr_block.rx_p = xcvr1_rx_p;
            xcvr_block.n = 24;
            xcvr_block.menu();
            break;
        }
        case 's': {
            xcvr_block_t sfp((alt_u32*)(AVM_SFP_BASE | ALT_CPU_DCACHE_BYPASS_MASK), AVM_SFP_SPAN);
            sfp.menu();
            break;
        }
        case 'c':
            PCIe40::menu();
            break;
        case 'R': {
            reconfig_t reconfig;
            for(int i = 0; i < AVM_XCVR0_SPAN; i += xcvr_block_t::XCVR_SPAN) {
                reconfig.pll(AVM_XCVR0_BASE + i);
            }
            for(int i = 0; i < AVM_XCVR1_SPAN; i += xcvr_block_t::XCVR_SPAN) {
                reconfig.pll(AVM_XCVR1_BASE + i);
            }
            break;
        }
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
