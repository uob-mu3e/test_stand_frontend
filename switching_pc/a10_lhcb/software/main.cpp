
#include "include/base.h"

#include "include/xcvr.h"
#include "include/a10/reconfig.h"

struct my_xcvr0_t : xcvr_block_t {
    reconfig_t reconfig;
    my_xcvr0_t() : xcvr_block_t((alt_u32*)(AVM_XCVR0_BASE | ALT_CPU_DCACHE_BYPASS_MASK)) {
    }
};
my_xcvr0_t xcvr0;

struct my_xcvr1_t : xcvr_block_t {
    reconfig_t reconfig;
    my_xcvr1_t() : xcvr_block_t((alt_u32*)(AVM_XCVR1_BASE | ALT_CPU_DCACHE_BYPASS_MASK)) {
    }
};
my_xcvr1_t xcvr1;

#include "include/i2c.h"
#include "include/si534x.h"

i2c_t i2c;

#include "PCIe40.h"

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
        case '1':
            xcvr0.menu();
            break;
        case '2':
            xcvr1.menu();
            break;
        case 's': {
            xcvr_block_t sfp((alt_u32*)(AVM_SFP_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            sfp.menu();
            break;
        }
        case 'c':
            PCIe40::menu();
            break;
        case 'R':
            for(int i = 0; i < AVM_XCVR0_SPAN; i += 0x10000) {
                pod.reconfig.pll(AVM_XCVR0_BASE + i);
            }
            for(int i = 0; i < AVM_XCVR1_SPAN; i += 0x10000) {
                pod.reconfig.pll(AVM_XCVR1_BASE + i);
            }
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
