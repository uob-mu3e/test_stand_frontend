
#include "include/base.h"

#include "include/xcvr.h"
#include "include/a10/reconfig.h"

struct my_xcvr_t : xcvr_block_t {
    reconfig_t reconfig;
    my_xcvr_t() : xcvr_block_t((alt_u32*)(AVM_XCVR1_BASE | ALT_CPU_DCACHE_BYPASS_MASK)) {
    }
};
my_xcvr_t pod;

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
        printf("  [1] => ...\n");
        printf("  [p] => POD\n");
        printf("  [s] => SFP\n");
        printf("  [c] => clocks\n");
        printf("  [R] => reconfig\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            break;
        case 'p':
            pod.menu();
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
            pod.reconfig.pll(AVM_XCVR1_BASE + 0x00000000);
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
