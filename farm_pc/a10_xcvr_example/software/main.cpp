
#include "include/base.h"

#include "include/a10/flash.h"
flash_t flash;

#include "include/a10/fan.h"
fan_t fan(0x01);

#include "include/xcvr.h"
#include "include/a10/reconfig.h"
reconfig_t reconfig;

int main() {
    base_init();

    fan.init();

    flash.init();

    while (1) {
        printf("  [1] => xcvr\n");
        printf("  [R] => reconfig");
        printf("  [9] => flash\n");
        printf("  [0] => fan\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            menu_xcvr((alt_u32*)(AVM_XCVR0_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case 'R':
            for(alt_u32 base = AVM_XCVR0_BASE; base < AVM_XCVR0_BASE + AVM_XCVR0_SPAN; base += 0x10000) {
                if(*(alt_u32*)base == 0xCCCCCCCC) continue;
                reconfig.pll(base);
            }
            break;
        case '9':
            flash.menu();
            break;
        case '0':
            fan.menu();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
