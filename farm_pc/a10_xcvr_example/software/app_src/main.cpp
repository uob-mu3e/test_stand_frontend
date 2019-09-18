
#include "../include/base.h"

#include "../include/a10/flash.h"
flash_t flash;

#include "../include/a10/fan.h"
fan_t fan;

#include "../include/xcvr.h"
#include "../include/a10/reconfig.h"
reconfig_t reconfig;

int main() {
    base_init();

    fan.init();

    flash.init();

    while (1) {
        printf("  [1] => flash\n");
        printf("  [2] => xcvr qsfp\n");
        printf("  [8] => fan\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            flash.menu();
            break;
        case '2':
            menu_xcvr((alt_u32*)(AVM_QSFP_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case '8':
            fan.menu();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
