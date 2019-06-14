
#include "../include/base.h"

#include "../include/i2c.h"
i2c_t i2c;

#include "../include/a10/fan.h"

#include "../include/xcvr.h"

alt_u32 alarm_callback(void*) {
    IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0xFF);
    // watchdog
    IOWR_ALTERA_AVALON_PIO_SET_BITS(PIO_BASE, alt_nticks() & 0xFF);

    return 10;
}

#include "../include/reconfig.h"

int main() {
    uart_init();

    fan_t fan;
    fan.init();

    alt_alarm alarm;
    int err = alt_alarm_start(&alarm, 10, alarm_callback, nullptr);
    if(err) {
        printf("ERROR: alt_alarm_start => %d\n%d\n", err);
    }

    reconfig_t reconfig;

    while (1) {
        printf("'%s' A10\n", ALT_DEVICE_FAMILY);
        printf("  [1] => xcvr qsfp\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            menu_xcvr((alt_u32*)(AVM_QSFP_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case 'r':
            reconfig.pll();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
