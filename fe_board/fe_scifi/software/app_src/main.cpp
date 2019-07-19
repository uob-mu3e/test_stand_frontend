
#include "../include/base.h"
#include "../include/xcvr.h"

#include "malibu.h"
#include "sc.h"

#include "../../../fe/software/app_src/si5345.h"
si5345_t si5345 { 0 };

alt_u32 alarm_callback(void*) {
    // watchdog
    IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0xFF);
    IOWR_ALTERA_AVALON_PIO_SET_BITS(PIO_BASE, alt_nticks() & 0xFF);

    sc_callback((alt_u32*)AVM_SC_BASE);

    return 10;
}

int main() {
    uart_init();

    printf("ALT_DEVICE_FAMILY = '%s'\n", ALT_DEVICE_FAMILY);
    printf("\n");

    si5345.init();

    alt_alarm alarm;
    int err = alt_alarm_start(&alarm, 0, alarm_callback, nullptr);
    if(err) {
        printf("ERROR: alt_alarm_start => %d\n%d\n", err);
    }

    while (1) {
        printf("\n");
        printf("FE_S4 (MALIBU):\n");
        printf("  [1] => xcvr qsfp\n");
        printf("  [2] => malibu\n");
        printf("  [3] => sc\n");
        printf("  [4] => xcvr pod\n");
        printf("  [5] => si5345\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            menu_xcvr((alt_u32*)(AVM_QSFP_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case '2':
            menu_malibu();
            break;
        case '3':
            menu_sc((alt_u32*)AVM_SC_BASE);
            break;
        case '4':
            menu_xcvr((alt_u32*)(AVM_POD_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case '5':
            si5345.menu();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
