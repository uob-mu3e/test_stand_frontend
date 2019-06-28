
#include "base.h"

alt_u32 alarm_callback(void*) {
    // watchdog
    IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0xFF);
    IOWR_ALTERA_AVALON_PIO_SET_BITS(PIO_BASE, alt_nticks() & 0xFF);

    return 10;
}

int main() {
    uart_init();

    alt_alarm alarm;
    int err = alt_alarm_start(&alarm, 10, alarm_callback, nullptr);
    if(err) {
        printf("ERROR: alt_alarm_start => %d\n%d\n", err);
    }

    while (1) {
        printf("'%s' example\n", ALT_DEVICE_FAMILY);
        printf("  [1] => menu 1\n");
        printf("  [2] => menu 2\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            menu_1();
            break;
        case '2':
            menu_2();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
