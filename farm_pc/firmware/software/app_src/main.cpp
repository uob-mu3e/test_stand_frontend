
#include "../include/base.h"

#include "../include/a10/flash.h"
flash_t flash;

#include "../include/a10/fan.h"
fan_t fan;

alt_u32 alarm_callback(void*) {
    IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0xFF);
    // watchdog
    IOWR_ALTERA_AVALON_PIO_SET_BITS(PIO_BASE, alt_nticks() & 0xFF);

    if(flash.callback() == -EAGAIN) return 1;

    return 10;
}

int main() {
    uart_init();

    fan.init();
    flash.init();

    alt_alarm alarm;
    int err = alt_alarm_start(&alarm, 10, alarm_callback, nullptr);
    if(err) {
        printf("ERROR: alt_alarm_start => %d\n", err);
    }

    while (1) {
        printf("  [1] => flash\n");
        printf("  [8] => fan\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            flash.menu();
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
