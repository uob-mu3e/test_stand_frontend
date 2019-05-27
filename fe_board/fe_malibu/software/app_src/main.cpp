
#include "system.h"

#ifndef ALT_CPU_DCACHE_BYPASS_MASK
    #define ALT_CPU_DCACHE_BYPASS_MASK 0
#endif

#include <altera_avalon_pio_regs.h>

#include <sys/alt_alarm.h>
#include <sys/alt_timestamp.h>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

alt_u32 alarm_callback(void*) {
    IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0xFF);
    // watchdog
    IOWR_ALTERA_AVALON_PIO_SET_BITS(PIO_BASE, alt_nticks() & 0xFF);

    return 10;
}

int uart = -1;

char wait_key(useconds_t us = 100000) {
    while(true) {
        char cmd;
        if(read(uart, &cmd, 1) > 0) return cmd;
        usleep(us);
    }
}

#include "xcvr.h"
#include "malibu.h"
#include "sc.h"

int main() {
    alt_alarm alarm;
    int err = alt_alarm_start(&alarm, 10, alarm_callback, nullptr);
    if(err) {
        printf("ERROR: alt_alarm_start => %d\n%d\n", err);
    }

    uart = open(JTAG_UART_NAME, O_NONBLOCK);
    if(uart < 0) {
        printf("ERROR: can't open %s\n", JTAG_UART_NAME);
        return 1;
    }

    while (1) {
        printf("'%s' FE_S4 (MALIBU)\n", ALT_DEVICE_FAMILY);
        printf("  [1] => xcvr\n");
        printf("  [2] => malibu\n");
        printf("  [3] => sc\n");

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
            menu_sc();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
