
#include "../include/base.h"

#include "si.h"
si_t si;

alt_u32 alarm_callback(void*) {
    // watchdog
    IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0xFF);
    IOWR_ALTERA_AVALON_PIO_SET_BITS(PIO_BASE, alt_nticks() & 0xFF);

    return 10;
}

void menu_spi_si5345() {
    while (1) {
        printf("SI5345\n", ALT_DEVICE_FAMILY);
        printf("  [i] => init chip\n");
        printf("  [q] => quit\n");
        printf("  [r] => read(0xFE)\n");
        printf("  [t] => test\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case 'r' :
            si.read(0xFE);
            break;
        case 'i':
            si.init();
            break;
        case 't' :
            si.test();
            break;
        case '?':
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

int main() {
    uart_init();

    printf("ALT_DEVICE_FAMILY = '%s'\n", ALT_DEVICE_FAMILY);
    printf("\n");

    alt_alarm alarm;
    int err = alt_alarm_start(&alarm, 0, alarm_callback, nullptr);
    if(err) {
        printf("ERROR: alt_alarm_start => %d\n%d\n", err);
    }

    while (1) {
        printf("\n");
        printf("MAX_10:\n");
        printf("  [0] => spi si chip\n");
        printf("  [1] => Hello\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            printf("spi:\n");
            menu_spi_si5345();
            break;
        case '1':
            printf("Hello\n");
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
