
#include "../include/base.h"

#include "adc.h"
adc_t adc;

int main() {
    base_init();

    while (1) {
        printf("\n");

        printf("\n");
        printf("  [0] => spi si chip\n");
        printf("  [1] => adc\n");

        printf("Select entry ...\n");
        char cmd = wait_key();

        switch(cmd) {
        case '0':
            printf("spi:\n");
            //menu_spi_si5345();
            break;
        case '1':
            adc.menu();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
