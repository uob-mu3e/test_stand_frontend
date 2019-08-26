
#include "../include/base.h"

#include "adc.h"
adc_t adc;

#include "flash.h"
flash_t flash;

int main() {
    base_init();

    while (1) {
        printf("\n");
        printf("  [a] => adc\n");
        printf("  [f] => flash (UFM)\n");
        printf("  [s] => spi\n");

        printf("Select entry ...\n");
        char cmd = wait_key();

        switch(cmd) {
        case 'a':
            adc.menu();
            break;
        case 'f':
            flash.menu();
            break;
        case 's':
//            menu_spi_si5345();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
