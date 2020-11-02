
#include "../include/base.h"

#include "adc.h"
adc_t adc;

int main() {
    base_init();

    while (1) {
        printf("\n");
        printf("  [a] => adc\n");

        printf("Select entry ...\n");
        char cmd = wait_key();

        switch(cmd) {
        case 'a':
            adc.menu();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
