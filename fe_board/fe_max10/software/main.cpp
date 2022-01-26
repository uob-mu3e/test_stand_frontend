
#include "include/base.h"

#include "adc.h"
adc_t adc;

#include "ufm.h"
ufm_t ufm;

#include "spiflash.h"
flash_t spiflash;

#include "status.h"
status_t status;

int main() {
    base_init();

    printf("Init ADC");
//    adc.init();

    while (1) {
        printf("\n");
        printf("  [a] => ADC\n");
        printf("  [b] => Status\n");
        printf("  [f] => User flash\n");
        printf("  [s] => SPI flash\n");

        printf("Select entry ...\n");
        char cmd = wait_key();

        switch(cmd) {
        case 'a':
//            adc.menu();
            break;
        case 'b':
            status.menu();
        case 'f':
            ufm.menu();
            break;
        case 's':
            spiflash.menu();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
