
#include "../include/base.h"

#include "adc.h"
adc_t adc;

#include "ufm.h"
ufm_t ufm;

#include "spiflash.h"
flash_t spiflash;

int main() {
    base_init();
    
    printf("Init ADC");
    adc_interrupt_disable(ADC_SAMPLE_STORE_CSR_BASE);
    adc_set_mode_run_continuously(ADC_SEQUENCER_CSR_BASE);
    adc_start(ADC_SEQUENCER_CSR_BASE);


    while (1) {
        printf("\n");
        printf("  [a] => ADC\n");
        printf("  [f] => User flash\n");
        printf("  [s] => SPI flash\n");
        
        printf("Select entry ...\n");
        char cmd = wait_key();

        switch(cmd) {
        case 'a':
            adc.menu();
            break;
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
