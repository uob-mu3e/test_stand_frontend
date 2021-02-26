#ifndef __ADC_H__
#define __ADC_H__

#include <altera_modular_adc_sequencer_regs.h>
#include "altera_modular_adc.h"

#include <sys/alt_irq.h>

const
alt_u8 ADC_LUT[383] = {
    165,165,165,164,164,164,163,163,163,162,162,162,161,161,161,160,160,160,
    159,159,159,158,158,158,157,157,157,156,156,156,155,155,155,154,154,154,
    153,153,152,152,152,151,151,151,150,150,150,149,149,149,148,148,148,147,
    147,147,146,146,146,145,145,144,144,144,143,143,143,142,142,142,141,141,
    141,140,140,140,139,139,138,138,138,137,137,137,136,136,136,135,135,135,
    134,134,133,133,133,132,132,132,131,131,131,130,130,129,129,128,128,128,
    128,127,127,126,126,126,125,125,125,124,124,124,123,123,122,122,122,121,
    121,121,120,120,119,119,119,118,118,118,117,117,116,116,116,115,115,114,
    114,114,113,113,113,112,112,111,111,111,110,110,109,109,109,108,108,107,
    107,107,106,106,105,105,105,104,104,103,103,103,102,102,101,101,101,100,
    100,99,99,99,98,98,97,97,96,96,96,95,95,94,94,94,93,93,92,92,91,91,91,90,
     90,89,89,88,88,88,87,87,86,86,85,85,85,84,84,83,83,82,82,82,81,81,80,80,
     79,79,78,78,78,77,77,76,76,75,75,74,74,73,73,73,72,72,71,71,70,70,69,69,
     68,68,67,67,67,66,66,65,65,64,64,63,63,62,62,61,61,60,60,59,59,58,58,57,
     57,56,56,55,55,55,54,54,53,53,52,52,51,51,50,50,49,49,48,48,47,47,46,46,
     45,45,44,43,43,42,42,41,41,40,40,39,39,38,38,37,37,36,36,35,35,34,34,33,
     33,32,31,31,30,30,29,29,28,28,27,27,26,26,25,24,24,23,23,22,22,21,21,20,
     20,19,18,18,17,17,16,16,15,15,14,13,13,12,12,11,11,10, 9, 9, 8, 8, 7, 7,
      6, 5, 5, 4, 4, 3, 2, 2, 1, 1, 0
};

struct adc_t {

    alt_u8 celsius_lookup(int adc_avg_in) {
//        printf("temp = %d\n", adc_avg_in);
        return ADC_LUT[adc_avg_in];
    }

    void menu() {
        while (1) {
            printf("\n");
            printf("  [1] => adc readout while\n");
            printf("  [2] => read one time adc data\n");
            printf("  [q] => exit\n");
            
            printf("Select entry ...\n");
            char cmd = wait_key();

            switch(cmd) {
            case '1':
                adc_readout(true);
                break;
            case '2':
                adc_readout(false);
                break;
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
        }
    }
    
    void adc_readout(bool loop) {
        
        start_adc_sequencer();
        
        while(1) {
            char cmd;
            if(read(uart, &cmd, 1) > 0) switch(cmd) {
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
            
            alt_u32 adc_data[ADC_SAMPLE_STORE_CSR_CSD_LENGTH];
            alt_u32	adc_comp_data[5] = {0};
            
            // get mean of 64 adc reads
            alt_u32 adc_data_avg[ADC_SAMPLE_STORE_CSR_CSD_LENGTH] = {0};
            for(int i = 0 ; i<64 ; i++){
                
                alt_adc_word_read(ADC_SAMPLE_STORE_CSR_BASE, adc_data, ADC_SAMPLE_STORE_CSR_CSD_LENGTH);
                for (int j = 0; j < ADC_SAMPLE_STORE_CSR_CSD_LENGTH ; j++){
                    adc_data_avg[j] += adc_data[j];
                }
            }
            
            for (int j = 0; j < ADC_SAMPLE_STORE_CSR_CSD_LENGTH ; j++){
                adc_data_avg[j] = adc_data_avg[j]/64;
            }
            
            printf("On-die temperature = %d\n", celsius_lookup(adc_data_avg[9] - 3417) - 40);
            
            for (int k = 0; k<(ADC_SAMPLE_STORE_CSR_CSD_LENGTH+1)/2; k++){
                adc_comp_data[k] = (adc_data[2*k] << 16)+adc_data[2*k +1];
                if (loop == false) printf("adc_comp_data%i: %x\n", k, adc_comp_data[k]);
            }
            
            IOWR_ALTERA_AVALON_PIO_DATA(ADC_D0_BASE,(adc_comp_data[0]));
            IOWR_ALTERA_AVALON_PIO_DATA(ADC_D1_BASE,(adc_comp_data[1]));
            IOWR_ALTERA_AVALON_PIO_DATA(ADC_D2_BASE,(adc_comp_data[2]));
            IOWR_ALTERA_AVALON_PIO_DATA(ADC_D3_BASE,(adc_comp_data[3]));
            IOWR_ALTERA_AVALON_PIO_DATA(ADC_D4_BASE,(adc_comp_data[4]));
            
            if (loop == false) return;
            usleep(200000);
        }
    }

    void init() {
        adc_interrupt_disable(ADC_SAMPLE_STORE_CSR_BASE);
        adc_set_mode_run_once(ADC_SEQUENCER_CSR_BASE);

        if(int err = alt_ic_isr_register(0, 3, callback, this, nullptr)) {
            printf("[adc] ERROR: alt_ic_isr_register => %d\n", err);
            return;
        }
        adc_interrupt_enable(ADC_SAMPLE_STORE_CSR_BASE);
        adc_start(ADC_SEQUENCER_CSR_BASE);

    }

    static void callback(void* context) {
   //     adc_interrupt_disable(ADC_SAMPLE_STORE_CSR_BASE);
        alt_u32 adc_data[ADC_SAMPLE_STORE_CSR_CSD_LENGTH];
        alt_u32	adc_comp_data[5];
        alt_adc_word_read(ADC_SAMPLE_STORE_CSR_BASE, adc_data, 
                ADC_SAMPLE_STORE_CSR_CSD_LENGTH);

        for (int k = 0; k<(ADC_SAMPLE_STORE_CSR_CSD_LENGTH+1)/2; k++){
            adc_comp_data[k] = (adc_data[2*k] << 16)+adc_data[2*k +1];      
        }    

        IOWR_ALTERA_AVALON_PIO_DATA(ADC_D0_BASE,(adc_comp_data[0]));
        IOWR_ALTERA_AVALON_PIO_DATA(ADC_D1_BASE,(adc_comp_data[1]));
        IOWR_ALTERA_AVALON_PIO_DATA(ADC_D2_BASE,(adc_comp_data[2]));
        IOWR_ALTERA_AVALON_PIO_DATA(ADC_D3_BASE,(adc_comp_data[3]));
        IOWR_ALTERA_AVALON_PIO_DATA(ADC_D4_BASE,(adc_comp_data[4]));   

 //       adc_interrupt_enable(ADC_SAMPLE_STORE_CSR_BASE);
        adc_start(ADC_SEQUENCER_CSR_BASE);
    }


    void start_adc_sequencer() {
        adc_interrupt_disable(ADC_SAMPLE_STORE_CSR_BASE);
        adc_set_mode_run_continuously(ADC_SEQUENCER_CSR_BASE);
        adc_interrupt_enable(ADC_SAMPLE_STORE_CSR_BASE); 
        adc_start(ADC_SEQUENCER_CSR_BASE);
       
    }

};



#endif // __ADC_H__
