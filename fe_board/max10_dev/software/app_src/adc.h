#ifndef __ADC_H__
#define __ADC_H__

#include <altera_modular_adc_sequencer_regs.h>

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

#include <time.h>
#include <sys/time.h>

struct adc_t {

    alt_u8 celsius_lookup(int adc_avg_in) {
//        printf("temp = %d\n", adc_avg_in);
        return ADC_LUT[adc_avg_in];
    }

    void menu() {
        while (1) {
            printf("\n");
            printf("  [1] => read temperature\n");
            printf("  [2] => read adc data\n");
            printf("  [3] => rolling avg\n");

            printf("Select entry ...\n");
            char cmd = wait_key();

            switch(cmd) {
            case '1':
                read_temp();
                break;
            case '2':
                data_writeout();
                break;
            case '3':
                wait_writeout();
                break;
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
        }
    }

    void start_adc_sequencer() {
        IOWR(ADC_SEQUENCER_CSR_BASE, 0, 0);
        usleep(1000);
        IOWR(ADC_SAMPLE_STORE_CSR_BASE, 64, 0);
        IOWR(ADC_SEQUENCER_CSR_BASE, 0, 1);
    }

    void read_temp() {
        printf(
            "******** PIO and On-Die Temp Sensor example ********\n"
            "The value of ADC Channel connected to Temperature Sensing Diode\n"
            "is collected and averaged over 64 Samples\n"
            "----------------------------------------------------------------\n"
        );

        start_adc_sequencer();

        // event loop never exits
        while(1) {
            char cmd;
            if(read(uart, &cmd, 1) > 0) switch(cmd) {
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }

            int adc_avg = 0;
            // get average of 64 samples
            for(int i = 0; i < 64; i++) {
                int adc_value = IORD(ADC_SAMPLE_STORE_CSR_BASE, 0);
                adc_avg += adc_value;
            }
            adc_avg /= 64;
            printf("On-die temperature = %d\n", celsius_lookup(adc_avg - 3417) - 40);

            usleep(200000);
        }
    }

    void wait_writeout() {
        printf("start in 10 xeconds ... disconnect JTAG\n");
        usleep(10000000); // 10 sec

        start_adc_sequencer();

        const int count = 400;
        unsigned int e_x_store[count];
        unsigned int e_x2_store[count];
        unsigned int n = 250;

        for(int j = 0; j < count; j++) {
            unsigned int x = 0;
            unsigned int e_x2 = 0;
            unsigned int e_x = 0;
            for(int i = 0; i < n; i++) {
                x = IORD(ADC_SAMPLE_STORE_CSR_BASE, 1);
                e_x2 += x*x;
                e_x += x;
//                printf("%d, %d, %d\n", x, e_x2, e_x);
//                usleep(100);
            }
            e_x_store[j] = e_x;
            e_x2_store[j]= e_x2;
        }

        usleep(10000000); // 10 sec for plugging back in
        printf("DONE\n");
        printf("n = %d\n", n);
        for(int j = 0; j < count; j++) {
            printf("%u, %u\n", e_x_store[j], e_x2_store[j]);
        }
    }

    void data_writeout() {
        printf(
            "******* Write out of Data in Adc storage ********\n"
            "The value of ADC Channel connected to Arduino J4-1"
            "is collected and printed onto the board\n"
            "----------------------------------------------------------------\n"
        );

        start_adc_sequencer();

        int channel = 1;

        for(int j =0; j <1000; j++) {
//            usleep(100000);
            int adc_avg = 0;
            for(int i = 0; i <1000; i++) {
//                    IOWR(ADC_SEQUENCER_CSR_BASE, 7, 1);
                    adc_avg = IORD(ADC_SAMPLE_STORE_CSR_BASE, 7);
                    printf("%d\n", adc_avg);
//                    printf("%d.%d, %d\n", time, time.tv_usec, adc_avg);
//                    usleep(100000);
            }
//            adc_avg = adc_avg / 1024;
//            gettimeofday(&time, NULL);
//            printf("%d\n", adc_avg);
//            printf("%u, %d.%d\n", adc_avg, time, time.tv_usec);
        }
    }
};

#endif // __ADC_H__
