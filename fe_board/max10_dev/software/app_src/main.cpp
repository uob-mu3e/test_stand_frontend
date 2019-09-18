
#include "../include/base.h"

#include "adc.h"
adc_t adc;



#include <time.h> 
#include <sys/time.h>


alt_u8 celsius_lookup(int adc_avg_in);
void adc_data_writeout();
void adc_wait_writeout();


alt_u32 alarm_callback(void*) {
    // watchdog
    IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0xFF);
    IOWR_ALTERA_AVALON_PIO_SET_BITS(PIO_BASE, alt_nticks() & 0xFF);

    return 10;
}


int main() {
    base_init();

    while (1) {
        printf("\n");
        printf("MAX_10:\n");
        printf("  [0] => spi si chip\n");
        printf("  [1] => read temperature on adc \n");
        printf("  [2] => read adc data\n");
        printf("  [3] => rolling avg\n");


        printf("Select entry ...\n");
        char cmd = wait_key();

        switch(cmd) {
        case 'a':
            adc.menu();
            break;

        case '1':

            printf("*******PIO and On-Die Temp Sensor example********\n"
                   "Change Switches 1, 2 and 3 to change LEDs 1, 2 and 3\n"
                   "The value of ADC Channel connected to Temperature Sensing Diode is collected every second and is averaged over 64 Samples\n"
                   "------------------------------------------------------------------------------------------------------\n");

            //Starting the ADC sequencer
            IOWR(ADC_SEQUENCER_CSR_BASE, 0, 0);
            usleep(1000);
            IOWR(ADC_SAMPLE_STORE_CSR_BASE, 64, 0);
            IOWR(ADC_SEQUENCER_CSR_BASE, 0, 1);

            //Event loop never exits
            while(1)
            {
                // Reads the value from Switch and Displays in the LED
                switch_datain = IORD_ALTERA_AVALON_PIO_DATA(SW_IO_BASE);
                IOWR_ALTERA_AVALON_PIO_DATA(LED_IO_BASE,switch_datain);
                //Giving a 1 Second delay
                usleep(1000000);

                adc_avg=0;
                //Getting an average of 64 samples
                for (int i=0;i<64;i++)
                {
                    int adc_value=IORD(ADC_SAMPLE_STORE_CSR_BASE, 0);
                    printf("%d\n",adc_value);
                    adc_avg=adc_avg+adc_value;
                }
                printf("\n");
    
                printf("On-die temperature = %d\n",(celsius_lookup(adc_avg/64-3417)-40));

            }

            break;
        case '2':
            adc_data_writeout();
            break;
        case '3':
            adc_wait_writeout();
            break;
        case '4':
            printf("bla");
            for (int i=0; i<5; i++)
            {
                IOWR(LED_IO_BASE,i,0);
                usleep(10000000);
            }
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        
        }
    }

    return 0;
}

void adc_wait_writeout() 
{
    //int switch_datain = IORD_ALTERA_AVALON_PIO_DATA(SW_IO_BASE);
    //printf("%d\n", switch_datain);
    IOWR_ALTERA_AVALON_PIO_DATA(LED_IO_BASE+1,1);

    IOWR(ADC_SEQUENCER_CSR_BASE, 0, 0);
    usleep(1000);
    IOWR(ADC_SAMPLE_STORE_CSR_BASE, 64, 0);
    IOWR(ADC_SEQUENCER_CSR_BASE, 0, 1);
    
    printf("Starting calculation in 10 Seconds. Disconnect JTag.\n");
    IOWR(LED_IO_BASE,0,1);
    IOWR(LED_IO_BASE,1,1);
    IOWR(LED_IO_BASE,2,1);
    IOWR(LED_IO_BASE,3,1);
    IOWR(LED_IO_BASE,4,1);
    usleep(10000000); //10sec
    unsigned int count = 400;
    unsigned int e_x_store[count];
    unsigned int e_x2_store[count];
    unsigned int n = 250;


    for (short j=0; j<count; j++){
        unsigned int x = 0;
        unsigned int e_x2 = 0;
        unsigned int e_x = 0;
        for (unsigned int i=0; i<n;i++){
            x = IORD(ADC_SAMPLE_STORE_CSR_BASE, 1);
            e_x2 += x*x;
            e_x += x;
            //printf("%d \t %d \t %d \n", x,e_x2,e_x);
            //usleep(100);
        }
        e_x_store[j] = e_x;
        e_x2_store[j]= e_x2;
    }
    IOWR(LED_IO_BASE,0,0);
    IOWR(LED_IO_BASE+1,0,0);
    IOWR(LED_IO_BASE+2,0,0);
    IOWR(LED_IO_BASE+3,0,0);
    IOWR(LED_IO_BASE+4,0,0);
    usleep(10000000); // 10 sec for plugging back in
    printf("im done\n");
    printf("n =%d\n", n);
    for(short j=0; j<count; j++){
        printf("%u,%u\n", e_x_store[j],e_x2_store[j]);
    }
}


void adc_data_writeout()
{
    printf("******* Write out of Data in Adc storage ********\n"
        "The value of ADC Channel connected to Arduino J4-1 is collected and printed onto the board\n"
        "------------------------------------------------------------------------------------------------------\n");
 
    //Starting the ADC sequencer
    IOWR(ADC_SEQUENCER_CSR_BASE, 0, 0);
    usleep(1000);
    IOWR(ADC_SAMPLE_STORE_CSR_BASE, 64, 0);
    IOWR(ADC_SEQUENCER_CSR_BASE, 0, 1);
    
    short channel = 1;
    
    struct timeval time;
    unsigned int adc_avg = 0;
    
    for (short j =0; j <1000; j++)
    //while (1)
    {

        //usleep(100000);
        adc_avg = 0;
        for (short i =0; i <1000; i++){
                //IOWR(ADC_SEQUENCER_CSR_BASE, 7, 1);
                adc_avg =  IORD(ADC_SAMPLE_STORE_CSR_BASE, 7);
                //fprintf(stderr, "%d\n", temp);
                //fprintf( "./test.txt", "%d\n", "string format", temp);
                printf("%d\n", adc_avg);
                //printf("%d.%d \t%d\n",time,time.tv_usec, adc_avg);
                //usleep(100000);
        }
        //adc_avg = adc_avg/1024;
        //gettimeofday(&time, NULL);
        //printf("%d\n", adc_avg);
        //printf("%u,%d.%d\n", adc_avg,time,time.tv_usec);
        //printf("%d\n",temparray);
    }
}

alt_u8 celsius_lookup(int adc_avg_in)
{
	const alt_u8 celsius_lookup_table[383]={
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
			  6, 5, 5, 4, 4, 3, 2, 2, 1, 1, 0 };

	//printf("temp = %d",adc_avg_in);
	return (celsius_lookup_table[adc_avg_in]);

}

