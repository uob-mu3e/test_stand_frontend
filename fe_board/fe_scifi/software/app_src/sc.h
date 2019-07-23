
#include <sys/alt_stdio.h>
#include "malibu/malibu_basic_cmd.h"

void sc_callback(volatile alt_u32* data) {
//check spi command register, trigger spi configuration if needed
    alt_u32 d0 = data[0x13];
    if(d0 != 0){
        printf("[sc_callback] SPICTRL_REGISTER = 0x%08X\n", d0);
        printf("SPI: START= %d ASIC=%u\n",(d0>>5)&1, d0&0x0f);
	if((d0>>5)&1==0) return; // no start request
        usleep(1000);
	uint32_t pattern[74];
	for(int i=0;i<74;i++) pattern[i]=*((volatile alt_u32*)AVM_SC_BASE+0x14+i);
	for(int i=0;i<74;i++) pattern[i]++;
	//for(int i=0;i<74;i++) printf("pattern[%2.2d]=%8.8x\n",i,pattern[i]); -- WARNING: statement causes nios to crash for some reason, be careful
        SPI_write_pattern2(0, pattern);
	for(int i=0;i<74;i++) *((volatile alt_u32*)AVM_SC_BASE+0x14+i)=pattern[i];


        data[0x13] = 0;
        printf("SPI: finished...\n");
    }
//check registers and forward to avalon test interface
    if(((volatile alt_u32*)AVM_TEST_BASE)[0xB]!=data[0x10])
            printf("Update value dummyctrl_reg:    0x%08X\n", data[0x10]);
    ((volatile alt_u32*)AVM_TEST_BASE)[0xB]=data[0x10];

    if(((volatile alt_u32*)AVM_TEST_BASE)[0xC]!=data[0x11])
            printf("Update value dpctrl_reg:    0x%08X\n", data[0x11]);
    ((volatile alt_u32*)AVM_TEST_BASE)[0xC]=data[0x11];

    if(((volatile alt_u32*)AVM_TEST_BASE)[0xD]!=data[0x12])
            printf("Update value dummyctrl_reg:    0x%08X\n", data[0x12]);
    ((volatile alt_u32*)AVM_TEST_BASE)[0xD]=data[0x12];

}

void menu_sc(volatile alt_u32* data) {
    while(1) {
        printf("  [r] => read sc ram\n");
        printf("  [w] => write sc ram\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case 'r':
            for(int i = 0; i < 32; i++) {
                printf("[0x%04X] = 0x%08X\n", i, data[i]);
            }
            break;
        case 'w':
            for(int i = 0; i < 32; i++) {
                data[i] = i;
            }
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
