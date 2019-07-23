
typedef alt_u8 uint8_t;
typedef alt_u16 uint16_t;

#include "malibu/malibu_basic_cmd.h"

void menu_malibu() {
    while(1) {
        printf("  [0] => reset asic\n");
        printf("  [1] => reset datapath\n");
        printf("  [2] => configure all off\n");
        printf("  [3] => data\n");
        printf("  [4] => get datapath status\n");
        printf("  [5] => get slow control registers\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
	    (((volatile alt_u32*)AVM_TEST_BASE)[0xD])=1;
            usleep(100);
	    (((volatile alt_u32*)AVM_TEST_BASE)[0xD])=0;
            //IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0x00010000);
            break;
        case '1':
	    (((volatile alt_u32*)AVM_TEST_BASE)[0xD])=2;
            usleep(100);
	    (((volatile alt_u32*)AVM_TEST_BASE)[0xD])=0;
            //IOWR_ALTERA_AVALON_PIO_CLEAR_BITS(PIO_BASE, 0x00010000);
            break;
        case '2':
            SPI_configure(0, stic3_config_ALL_OFF);
            break;
        case '3':
            printf("TODO...\n");
            break;
        case '4':
            printf("buffer_full/frame_desync/rx_pll_lock: 0x%03X\n", ((volatile alt_u32*)AVM_TEST_BASE)[0x8]);
            printf("rx_dpa_lock: 0x%08X\n", ((volatile alt_u32*)AVM_TEST_BASE)[0x9]);
            printf("rx_ready: 0x%08X\n", ((volatile alt_u32*)AVM_TEST_BASE)[0xA]);
            break;
        case '5':
            printf("dummyctrl_reg:    0x%08X\n", ((volatile alt_u32*)AVM_TEST_BASE)[0xB]);
            printf("    :datagen_en   0x%X\n", (((volatile alt_u32*)AVM_TEST_BASE)[0xB]>>0)&1);
            printf("    :datagen_fast 0x%X\n", (((volatile alt_u32*)AVM_TEST_BASE)[0xB]>>1)&1);
            printf("    :datagen_cnt  0x%X\n", (((volatile alt_u32*)AVM_TEST_BASE)[0xB]>>2)&0x3ff);

            printf("dpctrl_reg:       0x%08X\n", ((volatile alt_u32*)AVM_TEST_BASE)[0xC]);
            printf("    :mask         0b");
	    for(int i=16;i>0;i--) printf("%d", (((volatile alt_u32*)AVM_TEST_BASE)[0xC]>>i)&1);
            printf("\n");

            printf("    :prbs_dec     0x%X\n", (((volatile alt_u32*)AVM_TEST_BASE)[0xC]>>31)&1);
            printf("subdet_reset_reg: 0x%08X\n", ((volatile alt_u32*)AVM_TEST_BASE)[0xD]);
            break;
        case '6':
            for(int i=0;i<0xf;i++){printf("AVM_TEST_BASE[%x]=%16.16x\n",i,((volatile alt_u32*)AVM_TEST_BASE)[i]);};
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
