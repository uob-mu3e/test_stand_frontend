
#include <sys/alt_stdio.h>
#define RESET_OUT_BASE 0x700f0360

void menu_reset() {
    while(1) {
        printf("  [0] => use genesis\n");
        printf("  [1] => run_prep\n");
        printf("  [2] => sync\n");
        printf("  [3] => start run\n");
        printf("  [4] => end run\n");
        printf("  [5] => abort run\n");
        printf("  [6] => start reset\n");
        printf("  [7] => stop reset\n");
        printf("Select entry ...\n");
        char cmd = wait_key();
        
        switch(cmd) {
        case '0':
            IOWR_ALTERA_AVALON_PIO_DATA(RESET_OUT_BASE,0x000);
            break;    
        case '1':
            IOWR_ALTERA_AVALON_PIO_DATA(RESET_OUT_BASE,0x110);
            break;
        case '2':
            IOWR_ALTERA_AVALON_PIO_DATA(RESET_OUT_BASE,0x111);
            break;
        case '3':
            IOWR_ALTERA_AVALON_PIO_DATA(RESET_OUT_BASE,0x112);
            break;
        case '4':
            IOWR_ALTERA_AVALON_PIO_DATA(RESET_OUT_BASE,0x113);
            break;
        case '5':
            IOWR_ALTERA_AVALON_PIO_DATA(RESET_OUT_BASE,0x114);
            break;
        case '6':
            IOWR_ALTERA_AVALON_PIO_DATA(RESET_OUT_BASE,0x130);
            break;
        case '7':
            IOWR_ALTERA_AVALON_PIO_DATA(RESET_OUT_BASE,0x131);
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
