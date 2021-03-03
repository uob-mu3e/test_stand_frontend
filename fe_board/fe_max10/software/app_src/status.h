#include "system.h"
#include "altera_avalon_pio_regs.h"

#ifndef STATUS__H
#define STATUS__H




struct status_t {
    alt_alarm alarm;

    void init() {
    
    }

    void menu() {
        while (1) {
            printf("\n");
            printf("[status] -------- menu --------\n");
            printf("  [s] => status\n");
            printf("  [q] => exit\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 's':
                ReadStatusRegister();
                break;
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
        }
    }

static void ReadStatusRegister(){
    printf("Status: %x\n", (uint32_t)IORD_ALTERA_AVALON_PIO_DATA(STATUS_BASE));
}

};

#endif
