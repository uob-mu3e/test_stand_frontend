/*
 * author : Martin Mueller
 * date : 2020
 */
#include "stdlib.h"

void menu_tmpDisplay(volatile alt_u32* xcvr, char ID = 'A') {
    
    while (1) {
        char cmd;
        if(read(uart, &cmd, 1) > 0) switch(cmd) {
        case '0': case '1': case '2': case '3': case '4': case '5': case '6': case '7': // select channel
            xcvr[0x00] = (cmd - '0') & 0xFF;
            break;
        case '?':
            wait_key();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
        printf("select Firefly with 0-3 and 4-7");
        printf("Firefly [°C] :  %i\n", xcvr[0x26]/10000);
        printf("ArriaV   [?] :  %i\n", sc.ram->data[0xFFF8]);
        printf("\n");

        usleep(200000);
    }
}
