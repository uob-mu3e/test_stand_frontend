
#include "../include/a10/si5340.h"

si5340_t si5340;

void menu_si5340() {
    while (1) {
        printf("si5340:\n");
        printf("  [1] => 100 MHz\n");
        printf("  [2] => 125 MHz\n");
        printf("  [3] => 156.25 MHz\n");
        printf("  [r] => read regs\n");
        printf("  [w] => write NUM\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            si5340.set_f0(100000000); // 100 MHz
            break;
        case '2':
            si5340.set_f0(125000000); // 125 MHz
            break;
        case '3':
            si5340.set_f0(156250000); // 156.25 MHz
            break;
        case 'r':
            si5340.test();
            break;
        case 'w':
            si5340.write();
            break;
        case '?':
            wait_key();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
