
#include "../include/base.h"

#include "../include/i2c.h"

#include "../include/si.h"

int main() {
    base_init();

    i2c_t i2c;
    si_t si5345_1 { i2c.dev, 0x68 };

    while (1) {
        printf("\n");
        printf("[PCIe40] -------- menu --------\n");

        printf("\n");
        printf("  [1] => ...\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            break;
        case 'c':
            for(uint16_t i = 0x0000; i < 0x0100; i++)
            printf("[0x%04X] = 0x%02X\n", i, si5345_1.read(i));
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
