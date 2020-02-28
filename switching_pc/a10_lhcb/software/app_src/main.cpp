
#include "../include/base.h"

#include "../include/i2c.h"

#include "../include/si534x.h"

int main() {
    base_init();

    i2c_t i2c;
    si534x_t si5345_1 { i2c.dev, 0x68 };

    while (1) {
        printf("\n");
        printf("[PCIe40] -------- menu --------\n");

        printf("\n");
        printf("  [1] => ...\n");
        printf("  [c] => clocks\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            break;
        case 'c':
            IOWR_ALTERA_AVALON_PIO_DATA(I2C_CS_BASE, 1 << 1);
            si5345_1.menu();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
