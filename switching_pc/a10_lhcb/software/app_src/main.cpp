
#include "../include/base.h"

#include "../include/i2c.h"
#include "../include/si534x.h"

i2c_t i2c;
si534x_t si5345 { i2c.dev, 0x68 };

void clocks_menu() {
    while (1) {
        printf("\n");
        printf("[PCIe40] -------- menu --------\n");

        printf("\n");
        printf("  [1] => si5345_1\n");
        printf("  [2] => si5345_2\n");
        printf("  [m] => ...\n");
        printf("  [e] => ...\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            IOWR_ALTERA_AVALON_PIO_DATA(I2C_CS_BASE, 1 << 1); // i2c_cs(1) <= '1'
            break;
        case '2':
            IOWR_ALTERA_AVALON_PIO_DATA(I2C_CS_BASE, 1 << 2); // i2c_cs(2) <= '1'
            break;
        case 'm':
            si5345.menu();
            break;
        case 'e':
            // TODO: enable outputs
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

int main() {
    base_init();

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
            clocks_menu();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
