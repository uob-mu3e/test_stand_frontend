
#include "si.h"

#include "si5345_revb_registers.h"

struct si5345_t : si_t {

    si5345_t(int spi_cs)
        : si_t(spi_cs)
    {
    }

    void menu() {
        alt_u32 pn_base = (read(0x0003) << 8) | read(0x0002);
        if(pn_base != 0x5345) {
            printf("Invalid base part number: 0x%04X\n", pn_base);
            return;
        }

        while (1) {
            printf("status:\n");
            printf("  SYSINCAL = %d\n", read(0x000C));
            printf("  LOF/LOS = %d/%d\n", (read(0x000D) & 0x10) != 0, (read(0x000D) & 0x01) != 0);
            printf("  HOLD/LOL = %d/%d\n", (read(0x000E) & 0x20) != 0, (read(0x000E) & 0x02) != 0);
            printf("\n");

            printf("si5340:\n");
            printf("  [I] => init\n");
            printf("  [R] => reset\n");
            printf("  [r] => read regs\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 'I':
                init(si5345_revb_registers, sizeof(si5345_revb_registers) / sizeof(si5345_revb_registers[0]));
                break;
            case 'R':
                reset();
                break;
            case 'r':
                printf("si5345.read:\n");
                for(alt_u16 address = 0x0200; address < 0x0300; address++) {
                    printf("  [0x%02X] = 0x%02X\n", address, read(address));
                }
                break;
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
        }
    }

};
