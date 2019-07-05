
#include "../include/si.h"

#include "si5345_revb_registers.h"

struct si5345_t : si_t {

    const char* DESIGN_ID = "feb.v01";

    si5345_t(int spi_slave)
        : si_t(spi_slave)
    {
    }

    void init() {
        char id[9];
        id[8] = '\0';
        for(int i = 0; i < 8; i++) id[i] = (char)read(0x026B + i);
        if(strcmp(id, DESIGN_ID) == 0) return;

        si_t::init(si5345_revb_registers, sizeof(si5345_revb_registers) / sizeof(si5345_revb_registers[0]));
        for(int i = 0; i < 8; i++) write(0x026B + i, DESIGN_ID[i]);

        for(int i = 0; i < 8; i++) {
            alt_u8 sysincal = read(0x000C);
            if(sysincal == 0) break;
            printf("[si5345.init] SYSINCAL = %u => wait ...\n", sysincal);
            usleep(1000);
        }
    }

    void reset() {
        write(0x001C, 0x01);
    }

    void preamble() {
        write(0x0B24, 0xC0);
        write(0x0B25, 0x00);
        write(0x0540, 0x01);
    }

    void postamble() {
        write(0x0540, 0x00);
        write(0x0B24, 0xC3);
        write(0x0B25, 0x02);
    }

    void status() {
        printf("status:\n");
        printf("  SYSINCAL = %d\n", read(0x000C));
        printf("  LOF/LOS = %d/%d\n", (read(0x000D) & 0x10) != 0, (read(0x000D) & 0x01) != 0);
        printf("  HOLD/LOL = %d/%d\n", (read(0x000E) & 0x20) != 0, (read(0x000E) & 0x02) != 0);
    }

    void menu() {
        alt_u32 pn_base = (read(0x0003) << 8) | read(0x0002);
        if(pn_base != 0x5345) {
            printf("Invalid base part number: 0x%04X\n", pn_base);
            return;
        }

        while (1) {
            status();
            printf("\n");

            printf("si5340:\n");
            printf("  [I] => init\n");
            printf("  [R] => reset\n");
            printf("  [r] => read regs\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 'I':
                init();
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
