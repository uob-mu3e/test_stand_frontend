#ifndef __FE_SI5345_H__
#define __FE_SI5345_H__

#include "../include/si534x.h"

#include "si5345_revb_registers.h"

struct si5345_t : si534x_t {

    const char* DESIGN_ID = "feb.v01";

    si5345_t(alt_u32 spi_base, alt_u32 spi_slave)
        : si534x_t(spi_base, spi_slave)
    {
    }

    void init() {
        char id[9];
        id[8] = '\0';
        for(int i = 0; i < 8; i++) id[i] = (char)read(0x026B + i);
        if(strcmp(id, DESIGN_ID) == 0) return;

        si_t::init(si5345_revb_registers, sizeof(si5345_revb_registers) / sizeof(si5345_revb_registers[0]));
        for(int i = 0; i < 8; i++) write(0x026B + i, DESIGN_ID[i]);

        wait_sysincal();
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

    void menu() {
        alt_u32 pn_base = (read(0x0003) << 8) | read(0x0002);
        if(pn_base != 0x5345) {
            printf("Invalid base part number: 0x%04X\n", pn_base);
            return;
        }

        while (1) {
            status();
            printf("\n");

            printf("si5345:\n");
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
            case 'r': {
                printf("si5345.read:\n");
                for(alt_u16 address = 0x0200; address < 0x0300; address++) {
                    printf("  [0x%02X] = 0x%02X\n", address, read(address));
                }
                break;
            }
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
        }
    }

};

#endif // __FE_SI5345_H__
