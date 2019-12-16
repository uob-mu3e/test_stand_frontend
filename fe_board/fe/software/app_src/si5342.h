#ifndef __FE_SI5342_H__
#define __FE_SI5342_H__

#include "../include/si534x.h"

#include "si5342_revb_registers.h"

struct si5342_t : si534x_t {

    const char* DESIGN_ID = "feb.42.2";

    si5342_t(alt_u32 spi_base, alt_u32 spi_slave)
        : si534x_t(spi_base, spi_slave)
    {
    }

    void init() {
        char id[9];
        id[8] = '\0';
        for(int i = 0; i < 8; i++) id[i] = (char)read(0x026B + i);
        if(strcmp(id, DESIGN_ID) == 0) return;

        si_t::init(si5342_revb_registers, sizeof(si5342_revb_registers) / sizeof(si5342_revb_registers[0]));
        for(int i = 0; i < 8; i++) write(0x026B + i, DESIGN_ID[i]);

        wait_sysincal();
    }

    void menu() {
        alt_u32 pn_base = (read(0x0003) << 8) | read(0x0002);
        if(pn_base != 0x5342) {
            printf("Invalid base part number: 0x%04X\n", pn_base);
            return;
        }

        while (1) {
            status();
            printf("\n");

            printf("si5342:\n");
            printf("  [I] => init\n");
            printf("  [W] => write to NVM\n");
            printf("  [r] => read regs\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 'I':
                init();
                break;
            case 'W':
                nvm_write();
                return;
            case 'r':
                printf("si5342.read:\n");
                for(alt_u16 address = 0x0000; address < 0x0100; address++) {
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

#endif // __FE_SI5342_H__
