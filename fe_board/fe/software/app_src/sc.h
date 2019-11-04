#ifndef __FE_SC_H__
#define __FE_SC_H__

#include "sc_ram.h"

#include <sys/alt_irq.h>

struct sc_t {
    volatile sc_ram_t* ram = (sc_ram_t*)AVM_SC_BASE;

    alt_alarm alarm;

    void init() {
        printf("[sc] init\n");

        if(int err = alt_ic_isr_register(0, 16, callback, this, nullptr)) {
            printf("[sc] ERROR: alt_ic_isr_register => %d\n", err);
        }
    }

    void callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n);

    void callback() {
        alt_u32 cmdlen = ram->regs.fe.cmdlen;
        if(cmdlen == 0) return;

        // command (upper 16 bits) and data length (lower 16 bits)
        alt_u32 cmd = cmdlen >> 16;
        alt_u32 n = cmdlen & 0xFFFF;

        printf("[sc::callback] cmd = 0x%04X, n = 0x%04X\n", cmd, n);

        // data offset
        alt_u32 offset = ram->regs.fe.offset & 0xFFFF;

        if(!(offset >= 0 && offset + n <= sizeof(sc_ram_t::data) / sizeof(sc_ram_t::data[0]))) {
            printf("[sc::callback] ERROR: ...\n");
        }
        else {
            auto data = n > 0 ? (ram->data + offset) : nullptr;
            callback(cmd, data, n);
        }

        ram->regs.fe.cmdlen = 0;
    }

    static
    void callback(void* context) {
        ((sc_t*)context)->callback();
    }

    static
    void print_data(volatile alt_u32* data, int n) {
        for(int i = 0; i < n; i++) {
            alt_u32 d = data[i];
            printf("[0x%04X] = 0x%08X\n", ((alt_u32)&data[i] / 4) & 0xFFFF, d);

            int k = 1;
            while(i+k < n && data[i+k] == d) k++;
            if(k > 2) {
                printf("[0x....]\n");
                i += k - 2;
            }
        }
    }

    void menu() {
        while(1) {
            printf("\n");
            printf("[sc] -------- menu --------\n");

            printf("\n");
            printf("  [r] => read data and regs\n");
            printf("  [w] => write [i] = i for i < 16\n");
            printf("  [i] => test cmdlen irq\n");
            printf("  [q] => exit\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 'r':
                printf("\n");
                printf("DATA:\n");
                print_data(ram->data, sizeof(ram->data) / sizeof(alt_u32));
                printf("\n");
                printf("REGS:\n");
                print_data((volatile alt_u32*)&ram->regs, sizeof(ram->regs) / sizeof(alt_u32));
                break;
            case 'w':
                for(int i = 0; i < 16; i++) {
                    ram->data[i] = i;
                }
                break;
            case 't':
                ram->regs.fe.cmdlen = 0xffff0000;
                break;
            case 'q':
                return;
            default:
                printf("invalid command: '%c'\n", cmd);
            }
        }
    }
};

#endif // __FE_SC_H__