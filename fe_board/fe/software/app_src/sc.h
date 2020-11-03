#ifndef __FE_SC_H__
#define __FE_SC_H__

#include "sc_ram.h"

#include <sys/alt_irq.h>

#define FEB_REPLY_SUCCESS 0
#define FEB_REPLY_ERROR   1

#include "stdlib.h"

struct sc_t {
    volatile sc_ram_t* ram = (sc_ram_t*)AVM_SC_BASE;

    alt_alarm alarm;

    void init() {
        printf("[sc] init\n");

        if(int err = alt_ic_isr_register(0, 12, callback, this, nullptr)) {
            printf("[sc] ERROR: alt_ic_isr_register => %d\n", err);
        }
    }

    alt_u16 callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n);

    void callback() {
        alt_u32 cmdlen = ram->regs.fe.cmdlen;
        if(cmdlen == 0) return;

        // command (upper 16 bits) and data length (lower 16 bits)
        alt_u32 cmd = cmdlen >> 16;
        alt_u32 n = cmdlen & 0xFFFF;

        //printf("[sc::callback] cmd = 0x%04X, n = 0x%04X\n", cmd, n);

        // data offset
        alt_u32 offset = ram->regs.fe.offset & 0xFFFF;

        alt_u16 status = FEB_REPLY_ERROR;
        if(!(offset >= 0 && offset + n <= sizeof(sc_ram_t::data) / sizeof(sc_ram_t::data[0]))) {
            printf("[sc::callback] ERROR: ...\n");
        }
        else {
            auto data = n > 0 ? (ram->data + offset) : nullptr;
            status = callback(cmd, data, n);
            //printf("[sc::callback] status = 0x%04X\n", status);
        }

        // zero upper 16 bits of command register
        // lower 16 bits are used as status
        ram->regs.fe.cmdlen = 0xFFFF & status;
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
        alt_u32 feb_id = 0x0;
        char str[2] = {0};
        while(1) {
            printf("\n");
            printf("[sc] -------- menu --------\n");
            printf("ID: 0x%08x\n", ram->data[0xFFFB]);
            printf("\n");
            printf("  [r] => read data and regs\n");
            printf("  [w] => write [i] = i for i < 16\n");
            printf("  [t] => read fpga id\n");
            printf("  [f] => write fpga id\n");
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
                printf("FPGA ID: 0x%08X\n", ram->data[0xFFFB]);
                break;
            case 'f':
                feb_id = 0x0;
                printf("Enter feb id in hex: ");

                for(int i = 0; i<8; i++){
                    printf("payload: 0x%08x\n", feb_id);
                    str[0] = wait_key();
                    feb_id = feb_id*16+strtol(str,NULL,16);
                }

                printf("setting feb_id to 0x%08x\n", feb_id);
                ram->data[0xFFFB] = feb_id;
                break;
            case 'i':
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
