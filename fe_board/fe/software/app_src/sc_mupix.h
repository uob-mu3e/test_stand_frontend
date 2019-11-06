#ifndef __MUPIX_SC_H__
#define __MUPIX_SC_H__

#include "sc_ram.h"

#include <sys/alt_irq.h>
#include "default_mupix_dacs.h"

static
void print_data(volatile alt_u32* data, int n) {
    for(int i = 0; i < n; i++) {
        alt_u32 d = data[i];
        printf("[0x%04X] = 0x%08X\n", (alt_u32)&data[i] & 0xFFFF, d);

        int k = 1;
        while(i+k < n && data[i+k] == d) k++;
        if(k > 2) {
            printf("[0x....]\n");
            i += k - 2;
        }
    }
}

void menu_mupix() {
    volatile sc_ram_t* ram = (sc_ram_t*)AVM_SC_BASE;
    auto& regs = sc.ram->regs.mupix;
    int i = 65409; // 0xFF81
    

    
    while(1) {
        printf("  [0] => read th_pix reg\n");
        printf("  [1] => write th_pix step 1\n");
        printf("  [2] => write th_pix step 2\n");
        printf("  [3] => write th_pix step 3\n");
        printf("  [4] => write th_pix\n");
        printf("  [5] => stop spi master\n");
        printf("  [6] => start spi master\n");
        printf("  [7] => read data and regs\n");
        printf("  [8/9] => write chip dacs\n");
        printf("  [d] => set default chip dacs\n");
        printf("  [m] => monitor lvds\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            printf("data");
        break;
        case '1':
            ram->data[0xFF81] = 0x15001380;
        break;
        case '2':
            ram->data[0xFF81] = 0x1cf00000;
        break;
        case '3':
            ram->data[0xFF80] = 0xAAAAAAAA;
        break;
        case '4':
            ram->data[0xFF83] = 0x15001380;
            ram->data[0xFF84] = 0x1cf00000;
            ram->data[0xFF8C] = 0x00000001;
            //printf("0x%08X\n", ram->data[0xFF86]);
            //printf("0x%08X\n", ram->data[0xFF87]);
            ram->data[0xFF8C] = 0x00000000;
        break;
        case '5':
            ram->data[0xFF8C] = 0x00000000;
        break;
        case '6':
            ram->data[0xFF8C] = 0x00000001;
        break;
        case '7':
            printf("\n");
            printf("DATA:\n");
            print_data(ram->data, sizeof(ram->data) / sizeof(alt_u32));
            printf("\n");
            printf("REGS:\n");
            print_data((volatile alt_u32*)&ram->regs, sizeof(ram->regs) / sizeof(alt_u32));    
        break;
        case '8':
            ram->data[0xFF8D] = 0x005e0003;
            for (int i = 0; i<93; i++) {
                ram->data[0xFF8D] = 0x00000000;
            }
                ram->data[0xFF8E] = 0x00100001;
        break;
        case '9':
            ram->data[0xFF8D] = 0x005e0003;
            for (int i = 0; i<93; i++) {
                ram->data[0xFF8D] = 0xAAAAAAAA;
            }
                ram->data[0xFF8E] = 0x00100001;
        break;
        case 'd':
            ram->data[0xFF8D] = 0x005e0003;
            for(int i = 0; i < sizeof(default_mupix_dacs); i++) {
                ram->data[0xFF8D] = default_mupix_dacs[i];
            }
            ram->data[0xFF8E] = 0x00100001;
        break;
        case 'm ':
            for (int i = 0; i<68; i++) {
                ram->data[0xFF93] = i;
 //               print_data((volatile alt_u32*)&ram->data[0xFF94], 1);
            }
        break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

#endif // __MUPIX_SC_H__
