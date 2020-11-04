#ifndef __MUPIX_SC_H__
#define __MUPIX_SC_H__

#include "../../../fe/software/app_src/sc_ram.h"

#include <sys/alt_irq.h>
//#include "default_mupix_dacs.h"

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

alt_u32 get_n_downto_m_bits(alt_u32 n, alt_u32 m, volatile alt_u32 bits)
{
   
   alt_u32 r = 0;
   for (alt_u32 i=m; i<=n; i++)
       r |= 1 << i;

   return r & bits;
}

void menu_lvds(volatile sc_ram_t* ram) {
    alt_u32 receiver_state;
    alt_u32 receiver_pll_state;
    alt_u32 run_counter;
    alt_u32 error_counter;
    alt_u32 current_receiver = 0; 
    while (1) {
        char cmd;
        if(read(uart, &cmd, 1) > 0) switch(cmd) {
        case '0': case '1': case '2': case '3': // select receiver
            current_receiver = (cmd - '0') & 0xFF;
            break;
        case 'r': // reset
            ram->data[0xFF8F] = 0x00000000; // reset lvds links
            break;
        case 'q':
            return;
        case 'd':
            for (int i = 0; i<68; i++) {
                ram->data[0xFF93] = i;
                alt_u32 d = ram->data[0xFF94];
                printf("[0x%04X] = 0x%08X\n", (alt_u32)&ram->data[0xFF94] & 0xFFFF, d);
                
                
                print_data((volatile alt_u32*)&ram->data[0xFF94], 1);
            }
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }

        // read lvds receiver status of current receiver
        if (current_receiver < 16) {
            ram->data[0xFF93] = 0;
        } else {
            ram->data[0xFF93] = 1;
        }
        receiver_state = ram->data[0xFF94];
        
        // read pll status
        ram->data[0xFF93] = 2;
        receiver_pll_state = ram->data[0xFF94];
        
        // read lvds runcounter
        ram->data[0xFF93] = 4 + current_receiver;
        run_counter = ram->data[0xFF94];
        
        // read lvds errorcounter
        ram->data[0xFF93] = 36 + current_receiver;
        error_counter = ram->data[0xFF94];
        

        printf("Receiver 0x%02X\n", current_receiver);
        printf("                PLL_LOC RUN_CNT ERR_CNT\n");
        printf("                %02X 0x%08X 0x%08X\n", get_n_downto_m_bits(0, 1, receiver_pll_state), run_counter, error_counter);
        //printf("  tx    :   %s  0x%02X 0x%04X 0x%04X\n",
        //    xcvr[0x10] == 0x00 && xcvr[0x11] == 0x0001 && xcvr[0x12] == 0x0000 ? "OK" : "  ",
        //    xcvr[0x10], xcvr[0x11], xcvr[0x12]
        //);
        //printf("  rx    :   %s  0x%02X 0x%04X 0x%04X\n",
        //    xcvr[0x20] == 0x00 && (xcvr[0x21] & 0x1005) == 0x1005 && xcvr[0x22] == 0x0000 ? "OK" : "  ",
        //    xcvr[0x20], xcvr[0x21], xcvr[0x22]
        //);
        //printf("        :   LoL_cnt = %d, err_cnt = %d\n", xcvr[0x23], xcvr[0x24]);
         //printf("  data  :   0x%08X / 0x%01X\n", xcvr[0x2A], xcvr[0x2B]);
        printf("\n");

        usleep(200000);
    }
}

void menu_mupix() {
    volatile sc_ram_t* ram = (sc_ram_t*)AVM_SC_BASE;
    auto& regs = sc.ram->regs.mupix;
    int i = 65409; // 0xFF81
    alt_u32 receiver_state;
    alt_u32 receiver_pll_state;
    

    
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
        printf("  [r] => reset lvds links\n");
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
            //ram->data[0xFF83] = 0x15001380;
            //ram->data[0xFF84] = 0x1cf00000;
            //ram->data[0xFF83] = 0x12dc12dc;
            //ram->data[0xFF84] = 0x2f270000;
            ram->data[0xFF83] = 0x4b704b70;
            ram->data[0xFF84] = 0xbc9c0000;
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
        case 'r':
            ram->data[0xFF8F] = 0x00000000;
        break;
        case 'm':
           // ram->data[0xFF93] = 0;
           // for (int i = 0; i<16; i++) {
           //     receiver_state = ram->data[0xFF94];
           //     
           //    for (int j = 0; j < 32; ++j) {
           //         volatile alt_u32* bits[j] = (hex >> j) & 1;
           //     }
           //     
           //     printf("Receiver %08X\n State %08X\n", i, d);
           // }
           // print_data((volatile alt_u32*)&ram->data[0xFF94], 1);
           // ram->data[0xFF93] = 0;
            //printf(ram->data[0xFF93]);
            
            // print lvds receiver 0-16 status
            /*ram->data[0xFF93] = 0;
            receiver_state = ram->data[0xFF94];
            for (int i = 0; i<16; i++) {
                printf("Receiver 0x%02X RX_STATE: %02X\n", i, get_n_downto_m_bits(2*i + 2, 2*i + 0, receiver_state));
            }
            
            // print lvds receiver 17-32 status
            ram->data[0xFF93] = 1;
            receiver_state = ram->data[0xFF94];
            for (int i = 0; i<16; i++) {
                printf("Receiver 0x%02X RX_STATE: %02X\n", i+16, get_n_downto_m_bits(2*i + 2, 2*i + 0, receiver_state));
            }
            
            // print lvds receiver pll status
            ram->data[0xFF93] = 1;
            receiver_pll_state = ram->data[0xFF94];
            printf("PLL_LOCKED: %01X\n", get_n_downto_m_bits(1 , 0, receiver_pll_state));
            
            // print lvds receiver runcounter
            ram->data[0xFF93] = 1;
            
            for (int i = 0; i<68; i++) {
                ram->data[0xFF93] = i;
                print_data((volatile alt_u32*)&ram->data[0xFF94], 1);
            }*/
            menu_lvds(ram);
        break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

#endif // __MUPIX_SC_H__
