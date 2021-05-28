/*
 * author : Alexandr Kozlinskiy
 * date : 2019
 */

#ifndef __UTIL_XCVR_H__
#define __UTIL_XCVR_H__

struct xcvr_block_t {
    volatile alt_u32* base;

    char id;

    explicit
    xcvr_block_t(volatile alt_u32* base, char id = 'A') : base(base), id(id) {
        if(id < 'A') id = 'A';
        if(id > 'Z') id = 'Z';
    }

    void menu() {
        while (1) {
            volatile alt_u32* xcvr = base + (id - 'A') * 0x00010000/4;
            if(menu(xcvr) != 0) return;
            status(xcvr);
            usleep(200000);
        }
    }

    int menu(volatile alt_u32* xcvr) {
        char cmd;
        if(read(uart, &cmd, 1) > 0) switch(cmd) {
        case '[':
            if(id > 'A') --id;
            break;
        case ']':
            if(id < 'Z') ++id;
            break;
        case '0': case '1': case '2': case '3': // select channel
        case '4': case '5': case '6': case '7':
            xcvr[0x00] = (cmd - '0') & 0xFF;
            break;
        case 'r': // reset
            xcvr[0x10] = 0x01; // tx
            xcvr[0x20] = 0x01; // rx
            break;
        case 'l': // loopback
            xcvr[0x2F] ^= 0x01;
            break;
        case '?':
            wait_key();
            break;
        case 'q':
            return -1;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
        return 0;
    }

    void status(volatile alt_u32* xcvr) {
        int ch = xcvr[0x00] & 0xFF;

        printf("QSFP+-=][, selCH=0-7, reset=r, loopback=l\n");
        printf("xcvr[%c].ch[0x%02X], lpbk = %d\n", id, ch, xcvr[0x2F]);
        printf("                R_DA S_LS_R E__FDE\n");
        printf("  tx    :   %s  0x%02X 0x%04X 0x%04X\n",
            xcvr[0x10] == 0x00 && xcvr[0x11] == 0x0001 && xcvr[0x12] == 0x0000 ? "OK" : "  ",
            xcvr[0x10], xcvr[0x11], xcvr[0x12]
        );
        printf("  rx    :   %s  0x%02X 0x%04X 0x%04X\n",
            xcvr[0x20] == 0x00 && (xcvr[0x21] & 0x1005) == 0x1005 && xcvr[0x22] == 0x0000 ? "OK" : "  ",
            xcvr[0x20], xcvr[0x21], xcvr[0x22]
        );
        printf("        :   LoL_cnt = %d, err_cnt = %d\n", xcvr[0x23], xcvr[0x24]);
        printf("  data  :   0x%08X / 0x%01X\n", xcvr[0x2A], xcvr[0x2B]);

        if(xcvr[0x25] != 0xCCCCCCCC && xcvr[0x26] != 0xCCCCCCCC) {
            printf("  mW/C  :   %i / %i\n", xcvr[0x25], xcvr[0x26] / 10000);
        }

        printf("\n");
    }
};

void menu_xcvr(volatile alt_u32* base, char ID = 'A') {
    xcvr_block_t xcvr_block(base, ID);
    xcvr_block.menu();
}

#endif // __UTIL_XCVR_H__
