/*
 * author : Alexandr Kozlinskiy
 * date : 2019
 */

#ifndef __UTIL_XCVR_H__
#define __UTIL_XCVR_H__

struct xcvr_block_t {
    static const alt_u32 XCVR_SPAN = 0x10000;

    volatile alt_u32* base;
    alt_u32 span;

    char id = 'A';

    explicit
    xcvr_block_t(volatile alt_u32* base, alt_u32 span = 0) : base(base), span(span) {
    }

    void menu() {
        while (1) {
            volatile alt_u32* xcvr = base + (id - 'A') * XCVR_SPAN / 4;
            if(menu(xcvr) != 0) return;
            if(span == 0) status_table(xcvr);
            else {
                for(int i = 0; i < span; i += XCVR_SPAN) status_table(base + i / 4);
            }
            printf("\n");
            status(xcvr);
            printf("\n");
            usleep(200000);
        }
    }

    int menu(volatile alt_u32* xcvr) {
        printf("\n");
        printf("XCVR 0x%08X:\n", base);
        printf("  [[,]] => select group\n");
        printf("  [0-7] => select channel\n");
        printf("  [r,R] => reset\n");
        printf("  [l,L] => loopback\n");
        printf("\n");

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
        case 'R': { // reset (all channels)
            alt_u32 ch_prev = xcvr[0x00];
            for(alt_u32 ch = 0; ch < xcvr[0x01]; ch++) {
                xcvr[0x00] = ch;
                xcvr[0x10] = 0x01; // tx
                xcvr[0x20] = 0x01; // rx
            }
            xcvr[0x00] = ch_prev;
            break;
        }
        case 'l': // toggle loopback
            xcvr[0x2F] ^= 0x01;
            break;
        case 'L': { // toggle loopback (all channels)
            alt_u32 ch_prev = xcvr[0x00];
            for(alt_u32 ch = 0; ch < xcvr[0x01]; ch++) {
                xcvr[0x00] = ch;
                xcvr[0x2F] ^= 0x01;
            }
            xcvr[0x00] = ch_prev;
            break;
        }
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
    }

    void status_table(volatile alt_u32* xcvr) {
        alt_u32 ch_prev = xcvr[0x00];
        if(ch_prev > 0xFF) return;

        printf("-- xcvr[0x%08X] --\n", xcvr);

        printf("  ch");
        for(alt_u32 ch = 0; ch < xcvr[0x01]; ch++) {
            xcvr[0x00] = ch;
            int l = xcvr[0x2F];
            int e = xcvr[0x23] > 0 || xcvr[0x24] > 0;
            int E = xcvr[0x23] == 0xFF && xcvr[0x24] == 0xFFFF;
            int _ = xcvr[0x21] != 0;
            int S = xcvr[0x20] == 0 && (xcvr[0x21] & 0x1005) == 0x1005 && xcvr[0x22] == 0;
            printf(" | %d [%s%s%s...]", xcvr[0x00], S ? "S" : _ ? "_" : ".", l ? "l" : ".", E ? "E" : e ? "e" : ".");
        }
        printf(" |\n");

        printf("data");
        for(alt_u32 ch = 0; ch < xcvr[0x01]; ch++) {
            xcvr[0x00] = ch;
            printf(" | %08X/%01X", xcvr[0x2A], xcvr[0x2B]);
        }
        printf(" |\n");

        xcvr[0x00] = ch_prev;
    }
};

void menu_xcvr(volatile alt_u32* base, alt_u32 span = 0) {
    xcvr_block_t xcvr_block(base, span);
    xcvr_block.menu();
}

void menu_xcvr(alt_u32 base, alt_u32 span = 0) {
    menu_xcvr((alt_u32*)(base | ALT_CPU_DCACHE_BYPASS_MASK), span);
}

#endif // __UTIL_XCVR_H__
