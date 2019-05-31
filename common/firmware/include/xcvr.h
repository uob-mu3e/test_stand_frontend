
void menu_xcvr(volatile alt_u32* xcvr) {
    while (1) {
        char cmd;
        if(read(uart, &cmd, 1) > 0) switch(cmd) {
        case '0': case '1': case '2': case '3': // select channel
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
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }

        int ch = xcvr[0x00] & 0xFF;

        printf("xcvr[A].ch[0x%02X], lpbk = %d\n", ch, xcvr[0x2F]);
        printf("                R_DA S_LS_R E__FDE\n");
        printf("  tx    :   %s  0x%02X 0x%04X 0x%04X\n",
            xcvr[0x10] == 0x00 && xcvr[0x11] == 0x0001 && xcvr[0x12] == 0x0000 ? "OK" : "  ",
            xcvr[0x10], xcvr[0x11], xcvr[0x12]
        );
        printf("  rx    :   %s  0x%02X 0x%04X 0x%04X\n",
            xcvr[0x20] == 0x00 && xcvr[0x21] == 0x1F07 && xcvr[0x22] == 0x0000 ? "OK" : "  ",
            xcvr[0x20], xcvr[0x21], xcvr[0x22]
        );
        printf("        :   LoL_cnt = %d, err_cnt = %d\n", xcvr[0x23], xcvr[0x24]);
        printf("  data  :   0x%08X / 0x%01X\n", xcvr[0x2A], xcvr[0x2B]);
        printf("\n");

        usleep(200000);
    }
}
