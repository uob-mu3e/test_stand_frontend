
void sc_callback() {
    alt_u32 d0 = sc->data[0];
    if(d0 == 0) return;
    printf("[sc_callback] data[0] = 0x%08X\n", d0);

    // command (upper 16 bits) and buffer length (lower 16 bits)
    alt_u32 command = d0 >> 16;
    alt_u32 n = d0 & 0xFFFF;

    // offset to buffer
    alt_u32 offset = sc->data[1] & 0xFFFF;
    if(!(offset >= 0 && offset + n < AVM_SC_SPAN / 4)) {
        // out of bounds
        sc->data[0] = 0;
        return;
    }

    switch(command) {
    case 0x0101:
        malibu.powerup();
        break;
    case 0x0102:
        malibu.powerdown();
        break;
    case 0x0103:
        malibu.stic_configure(0, stic3_config_ALL_OFF);
        break;
    case 0x0104:
        malibu.stic_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
        break;
    case 0xFFFF:
        for(alt_u32 i = 0; i < n; i++) {
            printf("[sc_callback] data[0x%04X] = 0x%08X\n", i, sc->data[offset + i]);
        }
        break;
    default:
        printf("[sc_callback] unknown command\n");
    }

    sc->data[0] = 0;
}

void menu_sc() {
    while(1) {
        printf("  [r] => read sc ram\n");
        printf("  [w] => write sc ram\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case 'r':
            for(int i = 0; i < 32; i++) {
                printf("[0x%04X] = 0x%08X\n", i, sc->data[i]);
            }
            break;
        case 'w':
            for(int i = 0; i < 32; i++) {
                sc->data[i] = i;
            }
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
