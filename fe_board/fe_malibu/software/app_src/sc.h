
void sc_callback() {
    alt_u32 d0 = sc->data[0];
    if(d0 == 0) return;

    // command (upper 16 bits) and buffer length (lower 16 bits)
    alt_u32 cmd = d0 >> 16;
    alt_u32 n = d0 & 0xFFFF;
    printf("[sc_callback] cmd = 0x%04X, n = 0x%04X\n", cmd, n);

    // offset to buffer
    alt_u32 offset = sc->data[1] & 0xFFFF;
    if(!(offset >= 0 && offset + n <= AVM_SC_SPAN / 4)) {
        // out of bounds
        sc->data[0] = 0;
        return;
    }

    switch(cmd) {
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
    default:
        if((cmd & 0xFFF0) == 0x0110) {
            printf("try stic_configure\n");
            int stic = cmd & 0x000F;
            int ok = 1;
            for(int i = 0; i < sizeof(stic3_config_ALL_OFF) / sizeof(stic3_config_ALL_OFF[0]); i++) {
                alt_u8 b = ((alt_u8*)(sc->data + offset))[i];
                if(b != stic3_config_ALL_OFF[i]) ok = 0;
            }
            printf("ok = %d\n", ok);
            if(ok == 1) malibu.stic_configure(stic, ((alt_u8*)(sc->data + offset)));
        }
        else {
            printf("[sc_callback] unknown command\n");
        }
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
        case 't':
            for(int i = 0xFF40; i < 0xFF50; i++) {
                printf("[0x%04X] = 0x%08X\n", i, sc->data[i]);
            }
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
