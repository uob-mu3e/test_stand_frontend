
volatile alt_u32* sc_data = (alt_u32*)AVM_SC_BASE;

void sc_callback() {
    // command (upper 16 bits) and length (lower 16 bits)
    alt_u32 n = sc_data[0];
    if(n == 0) return;
    printf("[sc_callback] sc[0] = 0x%08X\n", n);

    alt_u32 cmd = n >> 16; n &= 0xFFFF;
    volatile alt_u32* data = sc_data + sc_data[1];
    if(!(sc_data <= data && data + n < sc_data + AVM_SC_SPAN / 4)) {
        sc_data[0] = 0;
        return;
    }

    switch(cmd) {
    case 0x0101:
        Malibu_Powerup();
        break;
    case 0x0102:
        Malibu_Powerdown();
        break;
    case 0x0103:
        PowerUpASIC(0);
        break;
    default:
        for(int i = 0; i < n; i++) {
            printf("[0x%04X] = 0x%08X\n", i, data[i]);
        }
    }

    sc_data[0] = 0;
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
                printf("[0x%04X] = 0x%08X\n", i, sc_data[i]);
            }
            break;
        case 'w':
            for(int i = 0; i < 32; i++) {
                sc_data[i] = i;
            }
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
