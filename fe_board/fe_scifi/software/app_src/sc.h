
void sc_callback(volatile alt_u32* data) {
    // command (upper 16 bits) and length (lower 16 bits)
    alt_u32 d0 = data[0];
    if(d0 == 0) return;
    printf("[sc_callback] data[0] = 0x%08X\n", d0);

    alt_u32 command = d0 >> 16;
    alt_u32 n = d0 & 0xFFFF;

    alt_u32 offset = data[1] & 0xFFFF;
    if(!(offset >= 16 && offset + n < AVM_SC_SPAN / 4)) {
        data[0] = 0;
        return;
    }

    switch(command) {
    case 0x0101:
        Malibu_Powerup();
        break;
    case 0x0102:
        Malibu_Powerdown();
        break;
    case 0x0103:
        PowerUpASIC(0);
        break;
    case 0xFFFF:
        for(alt_u32 i = 0; i < n; i++) {
            printf("[sc_callback] data[0x%04X] = 0x%08X\n", i, data[offset + i]);
        }
        break;
    default:
        printf("[sc_callback] unknown command\n");
    }

    data[0] = 0;
}

void menu_sc(volatile alt_u32* data) {
    while(1) {
        printf("  [r] => read sc ram\n");
        printf("  [w] => write sc ram\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case 'r':
            for(int i = 0; i < 32; i++) {
                printf("[0x%04X] = 0x%08X\n", i, data[i]);
            }
            break;
        case 'w':
            for(int i = 0; i < 32; i++) {
                data[i] = i;
            }
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
