
volatile alt_u32* sc_data = (alt_u32*)AVM_SC_BASE;

void sc_callback() {
    alt_u32 n = sc_data[0];
    if(n == 0) return;
    volatile alt_u32* data = sc_data + sc_data[1];
    if(!(sc_data <= data && data + n < sc_data + AVM_SC_SPAN / 4)) return;

    for(int i = 0; i < n; i++) {
        printf("[0x%04X] = 0x%08X\n", i, data[i]);
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
            for(int i = 0; i < 256; i++) {
                printf("[0x%04X] = 0x%08X\n", i, sc_data[i]);
            }
            break;
        case 'w':
            for(int i = 0; i < 256; i++) {
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
