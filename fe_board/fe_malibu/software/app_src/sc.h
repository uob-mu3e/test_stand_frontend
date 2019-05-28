
void menu_sc() {
    volatile alt_u32* data = (alt_u32*)AVM_SC_BASE;

    while(1) {
        printf("  [r] => read sc ram\n");
        printf("  [w] => write sc ram\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case 'r':
            for(int i = 0; i < 256; i++) {
                printf("[0x%04X] = 0x%08X | %s\n", i, data[i], i == data[i] ? "OK" : "");
            }
            break;
        case 'w':
            for(int i = 0; i < 256; i++) {
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
