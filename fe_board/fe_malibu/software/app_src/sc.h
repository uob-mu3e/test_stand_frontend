
void menu_sc() {
    while(1) {
        printf("  [0] => read sc ram\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            for(int i = 0; i < 256; i++) {
                alt_u32* data = (alt_u32*)AVM_SC_BASE;
                printf("0x%08X\n", data[i]);
            }
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
