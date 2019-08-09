
void menu_reset() {
    auto& reset_bypass = sc->regs.fe.reset_bypass;

    while(1) {
        printf("fe.reset_bypass = 0x04X\n", reset_bypass);

        printf("  [0] => use genesis\n");
        printf("  [1] => run_prep\n");
        printf("  [2] => sync\n");
        printf("  [3] => start run\n");
        printf("  [4] => end run\n");
        printf("  [5] => abort run\n");
        printf("  [6] => start reset\n");
        printf("  [7] => stop reset\n");
        printf("Select entry ...\n");
        char cmd = wait_key();
        
        switch(cmd) {
        case '0':
            reset_bypass = 0x0000;
            break;
        case '1':
            reset_bypass = 0x0110;
            break;
        case '2':
            reset_bypass = 0x0111;
            break;
        case '3':
            reset_bypass = 0x0112;
            break;
        case '4':
            reset_bypass = 0x0113;
            break;
        case '5':
            reset_bypass = 0x0114;
            break;
        case '6':
            reset_bypass = 0x0130;
            break;
        case '7':
            reset_bypass = 0x0131;
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
