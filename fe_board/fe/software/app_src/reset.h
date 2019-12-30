
void menu_reset() {
    auto& reset_bypass = sc.ram->regs.fe.reset_bypass;

    while(1) {
        printf("\n");
        printf("[reset] -------- menu --------\n");

        printf("\n");
        printf("fe.reset_bypass = 0x%04X\n", reset_bypass);
	printf("fe.reset_bypass: run state=");
        switch((reset_bypass >> 16) & 0x3f) {
		case 1<<0: printf("RUN_STATE_IDLE\n"); break;
		case 1<<1: printf("RUN_STATE_PREP\n"); break;
		case 1<<2: printf("RUN_STATE_SYNC\n"); break;
		case 1<<3: printf("RUN_STATE_RUNNING\n"); break;
		case 1<<4: printf("RUN_STATE_TERMINATING\n"); break;
		case 1<<5: printf("RUN_STATE_LINK_TEST\n"); break;
		case 1<<6: printf("RUN_STATE_SYNC_TEST\n"); break;
		case 1<<7: printf("RUN_STATE_RESET\n"); break;
		case 1<<8: printf("RUN_STATE_OUT_OF_DAQ\n"); break;
		default:   printf("UNKNOWN\n"); break;
	}
        if(reset_bypass & 0x03ff != 0){
		printf("Bypass enabled. Payload: %4.4x\n",sc.ram->regs.fe.reset_bypass_payload);
	}


        printf("\n");
        printf("  [0] => use genesis\n");
        printf("  [1] => run_prep\n");
        printf("  [2] => sync\n");
        printf("  [3] => start run\n");
        printf("  [4] => end run\n");
        printf("  [5] => abort run\n");
        printf("  [6] => start reset\n");
        printf("  [7] => stop reset\n");
        printf("  [8] => start link test\n");
        printf("  [9] => stop link test\n");

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
        case '8':
            reset_bypass = 0x0120;
            break;
        case '9':
            reset_bypass = 0x0121;
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
