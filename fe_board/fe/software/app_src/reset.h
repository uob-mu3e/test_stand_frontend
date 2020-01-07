#include "stdlib.h"

void menu_reset() {
    auto& reset_bypass = sc.ram->regs.fe.reset_bypass;
    auto& reset_bypass_payload = sc.ram->regs.fe.reset_bypass_payload;
    alt_u32 payload = 0x0;
    char str[2] = {0};

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
/*
	printf("fe.reset_bypass: transition_command=");
        switch((reset_bypass) & 0xffff) {
		case 0x0000 : printf("USE GENESIS\n"); break;
		case 0x0110 : printf("RUN_PREP\n"); break;
		case 0x0111 : printf("SYNC\n"); break;
		case 0x0112 : printf("START_RUN\n"); break;
		case 0x0113 : printf("END_RUN\n"); break;
		case 0x0114 : printf("ABORT_RUN\n"); break;
		case 0x0130 : printf("START_RESET\n"); break;
		case 0x0131 : printf("STOP_RESET\n"); break;
		case 0x0120 : printf("START_LINKTEST\n"); break;
		case 0x0121 : printf("STOP_LINKTEST\n"); break;
		default:   printf("UNKNOWN\n"); break;
	}
*/
	printf("fe.reset_bypass: command payload=0x%8.8x (%d)\n",sc.ram->regs.fe.reset_bypass_payload,sc.ram->regs.fe.reset_bypass_payload);


        printf("\n");
        //printf("  [0] => use genesis\n");
        printf("  [1] => run_prep\n");
        printf("  [2] => sync\n");
        printf("  [3] => start run\n");
        printf("  [4] => end run\n");
        printf("  [5] => abort run\n");
        printf("  [6] => start reset\n");
        printf("  [7] => stop reset\n");
        printf("  [8] => start link test\n");
        printf("  [9] => stop link test\n");

        printf("  [p] => set payload   payload: 0x%08x\n", payload);

        printf("Select entry ...\n");
        reset_bypass = 0x0000;
        char cmd = wait_key();
        switch(cmd) {
        //case '0':
        //    reset_bypass = 0x0000;
        //    break;
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
        case 'p':
            payload = 0x0;
            printf("Enter payload in hex: ");

            for(int i = 0; i<8; i++){
                printf("payload: 0x%08x\n", payload);
                str[0] = wait_key();
                payload = payload*16+strtol(str,NULL,16);
            }

            printf("setting payload to 0x%08x\n", payload);
            reset_bypass_payload = payload;
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
        reset_bypass = 0x0000;
    }
}
