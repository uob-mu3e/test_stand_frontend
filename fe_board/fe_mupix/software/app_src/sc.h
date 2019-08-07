
#include <sys/alt_stdio.h>

void menu_sc(volatile alt_u32* data) {
    int add = 0;
    while(1) {
        
        printf("  [r] => read sc ram\n");
        printf("  [w] => write sc ram\n");
        printf("  [p] => pixel test\n");
        printf("  [t] => set th\n");
        printf("  [m] => set th\n");
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
        case 'p':
            for(int i = 1; i < 10; i++) {
                data[i] = i;
            }
            data[add + 10] = 0xABAD1DEA;
            data[add] = 0xBADC0DED;
            add = add + 10;
            printf("Add: '%i'\n", add);
            break;
        case 't':
            data[4] = 0x1cf00000;
            data[3] = 0x15001380;
            //data[1] = 0x12341380;
            //data[0] = 0x00000001;
            break;
        case 'm':
            data[6] = 0xAAAAAAAA;
            data[5] = 0xBBBBBBBB;
            data[4] = 0x1cf00000;
            //data[1] = 0x15001380;
            data[3] = 0x15001666;
            data[2] = 0x00000001;
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}
