
#include "../include/base.h"

int main() {
    base_init();

    while (1) {
        printf("  [1] => ...\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
