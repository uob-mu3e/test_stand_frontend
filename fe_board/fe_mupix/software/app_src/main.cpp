
#include "../include/base.h"
#include "../include/xcvr.h"

#include "sc.h"

#include "../../../fe/software/app_src/si5345.h"
si5345_t si5345 { 0 };

int main() {
    base_init();

    si5345.init();

    while (1) {
        printf("\n");
        printf("fe_mupix:\n");
        printf("  [1] => xcvr qsfp\n");
        printf("  [3] => sc\n");
        printf("  [4] => xcvr pod\n");
        printf("  [5] => si5345\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            menu_xcvr((alt_u32*)(AVM_QSFP_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case '3':
            menu_sc((alt_u32*)AVM_SC_BASE);
            break;
        case '4':
            menu_xcvr((alt_u32*)(AVM_POD_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case '5':
            si5345.menu();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
