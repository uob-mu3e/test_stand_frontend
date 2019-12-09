
#include "../include/base.h"
#include "../include/xcvr.h"

#include "../../../fe/software/app_src/si5345.h"
si5345_t si5345 { SPI_SI_BASE, 0 };

#include "../../../fe/software/app_src/sc.h"
sc_t sc;

#include "../../../fe/software/app_src/mscb_user.h"
mscb_t mscb;
#include "../../../fe/software/app_src/reset.h"

#include "scifi_module.h"
scifi_module_t scifi_module(&sc,2);

//definition of callback function for slow control packets
alt_u16 sc_t::callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
    return scifi_module.callback(cmd,data,n);
}


int main() {
    base_init();

    si5345.init();
    mscb.init();
    sc.init();

    scifi_module.RSTSKWctrl_Clear();

    while (1) {
        printf("\n");
        printf("[fe_scifi] -------- menu --------\n");

        printf("\n");
        printf("  [1] => xcvr qsfp\n");
        printf("  [2] => scifi\n");
        printf("  [3] => sc\n");
        printf("  [4] => xcvr pod\n");
        printf("  [5] => si5345\n");
        printf("  [6] => mscb\n");
        printf("  [7] => reset system\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            menu_xcvr((alt_u32*)(AVM_QSFP_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case '2':
            scifi_module.menu();
            break;
        case '3':
            sc.menu();
            break;
        case '4':
            menu_xcvr((alt_u32*)(AVM_POD_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case '5':
            si5345.menu();
            break;
        case '6':
            mscb_main();
            break;
        case '7':
            menu_reset();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
