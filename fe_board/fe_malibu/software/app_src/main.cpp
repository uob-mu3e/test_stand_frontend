
#include "../include/base.h"
#include "../include/xcvr.h"

#include "../../../fe/software/app_src/si5345.h"
si5345_t si5345 { 0 }; // spi_slave = 0

#include "../../../fe/software/app_src/sc_ram.h"
volatile sc_ram_t* sc = (sc_ram_t*)AVM_SC_BASE;

#include "../../../fe/software/app_src/malibu.h"
#include "sc.h"
#include "../../../fe/software/app_src/mscb_user.h"
#include "../../../fe/software/app_src/reset.h"

alt_u32 alarm_callback(void*) {
    sc_callback();

    return 10;
}

int main() {
    base_init();

    si5345.init();

    alt_alarm alarm;
    int err = alt_alarm_start(&alarm, 0, alarm_callback, nullptr);
    if(err) {
        printf("ERROR: alt_alarm_start => %d\n", err);
    }

    while (1) {
        printf("\n");
        printf("[fe_malibu] -------- menu --------\n");

        printf("\n");
        printf("  [1] => xcvr qsfp\n");
        printf("  [2] => malibu\n");
        printf("  [3] => sc\n");
        printf("  [4] => xcvr pod\n");
        printf("  [5] => si5345\n");
        printf("  [6] => mscb (exit by reset only)\n");
        printf("  [7] => reset system\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            menu_xcvr((alt_u32*)(AVM_QSFP_BASE | ALT_CPU_DCACHE_BYPASS_MASK));
            break;
        case '2':
            menu_malibu();
            break;
        case '3':
            menu_sc();
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
