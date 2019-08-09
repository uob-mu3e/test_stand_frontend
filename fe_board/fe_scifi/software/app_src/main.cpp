
#include "../include/base.h"
#include "../include/xcvr.h"

#include "../include/i2c.h"
i2c_t i2c;

#include "../../../fe/software/app_src/si5345.h"
si5345_t si5345 { 4 }; // spi_slave = 4

#include "../../../fe/software/app_src/sc_ram.h"
volatile sc_ram_t* sc = (sc_ram_t*)AVM_SC_BASE;

#include "malibu.h"
#include "sc.h"
#include "../../../fe/software/app_src/mscb_user.h"

alt_u32 alarm_callback(void*) {
    sc_callback((alt_u32*)AVM_SC_BASE);

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
        printf("fe_scifi:\n");
        printf("  [1] => xcvr qsfp\n");
        printf("  [2] => malibu\n");
        printf("  [3] => sc\n");
        printf("  [4] => xcvr pod\n");
        printf("  [5] => si5345\n");
        printf("  [6] => mscb (exit by reset only)\n");

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
            menu_sc((alt_u32*)AVM_SC_BASE);
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
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
