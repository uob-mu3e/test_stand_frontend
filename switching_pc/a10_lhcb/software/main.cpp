
#include "include/base.h"

#include "include/xcvr.h"
#include "include/a10/reconfig.h"

struct my_xcvr0_t : xcvr_block_t {
    reconfig_t reconfig;
    my_xcvr0_t() : xcvr_block_t((alt_u32*)(AVM_XCVR0_BASE | ALT_CPU_DCACHE_BYPASS_MASK)) {
    }
};
my_xcvr0_t xcvr0;

struct my_xcvr1_t : xcvr_block_t {
    reconfig_t reconfig;
    my_xcvr1_t() : xcvr_block_t((alt_u32*)(AVM_XCVR1_BASE | ALT_CPU_DCACHE_BYPASS_MASK)) {
    }
};
my_xcvr1_t xcvr1;

#include "include/i2c.h"
#include "include/si534x.h"

i2c_t i2c;

#include "Si5345-RevD-SI5345_1-Registers.h"
#include "Si5345-RevD-SI5345_2-Registers.h"
si534x_t si5345 { i2c.dev, 0x68 };

void clocks_menu() {
    while (1) {
        printf("\n");
        printf("[PCIe40] -------- menu --------\n");

        printf("\n");
        printf("  [1] => si5345_1\n");
        printf("  [2] => si5345_2\n");
        printf("  [m] => menu\n");
        printf("  [I] => init\n");
        printf("  [q] => exit\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '1':
            IOWR_ALTERA_AVALON_PIO_DATA(I2C_MASK_BASE, 1 << 1); // i2c_mask(1) <= '1'
            break;
        case '2':
            IOWR_ALTERA_AVALON_PIO_DATA(I2C_MASK_BASE, 1 << 2); // i2c_mask(2) <= '1'
            break;
        case 'm':
            si5345.menu();
            break;
        case 'I':
            IOWR_ALTERA_AVALON_PIO_DATA(I2C_MASK_BASE, 1 << 1);
            si5345.init(si5345_1_registers, sizeof(si5345_1_registers) / sizeof(si5345_1_registers[0]));
            IOWR_ALTERA_AVALON_PIO_DATA(I2C_MASK_BASE, 1 << 2);
            si5345.init(si5345_2_registers, sizeof(si5345_2_registers) / sizeof(si5345_2_registers[0]));
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

int main() {
    base_init();

    // write SI (clock chip) configuration
    if(1) {
        const char* ID_SI5345_1 = "000efddf"; // `sha1sum Si5345-RevD-SI5345_1-Registers.h | cut -b 1-8`
        const char* ID_SI5345_2 = "f174722f"; // `sha1sum Si5345-RevD-SI5345_2-Registers.h | cut -b 1-8`

        printf("I [%s] check SI5345_1 config\n", __FUNCTION__);
        IOWR_ALTERA_AVALON_PIO_DATA(I2C_MASK_BASE, 1 << 1);
        if(si5345.cmp_design_id(ID_SI5345_1) != 0) {
            printf("I [%s] configure SI5345_1 ...\n", __FUNCTION__);
            si5345.init(si5345_1_registers, sizeof(si5345_1_registers) / sizeof(si5345_1_registers[0]));
            si5345.write_design_id(ID_SI5345_1);
            si5345.wait_sysincal();
        }

        printf("I [%s] check SI5345_2 config\n", __FUNCTION__);
        IOWR_ALTERA_AVALON_PIO_DATA(I2C_MASK_BASE, 1 << 2);
        if(si5345.cmp_design_id(ID_SI5345_2) != 0) {
            printf("I [%s] configure SI5345_2 ...\n", __FUNCTION__);
            si5345.init(si5345_2_registers, sizeof(si5345_2_registers) / sizeof(si5345_2_registers[0]));
            si5345.write_design_id(ID_SI5345_2);
            si5345.wait_sysincal();
        }

        // TODO: reconfig
    }

    while (1) {
        printf("\n");
        printf("[PCIe40] -------- menu --------\n");

        printf("\n");
        printf("  [0] => ...\n");
        printf("  [1] => XCVR 156\n");
        printf("  [2] => XCVR 250\n");
        printf("  [c] => clocks\n");
        printf("  [R] => reconfig\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            break;
        case '1':
            xcvr0.menu();
            break;
        case '2':
            xcvr1.menu();
            break;
        case 'c':
            clocks_menu();
            break;
        case 'R':
            xcvr0.reconfig.pll(AVM_XCVR1_BASE + 0x00000000);
            xcvr1.reconfig.pll(AVM_XCVR1_BASE + 0x00000000);
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
