#ifndef __PCIe40_h__
#define __PCIe40_h__

#include "Si5345-RevD-SI5345_1-Registers.h"
#include "Si5345-RevD-SI5345_2-Registers.h"
si534x_t si5345 { i2c.dev, 0x68 };

namespace PCIe40 {

void menu() {
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

void init() {
    // write SI (clock chip) configuration

    const char* ID_SI5345_1 = "000efddf"; // `sha1sum Si5345-RevD-SI5345_1-Registers.h | cut -b 1-8`
    const char* ID_SI5345_2 = "f174722f"; // `sha1sum Si5345-RevD-SI5345_2-Registers.h | cut -b 1-8`

    printf("I [%s] SI5345_1\n", __FUNCTION__);
    IOWR_ALTERA_AVALON_PIO_DATA(I2C_MASK_BASE, 1 << 1);
    if(si5345.cmp_design_id(ID_SI5345_1) != 0) {
        printf("+ configure ...\n");
        si5345.init(si5345_1_registers, sizeof(si5345_1_registers) / sizeof(si5345_1_registers[0]));
        si5345.write_design_id(ID_SI5345_1);
        si5345.wait_sysincal();
        printf("+ DONE\n");
    }
    else printf("+ OK\n");

    printf("I [%s] SI5345_2\n", __FUNCTION__);
    IOWR_ALTERA_AVALON_PIO_DATA(I2C_MASK_BASE, 1 << 2);
    if(si5345.cmp_design_id(ID_SI5345_2) != 0) {
        printf("+ configure ...\n");
        si5345.init(si5345_2_registers, sizeof(si5345_2_registers) / sizeof(si5345_2_registers[0]));
        si5345.write_design_id(ID_SI5345_2);
        si5345.wait_sysincal();
        printf("+ DONE\n");
    }
    else printf("+ OK\n");

    // TODO: reconfig
}

} // namespace PCIe40

#endif // __PCIe40_h__
