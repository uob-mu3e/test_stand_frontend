
#include "si_regs.h"
#include "in0_125_in1_125_out0_125_out1_125_out2_125_out3_125.h"

#include <altera_avalon_spi.h>

struct si_t {

    alt_u8 read(alt_u8 a) {
        alt_u8 w[3] = { 0x00, a, 0x80 }, r;
        alt_avalon_spi_command(SPI_BASE, 0, 3, w, 1, &r, 0);
        return r;
    }

    void write(alt_u8 a, alt_u8 d) {
        alt_u8 w[4] = { 0x00, a, 0x40, d };
        alt_avalon_spi_command(SPI_BASE, 0, 4, w, 0, 0, 0);
    }

    void test() {
        printf("si5345_test:\n");
        write(0x01, 0);
//        write(0x1E, 1);
        for(int i = 0; i < 0xFF; i++) {
            printf("  [0x%02X] = 0x%02X\n", i, read(i));
        }
    }

    void init() {
        const si5344_revd_register_t* regs = si5344_revd_registers;

        printf("si5345_init:\n");
        alt_u8 page = read(0x01);
        for(unsigned i = 0; i < SI5344_REVD_REG_CONFIG_NUM_REGS; i++) {
            alt_u32 a = regs[i].address;
            alt_u32 v = regs[i].value;

            if(page != a >> 8) {
                printf("  page <= 0x%02X\n", a >> 8);
                write(0x01, a >> 8);
                page = a >> 8;
            }
            while(read(0xFE) != 0x0F) {
                printf("  device not ready\n");
                usleep(1000);
            }
            printf("  [0x%02X] <= 0x%02X\n", a & 0xFF, v);
            write(a & 0xFF, v);
            for(int i = 0; read(0xFE) != 0x0F; i++) {
                if(i == 9) {
                    printf("  device not ready\n");
                    goto error;
                }
                usleep(1000);
            }
            if(read(a & 0xFF) != v) printf("NOT OK : 0x%02X\n", read(a & 0xFF));
//            printf("  [0x%02X] = 0x%02X\n", reg[1], read(reg[1]));
            if(i == 2) usleep(300000);
        }
        printf("si5345_init: DONE\n");
        
        error: ;
    }

};
