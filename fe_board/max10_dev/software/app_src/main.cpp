
#include "../include/base.h"

#include "adc.h"
adc_t adc;

int main() {
    base_init();

    auto csr = (volatile alt_u32*)FLASH_CSR_BASE;
    auto data = (volatile alt_u32*)(FLASH_DATA_SECTOR2_START_ADDR);

    while (1) {
        printf("\n");
        printf("flash.status = 0x%08X\n", csr[0]);
        printf("flash.control = 0x%08X\n", csr[1]);

        printf("\n");
        printf("  [0] => spi si chip\n");
        printf("  [1] => adc\n");
        printf("  [r] => read flash\n");
        printf("  [w] => write flash\n");

        printf("Select entry ...\n");
        char cmd = wait_key();

        switch(cmd) {
        case '0':
            printf("spi:\n");
            //menu_spi_si5345();
            break;
        case '1':
            adc.menu();
            break;
        case 'r':
            for(int i = 0; i < 16; i++) {
                if(i % 4 == 0) printf("\n[0x%04X]", &data[i]);
                printf(" %08X", data[i]);
            }
            printf("\n");
            break;
        case 'w':
            // 1. disable write protection
            csr[1] &= ~(1 << 24);
            // 2. program data
            data[0] = 0;
            // 3. check write busy field
            while((csr[0] & 0x3) == 0x2) {
                printf("busy\n");
            }
            // 4. check write successful field
            if((csr[0] & 0x8) == 0) {
                printf("fail\n");
            }
            // 6. enable write protection
            csr[1] |= (1 << 24);
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
