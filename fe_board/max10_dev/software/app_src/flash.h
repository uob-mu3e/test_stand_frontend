#ifndef __FLASH_H__
#define __FLASH_H__

struct flash_t {
    volatile alt_u32* csr = (volatile alt_u32*)FLASH_CSR_BASE;
    volatile alt_u32* sector2 = (volatile alt_u32*)FLASH_DATA_SECTOR2_START_ADDR;

    void menu() {
        while(1) {
            printf("\n");
            printf("[flash] -------- menu --------\n");

            printf("\n");
            printf("  status =");
            if((csr[0] & 0x3) == 0x0) printf(" IDLE");
            else if((csr[0] & 0x3) == 0x1) printf(" BUSY_ERASE");
            else if((csr[0] & 0x3) == 0x2) printf(" BUSY_WRITE");
            else if((csr[0] & 0x3) == 0x3) printf(" BUSY_READ");
            printf(" %sRS", (csr[0] & 0x04) ? "" : "~");
            printf(" %sWS", (csr[0] & 0x08) ? "" : "~");
            printf(" %sES", (csr[0] & 0x10) ? "" : "~");
            printf(" %d%d%d%d%d", (csr[0] >> 5) & 1, (csr[0] >> 6) & 1, (csr[0] >> 7) & 1, (csr[0] >> 8) & 1, (csr[0] >> 9) & 1);
            printf("\n");

            printf("  control =");
            printf(" 0x%05X", csr[1] & 0xFFFFF);
            printf(" 0x%01X", (csr[1] >> 20) & 0x7);
            printf(" %d%d%d%d%d", (csr[0] >> 23) & 1, (csr[0] >> 24) & 1, (csr[0] >> 25) & 1, (csr[0] >> 26) & 1, (csr[0] >> 27) & 1);
            printf("\n");

            printf("\n");
            printf("  [r] => read\n");
            printf("  [w] => write\n");

            printf("Select entry ...\n");
            char cmd = wait_key();
            switch(cmd) {
            case 'r':
                for(int i = 0; i < 16; i++) {
                    if(i % 4 == 0) printf("\n[0x%04X]", &sector2[i]);
                    printf("  %08X", sector2[i]);
                }
                printf("\n");
                break;
            case 'w':
                while(csr[0] & 0x3) {
                    printf("busy\n");
                }
                // 1. disable write protection
                csr[1] &= ~(1 << 24);
                // 2. program data
                sector2[0] = 0;
                // 3. check write busy field
                while(csr[0] & 0x3) {
                    printf("busy\n");
                }
                // 4. check write successful field
                if(!(csr[0] & 0x8)) {
                    printf("fail\n");
                }
                // 6. enable write protection
                csr[1] |= (1 << 24);
                break;
            case 'e':
                break;
            }
        }
    }
};

#endif // __FLASH_H__
