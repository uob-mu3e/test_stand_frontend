
#include "si5340_regs.h"

struct si5340_t {

    const alt_u8 dev = 0x77;

    alt_u8 set_page(alt_u8 page = -1) {
        if(page != -1) i2c.set(dev, 0x01, page);
        page = i2c.get(dev, 0x01);
        printf("  page 0x%02X :\n", page);
        return page;
    }

    void set_N0(alt_u64 n) {
        set_page(0x03);

        // 
        i2c.set(dev, 0x02, (n >> 0) & 0xFF);
        i2c.set(dev, 0x03, (n >> 8) & 0xFF);
        i2c.set(dev, 0x04, (n >> 16) & 0xFF);
        i2c.set(dev, 0x05, (n >> 24) & 0xFF);
        i2c.set(dev, 0x06, (n >> 32) & 0xFF);
        i2c.set(dev, 0x07, (n >> 40) & 0xFF);

        // update N0
        i2c.set(dev, 0x0c, 1);
    }

    void set_f0(alt_u32 f) {
        alt_u32 f_in = 48000000;
        while(f_in % 2 == 0 && f % 2 == 0) { f_in /= 2; f /= 2; }
        alt_u64 n = 0x93b4800000;
        while(n % 2 == 0 && f % 2 == 0) { n /= 2; f /= 2; }
        set_N0(n * f_in / f / 2);
    }

    void test() {
        i2c.set(dev, 0x01, 0); // set page 0
        printf("  DIE_REV       = 0x%02X\n", i2c.get(dev, 0x00));
        printf("  PN_BASE       = 0x%02X%02X\n", i2c.get(dev, 0x03), i2c.get(dev, 0x02));
        printf("  GRADE         = 0x%02X\n", i2c.get(dev, 0x04));
        printf("  DEVICE_REV    = 0x%02X\n", i2c.get(dev, 0x05));
        printf("  TOOL_VERSION  = 0x%02X%02X%02X\n", i2c.get(dev, 0x08), i2c.get(dev, 0x07), i2c.get(dev, 0x06));
        printf("  TEMP_GRADE    = 0x%02X\n", i2c.get(dev, 0x09));
        printf("  PKG_ID        = 0x%02X\n", i2c.get(dev, 0x0A));

        auto& regs = si5340_regs;
        alt_u8 page = -1;
        for(unsigned i = 0; i < sizeof(regs)/sizeof(regs[0]); i++) {
            auto& reg = regs[i];
            if(page != reg[0]) {
                i2c.set(dev, 0x01, reg[0]);
                page = i2c.get(dev, 0x01);
                printf("  page 0x%02X :\n", page);
            }
            alt_u8 a = reg[1];
            alt_u8 r = i2c.get(dev, a), m = reg[2];
            printf("    [0x%02X] = 0x%02X | 0x%02X\n", a, r, (r & ~m) | (reg[3] & m));
        }
    }

    void write() {
        auto& regs = si5340_regs;
        for(int i = 0; i < sizeof(regs) / sizeof(regs[0]); i++) {
            printf("[%04X]\n", i);
        }

        // 0x2373000000 - 100 MHz
        // 0x1C5C000000 - 125 MHz
        // 0x16B0000000 - 156.25 MHz
        set_N0(0x2373000000); // 100 MHz

        // soft reset
//        set_page(0x0);
//        i2c.set(dev, 0x1C, 1);

    }

};
