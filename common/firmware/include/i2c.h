/*
 * mm_i2c.h
 *
 * author : Alexandr Kozlinskiy
 * date : 2017-11-13
 */

#ifndef SOFTWARE_APP_SRC_I2C_H_
#define SOFTWARE_APP_SRC_I2C_H_

#include "system.h"

#include <altera_avalon_i2c.h>

#ifdef MM_I2C_MASTER_BASE
struct mm_i2c_t {
    volatile alt_u8* base = (alt_u8*)(MM_I2C_MASTER_BASE | ALT_CPU_DCACHE_BYPASS_MASK);

    alt_u8 get(alt_u8 addr, alt_u8 cmd) {
        return base[(addr << 8) | cmd];
    }

    void set(alt_u8 addr, alt_u8 cmd, alt_u8 x) {
        base[(addr << 8) | cmd] = x;
    }

    void print() {
        printf("I2C: %u, %02X\n", (i2c[0x8001] << 8) | i2c[0x8000], i2c[0x8003]);
        printf("\n");
    }
};
#endif // MM_I2C_MASTER_BASE

struct i2c_t {
    ALT_AVALON_I2C_DEV_t* dev = alt_avalon_i2c_open(I2C_NAME);

    void read(alt_u8 addr, alt_u8* r, alt_u32 n) {
        if(!dev) return;
        alt_avalon_i2c_master_target_set(dev, addr);
        int err = alt_avalon_i2c_master_rx(dev, r, n, ALT_AVALON_I2C_NO_INTERRUPTS);
        if(err != ALT_AVALON_I2C_SUCCESS) {
            printf("error: alt_avalon_i2c_master_rx => %d\n", err);
        }
    }

    void write(alt_u8 addr, alt_u8* w, alt_u32 n) {
        if(!dev) return;
        alt_avalon_i2c_master_target_set(dev, addr);
        int err = alt_avalon_i2c_master_tx(dev, w, n, ALT_AVALON_I2C_NO_INTERRUPTS);
        if(err != ALT_AVALON_I2C_SUCCESS) {
            printf("error: alt_avalon_i2c_master_tx => %d\n", err);
        }
    }

    alt_u8 r8(alt_u8 addr) {
        alt_u8 r = 0;
        read(addr, &r, 1);
        return r;
    }

    void w8(alt_u8 addr, alt_u8 w) {
        write(addr, &w, 1);
    }

    alt_u16 r16(alt_u8 addr) {
        alt_u8 r[2] {};
        read(addr, r, 2);
        return (r[0] << 8) | r[1];
    }

    alt_u8 get(alt_u8 addr, alt_u8 w) {
        w8(addr, w);
        return r8(addr);
    }

    void set(alt_u8 addr, alt_u8 w0, alt_u8 w1) {
        alt_u8 w[2] = { w0, w1 };
        write(addr, w, 2);
    }

    void print() {
    }
};

#endif /* SOFTWARE_APP_SRC_I2C_H_ */
