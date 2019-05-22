#include "system.h"

#include <sys/alt_stdio.h>

#include <sys/alt_alarm.h>
#include <sys/alt_timestamp.h>

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include "i2c.h"
i2c_t i2c;

#include "si.h"
si_t si;

struct fan_t {
    const alt_u32 fclk = 254000;

//    uint8_t mode = 0x2; // closed loop
    alt_u8 scale = 4;
    alt_u8 rps = 50;

    alt_u8 get_rps() const {
        return scale * fclk / ( 128 * 2 * (i2c.get(0x48, 0x00) + 1));
    }

    void set_rps(alt_u8 rps) {
        i2c.set(0x48, 0x00, scale * fclk / (128 * 2 * rps) - 1);
    }

    void init() {
        i2c.set(0x48, 0x16, 0x02); // tachometer count time
        i2c.set(0x48, 0x08, 0x07); // alarm enable
        i2c.set(0x48, 0x04, 0xF5); // gpio definition
        i2c.set(0x48, 0x02, 0x2A); // configuration
        i2c.set(0x48, 0x00, 0x4E); // fan speed
        set_rps(rps);
    }

    void print() const {
        printf(
            "FAN: conf = 0x%02X, gpio = 0x%02X, tach = 0x%02X\n"
            "     alarm = 0x%02X => 0x%02X\n"
            "     rps = 0x%02X (%u) => %u\n",
            i2c.get(0x48, 0x02), i2c.get(0x48, 0x04), i2c.get(0x48, 0x16),
            i2c.get(0x48, 0x08), i2c.get(0x48, 0x0A),
            i2c.get(0x48, 0x00), rps, i2c.get(0x48, 0x0C) / 2
        );
    }
};

struct temp_t {
    void print() {
        printf("TEMP: local = %u, remote = %u\n",
            i2c.get(0x18, 0x00),
            i2c.get(0x18, 0x01)
        );
    }
};

#include "cfi1616.h"
cfi1616_t flash;

volatile alt_u32* ctrl = (alt_u32*)(CTRL_REGION_BASE);
volatile alt_u8* data = (alt_u8*)(DATA_REGION_BASE);

alt_u32 hist_ts[16];

alt_u32 hibit(alt_u32 n) {
    if(n == 0) return 0;
    alt_u32 r = 0;
    if(n & 0xFFFF0000) { r += 16; n >>= 16; };
    if(n & 0xFF00) { r += 8; n >>= 8; };
    if(n & 0xF0) { r += 4; n >>= 4; };
    if(n & 8) return r + 3;
    if(n & 4) return r + 2;
    if(n & 2) return r + 1;
    return 0;
}

alt_u32 alarm_callback(void*) {
//    IOWR_ALTERA_AVALON_PIO_DATA(PIO_BASE, (alt_nticks() >> 8) & 0xFF);

    alt_timestamp_start();
    int state = flash.callback((alt_u8*)IORD(ctrl, 0), data, (alt_u32)IORD(ctrl, 1));
    alt_u32 ts_bin = hibit(alt_timestamp() / 125);
    if(ts_bin < 16) hist_ts[ts_bin]++;
    if(state == -EAGAIN) return 1;
    if(state == 0) {
        IOWR(ctrl, 0, 0);
        IOWR(ctrl, 1, 0);
    }

    return 10;
}

int uart = -1;

char wait_key(useconds_t us = 100000) {
    while(1) {
        char cmd;
        if(read(uart, &cmd, 1) > 0) return cmd;
        usleep(us);
    }
}

fan_t fan;
temp_t temp;

void menu_i2c() {
    
        fan.print();
        temp.print();
        i2c.print();

        // power monitor
        i2c.w8(0x40, 0x02);
        printf("pwr_bus: %u mV\n", 40959 * i2c.r16(0x40) / 0x7FFF); // 0x7FFF = 40.95875 V
        i2c.w8(0x40, 0x01);
        printf("pwr_shunt: %u uV\n", 81918 * i2c.r16(0x40) / 0x7FFF); // 0x7FFFF = 81.9175 mV

        // SI5340B
        //printf("si_revid: %02X\n", i2c.get(0x77, 0x00));
        //printf("si_devid: %02X%02X\n", i2c.get(0x77, 0x03), i2c.get(0x77, 0x02));

        
    
    while (1) {
        
        printf("I2C Menu\n", ALT_DEVICE_FAMILY);
        printf("  [i] => fan init\n");
        printf("  []] => ++fan\n");
        printf("  [[] => --fan\n");
        printf("  [?] => wait\n");
        printf("  [q] => return\n");

        printf("Select entry ...\n");
        
        char cmd = wait_key();
        switch(cmd) {
        case 'i': // fan i2c
            fan.init();
            break;
        case ']':
            fan.set_rps(++fan.rps);
            break;
        case '[':
            fan.set_rps(--fan.rps);
            break;
        case '?':
            wait_key();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

void menu_flash() {
    volatile alt_u8* addr_test = flash.base + 0x05E80000;

    while (1) {
        
        printf("FLASH Menu\n", ALT_DEVICE_FAMILY);
        printf("  [e] => erase\n");
        printf("  [p] => program\n");
        printf("  [?] => wait\n");
        printf("  [q] => return\n");

        printf("Select entry ...\n");
        
        char cmd;
        if(read(uart, &cmd, 1) > 0) switch(cmd) {
        case 'e': {
            int err;
            err = flash.unlock(addr_test);
            printf("%08X : unlock => %d\n", addr_test, err);
            err = flash.erase(addr_test);
            err = flash.sync(addr_test);
            printf("%08X : erase => %d\n", addr_test, err);
            break;
        }
        case 'p': {
            int err;
            for(int i = 0; i < 512; i++) data[i] = i;
            for(alt_u32 i = 0; i < flash.regions[1].blockSize; i += flash.bufferSize) {
                err = flash.program(addr_test + i, data, flash.bufferSize);
                err = flash.sync(addr_test + i);
                printf("%08X : program => %d\n", addr_test + i, err);
            }
            err = flash.lock(addr_test);
            printf("%08X : lock => %d\n", addr_test, err);
            break;
        }
        case '?':
            wait_key();
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }

        for(int i = 0; i <= 10; i++) {
            printf("%8u", 1 << i);
        }
        printf("\n");
        for(int i = 0; i <= 10; i++) {
            printf("%8u", hist_ts[i]);
        }
        printf("\n");

        usleep(200000);
    }
}

void menu_spi_si5345() {
    while (1) {
        printf("SI5345\n", ALT_DEVICE_FAMILY);
        printf("  [i] => init chip\n");
        printf("  [q] => quit\n");
        printf("  [r] => read(0xFE)\n");
        printf("  [t] => test\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case 'r' :
            si.read(0xFE);
            break;
        case 'i':
            si.init();
            break;
        case 't' :
            si.test();
            break;
        case '?':
            break;
        case 'q':
            return;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }
}

int main() {
    fan.init();

    flash.init((alt_u8*)(FLASH_BASE));
    alt_alarm alarm;
    int err = alt_alarm_start(&alarm, 10, alarm_callback, nullptr);
    if(err) {
        printf("ERROR: alt_alarm_start => %d\n%d\n", err);
    }

    uart = open(JTAG_UART_NAME, O_NONBLOCK);
    if(uart < 0) {
        printf("ERROR: can't open %s\n", JTAG_UART_NAME);
        return 1;
    }

    while (1) {
        printf("'%s' NIOS Menu\n", ALT_DEVICE_FAMILY);
        printf("  [0] => spi si chip\n");
        printf("  [1] => i2c fan\n");
        printf("  [2] => flash\n");

        printf("Select entry ...\n");
        char cmd = wait_key();
        switch(cmd) {
        case '0':
            menu_spi_si5345();
            break;
        case '1':
            printf("i2c:\n");
            menu_i2c();
            break;
        case '2':
            printf("flash:\n");
            menu_flash();
            break;
        default:
            printf("invalid command: '%c'\n", cmd);
        }
    }

    return 0;
}
