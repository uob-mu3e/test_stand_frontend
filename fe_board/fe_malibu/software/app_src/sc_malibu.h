
#include "../../../../common/include/firmware/feb.h"
using namespace mu3e::feb;

alt_u16 sc_t::callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
    switch(cmd) {
    case CMD_TILE_ON:
        malibu.powerup();
        break;
    case CMD_TILE_OFF:
        malibu.powerdown();
        break;
    case CMD_TILE_STIC_OFF:
        malibu.stic_configure(0, stic3_config_ALL_OFF);
        break;
    case CMD_TILE_STIC_PLL_TEST:
        malibu.stic_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
        break;
    case CMD_TILE_I2C_WRITE:
        malibu.i2c_write_u32(data, n);
        break;
    default:
        if((cmd & 0xFFF0) == CMD_TILE_STIC_CFG) {
            printf("try stic_configure\n");
            int stic = cmd & 0x000F;
            int ok = 1;
            for(int i = 0; i < sizeof(stic3_config_ALL_OFF) / sizeof(stic3_config_ALL_OFF[0]); i++) {
                alt_u8 b = ((alt_u8*)data)[i];
                if(b != stic3_config_ALL_OFF[i]) ok = 0;
            }
            printf("ok = %d\n", ok);
            if(ok == 1) malibu.stic_configure(stic, (alt_u8*)data);
        }
        else {
            printf("[sc_callback] unknown command\n");
        }
    }

    return 0;
}
