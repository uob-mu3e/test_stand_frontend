
#include "../../../../common/include/feb.h"
using namespace mu3e::daq::feb;

alt_u16 sc_t::callback(alt_u16 cmd, volatile alt_u32* data, alt_u16 n) {
    switch(cmd) {
    case CMD_TILE_ON:
        malibu.power_TMB(true);
        break;
    case CMD_TILE_OFF:
        malibu.power_TMB(false);
        break;
    case CMD_TILE_STIC_OFF:
        malibu.chip_configure(0, stic3_config_ALL_OFF);
        break;
    case CMD_TILE_STIC_PLL_TEST:
        malibu.chip_configure(0, stic3_config_PLL_TEST_ch0to6_noGenIDLE);
        break;
    case CMD_TILE_I2C_WRITE:
        malibu.i2c_write_u32(data, n);
        break;
    default:
        if((cmd & 0xFFF0) == CMD_TILE_STIC_CFG) {
            printf("try stic_configure\n");
            int asic = cmd & 0x000F;
            malibu.chip_configure(stic, (alt_u8*)data);
        }
        else {
            printf("[sc_callback] unknown command\n");
        }
    }

    return 0;
}
