#ifndef __MU3E_HARDWARE_SC_H__
#define __MU3E_HARDWARE_SC_H__

namespace mu3e { namespace daq { namespace feb {

const uint16_t CMD_MUTRIG_ASIC_CFG           = 0x0110;
const uint16_t CMD_MUTRIG_ASIC_OFF           = 0x0103; // configure all off
const uint16_t CMD_MUTRIG_CNT_READ           = 0x0105;
const uint16_t CMD_MUTRIG_CNT_RESET          = 0x0106;
const uint16_t CMD_MUTRIG_SKEW_RESET         = 0x0104;

const uint16_t CMD_TILE_STIC_CFG             = 0x0110;
const uint16_t CMD_TILE_STIC_OFF             = 0x0103;
const uint16_t CMD_TILE_STIC_PLL_TEST        = 0x0104;
const uint16_t CMD_TILE_ON                   = 0x0101;
const uint16_t CMD_TILE_OFF                  = 0x0102;
const uint16_t CMD_TILE_I2C_WRITE            = 0x0105;

const uint16_t CMD_MUPIX_CHIP_CFG            = 0x0110;
const uint16_t CMD_MUPIX_BOARD_CFG           = 0x0120;

const uint16_t CMD_PING                      = 0xFFFE;
const uint16_t CMD_FFFF                      = 0xFFFF;

} } } // namespace mu3e::daq::feb

#endif // __MU3E_HARDWARE_SC_H__
