-- register map for a10 counters

library ieee;
use ieee.std_logic_1164.all;

package a10_counters is

        -- 250 MHz counters
        constant SWB_STREAM_FIFO_FULL_PIXEL_CNT               :  integer := 16#00#;
        constant SWB_BANK_BUILDER_IDLE_NOT_HEADER_PIXEL_CNT   :  integer := 16#01#;
        constant SWB_BANK_BUILDER_RAM_FULL_PIXEL_CNT          :  integer := 16#02#;
        constant SWB_BANK_BUILDER_TAG_FIFO_FULL_PIXEL_CNT     :  integer := 16#03#;

        -- 156 MHz counters
        constant SWB_LINK_FIFO_ALMOST_FULL_PIXEL_CNT          :  integer := 16#04#;
        constant SWB_LINK_FIFO_FULL_PIXEL_CNT                 :  integer := 16#05#;
        constant SWB_SKIP_EVENT_PIXEL_CNT                     :  integer := 16#06#;
        constant SWB_EVENT_PIXEL_CNT                          :  integer := 16#07#;
        constant SWB_SUB_HEADER_PIXEL_CNT                     :  integer := 16#08#;

end package;
