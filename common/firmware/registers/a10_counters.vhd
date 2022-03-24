-- register map for a10 counters

library ieee;
use ieee.std_logic_1164.all;

package a10_counters is

        -- datapath counters
        constant NDATAPATH_CNTS : integer := 6;
        constant SWB_STREAM_FIFO_FULL_CNT               :  integer := 16#00#;
        constant SWB_BANK_BUILDER_IDLE_NOT_HEADER_CNT   :  integer := 16#01#;
        constant SWB_BANK_BUILDER_SKIP_EVENT_CNT        :  integer := 16#02#;
        constant SWB_BANK_BUILDER_EVENT_CNT             :  integer := 16#03#;
        constant SWB_BANK_BUILDER_TAG_FIFO_FULL_CNT     :  integer := 16#04#;
        constant SWB_EVENTS_TO_FARM_CNT                 :  integer := 16#05#;

        -- link counters
        constant NLINK_CNTS : integer := 5;
        constant SWB_LINK_FIFO_ALMOST_FULL_CNT    :  integer := 16#00#;
        constant SWB_LINK_FIFO_FULL_CNT           :  integer := 16#01#;
        constant SWB_SKIP_EVENT_CNT               :  integer := 16#02#;
        constant SWB_EVENT_CNT                    :  integer := 16#03#;
        constant SWB_SUB_HEADER_CNT               :  integer := 16#04#;

end package;
