-- Register Map
-- Note: 
-- For mupix_registers.tcl script to identify write register, use naming scheme: 		***_REGISTER_W
-- For mupix_registers.tcl script to identify read  register, use naming scheme: 		***_REGISTER_R
-- For mupix_registers.tcl script to identify bit range     , use naming scheme: 		***_RANGE
-- For mupix_registers.tcl script to identify single bit constant, use naming scheme: 	***_BIT

library ieee;
use ieee.std_logic_1164.all;
use work.mupix_constants.all;

package mupix_registers is

--////////////////////////////////////////////--
--/////////////WRITE REGISTER MAP/////////////--
--////////////////////////////////////////////--
    constant MP_READOUT_MODE_REGISTER_W     :  integer := 16#40#;
        constant INVERT_TS_BIT              :  integer := 0;
        constant INVERT_TS2_BIT             :  integer := 1;
        constant GRAY_TS_BIT                :  integer := 2;
        constant GRAY_TS2_BIT               :  integer := 3;
        subtype  CHIP_ID_MODE_RANGE         is integer range 5 downto 4;
        subtype  TOT_MODE_RANGE             is integer range 7 downto 5;
------------------------------------------------------------------
---------------------- Read register Map -------------------------
------------------------------------------------------------------

-----------------------------------------------------------------
-- TODO when datapath done: remove if unused
-----------------------------------------------------------------
constant LINK_MASK_REGISTER_W               : integer := 16#53#;
constant NREGISTERS_MUPIX_WR                : integer := 5;
constant RO_PRESCALER_REGISTER_W            : integer := 16#50#;    -- dec 0
constant DEBUG_CHIP_SELECT_REGISTER_W       : integer := 16#51#;
constant TIMESTAMP_GRAY_INVERT_REGISTER_W   : integer := 16#52#;
constant TS_INVERT_BIT                      : integer := 0;
constant TS2_INVERT_BIT                     : integer := 1;
constant TS_GRAY_BIT                        : integer := 2;
constant TS2_GRAY_BIT                       : integer := 3;
constant NREGISTERS_MUPIX_RD                : integer := 94;
constant RX_STATE_RECEIVER_0_REGISTER_R     : integer := 16#50#;    -- dec 0
constant RX_STATE_RECEIVER_1_REGISTER_R     : integer := 16#51#;    -- dec 1
constant LVDS_PLL_LOCKED_REGISTER_R         : integer := 16#52#;    -- dec 2
constant MULTICHIP_RO_OVERFLOW_REGISTER_R   : integer := 16#53#;    -- dec 3
constant LVDS_RUNCOUNTER_REGISTER_R         : integer := 16#54#;    -- dec 4 (to 48)
constant LVDS_ERRCOUNTER_REGISTER_R         : integer := 16#80#;    -- dec 49 (to 93)


end package mupix_registers;
