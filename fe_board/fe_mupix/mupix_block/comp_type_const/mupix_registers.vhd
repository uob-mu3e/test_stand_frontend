-- Register Map
-- Note: 
-- For mupix_registers.tcl script to identify write register, use naming scheme: 		***_REGISTER_W
-- For mupix_registers.tcl script to identify read  register, use naming scheme: 		***_REGISTER_R
-- For mupix_registers.tcl script to identify bit range     , use naming scheme: 		***_RANGE
-- For mupix_registers.tcl script to identify single bit constant, use naming scheme: 	***_BIT

-- REGISTERS above 60: datapath

library ieee;
use ieee.std_logic_1164.all;
use work.mupix_constants.all;

package mupix_registers is

constant NREGISTERS_MUPIX_DATAPATH_WR       : integer := 50;
constant NREGISTERS_MUPIX_DATAPATH_RD       : integer := 50;
constant MUPIX_DATAPATH_ADDR_START          : integer := 96; --(x"60")

--////////////////////////////////////////////--
--/////////////WRITE REGISTER MAP/////////////--
--////////////////////////////////////////////--
    constant MP_READOUT_MODE_REGISTER_W     :  integer := 16#40#;
        constant INVERT_TS_BIT              :  integer := 0; -- if set: TS is inverted
        constant INVERT_TS2_BIT             :  integer := 1; -- if set: TS2 is inverted
        constant GRAY_TS_BIT                :  integer := 2; -- if set: TS is grey-decoded
        constant GRAY_TS2_BIT               :  integer := 3; -- if set: TS2 is grey-decoded
        -- bits to select different chip id numbering modes (layers etc.)
        -- not in use at the moment
        subtype  CHIP_ID_MODE_RANGE         is integer range 5 downto 4;
        -- bits to select different TOT calculation modes
        -- Default is to send TS2 as TOT
        subtype  TOT_MODE_RANGE             is integer range 7 downto 5;

--datapath
    constant MP_DATA_GEN_CONTROL_REGISTER_W :  integer := 16#41#;
        -- hit output probability is 1/(2^(MP_DATA_GEN_HIT_P_RANGE+1)) for each cycle where a hit could be send
        -- (in 125 MHz minus protocol overhead sorter -> merger)
        subtype  MP_DATA_GEN_HIT_P_RANGE    is integer range 3 downto 0;
        constant MP_DATA_GEN_FULL_STEAM_BIT :  integer := 4; -- if set: hit output probability is 1
        constant MP_DATA_GEN_SYNC_BIT       :  integer := 5; -- if set: generator seed is the same on all boards else: generator seed depends on ref_addr from backplane
        constant MP_DATA_GEN_ENGAGE_BIT     :  integer := 16; -- if set: use hits from generator, datapath is not connected to link
        constant MP_DATA_GEN_ENABLE_BIT     :  integer := 31; -- if set: enable hit generation (set MP_DATA_GEN_ENGAGE_BIT to actually replace sorter output with these hits)
------------------------------------------------------------------
---------------------- Read register Map -------------------------
------------------------------------------------------------------

-- datapath
    constant MP_LVDS_DATA_VALID_REGISTER_R  : integer := 16#60#;
    constant MP_LVDS_DATA_VALID2_REGISTER_R : integer := 16#61#;

-----------------------------------------------------------------
-- TODO when datapath done: remove if unused
-----------------------------------------------------------------
constant LINK_MASK_REGISTER_W               : integer := 16#53#;
constant RO_PRESCALER_REGISTER_W            : integer := 16#50#;    -- dec 0
constant DEBUG_CHIP_SELECT_REGISTER_W       : integer := 16#51#;
constant TIMESTAMP_GRAY_INVERT_REGISTER_W   : integer := 16#52#;
constant TS_INVERT_BIT                      : integer := 0;
constant TS2_INVERT_BIT                     : integer := 1;
constant TS_GRAY_BIT                        : integer := 2;
constant TS2_GRAY_BIT                       : integer := 3;
constant RX_STATE_RECEIVER_0_REGISTER_R     : integer := 16#50#;    -- dec 0
constant RX_STATE_RECEIVER_1_REGISTER_R     : integer := 16#51#;    -- dec 1
constant LVDS_PLL_LOCKED_REGISTER_R         : integer := 16#52#;    -- dec 2
constant MULTICHIP_RO_OVERFLOW_REGISTER_R   : integer := 16#53#;    -- dec 3
constant LVDS_RUNCOUNTER_REGISTER_R         : integer := 16#54#;    -- dec 4 (to 48)
constant LVDS_ERRCOUNTER_REGISTER_R         : integer := 16#80#;    -- dec 49 (to 93)


end package mupix_registers;
