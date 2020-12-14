-- Register Map
-- Note: 
-- write register, use naming scheme:       ***_REGISTER_W
-- read  register, use naming scheme:       ***_REGISTER_R
-- bit range     , use naming scheme:       ***_RANGE
-- single bit constant, use naming scheme:  ***_BIT

-- REGISTERS above 60: datapath

-- M.Mueller, November 2020

library ieee;
use ieee.std_logic_1164.all;
use work.mupix_constants.all;

package mupix_registers is

constant MUPIX_DATAPATH_ADDR_START          : integer := 96; --(x"60") --(start of the mp_datapath addr-space, 0x40-MUPIX_DATAPATH_ADDR_START is mp_block )

--////////////////////////////////////////////--
--//////////////////REGISTER MAP//////////////--
--////////////////////////////////////////////--

-----------------------------------------------------------------
---- mupix_block ------------------------------------------------
-----------------------------------------------------------------


-----------------------------------------------------------------
---- datapath ---------------------------------------------------
-----------------------------------------------------------------

    constant MP_READOUT_MODE_REGISTER_W     :  integer := 16#60#;
        constant INVERT_TS_BIT              :  integer := 0; -- if set: TS is inverted
        constant INVERT_TS2_BIT             :  integer := 1; -- if set: TS2 is inverted
        constant GRAY_TS_BIT                :  integer := 2; -- if set: TS is grey-decoded
        constant GRAY_TS2_BIT               :  integer := 3; -- if set: TS2 is grey-decoded
        -- bits to select different chip id numbering modes (layers etc.)
        -- not in use at the moment
        subtype  CHIP_ID_MODE_RANGE         is integer range 5 downto 4;
        -- bits to select different TOT calculation modes
        -- Default is to send TS2 as TOT
        subtype  TOT_MODE_RANGE             is integer range 8 downto 6;
    constant MP_LVDS_LINK_MASK_REGISTER_W   :  integer := 16#61#;
    constant MP_LVDS_LINK_MASK2_REGISTER_W  :  integer := 16#62#;
    constant MP_LVDS_DATA_VALID_REGISTER_R  :  integer := 16#63#;
    constant MP_LVDS_DATA_VALID2_REGISTER_R :  integer := 16#64#;
    constant MP_DATA_GEN_CONTROL_REGISTER_W :  integer := 16#65#;
        -- hit output probability is 1/(2^(MP_DATA_GEN_HIT_P_RANGE+1)) for each cycle where a hit could be send
        -- (in 125 MHz minus protocol overhead sorter -> merger)
        subtype  MP_DATA_GEN_HIT_P_RANGE    is integer range 3 downto 0;
        constant MP_DATA_GEN_FULL_STEAM_BIT :  integer := 4; -- if set: hit output probability is 1
        constant MP_DATA_GEN_SYNC_BIT       :  integer := 5; -- if set: generator seed is the same on all boards else: generator seed depends on ref_addr from backplane
        constant MP_DATA_GEN_ENGAGE_BIT     :  integer := 16; -- if set: use hits from generator, datapath is not connected to link
        constant MP_DATA_GEN_SORT_IN_BIT    :  integer := 17; -- if set: generated hits are inserted after the data_unpacker
        constant MP_DATA_GEN_ENABLE_BIT     :  integer := 31; -- if set: enable hit generation (set MP_DATA_GEN_ENGAGE_BIT to actually replace sorter output with these hits)

end package mupix_registers;
