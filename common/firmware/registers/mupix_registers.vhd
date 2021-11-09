-- Register Map
-- Note: 
-- write register, use naming scheme:       ***_REGISTER_W
-- read  register, use naming scheme:       ***_REGISTER_R
-- bit range     , use naming scheme:       ***_RANGE
-- single bit constant, use naming scheme:  ***_BIT

-- REGISTERS above 80: datapath

-- M.Mueller, November 2020

library ieee;
use ieee.std_logic_1164.all;

package mupix_registers is

--(x"60") --(start of the mp_datapath addr-space, 0x40-MUPIX_DATAPATH_ADDR_START is mp_ctrl )
constant MUPIX_DATAPATH_ADDR_START          : integer := 128;
constant MUPIX_LVDS_STATUS_BLOCK_LENGTH     : integer := 36;
--////////////////////////////////////////////--
--//////////////////REGISTER MAP//////////////--
--////////////////////////////////////////////--

-----------------------------------------------------------------
---- mupix ctrl -------------------------------------------------
-----------------------------------------------------------------

    constant MP_CTRL_ENABLE_REGISTER_W          :  integer := 16#0400#;
        constant WR_BIAS_BIT                    :  integer := 0;
        constant WR_CONF_BIT                    :  integer := 1;
        constant WR_VDAC_BIT                    :  integer := 2;
        constant WR_COL_BIT                     :  integer := 3;
        constant WR_TEST_BIT                    :  integer := 4;
        constant WR_TDAC_BIT                    :  integer := 5;
        constant CLEAR_BIAS_FIFO_BIT            :  integer := 6;
        constant CLEAR_CONF_FIFO_BIT            :  integer := 7;
        constant CLEAR_VDAC_FIFO_BIT            :  integer := 8;
        constant CLEAR_COL_FIFO_BIT             :  integer := 9;
        constant CLEAR_TEST_FIFO_BIT            :  integer := 10;
        constant CLEAR_TDAC_FIFO_BIT            :  integer := 11;

    constant MP_CTRL_BIAS_REGISTER_W            :  integer := 16#0401#;
    constant MP_CTRL_CONF_REGISTER_W            :  integer := 16#0402#;
    constant MP_CTRL_VDAC_REGISTER_W            :  integer := 16#0403#;
    constant MP_CTRL_COL_REGISTER_W             :  integer := 16#0404#;
    constant MP_CTRL_TEST_REGISTER_W            :  integer := 16#0405#;
    constant MP_CTRL_TDAC_REGISTER_W            :  integer := 16#0406#;

    constant MP_CTRL_SLOW_DOWN_REGISTER_W       :  integer := 16#0407#;
    constant MP_CTRL_CHIP_MASK_REGISTER_W       :  integer := 16#0408#;
    constant MP_CTRL_INVERT_REGISTER_W          :  integer := 16#0409#;
        constant MP_CTRL_INVERT_29_BIT          :  integer := 0;
        constant MP_CTRL_INVERT_CSN_BIT         :  integer := 1;

    constant MP_CTRL_ALL_REGISTER_W             :  integer := 16#040A#;
    constant MP_CTRL_SPI_BUSY_REGISTER_R        :  integer := 16#040F#;

-----------------------------------------------------------------
---- mupix datapath ---------------------------------------------
-----------------------------------------------------------------

    constant MP_READOUT_MODE_REGISTER_W         :  integer := 16#0900#;
        -- if set: TS is inverted
        constant INVERT_TS_BIT                  :  integer := 0;
        -- if set: TS2 is inverted
        constant INVERT_TS2_BIT                 :  integer := 1;
        -- if set: TS is grey-decoded
        constant GRAY_TS_BIT                    :  integer := 2;
        -- if set: TS2 is grey-decoded
        constant GRAY_TS2_BIT                   :  integer := 3;
        -- bits to select different chip id numbering modes (layers etc.)
        -- not in use at the moment
        subtype  CHIP_ID_MODE_RANGE             is integer range 5 downto 4;
        -- bits to select different TOT calculation modes
        -- Default is to send TS2 as TOT
        subtype  TOT_MODE_RANGE                 is integer range 8 downto 6;
    constant MP_LVDS_LINK_MASK_REGISTER_W       :  integer := 16#0901#;
    constant MP_LVDS_LINK_MASK2_REGISTER_W      :  integer := 16#0902#;
    constant MP_LVDS_DATA_VALID_REGISTER_R      :  integer := 16#0903#;
    constant MP_LVDS_DATA_VALID2_REGISTER_R     :  integer := 16#0904#;
    constant MP_DATA_GEN_CONTROL_REGISTER_W     :  integer := 16#0905#;
        -- hit output probability is 1/(2^(MP_DATA_GEN_HIT_P_RANGE+1)) for each cycle where a hit could be send
        -- (in 125 MHz minus protocol overhead sorter -> merger)
        subtype  MP_DATA_GEN_HIT_P_RANGE        is integer range 3 downto 0;
        -- if set: hit output probability is 1
        constant MP_DATA_GEN_FULL_STEAM_BIT     :  integer := 4;
        -- if set: generator seed is the same on all boards else: generator seed depends on ref_addr from backplane
        constant MP_DATA_GEN_SYNC_BIT           :  integer := 5;
        -- if set: use hits from generator, datapath is not connected to link
        constant MP_DATA_GEN_ENGAGE_BIT         :  integer := 16;
        -- if set: generated hits are inserted after the data_unpacker
        constant MP_DATA_GEN_SORT_IN_BIT        :  integer := 17;
        -- if set: enable hit generation (set MP_DATA_GEN_ENGAGE_BIT to actually replace sorter output with these hits)
        constant MP_DATA_GEN_ENABLE_BIT         :  integer := 31;
    -- start of lvds status register block, 1 Word for each chip
    constant MP_LVDS_STATUS_START_REGISTER_W    :  integer := 16#0906#;
        subtype  MP_LVDS_STATUS_DISP_ERR_RANGE  is integer range 27 downto 0;
        constant MP_LVDS_STATUS_PLL_LOCKED_BIT  :  integer := 28;
        subtype  MP_LVDS_STATUS_STATE_RANGE     is integer range 30 downto 29;
        constant MP_LVDS_STATUS_READY_BIT       :  integer := 31;
    constant MP_LVDS_INVERT_REGISTER_W          :  integer := 16#0930#;
    constant MP_SORTER_DELAY_REGISTER_W         :  integer := 16#0931#;
    -- 40 counters
    constant MP_SORTER_COUNTER_REGISTER_R       :  integer := 16#0932#;
    constant MP_DATA_BYPASS_SELECT_REGISTER_W   :  integer := 16#094B#;
    constant MP_TS_HISTO_SELECT_REGISTER_W      :  integer := 16#094C#;
        subtype  MP_TS_HISTO_LINK_SELECT_RANGE  is integer range 15 downto 0;
        subtype  MP_TS_HISTO_N_SAMPLE_RANGE     is integer range 31 downto 16;
    constant MP_LAST_SORTER_HIT_REGISTER_R      :  integer := 16#094D#;
    constant MP_SORTER_INJECT_REGISTER_W        :  integer := 16#094E#;
        -- select the input of the sorter to inject to
        subtype MP_SORTER_INJECT_SELECT_RANGE   is integer range 7 downto 4;
        -- rising edge on this bit will trigger a single inject of the word MP_SORTER_INJECT_REGISTER_W at sorter input MP_SORTER_INJECT_REGISTER_W(MP_SORTER_INJECT_SELECT_RANGE)
        constant MP_SORTER_INJECT_ENABLE_BIT    :  integer := 8;
    constant MP_HIT_ENA_CNT_REGISTER_R          :  integer := 16#094F#;
    constant MP_HIT_ENA_CNT_SELECT_REGISTER_W   :  integer := 16#0950#;
    constant MP_HIT_ENA_CNT_SORTER_IN_REGISTER_R :  integer := 16#0951#;
    constant MP_HIT_ENA_CNT_SORTER_SELECT_REGISTER_W :  integer := 16#0952#;
    constant MP_HIT_ENA_CNT_SORTER_OUT_REGISTER_R : integer := 16#0953#;
    constant MP_RESET_LVDS_N_REGISTER_W         :  integer := 16#0954#;

end package mupix_registers;
