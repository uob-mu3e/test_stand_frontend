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
---- mupix ctrl (0x0400-0x0FFF)----------------------------------
-----------------------------------------------------------------

    constant MP_CTRL_ENABLE_REGISTER_W          :  integer := 16#0400#; -- DOC: Used to start the mupix configuration | MP_FEB
        constant WR_BIAS_BIT                    :  integer := 0;        -- DOC: start spi writing of the mupix bias reg (one can start all in parallel)| MP_FEB
        constant WR_CONF_BIT                    :  integer := 1;        -- DOC: start spi writing of the mupix conf reg | MP_FEB
        constant WR_VDAC_BIT                    :  integer := 2;        -- DOC: start spi writing of the mupix VDAC reg | MP_FEB
        constant WR_COL_BIT                     :  integer := 3;        -- DOC: start spi writing of the mupix COL reg | MP_FEB
        constant WR_TEST_BIT                    :  integer := 4;        -- DOC: start spi writing of the mupix TEST reg | MP_FEB
        constant WR_TDAC_BIT                    :  integer := 5;        -- DOC: start spi writing of the mupix TDAC reg | MP_FEB
        constant CLEAR_BIAS_FIFO_BIT            :  integer := 6;        -- DOC: clear Bias fifo in mp_ctrl | MP_FEB
        constant CLEAR_CONF_FIFO_BIT            :  integer := 7;        -- DOC: clear Bias fifo in mp_ctrl | MP_FEB
        constant CLEAR_VDAC_FIFO_BIT            :  integer := 8;        -- DOC: clear Bias fifo in mp_ctrl | MP_FEB
        constant CLEAR_COL_FIFO_BIT             :  integer := 9;        -- DOC: clear Bias fifo in mp_ctrl | MP_FEB
        constant CLEAR_TEST_FIFO_BIT            :  integer := 10;       -- DOC: clear Bias fifo in mp_ctrl | MP_FEB
        constant CLEAR_TDAC_FIFO_BIT            :  integer := 11;       -- DOC: clear Bias fifo in mp_ctrl | MP_FEB

    constant MP_CTRL_BIAS_REGISTER_W            :  integer := 16#0401#; -- DOC: If you want to write the mupix BIAS reg only, send data here | MP_FEB
    constant MP_CTRL_CONF_REGISTER_W            :  integer := 16#0402#; -- DOC: If you want to write the mupix CONF reg only, send data here | MP_FEB
    constant MP_CTRL_VDAC_REGISTER_W            :  integer := 16#0403#; -- DOC: If you want to write the mupix VDAC reg only, send data here | MP_FEB
    constant MP_CTRL_COL_REGISTER_W             :  integer := 16#0404#; -- DOC: If you want to write the mupix COL reg only, send data here | MP_FEB
    constant MP_CTRL_TEST_REGISTER_W            :  integer := 16#0405#; -- DOC: If you want to write the mupix TEST reg only, send data here | MP_FEB
    constant MP_CTRL_TDAC_REGISTER_W            :  integer := 16#0406#; -- DOC: If you want to write the mupix TDAC reg only, send data here | MP_FEB

    constant MP_CTRL_SLOW_DOWN_REGISTER_W       :  integer := 16#0407#; -- DOC: Division factor for the mupix spi clk | MP_FEB
    constant MP_CTRL_CHIP_MASK_REGISTER_W       :  integer := 16#0408#; -- DOC: MASK for the SPI writing to mupix (one can write in parallel) | MP_FEB
    constant MP_CTRL_INVERT_REGISTER_W          :  integer := 16#0409#; -- DOC: To be removed, inversions in mupix SPI | MP_FEB
        constant MP_CTRL_INVERT_29_BIT          :  integer := 0;        -- DOC: inverts oder of the mupix 29 bit spi shift reg | MP_FEB
        constant MP_CTRL_INVERT_CSN_BIT         :  integer := 1;        -- DOC: inverts mupix spi chip select bit | MP_FEB

    constant MP_CTRL_ALL_REGISTER_W             :  integer := 16#040A#; -- DOC: Write complete mupix configuration to this address | MP_FEB
    constant MP_CTRL_SPI_BUSY_REGISTER_R        :  integer := 16#040F#; -- DOC: Indicates if the mupix spi is busy, do not send new data | MP_FEB

-----------------------------------------------------------------
---- mupix datapath (0x1200-0xFBFF)------------------------------
-----------------------------------------------------------------

    constant MP_READOUT_MODE_REGISTER_W         :  integer := 16#1200#; -- DOC: to be removed | MP_FEB
        constant INVERT_TS_BIT                  :  integer := 0;        -- DOC: if set: TS is inverted | MP_FEB
        constant INVERT_TS2_BIT                 :  integer := 1;        -- DOC: if set: TS2 is inverted | MP_FEB
        constant GRAY_TS_BIT                    :  integer := 2;        -- DOC: if set: TS is grey-decoded | MP_FEB
        constant GRAY_TS2_BIT                   :  integer := 3;        -- DOC: if set: TS2 is grey-decoded | MP_FEB
        subtype  CHIP_ID_MODE_RANGE             is integer range 5 downto 4; -- DOC: bits to select different chip id numbering modes (not in use) | MP_FEB
        subtype  TOT_MODE_RANGE                 is integer range 8 downto 6; -- DOC: bits to select different TOT calculation modes (Default is to send TS2 as TOT, not in use) | MP_FEB
    constant MP_LVDS_LINK_MASK_REGISTER_W       :  integer := 16#1201#; -- DOC: masking of mupix lvds connections | MP_FEB
    constant MP_LVDS_LINK_MASK2_REGISTER_W      :  integer := 16#1202#; -- DOC: masking of mupix lvds connections | MP_FEB
    constant MP_DATA_GEN_CONTROL_REGISTER_W     :  integer := 16#1203#; -- DOC: controls the mupix data generator | MP_FEB
        subtype  MP_DATA_GEN_HIT_P_RANGE        is integer range 3 downto 0; -- DOC: generator hit output probability, 1/(2^(MP_DATA_GEN_HIT_P_RANGE+1)) for each cycle where a hit could be send | MP_FEB
        constant MP_DATA_GEN_FULL_STEAM_BIT     :  integer := 4;        -- DOC: if set: generator hit output probability is 1 | MP_FEB
        constant MP_DATA_GEN_SYNC_BIT           :  integer := 5;        -- DOC: if set: generator seed is the same on all boards else: generator seed depends on FPGA_ID | MP_FEB
        constant MP_DATA_GEN_ENGAGE_BIT         :  integer := 16;       -- DOC: if set: use hits from generator, datapath is not connected to link | MP_FEB
        constant MP_DATA_GEN_SORT_IN_BIT        :  integer := 17;       -- DOC: if set: generated hits are inserted after the data_unpacker (bevore sorter) | MP_FEB
        constant MP_DATA_GEN_ENABLE_BIT         :  integer := 31;       -- DOC: if set: data generator generates hits | MP_FEB
    constant MP_LVDS_INVERT_REGISTER_W          :  integer := 16#1204#;       -- DOC: inverting mupix lvds lines | MP_FEB 
    constant MP_DATA_BYPASS_SELECT_REGISTER_W   :  integer := 16#1205#;       -- DOC: bypass the mupix soter and put input to_integer(THISREG) directly on optical link (implemented but not connected in top) | MP_FEB
    constant MP_TS_HISTO_SELECT_REGISTER_W      :  integer := 16#1206#;       -- DOC: not in use | MP_FEB
        subtype  MP_TS_HISTO_LINK_SELECT_RANGE  is integer range 15 downto 0; -- DOC: not in use | MP_FEB
        subtype  MP_TS_HISTO_N_SAMPLE_RANGE     is integer range 31 downto 16;-- DOC: not in use | MP_FEB
    constant MP_LAST_SORTER_HIT_REGISTER_R      :  integer := 16#1207#;       -- DOC: register that contains the last mupix hit of the sorter output | MP_FEB
    constant MP_SORTER_INJECT_REGISTER_W        :  integer := 16#1208#;       -- DOC: used to inject single hits at the sorter inputs | MP_FEB
        -- select the input of the sorter to inject to
        subtype MP_SORTER_INJECT_SELECT_RANGE   is integer range 7 downto 4;  -- DOC: input of the sorter to inject to | MP_FEB
        -- rising edge on this bit will trigger a single inject of the word MP_SORTER_INJECT_REGISTER_W at sorter input MP_SORTER_INJECT_REGISTER_W(MP_SORTER_INJECT_SELECT_RANGE)
        constant MP_SORTER_INJECT_ENABLE_BIT    :  integer := 8;              -- DOC: rising_edge: single hit is injected | MP_FEB
    constant MP_HIT_ENA_CNT_REGISTER_R          :  integer := 16#1209#;       -- DOC: hit enable counter | MP_FEB
    constant MP_HIT_ENA_CNT_SELECT_REGISTER_W   :  integer := 16#120A#;       -- DOC: register to select the link for hit ena counter | MP_FEB
    constant MP_HIT_ENA_CNT_SORTER_IN_REGISTER_R :  integer := 16#120B#;      -- DOC: hit enable counter at the sorter input | MP_FEB
    constant MP_HIT_ENA_CNT_SORTER_SELECT_REGISTER_W :  integer := 16#120C#;  -- DOC: register to select the link for the sorter input hin ena counter | MP_FEB
    constant MP_HIT_ENA_CNT_SORTER_OUT_REGISTER_R : integer := 16#120D#;      -- DOC: hit counter at sorter output | MP_FEB
    constant MP_RESET_LVDS_N_REGISTER_W         :  integer := 16#120F#;       -- DOC: reset register for mupix lvds rx | MP_FEB

-----------------------------------------------------------------
---- mupix lvds rx (0x1100-0x11FF)-------------------------------
-----------------------------------------------------------------

    constant MP_LVDS_STATUS_START_REGISTER_W    :  integer := 16#1100#;       -- DOC: start of lvds status register block, 1 Word for each lvds link from here on | MP_FEB
        subtype  MP_LVDS_STATUS_DISP_ERR_RANGE  is integer range 27 downto 0; -- DOC: Disparity error counter in each lvds status register | MP_FEB 
        constant MP_LVDS_STATUS_PLL_LOCKED_BIT  :  integer := 28;             -- DOC: PLL locked bit in each lvds status register | MP_FEB 
        subtype  MP_LVDS_STATUS_STATE_RANGE     is integer range 30 downto 29;-- DOC: status Bit in each lvds status register | MP_FEB 
        constant MP_LVDS_STATUS_READY_BIT       :  integer := 31;             -- DOC: if set: this mupix lvds link is locked and ready | MP_FEB

-----------------------------------------------------------------
---- mupix sorter (0x1000-0x10FF)--------------------------------
-----------------------------------------------------------------

    constant MP_SORTER_COUNTER_REGISTER_R           :  integer := 16#1000#;       -- DOC: Hit counters in the sorter, 40 32 bit counters in total. For the inner pixel FEBs: 12 counters with in-time hits per chip, 12 counters with out-of-time hits per chip, 12 counters with overflows per chip, a counter with the number of output hits and the current credit value. The last two counters are currently reserved for future use | MP_FEB
    constant MP_SORTER_NINTIME_REGISTER_R           :  integer := 16#1000#;
    constant MP_SORTER_NOUTOFTIME_REGISTER_R        :  integer := 16#100C#;
    constant MP_SORTER_NOVERFLOW_REGISTER_R         :  integer := 16#1018#;
    constant MP_SORTER_NOUT_REGISTER_R              :  integer := 16#1024#;
    constant MP_SORTER_CREDIT_REGISTER_R            :  integer := 16#1025#;
    constant MP_SORTER_DELAY_REGISTER_W             :  integer := 16#1028#;       -- DOC: Minimum round-trip delay from sync reset going off to hit with TS > 0 appearing at sorter input in 8 ns steps | MP_FEB
    constant MP_SORTER_ZERO_SUPPRESSION_REGISTER_W  :  integer := 16#1029#;

end package mupix_registers;
