----------------------------------------
-- SciFi version of the Frontend Board
----------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;
use ieee.numeric_std.all;

use work.mudaq.all;

entity top is
    port (
        fpga_reset                  : in    std_logic;

        LVDS_clk_si1_fpga_A         : in    std_logic; -- 125 MHz base clock for LVDS PLLs - right // SI5345
        LVDS_clk_si1_fpga_B         : in    std_logic; -- 125 MHz base clock for LVDS PLLs - left // SI5345
        transceiver_pll_clock       : in    std_logic_vector(0 downto 0); --_vector(1 downto 0); -- 156.25 MHz  clock for transceiver PLL-- Using 1 or 2 with 156 Mhz gives warings about coupling in clock builder
      --extra_transceiver_pll_clocks: in    std_logic_vector(1 downto 0); -- 125 MHz base clock for transceiver PLLs // SI5345

        lvds_firefly_clk            : in    std_logic; -- 125 MHz base clock

        systemclock                 : in    std_logic; -- 50 MHz system clock // SI5345
        systemclock_bottom          : in    std_logic; -- 50 MHz system clock // SI5345
        clk_125_top                 : in    std_logic; -- 125 MHz clock spare // SI5345
        clk_125_bottom              : in    std_logic; -- 125 Mhz clock spare // SI5345
        spare_clk_osc               : in    std_logic; -- Spare clock // 50 MHz oscillator

        -- scifi DAB signals  (7 downto 4 and signalname2 is con3, the others are con2)
        scifi_din                   : in    std_logic_vector(7 downto 0);
        scifi_syncres               : out   std_logic;
        scifi_syncres2              : out   std_logic;
        scifi_csn                   : out   std_logic_vector(7 downto 0);
        scifi_spi_sclk              : out   std_logic;
        scifi_spi_miso              : in    std_logic;
        scifi_spi_mosi              : out   std_logic;
        scifi_temp_mutrig         : in    std_logic;
        scifi_temp_sipm           : in    std_logic;

        scifi_spi_sclk2             : out   std_logic;
        scifi_spi_miso2             : in    std_logic;
        scifi_spi_mosi2             : out   std_logic;
        -- not used at the current version of the DAB for sicfi
        scifi_cec_csn               : out   std_logic_vector(7 downto 0);
        scifi_cec_miso              : in    std_logic;
        scifi_fifo_ext              : out   std_logic;
        scifi_inject                : out   std_logic;
        scifi_cec_miso2             : in    std_logic;
        scifi_fifo_ext2             : out   std_logic;
        scifi_inject2               : out   std_logic;
        scifi_temp_mutrig2        : in    std_logic;
        scifi_temp_sipm2          : in    std_logic;

        -- Fireflies
        firefly1_tx_data            : out   std_logic_vector(3 downto 0); -- transceiver
        firefly2_tx_data            : out   std_logic_vector(3 downto 0); -- transceiver
        firefly1_rx_data            : in    std_logic;-- transceiver
        firefly2_rx_data            : in    std_logic_vector(2 downto 0);-- transceiver

        firefly1_lvds_rx_in         : in    std_logic;--_vector(1 downto 0); -- receiver for slow control or something else
        firefly2_lvds_rx_in         : in    std_logic;--_vector(1 downto 0); -- receiver for slow control or something else

        Firefly_ModSel_n            : out   std_logic_vector(1 downto 0);-- Module select: active low, when host wants to communicate (I2C) with module
        Firefly_Rst_n               : out   std_logic_vector(1 downto 0);-- Module reset: active low, complete reset of module. Module indicates reset done by "low" interrupt_n (data_not_ready is negated).
        Firefly_Scl                 : inout std_logic;-- I2C Clock: module asserts low for clock stretch, timing infos: page 47
        Firefly_Sda                 : inout std_logic;-- I2C Data
      --Firefly_LPM                 : out   std_logic;-- Firefly Low Power Mode: Modules power consumption should be below 1.5 W. active high. Overrideable by I2C commands. Override default: high power (page 19 of documentation).
        Firefly_Int_n               : in    std_logic_vector(1 downto 0);-- Firefly Interrupt: when low: operational fault or status critical. after reset: goes high, and data_not_ready is read with '0' (byte 2 bit 0) and flag field is read
        Firefly_ModPrs_n            : in    std_logic_vector(1 downto 0);-- Module present: Pulled to ground if module is present

        -- LEDs, test points and buttons
        PushButton                  : in    std_logic_vector(1 downto 0);
        FPGA_Test                   : out   std_logic_vector(7 downto 0);

        --LCD
        lcd_csn                     : out   std_logic;--//2.5V    //LCD Chip Select
        lcd_d_cn                    : out   std_logic;--//2.5V    //LCD Data / Command Select
        lcd_data                    : out   std_logic_vector(7 downto 0);--//2.5V    //LCD Data
        lcd_wen                     : out   std_logic;--//2.5V    //LCD Write Enable

        -- SI5345(0): 7 Transceiver clocks @ 125 MHz
        -- SI4345(1): Clocks for the Fibres
        -- 1 reference and 2 inputs for synch
        si45_oe_n                   : out   std_logic_vector(1 downto 0);-- active low output enable -> should always be '0'
        si45_intr_n                 : in    std_logic_vector(1 downto 0);-- fault monitor: interrupt pin: change in state of status indicators
        si45_lol_n                  : in    std_logic_vector(1 downto 0);-- fault monitor: loss of lock of DSPLL

        -- I2C sel is set to GND on PCB -> SPI interface
        si45_rst_n                  : out   std_logic_vector(1 downto 0);--	reset
        si45_spi_cs_n               : out   std_logic_vector(1 downto 0);-- chip select
        si45_spi_in                 : out   std_logic_vector(1 downto 0);-- data in
        si45_spi_out                : in    std_logic_vector(1 downto 0);-- data out
        si45_spi_sclk               : out   std_logic_vector(1 downto 0);-- clock

        -- change frequency by the FSTEPW parameter
        si45_fdec                   : out   std_logic_vector(1 downto 0);-- decrease
        si45_finc                   : out   std_logic_vector(1 downto 0);-- increase

        -- Midas slow control bus
        mscb_fpga_in                : in    std_logic;
        mscb_fpga_out               : out   std_logic;
        mscb_fpga_oe_n              : out   std_logic;

        -- Backplane slot signal
        ref_adr                     : in    std_logic_vector(7 downto 0);

        -- MAX10 IF
        max10_spi_sclk              : out   std_logic;
        max10_spi_mosi              : inout std_logic;
        max10_spi_miso              : inout std_logic;
        max10_spi_D1                : inout std_logic;
        max10_spi_D2                : inout std_logic;
        max10_spi_D3                : inout std_logic;
        max10_spi_csn               : out   std_logic
        );
end top;

architecture rtl of top is

    -- Debouncers
    signal pb_db                    : std_logic_vector(1 downto 0);

    constant N_LINKS                : integer := 2;
    constant N_ASICS                : integer := 4;
    constant N_MODULES              : integer := 2;

    signal fifo_write               : std_logic_vector(N_LINKS-1 downto 0);
    signal fifo_wdata               : std_logic_vector(36*(N_LINKS-1)+35 downto 0);

    signal scifi_reg                : work.util.rw_t;

    signal run_state_125            : run_state_t;
    signal run_state_156            : run_state_t;
    signal run_state_125_prev       : run_state_t;
    signal ack_run_prep_permission  : std_logic;
    signal common_fifos_almost_full : std_logic_vector(N_LINKS-1 downto 0);
    signal s_run_state_all_done     : std_logic;
    signal s_MON_rxrdy              : std_logic_vector(N_MODULES*N_ASICS-1 downto 0);

    -- SPI SMB
    --internal I/O signals to DAB with correct polarity
    signal scifi_int_din                   : std_logic_vector(4 downto 1);
    signal scifi_int_syncres               : std_logic;
    signal scifi_int_syncres2              : std_logic;
    signal scifi_int_csn                   : std_logic_vector(15 downto 0);
    signal scifi_int_spi_sclk              : std_logic;
    signal scifi_int_spi_miso              : std_logic;
    signal scifi_int_spi_mosi              : std_logic;
    signal scifi_csn_buf                   : std_logic_vector(7 downto 0);

    --signal scifi_int_cec_csn               : std_logic_vector(4 downto 1);
    --signal scifi_int_cec_miso              : std_logic;
    --signal scifi_int_fifo_ext              : std_logic;
    --signal scifi_int_inject                : std_logic;
    --signal scifi_int_bidir_test            : std_logic;
    
    signal sync_cnt : integer range 0 to 255 := 0;
    signal chip_reset : std_logic := '0';
    signal counter_vec : std_logic_vector(31 downto 0);
    signal counter : integer;
    
    signal pll_test : std_logic;

    signal fast_pll_clk : std_logic;

begin
--------------------------------------------------------------------
--------------------------------------------------------------------
---- SciFi SUB-DETECTOR FIRMWARE -----------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
    scifi_csn <= scifi_csn_buf;
	 
	 -- stuff which is not used at the moment but PINs need to be tested
	 lcd_data(2) <= scifi_temp_mutrig or scifi_temp_mutrig2 or scifi_temp_sipm or scifi_temp_sipm2 or scifi_cec_miso or scifi_cec_miso2;

-- assignments of DAB pins: special IOBUF, constant and polarity flips here
    scifi_fifo_ext              <= '0';
    scifi_inject                <= pll_test;
    scifi_fifo_ext2             <= '0';
    scifi_inject2               <= pll_test;
    scifi_cec_csn               <= (others => '1');

    scifi_csn_buf(0) <= not scifi_int_csn(0);
    scifi_csn_buf(1) <= not scifi_int_csn(1);
    scifi_csn_buf(2) <= not scifi_int_csn(2);
    scifi_csn_buf(3) <=     scifi_int_csn(3);
    scifi_csn_buf(4) <=     scifi_int_csn(4);
    scifi_csn_buf(5) <= not scifi_int_csn(5);
    scifi_csn_buf(6) <= not scifi_int_csn(6);
    scifi_csn_buf(7) <=     scifi_int_csn(7);

    scifi_syncres   <= not scifi_int_syncres;
    scifi_syncres2  <= not scifi_int_syncres2;
    
    scifi_spi_sclk      <= not scifi_int_spi_sclk;
    scifi_spi_mosi      <= not scifi_int_spi_mosi;
    scifi_spi_sclk2     <= not scifi_int_spi_sclk;
    scifi_spi_mosi2     <= not scifi_int_spi_mosi;

    scifi_int_spi_miso <=  (not scifi_spi_miso2) when (scifi_csn_buf(7 downto 4) /= x"F") else (not scifi_spi_miso);

    -- LVDS inputs signflip in receiver block generic
     
-- scifi detector firmware
    e_tile_path : entity work.scifi_path
    generic map (
        IS_SCITILE      => '0',
        N_MODULES       => N_MODULES,
        N_ASICS         => N_ASICS,
        N_LINKS         => N_LINKS,
        INPUT_SIGNFLIP  => x"FFFFFFFF",
        LVDS_PLL_FREQ   => 125.0,
        LVDS_DATA_RATE  => 1250.0--,
    )
    port map (
        i_reg_addr                  => scifi_reg.addr(7 downto 0),
        i_reg_re                    => scifi_reg.re,
        o_reg_rdata                 => scifi_reg.rdata,
        i_reg_we                    => scifi_reg.we,
        i_reg_wdata                 => scifi_reg.wdata,

        o_chip_reset(0)             => open,
        o_chip_reset(1)             => open,

        o_pll_test                  => pll_test,
        i_data                      => scifi_din,

        io_i2c_sda                  => open,
        io_i2c_scl                  => open,
        i_cec                       => '0',
        i_spi_miso                  => '0',
        i_i2c_int                   => '0',
        o_pll_reset                 => open,
        o_spi_scl                   => open,
        o_spi_mosi                  => open,

        o_fifo_write                => fifo_write,
        o_fifo_wdata                => fifo_wdata,

        i_common_fifos_almost_full  => common_fifos_almost_full,

        i_run_state                 => run_state_125,
        o_run_state_all_done        => s_run_state_all_done,

        o_MON_rxrdy                 => s_MON_rxrdy,

        i_clk_core                  => transceiver_pll_clock(0),
        i_clk_g125                  => lvds_firefly_clk,
        i_clk_ref_A                 => LVDS_clk_si1_fpga_A,
        i_clk_ref_B                 => LVDS_clk_si1_fpga_B,

        o_fast_pll_clk              => fast_pll_clk,
        o_test_led                  => lcd_data(4 downto 3),
        i_reset                     => not pb_db(0)--,
    );

    process(lvds_firefly_clk)
    begin
    if ( falling_edge(lvds_firefly_clk) ) then
        if ( run_state_125 = RUN_STATE_SYNC and sync_cnt <= 10 ) then
            scifi_int_syncres <= '1';
            scifi_int_syncres2 <= '1';
            sync_cnt <= sync_cnt + 1;
        else
            scifi_int_syncres <= '0';
            scifi_int_syncres2 <= '0';
            sync_cnt <= 0;
        end if;
    end if;
    end process;

--    counter_vec <= std_logic_vector(to_unsigned(counter, counter_vec'length));
--    process(LVDS_clk_si1_fpga_A)
--    begin
--    if ( rising_edge(LVDS_clk_si1_fpga_A) ) then
--        if ( run_state_125 = RUN_STATE_SYNC ) then
--            counter <= counter +1;
--            if(counter_vec(6)='1') then
--                scifi_int_syncres <= not scifi_int_syncres;
--                scifi_int_syncres2 <= not scifi_int_syncres2;
--                counter <= 0;
--            end if;
--        elsif (run_state_125 = RUN_STATE_RUNNING) then
--            counter <= counter +1;
--            if(counter_vec(6)='1') then
--                if(scifi_int_syncres = '1') then 
--                    scifi_int_syncres <= not scifi_int_syncres;
--                    scifi_int_syncres2 <= not scifi_int_syncres2;
--                end if;
--                counter <= 0;
--            end if;
--        else
--            scifi_int_syncres <= '0';
--            scifi_int_syncres2 <= '0';
--        end if;
--    end if;
--    end process;
--
--    process(fast_pll_clk)
--    begin
--    if ( rising_edge(fast_pll_clk) ) then
--        run_state_125_prev <= run_state_125;
--        if ( run_state_125 = RUN_STATE_SYNC and run_state_125_prev/=RUN_STATE_SYNC) then
--            scifi_int_syncres <= '1';
--            scifi_int_syncres2 <= '1';
--        else
--            scifi_int_syncres <= '0';
--            scifi_int_syncres2 <= '0';
--        end if;
--    end if;
--    end process;

    --scifi_int_syncres <= chip_reset;
    --scifi_int_syncres2 <= chip_reset;

--------------------------------------------------------------------
--------------------------------------------------------------------
---- COMMON FIRMWARE PART ------------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

    db1: entity work.debouncer
    port map(
        i_clk       => spare_clk_osc,
        i_reset_n   => '1',
        i_d(0)      => PushButton(0),
        o_q(0)      => pb_db(0)--,
    );

    db2: entity work.debouncer
    port map(
        i_clk       => spare_clk_osc,
        i_reset_n   => '1',
        i_d(0)      => PushButton(1),
        o_q(0)      => pb_db(1)--,
    );

    e_fe_block : entity work.fe_block_v2
    generic map (
        NIOS_CLK_MHZ_g  => 50.0,
        N_LINKS => N_LINKS--,
    )
    port map (
        i_fpga_id           => ref_adr,
        i_fpga_type         => "111010", -- This is MuPix, TODO: Adjust midas frontends to add "Dummy - type"

        io_i2c_ffly_scl     => Firefly_Scl,
        io_i2c_ffly_sda     => Firefly_Sda,
        o_i2c_ffly_ModSel_n => Firefly_ModSel_n,
        o_ffly_Rst_n        => Firefly_Rst_n,
        i_ffly_Int_n        => Firefly_Int_n,
        i_ffly_ModPrs_n     => Firefly_ModPrs_n,

        i_spi_miso          => scifi_int_spi_miso,
        o_spi_mosi          => scifi_int_spi_mosi,
        o_spi_sclk          => scifi_int_spi_sclk,
        o_spi_ss_n          => scifi_int_csn,

        i_spi_si_miso       => si45_spi_out,
        o_spi_si_mosi       => si45_spi_in,
        o_spi_si_sclk       => si45_spi_sclk,
        o_spi_si_ss_n       => si45_spi_cs_n,

        o_si45_oe_n         => si45_oe_n,
        i_si45_intr_n       => si45_intr_n,
        i_si45_lol_n        => si45_lol_n,
        o_si45_rst_n        => si45_rst_n,
        o_si45_fdec         => si45_fdec,
        o_si45_finc         => si45_finc,

        o_ffly1_tx          => firefly1_tx_data,
        o_ffly2_tx          => firefly2_tx_data,
        i_ffly1_rx          => firefly1_rx_data,
        i_ffly2_rx          => firefly2_rx_data,

        i_ffly1_lvds_rx     => firefly1_lvds_rx_in,
        i_ffly2_lvds_rx     => firefly2_lvds_rx_in,

        i_can_terminate     => s_run_state_all_done,

        i_fifo_write        => fifo_write,
        i_fifo_wdata        => fifo_wdata,

        o_fifos_almost_full => common_fifos_almost_full,

        i_mscb_data         => mscb_fpga_in,
        o_mscb_data         => mscb_fpga_out,
        o_mscb_oe           => mscb_fpga_oe_n,

        o_max10_spi_sclk    => max10_spi_miso, --max10_spi_sclk, Replacement, due to broken line
        io_max10_spi_mosi   => max10_spi_mosi,
        io_max10_spi_miso   => 'Z',
        io_max10_spi_D1     => max10_spi_D1,
        io_max10_spi_D2     => max10_spi_D2,
        io_max10_spi_D3     => max10_spi_D3,
        o_max10_spi_csn     => max10_spi_csn,

        o_subdet_reg_addr   => scifi_reg.addr(7 downto 0),
        o_subdet_reg_re     => scifi_reg.re,
        i_subdet_reg_rdata  => scifi_reg.rdata,
        o_subdet_reg_we     => scifi_reg.we,
        o_subdet_reg_wdata  => scifi_reg.wdata,

        -- reset system
        o_run_state_125             => run_state_125,
        i_ack_run_prep_permission   => '1',--and_reduce(s_MON_rxrdy), --TODO: check what s_MON_rxrdy does and if it works

        -- clocks
        i_nios_clk          => spare_clk_osc,
        o_nios_clk_mon      => lcd_data(0),
        i_clk_156           => transceiver_pll_clock(0),
        o_clk_156_mon       => lcd_data(1),
        i_clk_125           => lvds_firefly_clk,

        i_areset_n          => pb_db(0),

        i_testin            => pb_db(1)--,
    );

    max10_spi_sclk <= '1'; -- This is temporary until we only have v2.1 boards with the
    -- correct connection; for now we use it to know 2.1 from 2.0


    FPGA_Test(0) <= transceiver_pll_clock(0);
    FPGA_Test(1) <= lvds_firefly_clk;
    FPGA_Test(2) <= clk_125_top;
end rtl;
