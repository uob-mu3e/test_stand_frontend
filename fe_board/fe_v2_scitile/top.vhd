----------------------------------------
-- SciTile version of the Frontend Board
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

    -- Tile DAB signals for FEB connector #3  / Inner Ring on SSW
    tileA_din                    : in    std_logic_vector(13 downto 1);
    tileA_pll_test               : out   std_logic; -- test pulse injection
    tileA_pll_reset              : out   std_logic; -- main reset (synchronisation and ASIC state machines)
    --SPI interface for ASICs
    tileA_spi_sclk_n             : out   std_logic; --spare out is inverted on CON2 of FEB, the equivalent net on CON2 is not.
    tileA_spi_mosi_n             : out   std_logic;
    tileA_spi_miso_n             : in    std_logic;
    --I2C interface for TMB control/monitoring
    tileA_i2c_sda_io             : inout std_logic;
    tileA_i2c_scl_io             : inout std_logic;

    -- Tile DAB signals for FEB connector #2  / Outer Ring on SSW
    tileB_din                    : in    std_logic_vector(13 downto 1);
    tileB_pll_test               : out   std_logic; -- test pulse injection
    tileB_pll_reset              : out   std_logic; -- main reset (synchronisation and ASIC state machines)
    --SPI interface for ASICs
    tileB_spi_sclk               : out   std_logic;
    tileB_spi_mosi_n             : out   std_logic;
    tileB_spi_miso_n             : in    std_logic;
    --I2C interface for TMB control/monitoring
    tileB_i2c_sda_io             : inout std_logic;
    tileB_i2c_scl_io             : inout std_logic;


    --DEPRECATED signals
--    tile_i2c_int                : in    std_logic;
--    tile_chip_reset             : out   std_logic;



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


    -- non-inverted io signals
    signal tileA_spi_mosi           : std_logic;
    signal tileB_spi_mosi           : std_logic;
    signal tileA_spi_miso           : std_logic;
    signal tileB_spi_miso           : std_logic;

    -- clocks & resets
    signal clk_125, reset_125_n     : std_logic;
    signal clk_156, reset_156_n     : std_logic;

    -- Debouncers
    signal pb_db                    : std_logic_vector(1 downto 0);

    constant N_LINKS                : integer := 1;
    constant N_ASICS                : integer := 13;
    constant N_MODULES              : integer := 1;
    constant IS_TILE_B              : boolean := false;

    signal fifo_write               : std_logic_vector(N_LINKS-1 downto 0);
    signal fifo_wdata               : std_logic_vector(36*(N_LINKS-1)+35 downto 0);

    signal malibu_reg               : work.util.rw_t;

    signal run_state_125            : run_state_t;
    signal ack_run_prep_permission  : std_logic;
    signal common_fifos_almost_full : std_logic_vector(N_LINKS-1 downto 0);
    signal s_run_state_all_done     : std_logic;
    signal s_MON_rxrdy              : std_logic_vector(N_MODULES*N_ASICS-1 downto 0);

    -- TMB interface / internal signals after selecting connector
    signal tile_pll_test               : std_logic; -- test pulse injection
    signal tile_pll_reset              : std_logic_vector(0 downto 0); -- main reset (synchronisation and ASIC state machines)
    signal tile_pll_reset_shifted      : std_logic_vector(0 downto 0);
	 
    --SPI interface for ASICs
    signal tile_spi_sclk               : std_logic;
    signal tile_spi_mosi               : std_logic;
    signal tile_spi_miso               : std_logic;

    -- tile_din cannot be a signal just for the selected connector since we need all 26 inputs to go to the same rx_block
    signal tile_din                    : std_logic_vector(12 downto 0);

    -- i2c interface (fe_block to io buffers)
    signal tileA_i2c_scl, tileA_i2c_scl_oe, tileA_i2c_sda, tileA_i2c_sda_oe : std_logic;
    signal tileB_i2c_scl, tileB_i2c_scl_oe, tileB_i2c_sda, tileB_i2c_sda_oe : std_logic;
    signal tile_i2c_scl_oe, tile_i2c_sda_oe : std_logic;

    -- spi multiplexing
    signal tmb_ss_n : std_logic_vector(15 downto 0);
begin

    -- io inversions:
    tileA_spi_mosi_n <= not tileA_spi_mosi;
    tileB_spi_mosi_n <= not tileB_spi_mosi;
    tileA_spi_miso <= not tileA_spi_miso_n;
    tileB_spi_miso <= not tileB_spi_miso_n;




    e_reset_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_125_n, i_reset_n => pb_db(0), i_clk => clk_125 );

    clk_156 <= transceiver_pll_clock(0);

    e_reset_156_n : entity work.reset_sync
    port map ( o_reset_n => reset_156_n, i_reset_n => pb_db(0), i_clk => clk_156 );



--------------------------------------------------------------------
--------------------------------------------------------------------
----TILE SUB-DETECTOR FIRMWARE -------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------
-- IO buffers for I2C
    iobuf_tileA_sda: entity work.ip_iobuf
    port map(
        datain(0)   => '0',
        oe(0)       => tileA_i2c_sda_oe,
        dataout(0)  => tileA_i2c_sda,
        dataio(0)   => tileA_i2c_sda_io
    );

    iobuf_tileA_scl: entity work.ip_iobuf
    port map(
        datain(0)   => '0',
        oe(0)       => tileA_i2c_scl_oe,
        dataout(0)  => tileA_i2c_scl,
        dataio(0)   => tileA_i2c_scl_io--,
    );

    iobuf_tileB_sda: entity work.ip_iobuf
    port map(
        datain(0)   => '0',
        oe(0)       => tileB_i2c_sda_oe,
        dataout(0)  => tileB_i2c_sda,
        dataio(0)   => tileB_i2c_sda_io
    );

    iobuf_tileB_scl: entity work.ip_iobuf
    port map(
        datain(0)   => '0',
        oe(0)       => tileB_i2c_scl_oe,
        dataout(0)  => tileB_i2c_scl,
        dataio(0)   => tileB_i2c_scl_io--,
    );

    
-- Selection of connector.
    g_DAB_interconnect_A: if IS_TILE_B=false generate
        tile_din         <= tileA_din(13 downto 1);
        tileA_pll_test   <= tile_pll_test;
        tileA_pll_reset  <= tile_pll_reset_shifted(0);
        tileA_spi_sclk_n <= not tile_spi_sclk;
        tileA_spi_mosi   <= tile_spi_mosi;
        tile_spi_miso    <= tileA_spi_miso;

        tileA_i2c_scl_oe <= tile_i2c_scl_oe;
        tileA_i2c_sda_oe <= tile_i2c_sda_oe;

        tileB_pll_test   <= '0';
        tileB_pll_reset  <= '0';
        tileB_spi_sclk   <= '0';
        tileB_spi_mosi   <= '0';

        tileB_i2c_scl_oe <= '0';
        tileB_i2c_sda_oe <= '0';
    end generate;
    g_DAB_interconnect_B: if IS_TILE_B=true generate
        tile_din         <= tileB_din(13 downto 1);
        tileB_pll_test   <= tile_pll_test;
        tileB_pll_reset  <= tile_pll_reset_shifted(0);
        tileB_spi_sclk   <= tile_spi_sclk;
        tileB_spi_mosi   <= tile_spi_mosi;
        tile_spi_miso    <= tileB_spi_miso;

        tileB_i2c_scl_oe <= tile_i2c_scl_oe;
        tileB_i2c_sda_oe <= tile_i2c_sda_oe;

        tileA_pll_test   <= '0';
        tileA_pll_reset  <= '0';
        tileA_spi_sclk_n <= '1';
        tileA_spi_mosi   <= '0';

        tileA_i2c_scl_oe <= '0';
        tileA_i2c_sda_oe <= '0';
    end generate;

-- Fast reset io/phase shifting
--ip_altiobuf_reset_inst : ip_altiobuf_reset
--port map(
--	datain => tile_pll_reset,
--	io_config_clk => spare_clk_osc,
--	io_config_clkena => "1",
--	io_config_datain => '1',
--	io_config_update  => '1',
--	dataout => tile_pll_reset_shifted--,
--	--dataout_b 
--	);
tile_pll_reset_shifted<= tile_pll_reset;
	 



-- SPI input multiplexing (CEC / configuration)
-- only input multiplexing is done here, the rest is done on the TMB

-- main datapath
    e_tile_path : entity work.tile_path
    generic map (
        N_MODULES       => N_MODULES,
        N_ASICS         => N_ASICS,
        N_LINKS         => N_LINKS,
        N_INPUTSRX      => 13,
        IS_TILE_B       => IS_TILE_B,
        INPUT_SIGNFLIP  => x"0000"&"0001111110000001",
        LVDS_PLL_FREQ   => 125.0,
        LVDS_DATA_RATE  => 1250.0--,
    )
    port map (
        i_reg_addr                  => malibu_reg.addr(15 downto 0),
        i_reg_re                    => malibu_reg.re,
        o_reg_rdata                 => malibu_reg.rdata,
        i_reg_we                    => malibu_reg.we,
        i_reg_wdata                 => malibu_reg.wdata,

        o_chip_reset                => open, --tile_chip_reset, --deprecated
        o_pll_test                  => tile_pll_test,
        i_data                      => tile_din,

        i_i2c_int                   => '1', -- tile_i2c_int, --deprecated
        o_pll_reset                 => tile_pll_reset(0),

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

        o_test_led                  => lcd_data(4),
        i_reset_125_n               => reset_125_n,
        i_reset_156_n               => reset_156_n,
        i_reset                     => not pb_db(0)--,
    );

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
        NIOS_CLK_MHZ_g  => 50.0--,
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

        i_spi_miso          => tile_spi_miso,
        o_spi_mosi          => tile_spi_mosi,
        o_spi_sclk          => tile_spi_sclk,
        o_spi_ss_n          => tmb_ss_n,

        i_i2c_scl           => tileA_i2c_scl,
        o_i2c_scl_oe        => tile_i2c_scl_oe,
        i_i2c_sda           => tileA_i2c_sda,
        o_i2c_sda_oe        => tile_i2c_sda_oe,

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

        o_subdet_reg_addr   => malibu_reg.addr(15 downto 0),
        o_subdet_reg_re     => malibu_reg.re,
        i_subdet_reg_rdata  => malibu_reg.rdata,
        o_subdet_reg_we     => malibu_reg.we,
        o_subdet_reg_wdata  => malibu_reg.wdata,

        -- reset system
        o_run_state_125             => run_state_125,
        i_ack_run_prep_permission   => and_reduce(s_MON_rxrdy),

        -- clocks
        i_nios_clk          => spare_clk_osc,
        o_nios_clk_mon      => lcd_data(0),
        i_clk_156           => transceiver_pll_clock(0),
        o_clk_156_mon       => lcd_data(1),
        i_clk_125           => lvds_firefly_clk,

        i_areset_n          => '1',--pb_db(0),

        i_testin            => pb_db(1)--,
    );

    max10_spi_sclk <= '1'; -- This is temporary until we only have v2.1 boards with the
    -- correct connection; for now we use it to know 2.1 from 2.0


    FPGA_Test(0) <= transceiver_pll_clock(0);
    FPGA_Test(1) <= lvds_firefly_clk;
    FPGA_Test(2) <= clk_125_top;

end rtl;
