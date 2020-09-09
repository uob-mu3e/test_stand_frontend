----------------------------------------
-- Dummy version of the Frontend Board 
-- Common Firmware only, no detector-specific parts
-- Martin Mueller, September 2020
----------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.daq_constants.all;
use work.cmp.all;

entity top is 
    port (
        fpga_reset                  : in    std_logic;

        LVDS_clk_si1_fpga_A         : in    std_logic; -- 125 MHz base clock for LVDS PLLs - right // SI5345
        LVDS_clk_si1_fpga_B         : in    std_logic; -- 125 MHz base clock for LVDS PLLs - left // SI5345
        transceiver_pll_clock       : in    std_logic_vector(2 downto 2); --_vector(1 downto 0); -- 125 MHz base clock for transceiver PLLs // SI5345 -- change to 156 ?
      --extra_transceiver_pll_clocks: in    std_logic_vector(1 downto 0); -- 125 MHz base clock for transceiver PLLs // SI5345

        lvds_firefly_clk            : in    std_logic; -- 125 MHz base clock

        systemclock                 : in    std_logic; -- 50 MHz system clock // SI5345
        systemclock_bottom          : in    std_logic; -- 50 MHz system clock // SI5345
        clk_125_top                 : in    std_logic; -- 125 MHz clock spare // SI5345
        clk_125_bottom              : in    std_logic; -- 125 Mhz clock spare // SI5345
        spare_clk_osc               : in    std_logic; -- Spare clock // 50 MHz oscillator

        -- Block A: Connections for three chips -- layer 0
        clock_A                     : out   std_logic;
        data_in_A                   : in    std_logic_vector(9 downto 1);
        fast_reset_A                : out   std_logic;
        SIN_A                       : out   std_logic;

        -- Block B: Connections for three chips -- layer 0
        clock_B                     : out   std_logic;
        data_in_B                   : in    std_logic_vector(9 downto 1);
        fast_reset_B                : out   std_logic;
        SIN_B                       : out   std_logic;

        -- Block C: Connections for three chips -- layer 1
        clock_C                     : out   std_logic;
        data_in_C                   : in    std_logic_vector(9 downto 1);
        fast_reset_C                : out   std_logic;
        SIN_C                       : out   std_logic;

        -- Block D: Connections for three chips -- layer 1
        clock_D                     : out   std_logic;
        data_in_D                   : in    std_logic_vector(9 downto 1);
        fast_reset_D                : out   std_logic;
        SIN_D                       : out   std_logic;

        -- Block E: Connections for three chips -- layer 1
        clock_E                     : out   std_logic;
        data_in_E                   : in    std_logic_vector(9 downto 1);
        fast_reset_E                : out   std_logic;
        SIN_E                       : out   std_logic;

        -- Extra signals
        clock_aux                   : out   std_logic;
        spare_out                   : out   std_logic_vector(3 downto 2);

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
        max10_spi_mosi              : out   std_logic;
        max10_spi_miso              : inout std_logic;
        max10_spi_D1                : inout std_logic;
        max10_spi_D2                : inout std_logic;
        max10_spi_D3                : inout std_logic;
        max10_spi_csn               : out   std_logic
        );
end top;

architecture rtl of top is
 
    -- Debouncers
    signal pb_db                : std_logic_vector(1 downto 0);
    signal version_out          : std_logic_vector(31 downto 0);

begin

--v_reg: version_reg 
--    PORT MAP(
--        data_out    => version_out(27 downto 0)
--    );

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

    -- Quad SPI IF to MAX	
    -- Tristate bidirectionals for a start
    max10_spi_miso  <= 'Z';
    max10_spi_D1    <= 'Z';
    max10_spi_D2    <= 'Z';
    max10_spi_D3    <= 'Z';

    -- Chip select is high	
    max10_spi_csn   <= '1';

    si45_rst_n <= (others => '1');
    si45_oe_n <= (others => '0');
    si45_fdec <= (others => '0');
    si45_finc <= (others => '0');

    e_fe_block : entity work.fe_block_v2
    generic map (
        NIOS_CLK_MHZ_g  => 50.0--,
    )
    port map (
        i_fpga_id           => X"FEB0",
        i_fpga_type         => "111010", -- This is MuPix, TODO: Adjust midas frontends to add "Dummy - type"

        io_i2c_ffly_scl     => Firefly_Scl,
        io_i2c_ffly_sda     => Firefly_Sda,
        o_i2c_ffly_ModSel_n => Firefly_ModSel_n,
        o_ffly_Rst_n        => Firefly_Rst_n,
        i_ffly_Int_n        => Firefly_Int_n,
        i_ffly_ModPrs_n     => Firefly_ModPrs_n,

        i_spi_miso          => '1',
        o_spi_mosi          => open,
        o_spi_sclk          => open,
        o_spi_ss_n          => open,

        i_spi_si_miso       => si45_spi_out,
        o_spi_si_mosi       => si45_spi_in,
        o_spi_si_sclk       => si45_spi_sclk,
        o_spi_si_ss_n       => si45_spi_cs_n,

        o_ffly1_tx          => firefly1_tx_data,
        o_ffly2_tx          => firefly2_tx_data,
        i_ffly1_rx          => firefly1_rx_data,
        i_ffly2_rx          => firefly2_rx_data,

        i_ffly1_lvds_rx     => firefly1_lvds_rx_in,
        i_ffly2_lvds_rx     => firefly2_lvds_rx_in,

        i_fifo_write        => (others => '0'), -- TODO in "Not-Dummy": connect to detector-block
        i_fifo_wdata        => (others => '0'), -- TODO in "Not-Dummy": connect to detector-block

        i_mscb_data         => mscb_fpga_in,
        o_mscb_data         => mscb_fpga_out,
        o_mscb_oe           => mscb_fpga_oe_n,

        o_mupix_reg_addr    => open, -- TODO in "Not-Dummy": connect to detector-block
        o_mupix_reg_re      => open,
        i_mupix_reg_rdata   => X"CCCCCCCC",
        o_mupix_reg_we      => open,
        o_mupix_reg_wdata   => open,

        o_malibu_reg_addr   => open, -- TODO in "Not-Dummy": connect to detector-block
        o_malibu_reg_re     => open,
        i_malibu_reg_rdata  => X"CCCCCCCC",
        o_malibu_reg_we     => open,
        o_malibu_reg_wdata  => open,

        o_scifi_reg_addr    => open, -- TODO in "Not-Dummy": connect to detector-block
        o_scifi_reg_re      => open,
        i_scifi_reg_rdata   => X"CCCCCCCC",
        o_scifi_reg_we      => open,
        o_scifi_reg_wdata   => open,
        
        -- reset system
        o_run_state_125     => open,      -- TODO in "Not-Dummy": connect to detector-block
        i_ack_run_prep_permission => '1', -- TODO in "Not-Dummy": connect to detector-block

        -- clocks
        i_nios_clk          => spare_clk_osc,
        o_nios_clk_mon      => lcd_data(0),
        i_clk_156           => transceiver_pll_clock(2),
        o_clk_156_mon       => lcd_data(1),
        i_clk_125           => lvds_firefly_clk,
        o_clk_125_mon       => lcd_data(2),

        i_areset_n          => pb_db(0),
        
        i_testin            => pb_db(1)--,
    );

end rtl;
