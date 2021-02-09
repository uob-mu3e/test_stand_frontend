----------------------------------------
-- Mupix version of the Frontend Board
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
        transceiver_pll_clock       : in    std_logic_vector(0 downto 0); --_vector(1 downto 0); -- 156.25 MHz  clock for transceiver PLL-- Using 1 or 2 with 156 Mhz gives warings about coupling in clock builder
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
        mosi_A                      : out   std_logic;
        csn_A                       : out   std_logic_vector(2 downto 0);

        -- Block B: Connections for three chips -- layer 0
        clock_B                     : out   std_logic;
        data_in_B                   : in    std_logic_vector(9 downto 1);
        fast_reset_B                : out   std_logic;
        SIN_B                       : out   std_logic;
        mosi_B                      : out   std_logic;
        csn_B                       : out   std_logic_vector(2 downto 0);

        -- Block C: Connections for three chips -- layer 1
        clock_C                     : out   std_logic;
        data_in_C                   : in    std_logic_vector(9 downto 1);
        fast_reset_C                : out   std_logic;
        SIN_C                       : out   std_logic;
        mosi_C                      : out   std_logic;
        csn_C                       : out   std_logic_vector(2 downto 0);

        -- Block D: Connections for three chips -- layer 1
        clock_D                     : out   std_logic;
        data_in_D                   : in    std_logic_vector(9 downto 1);
        fast_reset_D                : out   std_logic;
        SIN_D                       : out   std_logic;
        mosi_D                      : out   std_logic;
        csn_D                       : out   std_logic_vector(2 downto 0);

        -- Block E: Connections for three chips -- layer 1
        -- clock_E                     : out   std_logic;
        -- data_in_E                   : in    std_logic_vector(9 downto 1);
        -- fast_reset_E                : out   std_logic;
        -- SIN_E                       : out   std_logic;

        -- Extra signals
        
        --clock_aux                   : out   std_logic; -- Pin in use for csn_A[2] M.Mueller
        --spare_out                   : out   std_logic_vector(3 downto 2); -- Pins in use for csn_* M.Mueller

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
        max10_spi_D3                : out   std_logic;
        max10_spi_csn               : out   std_logic
        );
end top;

architecture rtl of top is

    -- Debouncers
    signal pb_db                    : std_logic_vector(1 downto 0);

    constant NPORTS                 : integer := 4;
    constant N_LINKS                : integer := 1;

    signal fifo_write               : std_logic_vector(N_LINKS-1 downto 0);
    signal fifo_wdata               : std_logic_vector(36*(N_LINKS-1)+35 downto 0); 

    signal mupix_reg                : work.util.rw_t;

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO                     : std_logic := '0';
    attribute keep                  : boolean;
    attribute keep of ZERO          : signal is true;

    signal i2c_scl, i2c_scl_oe, i2c_sda, i2c_sda_oe : std_logic;
    signal spi_miso, spi_mosi, spi_sclk : std_logic;
    signal spi_ss_n                 : std_logic_vector(15 downto 0);

    signal run_state_125            : run_state_t;
    signal run_state_156            : run_state_t;
    signal ack_run_prep_permission  : std_logic;

    signal sync_reset_cnt           : std_logic;
    signal nios_clock               : std_logic;

    signal mp_ctrl_clock            : std_logic_vector(3  downto 0);
    signal mp_ctrl_SIN              : std_logic_vector(3  downto 0);
    signal mp_ctrl_mosi             : std_logic_vector(3  downto 0);
    signal mp_ctrl_csn              : std_logic_vector(11 downto 0);

begin

--------------------------------------------------------------------
--------------------------------------------------------------------
----MUPIX SUB-DETECTOR FIRMWARE ------------------------------------
--------------------------------------------------------------------
--------------------------------------------------------------------

    clock_A <= mp_ctrl_clock(3);
    clock_B <= mp_ctrl_clock(2);
    clock_C <= mp_ctrl_clock(1);
    clock_D <= mp_ctrl_clock(0);

    SIN_A <= mp_ctrl_SIN(3);
    SIN_B <= mp_ctrl_SIN(2);
    SIN_C <= mp_ctrl_SIN(1);
    SIN_D <= mp_ctrl_SIN(0);

    mosi_A <= mp_ctrl_mosi(3);
    mosi_B <= mp_ctrl_mosi(2);
    mosi_C <= mp_ctrl_mosi(1);
    mosi_D <= mp_ctrl_mosi(0);

    csn_A <= mp_ctrl_csn(11 downto 9);
    csn_B <= mp_ctrl_csn( 8 downto 6);
    csn_C <= mp_ctrl_csn( 5 downto 3);
    csn_D <= mp_ctrl_csn( 2 downto 0);

    e_mupix_block : entity work.mupix_block
    port map (
        i_fpga_id               => ref_adr,

        -- config signals to mupix
        o_clock                 => mp_ctrl_clock,
        o_SIN                   => mp_ctrl_SIN,
        o_mosi                  => mp_ctrl_mosi,
        o_csn                   => mp_ctrl_csn,

        -- mupix dac regs
        i_reg_add               => mupix_reg.addr(7 downto 0),
        i_reg_re                => mupix_reg.re,
        o_reg_rdata             => mupix_reg.rdata,
        i_reg_we                => mupix_reg.we,
        i_reg_wdata             => mupix_reg.wdata,

        -- data
        o_fifo_wdata            => fifo_wdata,
        o_fifo_write            => fifo_write(0),

        i_run_state_125           => run_state_125,
        i_run_state_156           => run_state_156,
        o_ack_run_prep_permission => ack_run_prep_permission,

        i_lvds_data_in          => data_in_D & data_in_C & data_in_B & data_in_A,

        i_reset                 => not pb_db(0),
        -- 156.25 MHz
        i_clk156                => transceiver_pll_clock(0),
        i_clk125                => lvds_firefly_clk,
        i_lvds_rx_inclock_A     => LVDS_clk_si1_fpga_A,
        i_lvds_rx_inclock_B     => LVDS_clk_si1_fpga_B,
        i_sync_reset_cnt        => sync_reset_cnt--,
    );

    process(lvds_firefly_clk)
    begin
    if falling_edge(lvds_firefly_clk) then
        if(run_state_125 = RUN_STATE_SYNC)then
            fast_reset_A    <= '1';
            fast_reset_B    <= '1';
            fast_reset_C    <= '1';
            fast_reset_D    <= '1';
            --fast_reset_E    <= '1';
            sync_reset_cnt  <= '1';
        else
            fast_reset_A    <= '0';
            fast_reset_B    <= '0';
            fast_reset_C    <= '0';
            fast_reset_D    <= '0';
            --fast_reset_E    <= '0';
            sync_reset_cnt  <= '0';
        end if;
    end if;
    end process;

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

        i_spi_miso          => '1',
        o_spi_mosi          => open,
        o_spi_sclk          => open,
        o_spi_ss_n          => open,

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

        i_fifo_write        => fifo_write,
        i_fifo_wdata        => fifo_wdata,

        i_mscb_data         => mscb_fpga_in,
        o_mscb_data         => mscb_fpga_out,
        o_mscb_oe           => mscb_fpga_oe_n,

        o_max10_spi_sclk    => max10_spi_sclk,
        io_max10_spi_mosi   => max10_spi_mosi,
        io_max10_spi_miso   => max10_spi_miso,
        io_max10_spi_D1     => max10_spi_D1,
        io_max10_spi_D2     => max10_spi_D2,
        o_max10_spi_D3      => max10_spi_D3,
        o_max10_spi_csn     => max10_spi_csn,

        o_subdet_reg_addr   => mupix_reg.addr(7 downto 0),
        o_subdet_reg_re     => mupix_reg.re,
        i_subdet_reg_rdata  => mupix_reg.rdata,
        o_subdet_reg_we     => mupix_reg.we,
        o_subdet_reg_wdata  => mupix_reg.wdata,

        -- reset system
        o_run_state_125     => run_state_125,
        o_run_state_156     => run_state_156,
        i_ack_run_prep_permission => ack_run_prep_permission,

        -- clocks
        i_nios_clk          => spare_clk_osc,
        o_nios_clk_mon      => lcd_data(0),
        i_clk_156           => transceiver_pll_clock(0),
        o_clk_156_mon       => lcd_data(1),
        i_clk_125           => lvds_firefly_clk,
        o_clk_125_mon       => lcd_data(2),

        i_areset_n          => pb_db(0),
        
        i_testin            => pb_db(1)--,
    );


    FPGA_Test(0) <= transceiver_pll_clock(0);
    FPGA_Test(1) <= lvds_firefly_clk;
    FPGA_Test(2) <= clk_125_top;
end rtl;
