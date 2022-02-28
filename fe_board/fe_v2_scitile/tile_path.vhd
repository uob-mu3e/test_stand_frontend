library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;

entity tile_path is
generic (
    N_MODULES : positive;
    N_ASICS : positive;
    N_LINKS : positive;
    N_CC : positive := 15; -- will be always 15
    INPUT_SIGNFLIP : std_logic_vector := (31 downto 0 => '0');
    LVDS_PLL_FREQ : real;
    LVDS_DATA_RATE : real--;
);
port (
    -- read latency - 1
    i_reg_addr      : in    std_logic_vector(15 downto 0);
    i_reg_re        : in    std_logic;
    o_reg_rdata     : out   std_logic_vector(31 downto 0);
    i_reg_we        : in    std_logic;
    i_reg_wdata     : in    std_logic_vector(31 downto 0);

    -- to detector module
    o_chip_reset    : out   std_logic;
    o_pll_test      : out   std_logic;
    i_data          : in    std_logic_vector(N_MODULES*N_ASICS-1 downto 0);
    i_i2c_int       : in    std_logic;
    o_pll_reset     : out   std_logic;

    -- data out to common firmware
    o_fifo_wdata   : out   std_logic_vector(36*N_LINKS-1 downto 0);
    o_fifo_write   : out   std_logic_vector(N_LINKS-1 downto 0);

    i_common_fifos_almost_full : in std_logic_vector(N_LINKS-1 downto 0); 

    -- reset system
    i_run_state     : in    run_state_t; --run state sync to i_clk_g125
    o_run_state_all_done : out std_logic; -- all fifos empty, all data read

    o_MON_rxrdy     : out   std_logic_vector(N_MODULES*N_ASICS-1 downto 0); -- receiver ready flags for monitoring, sync to lvds_userclocks(A/B depending on LVDS placement)

    -- 156.25 MHz
    i_clk_core      : in    std_logic; -- core system (QSFP) clock
    -- 125 MHz
    i_clk_ref_A     : in    std_logic; -- lvds reference only
    i_clk_ref_B     : in    std_logic; -- lvds reference only
    i_clk_g125      : in    std_logic; -- global 125 MHz clock, signals to ASIC from this

    o_test_led      : out   std_logic;
    i_reset         : in    std_logic--;
);
end entity;

architecture arch of tile_path is

    signal s_testpulse : std_logic;
    signal s_receivers_usrclk : std_logic;

    signal rx_pll_lock : std_logic;
    signal rx_dpa_lock, rx_dpa_lock_reg : std_logic_vector(i_data'range);
    signal rx_ready : std_logic_vector(i_data'range);
    signal frame_desync : std_logic_vector(1 downto 0);
    signal buffer_full : std_logic_vector(1 downto 0);

    -- counters
    signal s_cntreg_ctrl : std_logic_vector(31 downto 0);
    signal s_cntreg_num_g, s_cntreg_num : std_logic_vector(31 downto 0);
    signal s_cntreg_denom_g, s_cntreg_denom_g_156 : std_logic_vector(63 downto 0);
    signal s_cntreg_denom_b : std_logic_vector(63 downto 0);

    -- registers controlled from midas
    signal s_dummyctrl_reg : std_logic_vector(31 downto 0);
    signal s_dpctrl_reg : std_logic_vector(31 downto 0);
    signal s_subdet_reset_reg : std_logic_vector(31 downto 0);
    signal s_subdet_resetdly_reg : std_logic_vector(31 downto 0);
    signal s_subdet_resetdly_reg_written : std_logic;

    -- reset synchronizers
    signal s_datapath_rst,s_datapath_rst_n_156 : std_logic;
    signal s_lvds_rx_rst, s_lvds_rx_rst_n_125 : std_logic;

    -- chip reset synchronization/shift
    signal s_chip_rst : std_logic;
    signal s_chip_rst_shifted : std_logic_vector(3 downto 0);

    -- lapse counter
    signal s_en_lapse_counter : std_logic;
    signal s_upper_bnd, s_lower_bnd : std_logic_vector(N_CC - 1 downto 0);

    signal iram         : work.util.rw_t;
    signal scitile_regs : work.util.rw_t;

begin

    -- 100 kHz
    e_test_pulse : entity work.clkdiv
    generic map ( P => 1250 )
    port map ( o_clk => s_testpulse, i_reset_n => not i_run_state(RUN_STATE_BITPOS_SYNC), i_clk => i_clk_g125 );
    o_pll_test <= s_testpulse;

    s_cntreg_denom_b <= work.util.gray2bin(s_cntreg_denom_g_156);


    ---- REGISTER MAPPING ----

    e_lvl1_sc_node: entity work.sc_node
      generic map (
        SLAVE1_ADDR_MATCH_g => "00--------------"--,
      )
      port map (
        i_clk          => i_clk_core,
        i_reset_n      => not i_reset,

        i_master_addr  => i_reg_addr,
        i_master_re    => i_reg_re,
        o_master_rdata => o_reg_rdata,
        i_master_we    => i_reg_we,
        i_master_wdata => i_reg_wdata,

        o_slave0_addr  => scitile_regs.addr(15 downto 0),
        o_slave0_re    => scitile_regs.re,
        i_slave0_rdata => scitile_regs.rdata,
        o_slave0_we    => scitile_regs.we,
        o_slave0_wdata => scitile_regs.wdata,

        o_slave1_addr  => iram.addr(15 downto 0),
        o_slave1_re    => open,
        i_slave1_rdata => iram.rdata,
        o_slave1_we    => iram.we,
        o_slave1_wdata => iram.wdata--,
      );

    e_scitile_reg_mapping : work.scitile_reg_mapping
    generic map (
        N_MODULES => N_MODULES,
        N_ASICS   => N_ASICS,
        N_CC      => N_CC--,
    )
    port map (
        i_clk                       => i_clk_core,
        i_reset_n                   => not i_reset,
        
        i_receivers_usrclk          => s_receivers_usrclk,

        i_reg_add                   => scitile_regs.addr(15 downto 0),
        i_reg_re                    => scitile_regs.re,
        o_reg_rdata                 => scitile_regs.rdata,
        i_reg_we                    => scitile_regs.we,
        i_reg_wdata                 => scitile_regs.wdata,

        -- inputs  156--------------------------------------------
        i_cntreg_num                => work.util.gray2bin(s_cntreg_num_g), -- on receivers_usrclk domain
        i_cntreg_denom_b            => work.util.gray2bin(s_cntreg_denom_g), -- on receivers_usrclk domain
        i_rx_pll_lock               => rx_pll_lock,
        i_frame_desync              => frame_desync,
        i_rx_dpa_lock_reg           => rx_dpa_lock, -- on receivers_usrclk domain
        i_rx_ready                  => rx_ready,

        -- outputs  156-------------------------------------------
        o_cntreg_ctrl               => s_cntreg_ctrl,
        o_dummyctrl_reg             => s_dummyctrl_reg,
        o_dpctrl_reg                => s_dpctrl_reg,
        o_subdet_reset_reg          => s_subdet_reset_reg,
        o_subdet_resetdly_reg_written => s_subdet_resetdly_reg_written,
        o_subdet_resetdly_reg       => s_subdet_resetdly_reg,

        o_en_lapse_counter          => s_en_lapse_counter,
        o_upper_bnd                 => s_upper_bnd,
        o_lower_bnd                 => s_lower_bnd--,

    );

    e_iram : entity work.ram_1r1w
    generic map (
        g_DATA_WIDTH => 32,
        g_ADDR_WIDTH => 14--,
    )
    port map (
        i_raddr => iram.addr(13 downto 0),
        o_rdata => iram.rdata,
        i_rclk  => i_clk_core,

        i_waddr => iram.addr(13 downto 0),
        i_wdata => iram.wdata,
        i_we    => iram.we,
        i_wclk  => i_clk_core--,
    );


    s_chip_rst <= s_subdet_reset_reg(0) or i_run_state(RUN_STATE_BITPOS_SYNC);
    s_datapath_rst <= i_reset or s_subdet_reset_reg(1) or i_run_state(RUN_STATE_BITPOS_PREP);
    s_lvds_rx_rst <= i_reset or s_subdet_reset_reg(2)  or i_run_state(RUN_STATE_BITPOS_RESET);

    rst_sync_dprst : entity work.reset_sync
    port map( i_reset_n => not s_datapath_rst, o_reset_n => s_datapath_rst_n_156, i_clk => i_clk_core);
    rst_sync_lvdsrst : entity work.reset_sync
    port map( i_reset_n => not s_lvds_rx_rst, o_reset_n => s_lvds_rx_rst_n_125, i_clk => i_clk_g125);


--    u_resetshift: entity work.clockalign_block
--    generic map ( CLKDIV => 2 )
--    port map (
--        i_clk_config    => i_clk_core,
--        i_rst           => i_reset,
--
--        i_pll_clk       => i_clk_g125,
--        i_pll_arst      => i_reset,
--
--        i_flag          => s_subdet_resetdly_reg_written,
--        i_data          => s_subdet_resetdly_reg,
--
--        i_sig           => s_chip_rst,
--        o_sig           => s_chip_rst_shifted,
--        o_pll_clk       => open
--    );
    --o_chip_reset <= s_chip_rst_shifted(N_MODULES-1 downto 0);
    o_chip_reset <= '0'; -- unused spare line to ASIC
    o_pll_reset <= s_chip_rst; -- TODO: connect to reset system again / implement shifting if needed

    e_mutrig_datapath : entity work.mutrig_datapath
    generic map (
        N_MODULES => N_MODULES,
        N_ASICS => N_ASICS,
        N_LINKS => N_LINKS,
        N_CC => N_CC,
        LVDS_PLL_FREQ => LVDS_PLL_FREQ,
        LVDS_DATA_RATE => LVDS_DATA_RATE,
        INPUT_SIGNFLIP => INPUT_SIGNFLIP,
        C_CHANNELNO_PREFIX_A => "00",
        C_CHANNELNO_PREFIX_B => "01"--,
    )
    port map (
        i_rst_core => not s_datapath_rst_n_156,
        i_rst_rx => not s_lvds_rx_rst_n_125,
        i_stic_txd => i_data,
        i_refclk_125_A => i_clk_ref_A,
        i_refclk_125_B => i_clk_ref_B,
        i_ts_clk => i_clk_g125,
        i_ts_rst => i_run_state(RUN_STATE_BITPOS_SYNC),

        -- interface to asic fifos
        i_clk_core => i_clk_core,
        o_fifo_data => o_fifo_wdata,
        o_fifo_wr => o_fifo_write,
        i_common_fifos_almost_full => i_common_fifos_almost_full,

        -- slow control
        i_SC_disable_dec => s_dpctrl_reg(31),
        i_SC_rx_wait_for_all => s_dpctrl_reg(30),
        i_SC_rx_wait_for_all_sticky => s_dpctrl_reg(29),
        i_SC_mask => s_dpctrl_reg(N_MODULES*N_ASICS-1 downto 0),
        i_SC_datagen_enable => s_dummyctrl_reg(1),
        i_SC_datagen_shortmode => s_dummyctrl_reg(2),
        i_SC_datagen_count => s_dummyctrl_reg(12 downto 3),
        
        --run control
        i_RC_may_generate => i_run_state(RUN_STATE_BITPOS_RUNNING),
        o_RC_all_done => o_run_state_all_done,

        -- lapse lapse counter
        i_en_lapse_counter => s_en_lapse_counter,
        i_lower_bnd => s_lower_bnd,
        i_upper_bnd => s_upper_bnd,

        -- monitors
        o_receivers_usrclk => s_receivers_usrclk,
        o_receivers_pll_lock => rx_pll_lock,
        o_receivers_dpa_lock => rx_dpa_lock,
        o_receivers_ready => rx_ready,
        o_frame_desync => frame_desync,

        i_SC_reset_counters => s_cntreg_ctrl(15),
        i_SC_counterselect => s_cntreg_ctrl(6 downto 0),
        o_counter_numerator => s_cntreg_num_g,
        o_counter_denominator_low => s_cntreg_denom_g(31 downto 0),
        o_counter_denominator_high => s_cntreg_denom_g(63 downto 32)
    );

    o_MON_rxrdy <= rx_ready;

end architecture;
