library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

use work.daq_constants.all;
use work.util.all;

entity scifi_path is
generic (
    N_m : positive := 2--;  --number of modules
);
port (
    -- read latency - 1
    i_reg_addr      : in    std_logic_vector(3 downto 0);
    i_reg_re        : in    std_logic;
    o_reg_rdata     : out   std_logic_vector(31 downto 0);
    i_reg_we        : in    std_logic;
    i_reg_wdata     : in    std_logic_vector(31 downto 0);

    o_chip_reset    : out   std_logic_vector(N_m-1 downto 0);
    o_pll_test      : out   std_logic;
    i_data          : in    std_logic_vector(N_m*4-1 downto 0);

    o_fifoA_rdata    : out   std_logic_vector(35 downto 0);
    o_fifoA_rempty   : out   std_logic;
    i_fifoA_rack     : in    std_logic;

    o_fifoB_rdata    : out   std_logic_vector(35 downto 0);
    o_fifoB_rempty   : out   std_logic;
    i_fifoB_rack     : in    std_logic;

    i_reset         : in    std_logic;
    -- 156.25 MHz
    i_clk_core      : in    std_logic; --core system (QSFP) clock
    -- 125 MHz
    i_clk_ref_A     : in    std_logic; --lvds reference only
    i_clk_ref_B     : in    std_logic; --lvds reference only
    i_clk_g125      : in    std_logic; --global 125MHz clock, signals to ASIC from this

    --reset system
    i_run_state      : in    run_state_t; --run state sync to i_clk_g125
    o_run_state_all_done : out std_logic; --all fifos empty, all data read

    o_MON_rxrdy     : out   std_logic_vector(N_m*4 - 1 downto 0)--; --receiver ready flags for monitoring, sync to lvds_userclocks(A/B depending on LVDS placement)
);
end entity;

architecture arch of scifi_path is
    signal s_testpulse : std_logic;

    signal rx_pll_lock : std_logic;
    signal rx_dpa_lock : std_logic_vector(i_data'range);
    signal rx_ready : std_logic_vector(i_data'range);
    signal frame_desync : std_logic_vector(1 downto 0);
    signal buffer_full : std_logic_vector(1 downto 0);

    -- counters
    signal s_cntreg_ctrl : std_logic_vector(31 downto 0);
    signal s_cntreg_num_g,       s_cntreg_num       : std_logic_vector(31 downto 0);
    signal s_cntreg_denom_low_g, s_cntreg_denom_low : std_logic_vector(31 downto 0);
    signal s_cntreg_denom_high_g,s_cntreg_denom_high: std_logic_vector(31 downto 0);

    -- registers controlled from midas
    signal s_dummyctrl_reg : std_logic_vector(31 downto 0);
    signal s_dpctrl_reg : std_logic_vector(31 downto 0);
    signal s_subdet_reset_reg : std_logic_vector(31 downto 0);
    signal s_subdet_resetdly_reg : std_logic_vector(31 downto 0);
    signal s_subdet_resetdly_reg_written : std_logic;
    --reset synchronizers
    signal s_datapath_rst,s_datapath_rst_n_156 : std_logic;
    signal s_lvds_rx_rst, s_lvds_rx_rst_n_125  : std_logic;

    --chip reset synchronization/shift
    signal s_chip_rst : std_logic;
    signal s_chip_rst_shifted : std_logic_vector(3 downto 0);
begin

    e_test_pulse : entity work.clkdiv
    generic map ( P => 1250 )
    port map ( o_clk => s_testpulse, i_reset_n => not i_run_state(RUN_STATE_BITPOS_SYNC), i_clk => i_clk_g125 );
    o_pll_test <= s_testpulse;


    process(i_clk_core, i_reset)
    begin
    if ( i_reset = '1' ) then
            s_dummyctrl_reg <= (others=>'0');
            s_dpctrl_reg <= (others=>'0');
            s_subdet_reset_reg <= (others=>'0');
            s_subdet_resetdly_reg <= (others=>'0');
        --
    elsif rising_edge(i_clk_core) then
        o_reg_rdata <= X"CCCCCCCC";
        s_subdet_resetdly_reg_written <= '0';
	s_cntreg_denom_low<=s_cntreg_denom_low_g;
	s_cntreg_denom_high<=s_cntreg_denom_high_g;
	s_cntreg_num<=s_cntreg_num_g;
        -- counters
        if ( i_reg_re = '1' and i_reg_addr = X"0" ) then
            o_reg_rdata <= s_cntreg_ctrl;
        end if;
        if ( i_reg_we = '1' and i_reg_addr = X"0" ) then
            s_cntreg_ctrl <= i_reg_wdata;
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"1" ) then
            o_reg_rdata <= gray2bin(s_cntreg_num);
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"2" ) then
            o_reg_rdata <= gray2bin(s_cntreg_denom_low);
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"3" ) then
            o_reg_rdata <= gray2bin(s_cntreg_denom_high);
        end if;

        -- monitors
        if ( i_reg_re = '1' and i_reg_addr = X"4" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(0) <= rx_pll_lock;
            o_reg_rdata(5 downto 4) <= frame_desync;
            o_reg_rdata(9 downto 8) <= buffer_full;
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"5" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(rx_dpa_lock'range) <= rx_dpa_lock;
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"6" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(rx_ready'range) <= rx_ready;
        end if;

        -- output write
        if ( i_reg_we = '1' and i_reg_addr = X"8" ) then
            s_dummyctrl_reg <= i_reg_wdata;
        end if;
        if ( i_reg_we = '1' and i_reg_addr = X"9" ) then
            s_dpctrl_reg <= i_reg_wdata;
        end if;
        if ( i_reg_we = '1' and i_reg_addr = X"A" ) then
            s_subdet_reset_reg <= i_reg_wdata;
        end if;
        if ( i_reg_we = '1' and i_reg_addr = X"B" ) then
            s_subdet_resetdly_reg <= i_reg_wdata;
            s_subdet_resetdly_reg_written <= '1';
        end if;
        -- output read
        if ( i_reg_re = '1' and i_reg_addr = X"8" ) then
            o_reg_rdata <= s_dummyctrl_reg;
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"9" ) then
            o_reg_rdata <= s_dpctrl_reg;
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"A" ) then
            o_reg_rdata <= s_subdet_reset_reg;
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"B" ) then
            o_reg_rdata <= s_subdet_resetdly_reg;
        end if;

        --
    end if;
    end process;

    s_chip_rst <= s_subdet_reset_reg(0) or i_run_state(RUN_STATE_BITPOS_SYNC); --TODO: remove register
    s_datapath_rst <= i_reset or s_subdet_reset_reg(1) or i_run_state(RUN_STATE_BITPOS_PREP); --TODO: remove register
    s_lvds_rx_rst <= i_reset or s_subdet_reset_reg(2);

rst_sync_dprst: entity work.reset_sync
    port map( i_reset_n   => s_datapath_rst, o_reset_n   => s_datapath_rst_n_156, i_clk       => i_clk_core);
rst_sync_lvdsrst: entity work.reset_sync
    port map( i_reset_n   => s_lvds_rx_rst, o_reset_n   => s_lvds_rx_rst_n_125, i_clk       => i_clk_g125);


    u_resetshift: entity work.clockalign_block
    generic map ( CLKDIV => 2 )
    port map (
		i_clk_config => i_clk_core,
		i_rst        => i_reset,

		i_pll_clk    => i_clk_g125,
		i_pll_arst   => i_reset,

		i_flag       => s_subdet_resetdly_reg_written,
		i_data       => s_subdet_resetdly_reg,

		i_sig => s_chip_rst,
		o_sig => s_chip_rst_shifted,
		o_pll_clk    => open
    );
    o_chip_reset <= s_chip_rst_shifted(N_m-1 downto 0);


    e_mutrig_datapath : entity work.mutrig_datapath
    generic map (
        N_MODULES => N_m,
        N_ASICS => 4,
        LVDS_PLL_FREQ => 125.0,
        LVDS_DATA_RATE => 1250.0,
        INPUT_SIGNFLIP => "1111"&"1111",
		  C_CHANNELNO_PREFIX_A => "00",
		  C_CHANNELNO_PREFIX_B => "01"
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
        o_A_fifo_empty => o_fifoA_rempty,
        o_A_fifo_data => o_fifoA_rdata,
        i_A_fifo_rd => i_fifoA_rack,

        o_B_fifo_empty => o_fifoB_rempty,
        o_B_fifo_data => o_fifoB_rdata,
        i_B_fifo_rd => i_fifoB_rack,

        -- slow control
        i_SC_disable_dec => s_dpctrl_reg(31),
        i_SC_rx_wait_for_all => s_dpctrl_reg(30),
        i_SC_rx_wait_for_all_sticky => s_dpctrl_reg(29),
        i_SC_mask => s_dpctrl_reg(4*N_m-1 downto 0),
        i_SC_datagen_enable => s_dummyctrl_reg(1),
        i_SC_datagen_shortmode => s_dummyctrl_reg(2),
        i_SC_datagen_count => s_dummyctrl_reg(12 downto 3),
        --run control
	i_RC_may_generate => i_run_state(RUN_STATE_BITPOS_RUNNING), 
	o_RC_all_done     => o_run_state_all_done,

        -- monitors
        o_receivers_usrclk => open,
        o_receivers_pll_lock => rx_pll_lock,
        o_receivers_dpa_lock=> rx_dpa_lock,
        o_receivers_ready => rx_ready,
        o_frame_desync => frame_desync,
        o_buffer_full => buffer_full,

        i_SC_reset_counters => s_cntreg_ctrl(15),
        i_SC_counterselect => s_cntreg_ctrl(5 downto 0),
        o_counter_numerator => s_cntreg_num_g,
        o_counter_denominator_low => s_cntreg_denom_low_g,
        o_counter_denominator_high =>s_cntreg_denom_high_g
    );

    o_MON_rxrdy <= rx_ready;
end architecture;
