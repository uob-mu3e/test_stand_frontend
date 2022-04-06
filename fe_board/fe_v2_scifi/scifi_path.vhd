library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

use work.mudaq.all;

entity scifi_path is
generic (
    IS_SCITILE : std_logic := '1';
    N_MODULES : positive;
    N_ASICS : positive;
    N_LINKS : positive;
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
    o_chip_reset    : out   std_logic_vector(N_MODULES-1 downto 0);
    o_pll_test      : out   std_logic;
    i_data          : in    std_logic_vector(N_MODULES*N_ASICS-1 downto 0);
    io_i2c_sda      : inout std_logic;
    io_i2c_scl      : inout std_logic;
    i_cec           : in    std_logic;
    i_spi_miso      : in    std_logic;
    i_i2c_int       : in    std_logic;
    o_pll_reset     : out   std_logic;
    o_spi_scl       : out   std_logic;
    o_spi_mosi      : out   std_logic;

    -- data out to common firmware
    o_fifo_wdata    : out   std_logic_vector(36*N_LINKS-1 downto 0);
    o_fifo_write    : out   std_logic_vector(N_LINKS-1 downto 0);

    i_common_fifos_almost_full : in std_logic_vector(N_LINKS-1 downto 0); 

    -- reset system
    i_run_state             : in  run_state_t; --run state sync to i_clk_g125
    o_run_state_all_done    : out std_logic; -- all fifos empty, all data read

    o_MON_rxrdy             : out std_logic_vector(N_MODULES*N_ASICS-1 downto 0); -- receiver ready flags for monitoring, sync to lvds_userclocks(A/B depending on LVDS placement)

    -- 125 MHz
    i_clk_ref_A             : in  std_logic; -- lvds reference only
    i_clk_ref_B             : in  std_logic; -- lvds reference only

    o_fast_pll_clk          : out std_logic;

    o_test_led              : out std_logic_vector(1 downto 0);

    i_reset_156_n           : in  std_logic;
    i_clk_156               : in  std_logic;
    i_reset_125_n           : in  std_logic;
    i_clk_125               : in  std_logic--;
);
end entity;

architecture arch of scifi_path is

    -- MuTrig PLL test
    signal s_testpulse : std_logic;

    -- rx signals
    signal rx_pll_lock                  : std_logic;
    signal rx_dpa_lock, rx_dpa_lock_reg : std_logic_vector(i_data'range);
    signal rx_ready                     : std_logic_vector(i_data'range);
    signal frame_desync                 : std_logic_vector(1 downto 0);
    signal buffer_full                  : std_logic_vector(1 downto 0);

    -- lapse counter
    signal ctrl_lapse_counter_reg       : std_logic_vector(31 downto 0);

    -- counters
    signal s_fifos_full                 : std_logic_vector(N_MODULES*N_ASICS-1 downto 0);
    signal s_counters                   : work.util.slv32_array_t(10 * N_MODULES*N_ASICS-1 downto 0);

    -- registers controlled from midas
    signal s_cntreg_ctrl                    : std_logic_vector(31 downto 0);
    signal s_dummyctrl_reg                  : std_logic_vector(31 downto 0);
    signal s_dpctrl_reg                     : std_logic_vector(31 downto 0);
    signal s_subdet_reset_reg               : std_logic_vector(31 downto 0);
    signal s_subdet_resetdly_reg            : std_logic_vector(31 downto 0);
    signal s_subdet_resetdly_reg_written    : std_logic;
    -- reset synchronizers
    signal s_datapath_rst, s_datapath_rst_n : std_logic;
    signal s_lvds_rx_rst, s_lvds_rx_rst_n   : std_logic;

    -- chip reset synchronization/shift
    signal s_chip_rst           : std_logic;
    signal s_chip_rst_shifted   : std_logic_vector(3 downto 0);

    signal fast_pll_clk         : std_logic;

    -- TODO: remove 
    signal a        : std_logic;
    signal b        : std_logic;
    signal sda_ena  : std_logic;
    signal sda_in   : std_logic;
    signal sda_out  : std_logic;
    signal scl_ena  : std_logic;
    signal scl_in   : std_logic;
    signal scl_out  : std_logic;
    signal doNotCompileAway : std_logic_vector(4 downto 0);

    -- transition counts spi
    signal miso_transition_count    : std_logic_vector(31 downto 0);
    signal miso_156                 : std_logic;
    signal miso_156_last            : std_logic;
    signal iram                     : work.util.rw_t;
    signal scifi_regs               : work.util.rw_t;

begin
--------------------------------------------------------------------
--- TODO: REMOVE THIS 
--- do not compile away stuff for pinout test
--------------------------------------------------------------------

    doNotCompileAway <= i_cec & i_spi_miso & i_i2c_int & scl_in & sda_in;
    dnca: entity work.doNotCompileAwayMux
    generic map (
        WIDTH_g   => 4--,
    )
    port map (
        i_clk               => i_clk_156,
        i_reset_n           => i_reset_156_n,
        i_doNotCompileAway  => doNotCompileAway,
        o_led               => o_test_led(0)--,
    );

    o_pll_reset <= not i_reset_156_n;
    o_spi_scl   <= not i_reset_156_n;
    o_spi_mosi  <= not i_reset_156_n;

    -- synthesis read_comments_as_HDL on
    -- buf1: entity work.ip_iobuf
    -- port map(
    --     datain(0)   => sda_out,
    --     oe(0)       => sda_ena,
    --     dataout(0)  => sda_in,
    --     dataio(0)   => io_i2c_sda
    -- );

    -- buf2: entity work.ip_iobuf
    -- port map(
    --     datain(0)   => scl_out,
    --     oe(0)       => scl_ena,
    --     dataout(0)  => scl_in,
    --     dataio(0)   => io_i2c_scl
    -- );
    -- synthesis read_comments_as_HDL off

    process(i_clk_156)
    begin
    if rising_edge(i_clk_156) then
        sda_ena <= '0';
        scl_ena <= '0';
        if(a='0' and b='0') then
            a<='1';
            sda_out <= '0';
            scl_out <= '0';
            sda_ena <= '1';
            scl_ena <= '1';
        elsif(a='1' and b='0') then
            b<= '1';
            sda_out <= '1';
            scl_out <= '1';
            sda_ena <= '1';
            scl_ena <= '1';
        else
            a<= '0';
            b<= '0';
        end if;
    end if;
    end process;

    process(i_clk_156, i_reset_156_n)
    begin
    if ( i_reset_156_n /= '1' ) then
            miso_transition_count <= (others => '0');
            miso_156 <= '0';
            miso_156_last <= '0';
    elsif rising_edge(i_clk_156) then
            miso_156 <= i_spi_miso;
            miso_156_last <= miso_156;
            if(miso_156 /= miso_156_last) then
                miso_transition_count <= miso_transition_count + '1';
            end if;
    end if;
    end process;



--------------------------------------------------------------------
--------------------------------------------------------------------


    -- 100 kHz for PLL test
    e_test_pulse : entity work.clkdiv
    generic map ( P => 1250 )
    port map ( o_clk => s_testpulse, i_reset_n => i_reset_125_n, i_clk => i_clk_125 ); -- i_run_state(RUN_STATE_BITPOS_SYNC), i_clk => i_clk_125 );
    o_pll_test <= '0' when s_cntreg_ctrl(31) = '0' else s_testpulse;

    o_test_led(1) <= s_cntreg_ctrl(0);

    ---- REGISTER MAPPING ----

    e_lvl1_sc_node : entity work.sc_node
      generic map (
        SLAVE1_ADDR_MATCH_g => "00--------------"--,
      )
      port map (
        i_master_addr  => i_reg_addr,
        i_master_re    => i_reg_re,
        o_master_rdata => o_reg_rdata,
        i_master_we    => i_reg_we,
        i_master_wdata => i_reg_wdata,

        o_slave0_addr  => scifi_regs.addr(15 downto 0),
        o_slave0_re    => scifi_regs.re,
        i_slave0_rdata => scifi_regs.rdata,
        o_slave0_we    => scifi_regs.we,
        o_slave0_wdata => scifi_regs.wdata,

        o_slave1_addr  => iram.addr(15 downto 0),
        o_slave1_re    => open,
        i_slave1_rdata => iram.rdata,
        o_slave1_we    => iram.we,
        o_slave1_wdata => iram.wdata,

        i_reset_n       => i_reset_156_n,
        i_clk           => i_clk_156--,
    );

    e_scifi_reg_mapping : entity work.scifi_reg_mapping
    generic map (
        N_MODULES => N_MODULES,
        N_ASICS   => N_ASICS--,
    )
    port map (
        i_reg_add                   => scifi_regs.addr(15 downto 0),
        i_reg_re                    => scifi_regs.re,
        o_reg_rdata                 => scifi_regs.rdata,
        i_reg_we                    => scifi_regs.we,
        i_reg_wdata                 => scifi_regs.wdata,

        -- inputs
        i_counters                  => s_counters,
        i_rx_pll_lock               => rx_pll_lock,
        i_frame_desync              => frame_desync,
        i_rx_dpa_lock_reg           => rx_dpa_lock,
        i_rx_ready                  => rx_ready,
        i_miso_transition_count     => miso_transition_count,
        i_fifos_full                => s_fifos_full,

        -- outputs
        o_cntreg_ctrl                   => s_cntreg_ctrl,
        o_dummyctrl_reg                 => s_dummyctrl_reg,
        o_dpctrl_reg                    => s_dpctrl_reg,
        o_subdet_reset_reg              => s_subdet_reset_reg,
        o_subdet_resetdly_reg_written   => s_subdet_resetdly_reg_written,
        o_subdet_resetdly_reg           => s_subdet_resetdly_reg,
        o_ctrl_lapse_counter_reg        => ctrl_lapse_counter_reg,

        i_clk_125                       => i_clk_125,
        i_reset_n                       => i_reset_156_n,
        i_clk                           => i_clk_156--,
    );

    e_iram : entity work.ram_1r1w
    generic map (
        g_DATA_WIDTH => 32,
        g_ADDR_WIDTH => 14--,
    )
    port map (
        i_raddr => iram.addr(13 downto 0),
        o_rdata => iram.rdata,
        i_rclk  => i_clk_156,

        i_waddr => iram.addr(13 downto 0),
        i_wdata => iram.wdata,
        i_we    => iram.we,
        i_wclk  => i_clk_156--,
    );

    s_chip_rst      <= (not i_reset_125_n) or s_subdet_reset_reg(0) or i_run_state(RUN_STATE_BITPOS_SYNC);
    s_datapath_rst  <= (not i_reset_125_n) or s_subdet_reset_reg(1) or i_run_state(RUN_STATE_BITPOS_PREP);
    s_lvds_rx_rst   <= (not i_reset_125_n) or s_subdet_reset_reg(2) or i_run_state(RUN_STATE_BITPOS_RESET);

    rst_sync_dprst : entity work.reset_sync
    port map( i_reset_n => not s_datapath_rst, o_reset_n => s_datapath_rst_n, i_clk => i_clk_125);

    rst_sync_lvdsrst : entity work.reset_sync
    port map( i_reset_n => not s_lvds_rx_rst, o_reset_n => s_lvds_rx_rst_n, i_clk => i_clk_125);

    --u_resetshift: entity work.clockalign_block
    --generic map ( CLKDIV => 2 )
    --port map (
    --    i_clk_config    => i_clk_156,
    --    i_rst           => not i_reset_156_n,
--
    --    i_pll_clk       => i_clk_125,
    --    i_pll_arst      => not i_reset_125_n,
--
    --    i_flag          => s_subdet_resetdly_reg_written,
    --    i_data          => s_subdet_resetdly_reg,
--
    --    i_sig           => s_chip_rst,
    --    o_sig           => s_chip_rst_shifted,
    --    o_pll_clk(0)    => o_fast_pll_clk
    --);
    o_chip_reset <= (others => s_chip_rst);--; (others =>s_chip_rst_shifted(0)); --s_chip_rst_shifted(N_MODULES-1 downto 0);TODO: fix this !!

    e_mutrig_datapath : entity work.mutrig_datapath
    generic map (
        IS_SCITILE => IS_SCITILE,
        N_MODULES => N_MODULES,
        N_ASICS => N_ASICS,
        N_LINKS => N_LINKS,
        N_CC => 15,
        LVDS_PLL_FREQ => LVDS_PLL_FREQ,
        LVDS_DATA_RATE => LVDS_DATA_RATE,
        INPUT_SIGNFLIP => INPUT_SIGNFLIP,
        GEN_DUMMIES => TRUE,
        C_ASICNO_PREFIX_A => "00",
        C_ASICNO_PREFIX_b => "01"--,
    )
    port map (
        i_rst_core      => not s_datapath_rst_n,
        i_rst_rx        => not s_lvds_rx_rst_n,
        i_stic_txd      => i_data,
        i_refclk_125_A  => i_clk_ref_A,
        i_refclk_125_B  => i_clk_ref_B,
        i_ts_clk        => i_clk_125,
        i_ts_rst        => i_run_state(RUN_STATE_BITPOS_SYNC),

        -- interface to asic fifos
        o_fifo_data     => o_fifo_wdata,
        o_fifo_wr       => o_fifo_write,

        i_common_fifos_almost_full => i_common_fifos_almost_full,

        -- slow control
        i_SC_disable_dec            => s_dpctrl_reg(31),
        i_SC_rx_wait_for_all        => s_dpctrl_reg(30),
        i_SC_rx_wait_for_all_sticky => s_dpctrl_reg(29),
        i_SC_mask                   => s_dpctrl_reg(N_MODULES*N_ASICS-1 downto 0),
        i_SC_mask_rx                => s_dpctrl_reg(N_MODULES*N_ASICS-1 downto 0),
        i_SC_datagen_enable         => s_dummyctrl_reg(1),
        i_SC_datagen_shortmode      => s_dummyctrl_reg(2),
        i_SC_datagen_count          => s_dummyctrl_reg(12 downto 3),

        -- run control
        i_RC_may_generate           => i_run_state(RUN_STATE_BITPOS_RUNNING),
        o_RC_all_done               => o_run_state_all_done,
        i_en_lapse_counter          => ctrl_lapse_counter_reg(31),
        i_upper_bnd                 => ctrl_lapse_counter_reg(14 downto 0),
        i_lower_bnd                 => ctrl_lapse_counter_reg(29 downto 15),

        -- monitors
        o_receivers_pll_lock        => rx_pll_lock,
        o_receivers_dpa_lock        => rx_dpa_lock,
        o_receivers_ready           => rx_ready,
        o_frame_desync              => frame_desync,

        i_SC_reset_counters         => s_cntreg_ctrl(15),
        o_fifos_full                => s_fifos_full,
        o_counters                  => s_counters,

        i_reset_156_n               => i_reset_156_n,
        i_clk_156                   => i_clk_156,
        i_reset_125_n               => i_reset_125_n,
        i_clk_125                   => i_clk_125--,
    );

    o_MON_rxrdy <= rx_ready;

end architecture;
