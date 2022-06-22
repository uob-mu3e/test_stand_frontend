-- M.Mueller, May 2021

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.mudaq.all;
use work.scifi_registers.all;

entity scifi_reg_mapping is
generic (
    N_MODULES   : positive;
    N_ASICS     : positive;
    N_LINKS     : positive := 1--;
);
port (

    i_reg_add                   : in  std_logic_vector(15 downto 0);
    i_reg_re                    : in  std_logic;
    o_reg_rdata                 : out std_logic_vector(31 downto 0);
    i_reg_we                    : in  std_logic;
    i_reg_wdata                 : in  std_logic_vector(31 downto 0);

    -- inputs
    i_counters                  : in  work.util.slv32_array_t(10 * N_MODULES*N_ASICS-1 downto 0);
    i_rx_pll_lock               : in  std_logic;
    i_frame_desync              : in  std_logic_vector(1 downto 0);
    i_rx_dpa_lock_reg           : in  std_logic_vector(N_MODULES*N_ASICS - 1 downto 0);
    i_rx_ready                  : in  std_logic_vector(N_MODULES*N_ASICS - 1 downto 0);
    i_miso_transition_count     : in  std_logic_vector(31 downto 0);
    i_fifos_full                : in  std_logic_vector(N_MODULES*N_ASICS - 1 downto 0);
    i_cc_diff                   : in  std_logic_vector(14 downto 0);
    i_ch_rate                   : in  work.util.slv32_array_t(127 downto 0);

    -- outputs
    o_cntreg_ctrl               : out std_logic_vector(31 downto 0);
    o_dummyctrl_reg             : out std_logic_vector(31 downto 0);
    o_dpctrl_reg                : out std_logic_vector(31 downto 0);
    o_subdet_reset_reg          : out std_logic_vector(31 downto 0);
    o_subdet_resetdly_reg_written : out std_logic;
    o_subdet_resetdly_reg       : out std_logic_vector(31 downto 0);
    o_ctrl_lapse_counter_reg    : out std_logic_vector(31 downto 0);
    o_link_data_reg             : out std_logic_vector(31 downto 0);

    i_clk_125                   : in    std_logic;

    i_reset_n                   : in    std_logic;
    i_clk                       : in    std_logic--;
);
end entity;

architecture rtl of scifi_reg_mapping is

    -- data path counters / monitor
    signal fifos_full               : std_logic_vector(N_MODULES*N_ASICS - 1 downto 0);
    signal counters156              : work.util.slv32_array_t(10 * N_MODULES*N_ASICS-1 downto 0);
    signal counter156               : std_logic_vector(31 downto 0);
    signal ch_rate                  : work.util.slv32_array_t(127 downto 0);

    -- data path ctrl
    signal cntreg_ctrl          : std_logic_vector(31 downto 0);
    signal dummyctrl_reg        : std_logic_vector(31 downto 0);
    signal dpctrl_reg           : std_logic_vector(31 downto 0);
    signal subdet_reset_reg     : std_logic_vector(31 downto 0);
    signal subdet_reset_reg_125 : std_logic_vector(31 downto 0);
    signal subdet_resetdly_reg  : std_logic_vector(31 downto 0);
    signal link_data_reg        : std_logic_vector(31 downto 0);

    -- rx monitor
    signal sync_rx_dpa_lock_reg : std_logic_vector(N_MODULES*N_ASICS - 1 downto 0);
    signal rx_ready             : std_logic_vector(N_MODULES*N_ASICS - 1 downto 0);
    signal frame_desync         : std_logic_vector(1 downto 0);
    signal ctrl_lapse_counter_reg : std_logic_vector(31 downto 0);
    signal cc_diff              : std_logic_vector(14 downto 0);

begin

    o_subdet_reset_reg(0)           <= subdet_reset_reg_125(0);
    o_subdet_reset_reg(31 downto 1) <= subdet_reset_reg(31 downto 1);
    o_subdet_resetdly_reg           <= subdet_resetdly_reg;

    -- sync from 125 clock to 156
    ----------------------------------------------------------------------------
    e_sync_rx_dpa_lock_reg : entity work.ff_sync
    generic map ( W => sync_rx_dpa_lock_reg'length )
    port map (
        i_d => i_rx_dpa_lock_reg, o_q => sync_rx_dpa_lock_reg,
        i_reset_n => i_reset_n, i_clk => i_clk--,
    );
    e_rx_ready : entity work.ff_sync
    generic map ( W => rx_ready'length )
    port map (
        i_d => i_rx_ready, o_q => rx_ready,
        i_reset_n => i_reset_n, i_clk => i_clk--,
    );
    e_frame_desync : entity work.ff_sync
    generic map ( W => frame_desync'length )
    port map (
        i_d => i_frame_desync, o_q => frame_desync,
        i_reset_n => i_reset_n, i_clk => i_clk--,
    );
    e_fifo_full : entity work.ff_sync
    generic map ( W => fifos_full'length )
    port map (
        i_d => i_fifos_full, o_q => fifos_full,
        i_reset_n => i_reset_n, i_clk => i_clk--,
    );
    e_cc_diff : entity work.ff_sync
        generic map ( W => i_cc_diff'length )
    port map (
        i_d => i_cc_diff, o_q => cc_diff,
        i_reset_n => i_reset_n, i_clk => i_clk--,
    );
    gen_counters : for i in 10 * N_MODULES*N_ASICS - 1 downto 0 generate
        e_counters : entity work.ff_sync
        generic map ( W => counters156(i)'length )
        port map (
            i_d => i_counters(i), o_q => counters156(i),
            i_reset_n => i_reset_n, i_clk => i_clk--,
        );
    end generate;
    gen_ch_rate : for i in 127 downto 0 generate
        e_ch_rate : entity work.ff_sync
        generic map ( W => ch_rate(i)'length )
        port map (
            i_d => i_ch_rate(i), o_q => ch_rate(i),
            i_reset_n => i_reset_n, i_clk => i_clk--,
        );
    end generate;
    ----------------------------------------------------------------------------

    -- sync from 156 to 125 clock
    ----------------------------------------------------------------------------
    e_cntreg_ctrl : entity work.ff_sync
    generic map ( W => cntreg_ctrl'length )
    port map (
        i_d => cntreg_ctrl, o_q => o_cntreg_ctrl,
        i_reset_n => i_reset_n, i_clk => i_clk_125--,
    );
    e_dummyctrl_reg : entity work.ff_sync
    generic map ( W => dummyctrl_reg'length )
    port map (
        i_d => dummyctrl_reg, o_q => o_dummyctrl_reg,
        i_reset_n => i_reset_n, i_clk => i_clk_125--,
    );
    e_dpctrl_reg : entity work.ff_sync
    generic map ( W => dpctrl_reg'length )
    port map (
        i_d => dpctrl_reg, o_q => o_dpctrl_reg,
        i_reset_n => i_reset_n, i_clk => i_clk_125--,
    );
    e_ctrl_lapse_reg : entity work.ff_sync
    generic map ( W => ctrl_lapse_counter_reg'length )
    port map (
        i_d => ctrl_lapse_counter_reg, o_q => o_ctrl_lapse_counter_reg,
        i_reset_n => i_reset_n, i_clk => i_clk_125--,
    );
    e_ctrl_reset_reg : entity work.ff_sync
    generic map ( W => subdet_reset_reg_125'length )
    port map (
        i_d => subdet_reset_reg, o_q => subdet_reset_reg_125,
        i_reset_n => i_reset_n, i_clk => i_clk_125--,
    );
    e_link_data_reg : entity work.ff_sync
    generic map ( W => link_data_reg'length )
    port map (
        i_d => link_data_reg, o_q => o_link_data_reg,
        i_reset_n => i_reset_n, i_clk => i_clk_125--,
    );
    ----------------------------------------------------------------------------

    process(i_clk, i_reset_n)
        variable regaddr : integer;
    begin
    if ( i_reset_n /= '1' ) then
            dummyctrl_reg           <= (others=>'0');
            counter156              <= (others=>'0');
            dpctrl_reg              <= (others=>'0');
            subdet_reset_reg        <= (others=>'0');
            subdet_resetdly_reg     <= (others=>'0');
            ctrl_lapse_counter_reg  <= (others=>'0');
    elsif rising_edge(i_clk) then
            o_reg_rdata         <= X"CCCCCCCC";
            regaddr             := to_integer(unsigned(i_reg_add));
            o_subdet_resetdly_reg_written <= '0';

            if ( i_reg_re = '1' and regaddr = SCIFI_CNT_CTRL_REGISTER_W ) then
                o_reg_rdata <= cntreg_ctrl;
            end if;
            if ( i_reg_we = '1' and regaddr = SCIFI_CNT_CTRL_REGISTER_W ) then
                cntreg_ctrl <= i_reg_wdata;
            end if;

            if ( i_reg_we = '1' and regaddr = SCIFI_CNT_ADDR_REGISTER_W ) then
                counter156 <= counters156(to_integer(unsigned(i_reg_wdata)));
            end if;

            if ( i_reg_re = '1' and regaddr = SCIFI_CNT_VALUE_REGISTER_R ) then
                o_reg_rdata <= counter156;
            end if;

            if ( i_reg_re = '1' and regaddr = SCIFI_MON_STATUS_REGISTER_R ) then
                o_reg_rdata <= (others => '0');
                o_reg_rdata(0) <= i_rx_pll_lock;
                o_reg_rdata(5 downto 4) <= frame_desync;
                o_reg_rdata(9 downto 8) <= "00";
                o_reg_rdata(N_MODULES*N_ASICS - 1 + 10 downto 10) <= fifos_full(N_MODULES*N_ASICS - 1 downto 0);
            end if;

            if ( i_reg_re = '1' and regaddr = SCIFI_MON_RX_DPA_LOCK_REGISTER_R ) then
                o_reg_rdata <= (others => '0');
                o_reg_rdata(N_MODULES*N_ASICS - 1 downto 0) <= sync_rx_dpa_lock_reg;
            end if;

            if ( i_reg_re = '1' and regaddr = SCIFI_MON_RX_READY_REGISTER_R ) then
                o_reg_rdata <= (others => '0');
                o_reg_rdata(N_MODULES*N_ASICS - 1 downto 0) <= rx_ready;
            end if;

            if ( i_reg_we = '1' and regaddr = SCIFI_CTRL_DUMMY_REGISTER_W ) then
                dummyctrl_reg <= i_reg_wdata;
            end if;
            if ( i_reg_re = '1' and regaddr = SCIFI_CTRL_DUMMY_REGISTER_W ) then
                o_reg_rdata <= dummyctrl_reg;
            end if;

            if ( i_reg_we = '1' and regaddr = SCIFI_CTRL_DP_REGISTER_W ) then
                dpctrl_reg <= i_reg_wdata;
            end if;
            if ( i_reg_re = '1' and regaddr = SCIFI_CTRL_DP_REGISTER_W ) then
                o_reg_rdata <= dpctrl_reg;
            end if;

            if ( i_reg_we = '1' and regaddr = SCIFI_CTRL_RESET_REGISTER_W ) then
                subdet_reset_reg <= i_reg_wdata;
            end if;
            if ( i_reg_re = '1' and regaddr = SCIFI_CTRL_RESET_REGISTER_W ) then
                o_reg_rdata <= subdet_reset_reg;
            end if;

            if ( i_reg_we = '1' and regaddr = SCIFI_CTRL_RESETDELAY_REGISTER_W ) then
                subdet_resetdly_reg <= i_reg_wdata;
                o_subdet_resetdly_reg_written <= '1';
            end if;
            if ( i_reg_re = '1' and regaddr = SCIFI_CTRL_RESETDELAY_REGISTER_W ) then
                o_reg_rdata <= subdet_resetdly_reg;
            end if;

            if ( i_reg_re = '1' and regaddr = SCIFI_CNT_MISO_TRANSITION_REGISTER_R ) then
                o_reg_rdata <= i_miso_transition_count;
            end if;

            if ( i_reg_we = '1' and regaddr = SCIFI_CTRL_LAPSE_COUNTER_REGISTER_W ) then
                ctrl_lapse_counter_reg <= i_reg_wdata;
            end if;

            if ( i_reg_re = '1' and regaddr = SCIFI_CC_DIFF_REGISTER_R ) then
                o_reg_rdata(14 downto 0) <= cc_diff;
            end if;

            if ( i_reg_we = '1' and regaddr = SCIFI_LINK_DATA_REGISTER_W ) then
                link_data_reg <= i_reg_wdata;
            end if;
            if ( i_reg_re = '1' and regaddr = SCIFI_LINK_DATA_REGISTER_W ) then
                o_reg_rdata <= link_data_reg;
            end if;

            loopChRate : for i in 127 downto 0 loop
                if ( i_reg_re = '1' and regaddr = SCIFI_CH_RATE_REGISTER_R + 1 ) then
                    o_reg_rdata <= ch_rate(i);
                end if;
            end loop;
    end if;
    end process;

end architecture;
