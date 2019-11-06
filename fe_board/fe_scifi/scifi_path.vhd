library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

entity scifi_path is
generic (
    N_g : positive := 1;    --number of asics
    N_m : positive := 1--;  --number of modules
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
    i_data          : in    std_logic_vector(N_g-1 downto 0);

    o_fifo_rdata    : out   std_logic_vector(35 downto 0);
    o_fifo_rempty   : out   std_logic;
    i_fifo_rack     : in    std_logic;

    i_reset         : in    std_logic;
    -- 156.25 MHz
    i_clk_core      : in    std_logic;
    -- 125 MHz
    i_clk_ref_A     : in    std_logic;
    i_clk_ref_B     : in    std_logic;

    o_MON_rxrdy     : out   std_logic_vector(N_g - 1 downto 0)--;
);
end entity;

architecture arch of scifi_path is
    signal s_testpulse : std_logic;

    signal fifo_rempty : std_logic;
    signal fifo_rdata : std_logic_vector(35 downto 0);
    signal fifo_rack_ext : std_logic;

    signal rx_pll_lock : std_logic;
    signal rx_dpa_lock : std_logic_vector(i_data'range);
    signal rx_ready : std_logic_vector(i_data'range);
    signal frame_desync : std_logic;
    signal buffer_full : std_logic;

    -- registers controlled from midas
    signal s_dummyctrl_reg : std_logic_vector(31 downto 0);
    signal s_dpctrl_reg : std_logic_vector(31 downto 0);
    signal s_subdet_reset_reg : std_logic_vector(31 downto 0);
    signal s_subdet_resetdly_reg : std_logic_vector(31 downto 0);
    signal s_subdet_resetdly_reg_written : std_logic;
    --chip reset synchronization/shift
    signal s_chip_rst_refclk_sync : std_logic_vector(1 downto 0);
    signal s_chip_reset_out : std_logic_vector(3 downto 0);
begin

    e_test_pulse : entity work.clkdiv
    generic map ( P => 1250 )
    port map ( clkout => s_testpulse, rst_n => not i_reset, clk => i_clk_ref_A );
    o_pll_test <= s_testpulse;

    o_fifo_rdata <= fifo_rdata;
    o_fifo_rempty <= fifo_rempty;
    fifo_rack_ext <= '1' when ( i_reg_we = '1' and i_reg_addr = X"1" and fifo_rempty = '0' ) else '0';

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

        -- data
        if ( i_reg_re = '1' and i_reg_addr = X"0" ) then
            o_reg_rdata <= fifo_rdata(31 downto 0);
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"1" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(3 downto 0) <= fifo_rdata(35 downto 32);
            o_reg_rdata(16) <= fifo_rempty;
        end if;

        -- monitors
        if ( i_reg_re = '1' and i_reg_addr = X"4" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(0) <= rx_pll_lock;
            o_reg_rdata(4) <= frame_desync;
            o_reg_rdata(8) <= buffer_full;
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

        ----------------------------------------------------------------------------
    --generation of reset signal synchronized to reference clock, make independent of nios clock
    --shift each reset output by configurable delay
    p_fee_reset_sync: process(i_clk_ref_A)
    begin
        if rising_edge(i_clk_ref_A) then
            s_chip_rst_refclk_sync <= s_chip_rst_refclk_sync(0) & s_subdet_reset_reg(0);
        end if;
    end process;


    u_resetshift: entity work.clockalign_block
    generic map ( CLKDIV => 2 )
    port map (
		i_clk_config => i_clk_core,
		i_rst        => i_reset,

		i_pll_clk    => i_clk_ref_A,
		i_pll_arst   => i_reset,

		i_flag       => s_subdet_resetdly_reg_written,
		i_data       => s_subdet_resetdly_reg,

		i_sig => s_chip_rst_refclk_sync(1),
		o_sig => o_chip_reset,
		o_pll_clk    => open
    );

    e_mutrig_datapath : entity work.mutrig_datapath
    generic map (
        N_ASICS => N_g,
        LVDS_PLL_FREQ => 125.0,
        LVDS_DATA_RATE => 1250.0,
        INPUT_SIGNFLIP => (N_g-1 downto 0 => '1')--,
    )
    port map (
        i_rst => i_reset or s_subdet_reset_reg(1),
        i_stic_txd => i_data(N_g-1 downto 0),
        i_refclk_125_A => i_clk_ref_A,
        i_refclk_125_B => i_clk_ref_B,
        i_ts_clk => i_clk_ref_A, --TODO: better source?
        i_ts_rst => i_reset,

        -- interface to asic fifos
        i_clk_core => i_clk_core,
        o_fifo_empty => fifo_rempty,
        o_fifo_data => fifo_rdata,
        i_fifo_rd => i_fifo_rack or fifo_rack_ext,

        -- slow control
        i_SC_disable_dec => not s_dpctrl_reg(31),
        i_SC_mask => s_dpctrl_reg(N_g-1 downto 0),
        i_SC_datagen_enable => s_dummyctrl_reg(1),
        i_SC_datagen_shortmode => s_dummyctrl_reg(2),
        i_SC_datagen_count => s_dummyctrl_reg(12 downto 3),
        i_SC_rx_wait_for_all => s_dpctrl_reg(30),

        -- monitors
        o_receivers_usrclk => open,
        o_receivers_pll_lock => rx_pll_lock,
        o_receivers_dpa_lock=> rx_dpa_lock,
        o_receivers_ready => rx_ready,
        o_frame_desync => frame_desync,
        o_buffer_full => buffer_full--,
    );
    o_MON_rxrdy <= rx_ready;
end architecture;
