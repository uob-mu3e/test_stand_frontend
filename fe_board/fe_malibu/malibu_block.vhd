library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity malibu_block is
generic (
    N_g : positive := 1--;
);
port (
    -- read latency - 1
    i_reg_addr      : in    std_logic_vector(3 downto 0);
    i_reg_re        : in    std_logic;
    o_reg_rdata     : out   std_logic_vector(31 downto 0);
    i_reg_we        : in    std_logic;
    i_reg_wdata     : in    std_logic_vector(31 downto 0);

    o_ck_fpga_0     : out   std_logic;
    o_chip_reset    : out   std_logic;
    o_pll_test      : out   std_logic;
    i_data          : in    std_logic_vector(N_g-1 downto 0);

    o_fifo_rdata    : out   std_logic_vector(35 downto 0);
    o_fifo_rempty   : out   std_logic;
    i_fifo_rack     : in    std_logic;

    i_reset         : in    std_logic;
    -- 156.25 MHz
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of malibu_block is

    signal refclk : std_logic;

    signal fifo_rempty : std_logic;
    signal fifo_rdata : std_logic_vector(35 downto 0);
    signal fifo_wfull : std_logic;

    signal rx_pll_lock : std_logic;
    signal rx_dpa_lock : std_logic_vector(i_data'range);
    signal rx_ready : std_logic_vector(i_data'range);
    signal frame_desync : std_logic;

begin

    e_test_pulse : entity work.clkdiv
    generic map ( P => 125 )
    port map ( clkout => o_pll_test, rst_n => not i_reset, clk => i_clk );

    o_fifo_rdata <= fifo_rdata;
    o_fifo_rempty <= fifo_rempty;

    process(i_clk, i_reset)
    begin
    if ( i_reset = '1' ) then
        o_chip_reset <= '0';
        --
    elsif rising_edge(i_clk) then
        o_reg_rdata <= X"CCCCCCCC";

        if ( i_reg_re = '1' and i_reg_addr = X"0" ) then
            o_reg_rdata <= fifo_rdata(31 downto 0);
        end if;

        if ( i_reg_re = '1' and i_reg_addr = X"1" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(3 downto 0) <= fifo_rdata(35 downto 32);
            o_reg_rdata(16) <= fifo_rempty;
            o_reg_rdata(17) <= fifo_wfull;
        end if;

        if ( i_reg_re = '1' and i_reg_addr = X"4" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(0) <= rx_pll_lock;
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"5" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(rx_ready'range) <= rx_ready;
        end if;
        if ( i_reg_re = '1' and i_reg_addr = X"6" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(rx_dpa_lock'range) <= rx_dpa_lock;
        end if;

        if ( i_reg_re = '1' and i_reg_addr = X"8" ) then
            o_reg_rdata <= (others => '0');
            o_reg_rdata(0) <= frame_desync;
        end if;
        --
    end if;
    end process;

    -- use 156.25 MHz instead of 160 MHz
    refclk <= i_clk;

    o_ck_fpga_0 <= refclk;

    e_mutrig_datapath : entity work.mutrig_datapath
    generic map (
        N_ASICS => N_g,
        LVDS_PLL_FREQ => 160.0,
        LVDS_DATA_RATE => 160--,
    )
    port map (
        i_rst => i_reset,
        i_stic_txd => i_data(N_g-1 downto 0),
        i_refclk_125 => refclk,
        i_ts_clk => refclk,
        i_ts_rst => i_reset,

        --interface to asic fifos
        i_clk_core => i_clk,
        o_fifo_empty => fifo_rempty,
        o_fifo_data => fifo_rdata,
        i_fifo_rd => i_fifo_rack,

        --slow control
        i_SC_disable_dec => '0',
        i_SC_mask => (others => '0'),
        i_SC_datagen_enable => '0',
        i_SC_datagen_shortmode => '0',
        i_SC_datagen_count => (others => '0'),

        --monitors
        o_receivers_usrclk => open,
        o_receivers_pll_lock => rx_pll_lock,
        o_receivers_dpa_lock=> rx_dpa_lock,
        o_receivers_ready => rx_ready,
        o_frame_desync => frame_desync,
        o_buffer_full => fifo_wfull--,
    );

end architecture;
