library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity scifi_path is
generic (
    N_g : positive := 1--;
);
port (
    -- avalon slave
    -- address space - 64 bytes (16 words)
    -- address units - words
    -- read latency - 1
    i_avs_address       : in    std_logic_vector(3 downto 0);
    i_avs_read          : in    std_logic;
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic;
    i_avs_writedata     : in    std_logic_vector(31 downto 0);
    o_avs_waitrequest   : out   std_logic;

    o_ck_fpga_0         : out   std_logic;
    o_chip_reset        : out   std_logic;
    o_pll_test          : out   std_logic;
    i_data              : in    std_logic_vector(N_g-1 downto 0);

    o_fifo_data         : out   std_logic_vector(35 downto 0);
    o_fifo_empty        : out   std_logic;
    i_fifo_rack         : out   std_logic;

    i_reset             : in    std_logic;
    -- 156.25 MHz   
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of scifi_path is

    signal refclk : std_logic;

    signal fifo_data : std_logic_vector(35 downto 0);
    signal fifo_empty : std_logic;
    signal fifo_rack_ext : std_logic;

    signal rx_pll_lock : std_logic;
    signal rx_dpa_lock : std_logic_vector(i_data'range);
    signal rx_ready : std_logic_vector(i_data'range);
    signal frame_desync : std_logic;
    signal buffer_full : std_logic;

    --registers controlled from midas
    signal s_dummyctrl_reg : std_logic_vector(31 downto 0);
    signal s_dpctrl_reg : std_logic_vector(31 downto 0);
    signal s_subdet_reset_reg : std_logic_vector(31 downto 0);
    
begin

    e_test_pulse : entity work.clkdiv
    generic map ( P => 125 )
    port map ( clkout => o_pll_test, rst_n => not i_reset, clk => i_clk );

    o_fifo_data <= fifo_data;
    o_fifo_empty <= fifo_empty;
    fifo_rack_ext <= '1' when ( i_avs_write = '1' and i_avs_address = X"0" and o_fifo_empty = '0' ) else '0';

    o_avs_waitrequest <= '0';

    process(i_clk, i_reset)
    begin
    if ( i_reset = '1' ) then
            s_dummyctrl_reg <= (others=>'0');
            s_dpctrl_reg <= (others=>'0');
            s_subdet_reset_reg <= (others=>'0');
        --
    elsif rising_edge(i_clk) then
        o_avs_readdata <= X"CCCCCCCC";
--monitors
        if ( i_avs_read = '1' and i_avs_address = X"0" and fifo_empty = '0' ) then
            o_avs_readdata <= fifo_data(31 downto 0);
        end if;
        if ( i_avs_read = '1' and i_avs_address = X"1" and fifo_empty = '0' ) then
            o_avs_readdata <= X"0000000" & fifo_data(35 downto 32);
        end if;

        if ( i_avs_read = '1' and i_avs_address = X"8" ) then
            o_avs_readdata <= (others => '0');
            o_avs_readdata(0) <= rx_pll_lock;
            o_avs_readdata(4) <= frame_desync;
            o_avs_readdata(8) <= buffer_full;
        end if;
        if ( i_avs_read = '1' and i_avs_address = X"9" ) then
            o_avs_readdata <= (others => '0');
            o_avs_readdata(rx_dpa_lock'range) <= rx_dpa_lock;
        end if;
        if ( i_avs_read = '1' and i_avs_address = X"A" ) then
            o_avs_readdata <= (others => '0');
            o_avs_readdata(rx_ready'range) <= rx_ready;
        end if;
--output
        if ( i_avs_write = '1' and i_avs_address = X"B" ) then
            s_dummyctrl_reg <= i_avs_writedata;
        end if;
        if ( i_avs_write = '1' and i_avs_address = X"C" ) then
            s_dpctrl_reg <= i_avs_writedata;
        end if;
        if ( i_avs_write = '1' and i_avs_address = X"D" ) then
            s_subdet_reset_reg <= i_avs_writedata;
        end if;
--output readback
        if ( i_avs_read = '1' and i_avs_address = X"B" ) then
            o_avs_readdata <= s_dummyctrl_reg;
        end if;
        if ( i_avs_read = '1' and i_avs_address = X"C" ) then
            o_avs_readdata <= s_dpctrl_reg;
        end if;
        if ( i_avs_read = '1' and i_avs_address = X"D" ) then
            o_avs_readdata <= s_subdet_reset_reg;
        end if;


      
        --
    end if;
    end process;

    o_chip_reset <= s_subdet_reset_reg(0);

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
        i_rst => i_reset or s_subdet_reset_reg(1),
        i_stic_txd => i_data(N_g-1 downto 0),
        i_refclk_125 => refclk,
        i_ts_clk => refclk,
        i_ts_rst => i_reset,

        --interface to asic fifos
        i_clk_core => i_clk,
        o_fifo_empty => fifo_empty,
        o_fifo_data => fifo_data,
        i_fifo_rd => i_fifo_rack or fifo_rack_ext,

        --slow control
        i_SC_disable_dec => not s_dpctrl_reg(31),
        i_SC_mask => s_dpctrl_reg(3 downto 0),
        i_SC_datagen_enable => s_dummyctrl_reg(1),
        i_SC_datagen_shortmode => s_dummyctrl_reg(2),
        i_SC_datagen_count => s_dummyctrl_reg(12 downto 3),

        --monitors
        o_receivers_usrclk => open,
        o_receivers_pll_lock => rx_pll_lock,
        o_receivers_dpa_lock=> rx_dpa_lock,
        o_receivers_ready => rx_ready,
        o_frame_desync => frame_desync,
        o_buffer_full => buffer_full--,
    );

end architecture;
