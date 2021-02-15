library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity xcvr_block is
generic (
    N_XCVR_g : positive := 8;
    N_CHANNELS_g : positive := 6;
    REFCLK_MHZ_g : positive := 125;
    DATA_RATE_MBPS_g : positive := 10000;
    CLK_MHZ_g : positive := 100--;
);
port (
    i_rx_serial         : in    std_logic_vector(N_XCVR_g*N_CHANNELS_g-1 downto 0);
    o_tx_serial         : out   std_logic_vector(N_XCVR_g*N_CHANNELS_g-1 downto 0);

    i_refclk            : in    std_logic_vector(N_XCVR_g-1 downto 0);

    o_rx_data           : out   work.util.slv32_array_t(N_XCVR_g*N_CHANNELS_g-1 downto 0);
    o_rx_datak          : out   work.util.slv4_array_t(N_XCVR_g*N_CHANNELS_g-1 downto 0);
    i_tx_data           : in    work.util.slv32_array_t(N_XCVR_g*N_CHANNELS_g-1 downto 0);
    i_tx_datak          : in    work.util.slv4_array_t(N_XCVR_g*N_CHANNELS_g-1 downto 0);

    o_rx_clk            : out   std_logic_vector(N_XCVR_g*N_CHANNELS_g-1 downto 0);
    i_rx_clk            : in    std_logic_vector(N_XCVR_g*N_CHANNELS_g-1 downto 0);
    o_tx_clk            : out   std_logic_vector(N_XCVR_g*N_CHANNELS_g-1 downto 0);
    i_tx_clk            : in    std_logic_vector(N_XCVR_g*N_CHANNELS_g-1 downto 0);

    -- avalon slave interface
    -- # address units words
    -- # read latency 0
    i_avs_address       : in    std_logic_vector(17 downto 0);
    i_avs_read          : in    std_logic;
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic;
    i_avs_writedata     : in    std_logic_vector(31 downto 0);
    o_avs_waitrequest   : out   std_logic;

    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of xcvr_block is

    type data_array_t is array ( natural range <> ) of std_logic_vector(N_CHANNELS_g*32-1 downto 0);
    signal rx_data, tx_data : data_array_t(N_XCVR_g-1 downto 0);

    type datak_array_t is array ( natural range <> ) of std_logic_vector(N_CHANNELS_g*4-1 downto 0);
    signal rx_datak, tx_datak : datak_array_t(N_XCVR_g-1 downto 0);

    signal avs_waitrequest : std_logic;
    signal av : work.util.avmm_array_t(N_XCVR_g-1 downto 0);
    signal av_i : integer;

begin

    generate_xcvr : for i in 0 to N_XCVR_g-1 generate
    begin
        generate_data : for j in 0 to N_CHANNELS_g-1 generate
        begin
            o_rx_data(i*N_CHANNELS_g+j) <= rx_data(i)(32*j+31 downto 0 + 32*j);
            o_rx_datak(i*N_CHANNELS_g+j) <= rx_datak(i)(4*j+3 downto 0 + 4*j);
            tx_data(i)(32*j+31 downto 0 + 32*j) <= i_tx_data(i*N_CHANNELS_g+j);
            tx_datak(i)(4*j+3 downto 0 + 4*j) <= i_tx_datak(i*N_CHANNELS_g+j);
        end generate;

        e_xcvr : entity work.xcvr_a10
        generic map (
            NUMBER_OF_CHANNELS_g => N_CHANNELS_g,
            INPUT_CLOCK_FREQUENCY_g => REFCLK_MHZ_g * 1000000,
            DATA_RATE_g => DATA_RATE_MBPS_g,
            CLK_MHZ_g => CLK_MHZ_g--,
        )
        port map (
            i_rx_serial => i_rx_serial(N_CHANNELS_g*i + N_CHANNELS_g-1 downto 0 + N_CHANNELS_g*i),
            o_tx_serial => o_tx_serial(N_CHANNELS_g*i + N_CHANNELS_g-1 downto 0 + N_CHANNELS_g*i),

            i_pll_clk   => i_refclk(i),
            i_cdr_clk   => i_refclk(i),

            o_rx_data   => rx_data(i),
            o_rx_datak  => rx_datak(i),
            i_tx_data   => tx_data(i),
            i_tx_datak  => tx_datak(i),

            o_rx_clkout => o_rx_clk(N_CHANNELS_g*i + N_CHANNELS_g-1 downto 0 + N_CHANNELS_g*i),
            i_rx_clkin  => i_rx_clk(N_CHANNELS_g*i + N_CHANNELS_g-1 downto 0 + N_CHANNELS_g*i),
            o_tx_clkout => o_tx_clk(N_CHANNELS_g*i + N_CHANNELS_g-1 downto 0 + N_CHANNELS_g*i),
            i_tx_clkin  => i_tx_clk(N_CHANNELS_g*i + N_CHANNELS_g-1 downto 0 + N_CHANNELS_g*i),

            i_avs_address     => av(i).address(13 downto 0),
            i_avs_read        => av(i).read,
            o_avs_readdata    => av(i).readdata,
            i_avs_write       => av(i).write,
            i_avs_writedata   => av(i).writedata,
            o_avs_waitrequest => av(i).waitrequest,

            i_reset     => not i_reset_n,
            i_clk       => i_clk--,
        );
    end generate;

    o_avs_waitrequest <= avs_waitrequest;

    av_i <= to_integer(unsigned(i_avs_address(i_avs_address'left downto 14)));

    -- avmm routing
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        o_avs_readdata <= X"CCCCCCCC";
        avs_waitrequest <= '1';
        for i in 0 to N_XCVR_g-1 loop
            av(i).read <= '0';
            av(i).write <= '0';
        end loop;
        --
    elsif rising_edge(i_clk) then
        o_avs_readdata <= X"CCCCCCCC";
        avs_waitrequest <= '1';

        for i in av'range loop
            if ( i = av_i and av(i).read = '0' and av(i).write = '0' and avs_waitrequest = '1' ) then
                -- start read/write request
                av(i).address(i_avs_address'range) <= i_avs_address;
                av(i).read <= i_avs_read;
                av(i).write <= i_avs_write;
                av(i).writedata <= i_avs_writedata;
            end if;

            if ( av(i).read /= av(i).write and av(i).waitrequest = '0' ) then
                -- done
                av(i).read <= '0';
                o_avs_readdata <= av(i).readdata;
                av(i).write <= '0';
                avs_waitrequest <= '0';
            end if;
        end loop;

        --
    end if;
    end process;

end architecture;
