library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity xcvr_block is
generic (
    N_CHANNELS_g : positive := 6;
    N_XCVR_g : positive := 8;
    REFCLK_MHZ_g : positive := 125;
    CLK_MHZ_g : positive := 125--;
);
port (
    i_rx_serial         : in    std_logic_vector(N_XCVR_g*N_CHANNELS_g-1 downto 0);
    o_tx_serial         : out   std_logic_vector(N_XCVR_g*N_CHANNELS_g-1 downto 0);

    i_refclk            : in    std_logic_vector(N_XCVR_g-1 downto 0);

    -- avalon slave interface
    -- # address units words
    -- # read latency 0
    i_avs_address       : in    std_logic_vector(13 downto 0);
    i_avs_read          : in    std_logic;
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic;
    i_avs_writedata     : in    std_logic_vector(31 downto 0);
    o_avs_waitrequest   : out   std_logic;

    o_data              : out   std_logic_vector(N_XCVR_g*N_CHANNELS_g*32-1 downto 0);
    o_datak             : out   std_logic_vector(N_XCVR_g*N_CHANNELS_g*4-1 downto 0);
    i_data              : in    std_logic_vector(N_XCVR_g*N_CHANNELS_g*32-1 downto 0);
    i_datak             : in    std_logic_vector(N_XCVR_g*N_CHANNELS_g*4-1 downto 0);

    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of xcvr_block is

begin

    generate_xcvr : for i in 0 to N_XCVR_g-1 generate
    begin
    e_xcvr : entity work.xcvr_a10
    generic map (
        NUMBER_OF_CHANNELS_g => N_CHANNELS_g,
        INPUT_CLOCK_FREQUENCY_g => REFCLK_MHZ_g * 1000000,
        DATA_RATE_g => 5000,
        CLK_MHZ_g => CLK_MHZ_g--,
    )
    port map (
        i_tx_data   => i_data,
        i_tx_datak  => i_datak,

        o_rx_data   => o_data,
        o_rx_datak  => o_datak,

        o_tx_clkout => open,
        i_tx_clkin  => (others => i_refclk(i)),
        o_rx_clkout => open,
        i_rx_clkin  => (others => i_refclk(i)),

        o_tx_serial => o_tx_serial(N_CHANNELS_g*i + N_CHANNELS_g-1 downto 0 + N_CHANNELS_g*i),
        i_rx_serial => i_rx_serial(N_CHANNELS_g*i + N_CHANNELS_g-1 downto 0 + N_CHANNELS_g*i),

        i_pll_clk   => i_refclk(i),
        i_cdr_clk   => i_refclk(i),

        i_avs_address     => i_avs_address,
        i_avs_read        => i_avs_read,
        o_avs_readdata    => o_avs_readdata,
        i_avs_write       => i_avs_write,
        i_avs_writedata   => i_avs_writedata,
        o_avs_waitrequest => o_avs_waitrequest,

        i_reset     => not i_reset_n,
        i_clk       => i_clk--,
    );
    end generate;

end architecture;
