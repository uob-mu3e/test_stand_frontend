library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    --  Reference clocks  
    A10_REFCLK_GBT_P_0                  : IN    STD_LOGIC;

    --  XCVR serial lines
    rx_gbt                              : IN    STD_LOGIC_VECTOR(47 DOWNTO 0);
    tx_gbt                              : OUT   STD_LOGIC_VECTOR(47 DOWNTO 0);

    --  Reset from push button through Max5
    A10_M5FL_CPU_RESET_N                : IN    STD_LOGIC;

    --  COLOR LEDS
    A10_LED_3C_1                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_2                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_3                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_4                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);

    --  LEDS
    A10_LED                             : OUT   STD_LOGIC_VECTOR(7 DOWNTO 0);

    --  general purpose internal clock
    CLK_A10_100MHZ_P                    : IN    STD_LOGIC--; -- from internal 100 MHz oscillator
);
end entity;

architecture rtl of top is

    signal nios_clk     : std_logic;
    signal nios_reset_n : std_logic;

    signal av_pod0 : work.util.avalon_t;

begin

    nios_clk <= CLK_A10_100MHZ_P;

    e_nios_reset_n : entity work.reset_sync
    port map ( rstout_n => nios_reset_n, arst_n => A10_M5FL_CPU_RESET_N, clk => nios_clk );

    e_nios_clk_hz : entity work.clkdiv
    generic map ( P => 100 * 10**6 )
    port map (
        clkout => A10_LED(0),
        rst_n => nios_reset_n,
        clk => nios_clk--,
    );

    i_nios : component work.cmp.nios
    port map (
        avm_qsfp_address        => av_pod0.address(13 downto 0),
        avm_qsfp_read           => av_pod0.read,
        avm_qsfp_readdata       => av_pod0.readdata,
        avm_qsfp_write          => av_pod0.write,
        avm_qsfp_writedata      => av_pod0.writedata,
        avm_qsfp_waitrequest    => av_pod0.waitrequest,

        rst_reset_n => nios_reset_n,
        clk_clk     => nios_clk--,
    );

    e_pod0 : entity work.xcvr_a10
    generic map (
        INPUT_CLOCK_FREQUENCY_g => 125000000,
        DATA_RATE_g => 5000,
        CLK_MHZ_g => 125--,
    )
    port map (
        i_tx_data   => X"03CAFEBC"
                     & X"02BABEBC"
                     & X"01DEADBC"
                     & X"00BEEFBC",
        i_tx_datak  => "0001"
                     & "0001"
                     & "0001"
                     & "0001",

        o_rx_data   => open,
        o_rx_datak  => open,

        o_tx_clkout => open,
        i_tx_clkin  => (others => A10_REFCLK_GBT_P_0),
        o_rx_clkout => open,
        i_rx_clkin  => (others => A10_REFCLK_GBT_P_0),

        o_tx_serial => tx_gbt(3 downto 0),
        i_rx_serial => rx_gbt(3 downto 0),

        i_pll_clk   => A10_REFCLK_GBT_P_0,
        i_cdr_clk   => A10_REFCLK_GBT_P_0,

        i_avs_address     => av_pod0.address(13 downto 0),
        i_avs_read        => av_pod0.read,
        o_avs_readdata    => av_pod0.readdata,
        i_avs_write       => av_pod0.write,
        i_avs_writedata   => av_pod0.writedata,
        o_avs_waitrequest => av_pod0.waitrequest,

        i_reset     => not nios_reset_n,
        i_clk       => nios_clk--,
    );

end architecture;
