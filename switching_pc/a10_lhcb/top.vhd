library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
port (
    --  PODs
    rx_gbt                              : IN    STD_LOGIC_VECTOR(47 DOWNTO 0);
    tx_gbt                              : OUT   STD_LOGIC_VECTOR(47 DOWNTO 0);
    A10_REFCLK_GBT_P_0                  : IN    STD_LOGIC;

    --  Reset from push button through Max5
    A10_M5FL_CPU_RESET_N                : IN    STD_LOGIC;

    --  COLOR LEDS
    A10_LED_3C_1                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_2                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_3                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_4                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);

    --  LEDS
    A10_LED                             : OUT   STD_LOGIC_VECTOR(7 DOWNTO 0);

    A10_SI53340_2_CLK_40_P              : in    std_logic;

    A10_SI5345_1_JITTER_CLOCK_P         : out   std_logic;
    A10_SI5345_2_JITTER_CLOCK_P         : out   std_logic;

    --  general purpose internal clock
    CLK_A10_100MHZ_P                    : IN    STD_LOGIC--; -- from internal 100 MHz oscillator
);
end entity;

architecture rtl of top is

    signal nios_clk, nios_reset_n : std_logic;

    signal av_pod0 : work.util.avalon_t;

begin

    nios_clk <= CLK_A10_100MHZ_P;

    e_nios_reset_n : entity work.reset_sync
    port map ( o_reset_n => nios_reset_n, i_reset_n => A10_M5FL_CPU_RESET_N, i_clk => nios_clk );

    e_nios_clk_hz : entity work.clkdiv
    generic map ( P => 100 * 10**6 )
    port map (
        o_clk => A10_LED(0),
        i_reset_n => nios_reset_n,
        i_clk => nios_clk--,
    );

    i_nios : component work.cmp.nios
    port map (
        avm_pod_reset_reset_n   => nios_reset_n,
        avm_pod_clock_clk       => nios_clk,
        avm_pod_address         => av_pod0.address(13 downto 0),
        avm_pod_read            => av_pod0.read,
        avm_pod_readdata        => av_pod0.readdata,
        avm_pod_write           => av_pod0.write,
        avm_pod_writedata       => av_pod0.writedata,
        avm_pod_waitrequest     => av_pod0.waitrequest,

        rst_reset_n => nios_reset_n,
        clk_clk     => nios_clk--,
    );

    generate_pod : for i in 0 downto 0 generate
    begin
    e_pod0 : entity work.xcvr_a10
    generic map (
        NUMBER_OF_CHANNELS_g => 6,
        INPUT_CLOCK_FREQUENCY_g => 240000000,
        DATA_RATE_g => 4800,
        CLK_MHZ_g => 100--,
    )
    port map (
        i_tx_data   => X"050000BC"
                     & X"040000BC"
                     & X"030000BC"
                     & X"020000BC"
                     & X"010000BC"
                     & X"000000BC",
        i_tx_datak  => "0001"
                     & "0001"
                     & "0001"
                     & "0001"
                     & "0001"
                     & "0001",

        o_rx_data   => open,
        o_rx_datak  => open,

        o_tx_clkout => open,
        i_tx_clkin  => (others => A10_REFCLK_GBT_P_0),
        o_rx_clkout => open,
        i_rx_clkin  => (others => A10_REFCLK_GBT_P_0),

        o_tx_serial => tx_gbt(5 downto 0),
        i_rx_serial => rx_gbt(5 downto 0),

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
    end generate;

    process(nios_clk)
    begin
    if rising_edge(nios_clk) then
    end if;
    end process;



    A10_SI5345_1_JITTER_CLOCK_P <= A10_SI53340_2_CLK_40_P;
    A10_SI5345_2_JITTER_CLOCK_P <= A10_SI53340_2_CLK_40_P;

end architecture;
