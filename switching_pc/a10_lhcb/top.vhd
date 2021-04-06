library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top is
generic (
    g_PCIE0_X : positive := 8;
    g_PCIE1_X : positive := 8--;
);
port (
    -- LEDs
    A10_LED                             : OUT   STD_LOGIC_VECTOR(7 DOWNTO 0);

    -- Color LEDs
    A10_LED_3C_1                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_2                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_3                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);
    A10_LED_3C_4                        : OUT   STD_LOGIC_VECTOR(2 DOWNTO 0);

    



    -- POD
    rx_gbt                              : IN    STD_LOGIC_VECTOR(47 DOWNTO 0);
    tx_gbt                              : OUT   STD_LOGIC_VECTOR(47 DOWNTO 0);
    A10_REFCLK_GBT_P_0                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_1                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_2                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_3                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_4                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_5                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_6                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_7                  : IN    STD_LOGIC;



    -- PCIe 0
    i_pcie0_rx                          : in    std_logic_vector(g_PCIE0_X-1 downto 0);
    o_pcie0_tx                          : out   std_logic_vector(g_PCIE0_X-1 downto 0);
    i_pcie0_perst_n                     : in    std_logic;
    i_pcie0_refclk                      : in    std_logic;

    -- PCIe 1
    i_pcie1_rx                          : in    std_logic_vector(g_PCIE1_X-1 downto 0);
    o_pcie1_tx                          : out   std_logic_vector(g_PCIE1_X-1 downto 0);
    i_pcie1_perst_n                     : in    std_logic;
    i_pcie1_refclk                      : in    std_logic;



    -- SI5345_1
    A10_SI5345_1_SMB_SCL                : inout std_logic;
    A10_SI5345_1_SMB_SDA                : inout std_logic;
    A10_SI5345_1_JITTER_CLOCK_P         : out   std_logic;

    -- SI5345_2
    A10_SI5345_2_SMB_SCL                : inout std_logic;
    A10_SI5345_2_SMB_SDA                : inout std_logic;
    A10_SI5345_2_JITTER_CLOCK_P         : out   std_logic;



    -- Reset from push button through Max5
    A10_M5FL_CPU_RESET_N                : IN    STD_LOGIC;

    -- general purpose internal clock
    CLK_A10_100MHZ_P                    : IN    STD_LOGIC--; -- from internal 100 MHz oscillator
);
end entity;

architecture arch of top is

    signal led : std_logic_vector(7 downto 0) := (others => '0');



    -- local 100 MHz clock
    signal clk_100, reset_100_n : std_logic;

    -- global 125 MHz clock
    signal clk_125, reset_125_n : std_logic;

    signal clk_156, reset_156_n, clk_250, reset_250_n : std_logic;
    signal pcie0_clk, pcie1_clk : std_logic;



    signal pod_rx_data : work.util.slv32_array_t(47 downto 0);
    signal pod_tx_data : work.util.slv32_array_t(47 downto 0) := (
        others => X"000000BC"--,
    );
    signal pod_rx_datak : work.util.slv4_array_t(47 downto 0);
    signal pod_tx_datak : work.util.slv4_array_t(47 downto 0) := (
        others => "0001"--,
    );



    signal pcie_wregs, pcie_rregs : work.mudaq.reg32array;

begin

    A10_LED <= not led;

    A10_SI5345_1_JITTER_CLOCK_P <= CLK_A10_100MHZ_P;
    A10_SI5345_2_JITTER_CLOCK_P <= CLK_A10_100MHZ_P;



    clk_100 <= CLK_A10_100MHZ_P;

    e_reset_100_n : entity work.reset_sync
    port map ( o_reset_n => reset_100_n, i_reset_n => A10_M5FL_CPU_RESET_N, i_clk => clk_100 );

    clk_125 <= A10_REFCLK_GBT_P_0; -- TODO: use global 125 MHz clock

    e_reset_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_125_n, i_reset_n => A10_M5FL_CPU_RESET_N, i_clk => clk_125 );



    a10_block : entity work.a10_block
    generic map (
        g_XCVR0_CHANNELS => 24,
        g_XCVR0_N => 4,
        g_XCVR1_CHANNELS => 24,
        g_XCVR1_N => 4,
        g_PCIE0_X => g_PCIE0_X,
        g_PCIE1_X => g_PCIE1_X,
        g_CLK_MHZ => 100.0--,
    )
    port map (
        -- I2C
        io_i2c_scl(1)                   => A10_SI5345_1_SMB_SCL,
        io_i2c_sda(1)                   => A10_SI5345_1_SMB_SDA,
        io_i2c_scl(2)                   => A10_SI5345_2_SMB_SCL,
        io_i2c_sda(2)                   => A10_SI5345_2_SMB_SDA,



        -- XCVR0 (6250 Mbps @ 156.25 MHz)
        i_xcvr0_rx                      => rx_gbt(47 downto 24),
        o_xcvr0_tx                      => tx_gbt(47 downto 24),
        i_xcvr0_refclk                  =>
            A10_REFCLK_GBT_P_7 & A10_REFCLK_GBT_P_6 & A10_REFCLK_GBT_P_5 & A10_REFCLK_GBT_P_4,
        o_xcvr0_rx_data                 => pod_rx_data(47 downto 24),
        o_xcvr0_rx_datak                => pod_rx_datak(47 downto 24),
        i_xcvr0_tx_data                 => pod_tx_data(47 downto 24),
        i_xcvr0_tx_datak                => pod_tx_datak(47 downto 24),



        -- XCVR1 (10000 Mbps @ 250 MHz)
        i_xcvr1_rx                      => rx_gbt(23 downto 0),
        o_xcvr1_tx                      => tx_gbt(23 downto 0),
        i_xcvr1_refclk                  =>
            A10_REFCLK_GBT_P_3 & A10_REFCLK_GBT_P_2 & A10_REFCLK_GBT_P_1 & A10_REFCLK_GBT_P_0,
        o_xcvr1_rx_data                 => pod_rx_data(23 downto 0),
        o_xcvr1_rx_datak                => pod_rx_datak(23 downto 0),
        i_xcvr1_tx_data                 => pod_tx_data(23 downto 0),
        i_xcvr1_tx_datak                => pod_tx_datak(23 downto 0),



        -- PCIe0
        i_pcie0_rx                      => i_pcie0_rx,
        o_pcie0_tx                      => o_pcie0_tx,
        i_pcie0_perst_n                 => i_pcie0_perst_n,
        i_pcie0_refclk                  => i_pcie0_refclk,
        o_pcie0_clk                     => pcie0_clk,

        i_pcie0_dma0_clk                => pcie0_clk,
        i_pcie0_wmem_clk                => pcie0_clk,
        i_pcie0_rmem_clk                => pcie0_clk,



        -- PCIe0
        i_pcie1_rx                      => i_pcie1_rx,
        o_pcie1_tx                      => o_pcie1_tx,
        i_pcie1_perst_n                 => i_pcie1_perst_n,
        i_pcie1_refclk                  => i_pcie1_refclk,
        o_pcie1_clk                     => pcie1_clk,

        i_pcie1_dma0_clk                => pcie0_clk,
        i_pcie1_wmem_clk                => pcie0_clk,
        i_pcie1_rmem_clk                => pcie0_clk,



        o_reset_156_n                   => reset_156_n,
        o_clk_156                       => clk_156,

        o_reset_250_n                   => reset_250_n,
        o_clk_250                       => clk_250,

        i_reset_125_n                   => reset_125_n,
        i_clk_125                       => clk_125,

        i_reset_n                       => reset_100_n,
        i_clk                           => clk_100--,
    );

end architecture;
