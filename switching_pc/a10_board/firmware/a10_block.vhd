library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity a10_block is
generic (
    g_XCVR0_CHANNELS    : positive := 16;
    g_XCVR0_N           : positive := 4--;
);
port (
    -- flash interface
    io_flash_data       : inout std_logic_vector(31 downto 0);
    o_flash_address     : out   std_logic_vector(31 downto 0);
    o_flash_read_n      : out   std_logic;
    o_flash_write_n     : out   std_logic;
    o_flash_cs_n        : out   std_logic;
    o_flash_reset_n     : out   std_logic;

    -- i2c
    io_i2c_scl          : inout std_logic_vector(31 downto 0);
    io_i2c_sda          : inout std_logic_vector(31 downto 0);

    -- spi
    i_spi_miso          : in    std_logic_vector(31 downto 0);
    o_spi_mosi          : out   std_logic_vector(31 downto 0);
--    o_spi_sclk
--    o_spi_ss_n

    o_nios_hz           : out   std_logic;

    -- xcvr 0 (6250 Mbps @ 156.25 MHz)
    i_xcvr0_rx          : in    std_logic_vector(g_XCVR0_CHANNELS-1 downto 0);
    o_xcvr0_tx          : out   std_logic_vector(g_XCVR0_CHANNELS-1 downto 0);
    o_xcvr0_rx_data     : out   work.util.slv32_array_t(g_XCVR0_CHANNELS-1 downto 0);
    o_xcvr0_rx_datak    : out   work.util.slv4_array_t(g_XCVR0_CHANNELS-1 downto 0);
    i_xcvr0_tx_data     : in    work.util.slv32_array_t(g_XCVR0_CHANNELS-1 downto 0);
    i_xcvr0_tx_datak    : in    work.util.slv4_array_t(g_XCVR0_CHANNELS-1 downto 0);
    i_clk_156           : in    std_logic;

    -- xcvr 1 (10000 Mbps @ 250 MHz)
    i_clk_250           : in    std_logic;

    -- PCIe 0
    i_pcie0_rx          : in    std_logic_vector(7 downto 0);
    o_pcie0_tx          : out   std_logic_vector(7 downto 0);
    i_pcie0_perst_n     : in    std_logic;
    i_pcie0_refclk      : in    std_logic; -- ref 100 MHz clock

    -- PCIe 1
    i_pcie1_refclk      : in    std_logic; -- ref 100 MHz clock

    -- global 125 MHz clock
    i_clk_125           : in    std_logic;
    i_reset_125_n       : in    std_logic--;
);
end entity;

architecture arch of a10_block is

begin

    -- nios reset sequence

    -- nios

    -- i2c mux

    -- spi mux

    -- xcvr_block 6250 Mbps @ 156.25 MHz

    -- xcvr_block 10000 Mbps @ 250 MHz

    -- pcie0

    -- pcie1

end architecture;
