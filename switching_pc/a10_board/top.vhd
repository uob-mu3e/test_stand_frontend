library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity top is
port (
    BUTTON              : in    std_logic_vector(3 downto 0);
    SW                  : in    std_logic_vector(1 downto 0);

    HEX0_D              : out   std_logic_vector(6 downto 0) := (others => '1');
--    HEX0_DP             : out   std_logic := '1';
    HEX1_D              : out   std_logic_vector(6 downto 0) := (others => '1');
--    HEX1_DP             : out   std_logic := '1';

    LED                 : out   std_logic_vector(3 downto 0) := (others => '1');
    LED_BRACKET         : out   std_logic_vector(3 downto 0) := (others => '1');

    SMA_CLKOUT          : out   std_logic;
    SMA_CLKIN           : in    std_logic;

    RS422_DE            : out   std_logic;
    RS422_DIN           : in    std_logic; -- 1.8-V
    RS422_DOUT          : out   std_logic;
--    RS422_RE_n          : out   std_logic;
--    RJ45_LED_L          : out   std_logic;
    RJ45_LED_R          : out   std_logic;

    -- //////// FAN ////////
    FAN_I2C_SCL         : inout std_logic;
    FAN_I2C_SDA         : inout std_logic;

    -- //////// FLASH ////////
    FLASH_A             : out   std_logic_vector(26 downto 1);
    FLASH_D             : inout std_logic_vector(31 downto 0);
    FLASH_OE_n          : inout std_logic;
    FLASH_WE_n          : out   std_logic;
    FLASH_CE_n          : out   std_logic_vector(1 downto 0);
    FLASH_ADV_n         : out   std_logic;
    FLASH_CLK           : out   std_logic;
    FLASH_RESET_n       : out   std_logic;

    -- //////// POWER ////////
    POWER_MONITOR_I2C_SCL   : inout std_logic;
    POWER_MONITOR_I2C_SDA   : inout std_logic;

    -- //////// TEMP ////////
    TEMP_I2C_SCL        : inout std_logic;
    TEMP_I2C_SDA        : inout std_logic;

    -- //////// Transiver ////////
    QSFPA_TX_p          : out   std_logic_vector(3 downto 0);
    QSFPB_TX_p          : out   std_logic_vector(3 downto 0);
    QSFPC_TX_p          : out   std_logic_vector(3 downto 0);
    QSFPD_TX_p          : out   std_logic_vector(3 downto 0);

    QSFPA_RX_p          : in    std_logic_vector(3 downto 0);
    QSFPB_RX_p          : in    std_logic_vector(3 downto 0);
    QSFPC_RX_p          : in    std_logic_vector(3 downto 0);
    QSFPD_RX_p          : in    std_logic_vector(3 downto 0);

    QSFPA_REFCLK_p      : in    std_logic;
    QSFPB_REFCLK_p      : in    std_logic;
    QSFPC_REFCLK_p      : in    std_logic;
    QSFPD_REFCLK_p      : in    std_logic;

    QSFPA_LP_MODE       : out   std_logic;
    QSFPB_LP_MODE       : out   std_logic;
    QSFPC_LP_MODE       : out   std_logic;
    QSFPD_LP_MODE       : out   std_logic;

    QSFPA_MOD_SEL_n     : out   std_logic;
    QSFPB_MOD_SEL_n     : out   std_logic;
    QSFPC_MOD_SEL_n     : out   std_logic;
    QSFPD_MOD_SEL_n     : out   std_logic;

    QSFPA_RST_n         : out   std_logic;
    QSFPB_RST_n         : out   std_logic;
    QSFPC_RST_n         : out   std_logic;
    QSFPD_RST_n         : out   std_logic;



    -- //////// PCIE ////////
    PCIE_RX_p           : in    std_logic_vector(7 downto 0);
    PCIE_TX_p           : out   std_logic_vector(7 downto 0);
    PCIE_PERST_n        : in    std_logic;
    PCIE_REFCLK_p       : in    std_logic;
    PCIE_SMBCLK         : in    std_logic;
    PCIE_SMBDAT         : inout std_logic;
    PCIE_WAKE_n         : out   std_logic;

    CPU_RESET_n         : in    std_logic;
    CLK_50_B2J          : in    std_logic--;
);
end entity;

architecture rtl of top is

    -- constants
    constant SWB_ID : std_logic_vector(7 downto 0) := x"01";
    constant g_NLINKS_FEB_TOTL   : positive := 12;
    constant g_NLINKS_FARM_TOTL  : positive := 3;
    constant g_NLINKS_FARM_PIXEL : positive := 2;
    constant g_NLINKS_DATA_PIXEL_US : positive := 5;
    constant g_NLINKS_DATA_PIXEL_DS : positive := 5;
    constant g_NLINKS_FARM_SCIFI : positive := 1;
    constant g_NLINKS_DATA_SCIFI : positive := 2;
    constant g_NLINKS_FARM_TILE  : positive := 8;
    constant g_NLINKS_DATA_TILE  : positive := 12;

    -- free running clock (used as nios clock)
    signal clk_50 : std_logic;
    signal reset_50_n : std_logic;

    -- global 125 MHz clock
    signal clk_125 : std_logic;
    signal reset_125_n : std_logic;

    -- 156.25 MHz data clock (derived from global 125 MHz clock)
    signal clk_156 : std_logic;
    signal reset_156_n : std_logic;

    -- 250 MHz pcie clock
    signal pcie0_clk : std_logic;
    signal pcie0_reset_n : std_logic;

    -- flash
    signal flash_cs_n : std_logic;

    -- pcie read / write registers
    signal pcie0_resets_n   : std_logic_vector(31 downto 0);
    signal pcie0_writeregs  : work.util.slv32_array_t(63 downto 0);
    signal pcie0_readregs   : work.util.slv32_array_t(63 downto 0);

    -- pcie read / write memory
    signal readmem_writedata    : std_logic_vector(31 downto 0);
    signal readmem_writeaddr    : std_logic_vector(15 downto 0);
    signal readmem_wren         : std_logic;
    signal writememreadaddr     : std_logic_vector(15 downto 0);
    signal writememreaddata     : std_logic_vector(31 downto 0);

    -- pcie dma
    signal dma_data_wren, dmamem_endofevent, pcie0_dma0_hfull : std_logic;
    signal dma_data : std_logic_vector(255 downto 0);

    signal feb_rx_data, feb_tx_data : work.util.slv32_array_t(g_NLINKS_FEB_TOTL-1 downto 0) := (others => x"000000BC");
    signal feb_rx_datak, feb_tx_datak : work.util.slv4_array_t(g_NLINKS_FEB_TOTL-1 downto 0) := (others => "0001");
    signal feb_rx, feb_tx : work.mu3e.link_array_t(g_NLINKS_FEB_TOTL-1 downto 0) := (others => work.mu3e.LINK_ZERO);

    -- pll locked signal top
    signal locked_50to125 : std_logic;

begin

    --! local 50 MHz clock (oscillator)
    clk_50 <= CLK_50_B2J;

    --! generate reset for 50 MHz
    e_reset_50_n : entity work.reset_sync
    port map ( o_reset_n => reset_50_n, i_reset_n => CPU_RESET_n, i_clk => clk_50 );

    --! generate reset for 125 MHz
    e_reset_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_125_n, i_reset_n => CPU_RESET_n, i_clk => clk_125 );

    --! generate and route 125 MHz clock to SMA output
    --! (can be connected to SMA input as global clock)
    e_pll_50to125 : component work.cmp.ip_pll_50to125
    port map (
        locked => locked_50to125,
        outclk_0 => SMA_CLKOUT,
        refclk => clk_50,
        rst => not reset_50_n
    );

    --! 125 MHz global clock (from SMA input)
    e_clk_125 : work.cmp.ip_clkctrl
    port map (
        inclk => SMA_CLKIN,
        outclk => clk_125--,
    );

    e_clk_pcieref_hz : entity work.clkdiv
    generic map ( P => 100000000 )
    port map ( o_clk => LED_BRACKET(0), i_reset_n => reset_50_n, i_clk => PCIE_REFCLK_p );

    --! A10 block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    a10_block : entity work.a10_block
    generic map (
        g_XCVR0_CHANNELS => 8,
        g_XCVR0_N => 2,
        g_XCVR1_CHANNELS => 0,
        g_XCVR1_N => 0,
        g_XCVR2_CHANNELS => 4,
        g_XCVR3_CHANNELS => 4,
        g_PCIE0_X => 8,
        g_FARM    => 0,
        g_CLK_MHZ => 50.0--,
    )
    port map (
        -- flash interface
        o_flash_address(27 downto 2)    => FLASH_A,
        io_flash_data                   => FLASH_D,
        o_flash_read_n                  => FLASH_OE_n,
        o_flash_write_n                 => FLASH_WE_n,
        o_flash_cs_n                    => flash_cs_n,
        o_flash_reset_n                 => FLASH_RESET_n,

        -- I2C
        io_i2c_scl(0)                   => FAN_I2C_SCL,
        io_i2c_sda(0)                   => FAN_I2C_SDA,
        io_i2c_scl(1)                   => TEMP_I2C_SCL,
        io_i2c_sda(1)                   => TEMP_I2C_SDA,
        io_i2c_scl(2)                   => POWER_MONITOR_I2C_SCL,
        io_i2c_sda(2)                   => POWER_MONITOR_I2C_SDA,

        -- SPI
        i_spi_miso(0)                   => RS422_DIN,
        o_spi_mosi(0)                   => RS422_DOUT,
        o_spi_sclk(0)                   => RJ45_LED_R,
        o_spi_ss_n(0)                   => RS422_DE,

        -- LEDs
        o_LED(1)                        => LED_BRACKET(1),

        -- XCVR0 (6250 Mbps @ 156.25 MHz)
        i_xcvr0_rx( 3 downto  0)        => QSFPA_RX_p,
        i_xcvr0_rx( 7 downto  4)        => QSFPB_RX_p,
        o_xcvr0_tx( 3 downto  0)        => QSFPA_TX_p,
        o_xcvr0_tx( 7 downto  4)        => QSFPB_TX_p,
        i_xcvr0_refclk                  => (others => clk_125),

        o_xcvr0_rx_data                 => feb_rx_data(7 downto 0),
        o_xcvr0_rx_datak                => feb_rx_datak(7 downto 0),
        i_xcvr0_tx_data                 => feb_tx_data(7 downto 0),
        i_xcvr0_tx_datak                => feb_tx_datak(7 downto 0),
        i_xcvr0_clk                     => pcie0_clk,

        -- XCVR2 (1250 Mbps @ 125 MHz)
        i_xcvr2_rx( 3 downto  0)        => QSFPD_RX_p,
        o_xcvr2_tx( 3 downto  0)        => QSFPD_TX_p,
        i_xcvr2_refclk                  => clk_125,
        i_xcvr2_clk                     => clk_125,

        -- XCVR3 (125 MHz clk)
        i_xcvr3_rx( 3 downto  0)        => QSFPC_RX_p,
        o_xcvr3_tx( 3 downto  0)        => QSFPC_TX_p,
        i_xcvr3_refclk                  => clk_125,
        i_xcvr3_clk                     => clk_125,

        -- PCIe0
        i_pcie0_rx                      => PCIE_RX_p,
        o_pcie0_tx                      => PCIE_TX_p,
        i_pcie0_perst_n                 => PCIE_PERST_n,
        i_pcie0_refclk                  => PCIE_REFCLK_p,
        o_pcie0_reset_n                 => pcie0_reset_n,
        o_pcie0_clk                     => pcie0_clk,
        o_pcie0_clk_hz                  => LED(0),

        -- PCIe0 DMA0
        i_pcie0_dma0_wdata              => dma_data,
        i_pcie0_dma0_we                 => dma_data_wren,
        i_pcie0_dma0_eoe                => dmamem_endofevent,
        o_pcie0_dma0_hfull              => pcie0_dma0_hfull,
        i_pcie0_dma0_clk                => pcie0_clk,

        -- PCIe0 read interface to writable memory
        i_pcie0_wmem_addr               => writememreadaddr,
        o_pcie0_wmem_rdata              => writememreaddata,
        i_pcie0_wmem_clk                => pcie0_clk,

        -- PCIe0 write interface to readable memory
        i_pcie0_rmem_addr               => readmem_writeaddr,
        i_pcie0_rmem_wdata              => readmem_writedata,
        i_pcie0_rmem_we                 => readmem_wren,
        i_pcie0_rmem_clk                => pcie0_clk,

        -- PCIe0 update interface for readable registers
        i_pcie0_rregs                   => pcie0_readregs,

        -- PCIe0 read interface for writable registers
        o_pcie0_wregs                   => pcie0_writeregs,
        i_pcie0_wregs_clk               => pcie0_clk,
        o_pcie0_resets_n                => pcie0_resets_n,

        -- resets clk
        o_clk_156_hz                    => LED_BRACKET(2),

        i_reset_125_n                   => reset_125_n,
        i_clk_125                       => clk_125,
        o_clk_125_hz                    => LED_BRACKET(3),

        i_reset_n                       => reset_50_n,
        i_clk                           => clk_50--,
    );


    --! A10 development board setups
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    FLASH_CE_n <= (flash_cs_n, flash_cs_n);
    FLASH_ADV_n <= '0';
    FLASH_CLK <= '0';

    QSFPA_LP_MODE <= '0';
    QSFPB_LP_MODE <= '0';
    QSFPC_LP_MODE <= '0';
    QSFPD_LP_MODE <= '0';

    QSFPA_MOD_SEL_n <= '1';
    QSFPB_MOD_SEL_n <= '1';
    QSFPC_MOD_SEL_n <= '1';
    QSFPD_MOD_SEL_n <= '1';

    QSFPA_RST_n <= '1';
    QSFPB_RST_n <= '1';
    QSFPC_RST_n <= '1';
    QSFPD_RST_n <= '1';

    generate_feb_links : for i in 0 to g_NLINKS_FEB_TOTL - 1 generate
        feb_rx(i).data     <= feb_rx_data(i);
        feb_rx(i).datak    <= feb_rx_datak(i);
        feb_tx_data(i)     <= feb_tx(i).data;
        feb_tx_datak(i)    <= feb_tx(i).datak;
    end generate;


    --! SWB Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_swb_block : entity work.swb_block
    generic map (
        g_NLINKS_FEB_TOTL       => g_NLINKS_FEB_TOTL,
        g_NLINKS_FARM_TOTL      => g_NLINKS_FARM_TOTL,
        g_NLINKS_FARM_PIXEL     => g_NLINKS_FARM_PIXEL,
        g_NLINKS_DATA_PIXEL     => g_NLINKS_DATA_PIXEL_US + g_NLINKS_DATA_PIXEL_DS,
        g_NLINKS_DATA_PIXEL_US  => g_NLINKS_DATA_PIXEL_US,
        g_NLINKS_DATA_PIXEL_DS  => g_NLINKS_DATA_PIXEL_DS,
        g_NLINKS_FARM_SCIFI     => g_NLINKS_FARM_SCIFI,
        g_NLINKS_DATA_SCIFI     => g_NLINKS_DATA_SCIFI,
        SWB_ID                  => SWB_ID--,
    )
    port map (
        i_feb_rx        => feb_rx,
        o_feb_tx        => feb_tx,

        i_writeregs     => pcie0_writeregs,
        o_readregs      => pcie0_readregs,
        i_resets_n      => pcie0_resets_n,

        i_wmem_rdata    => writememreaddata,
        o_wmem_addr     => writememreadaddr,

        o_rmem_wdata    => readmem_writedata,
        o_rmem_addr     => readmem_writeaddr,
        o_rmem_we       => readmem_wren,

        i_dmamemhalffull=> pcie0_dma0_hfull,
        o_dma_wren      => dma_data_wren,
        o_endofevent    => dmamem_endofevent,
        o_dma_data      => dma_data,

        o_farm_tx       => open,

        i_reset_n       => pcie0_reset_n,
        i_clk           => pcie0_clk--,
    );

end architecture;
