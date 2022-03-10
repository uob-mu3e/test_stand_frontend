library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mudaq.all;
use work.a10_pcie_registers.all;

entity top is
port (
    BUTTON              : in    std_logic_vector(3 downto 0);
    SW                  : in    std_logic_vector(1 downto 0);

    HEX0_D              : out   std_logic_vector(6 downto 0);
--    HEX0_DP             : out   std_logic;
    HEX1_D              : out   std_logic_vector(6 downto 0);
--    HEX1_DP             : out   std_logic;

    LED                 : out   std_logic_vector(3 downto 0) := "0000";
    LED_BRACKET         : out   std_logic_vector(3 downto 0) := "0000";

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
    CLK_50_B2J          : in    std_logic;

    --//// DDR4 A /////////////
    DDR4A_CK            : out   std_logic_vector(0 downto 0);
    DDR4A_CK_n          : out   std_logic_vector(0 downto 0);
    DDR4A_A             : out   std_logic_vector(16 downto 0);
    DDR4A_ACT_n         : out   std_logic;
    DDR4A_BA            : out   std_logic_vector(1 downto 0);
    DDR4A_BG            : out   std_logic_vector(1 downto 0);
    DDR4A_CKE           : out   std_logic_vector(0 downto 0);
    DDR4A_CS_n          : out   std_logic_vector(0 downto 0);
    DDR4A_ODT           : out   std_logic_vector(0 downto 0);
    DDR4A_RESET_n       : out   std_logic;
    DDR4A_ALERT_n       : in    std_logic;
    DDR4A_WE_n          : out   std_logic;
    DDR4A_RAS_n         : out   std_logic;
    DDR4A_CAS_n         : out   std_logic;
    DDR4A_DQS           : inout std_logic_vector(7 downto 0);
    DDR4A_DQS_n         : inout std_logic_vector(7 downto 0);
    DDR4A_DQ            : inout std_logic_vector(63 downto 0);
    DDR4A_DM            : out   std_logic_vector(7 downto 0);
    DDR4A_DBI_n         : inout std_logic_vector(7 downto 0);
    RZQ_DDR4_A          : in    std_logic;
    DDR4A_REFCLK_p      : in    std_logic;
    DDR4A_EVENT_n       : in    std_logic;
    DDR4A_SCL           : out   std_logic;
    DDR4A_SDA           : inout std_logic;
    
    --//// DDR4 B/////////////
    DDR4B_CK            : out   std_logic_vector(0 downto 0);
    DDR4B_CK_n          : out   std_logic_vector(0 downto 0);
    DDR4B_A             : out   std_logic_vector(16 downto 0);
    DDR4B_ACT_n         : out   std_logic;
    DDR4B_BA            : out   std_logic_vector(1 downto 0);
    DDR4B_BG            : out   std_logic_vector(1 downto 0);
    DDR4B_CKE           : out   std_logic_vector(0 downto 0);
    DDR4B_CS_n          : out   std_logic_vector(0 downto 0);
    DDR4B_ODT           : out   std_logic_vector(0 downto 0);
    DDR4B_RESET_n       : out   std_logic;
    DDR4B_ALERT_n       : in    std_logic;
    DDR4B_WE_n          : out   std_logic;
    DDR4B_RAS_n         : out   std_logic;
    DDR4B_CAS_n         : out   std_logic;
    DDR4B_DQS           : inout std_logic_vector(7 downto 0);
    DDR4B_DQS_n         : inout std_logic_vector(7 downto 0);
    DDR4B_DQ            : inout std_logic_vector(63 downto 0);
    DDR4B_DM            : out   std_logic_vector(7 downto 0);
    DDR4B_DBI_n         : inout std_logic_vector(7 downto 0);
    RZQ_DDR4_B          : in    std_logic;
    DDR4B_REFCLK_p      : in    std_logic;
    DDR4B_EVENT_n       : in    std_logic;
    DDR4B_SCL           : out   std_logic;
    DDR4B_SDA           : inout std_logic--;
);
end entity;

architecture rtl of top is

    -- free running clock (used as nios clock)
    signal clk_50       : std_logic;
    signal reset_50_n   : std_logic;

    -- global 125 MHz clock
    signal clk_125      : std_logic;
    signal reset_125_n  : std_logic;

    -- 156.25 MHz data clock (derived from global 125 MHz clock)
    signal clk_250      : std_logic;
    signal reset_250_n  : std_logic;

    -- 250 MHz pcie clock 
    signal reset_pcie0_n    : std_logic;
    signal pcie_fastclk_out : std_logic;

    -- flash
    signal flash_cs_n : std_logic;

    -- pcie read / write registers
    signal pcie0_resets_n_A   : std_logic_vector(31 downto 0);
    signal pcie0_resets_n_B   : std_logic_vector(31 downto 0);
    signal pcie0_resets_n_C   : std_logic_vector(31 downto 0);
    signal pcie0_writeregs_A  : work.util.slv32_array_t(63 downto 0);
    signal pcie0_writeregs_B  : work.util.slv32_array_t(63 downto 0);
    signal pcie0_writeregs_C  : work.util.slv32_array_t(63 downto 0);
    signal pcie0_regwritten_A : std_logic_vector(63 downto 0);
    signal pcie0_regwritten_B : std_logic_vector(63 downto 0);
    signal pcie0_regwritten_C : std_logic_vector(63 downto 0);
    signal pcie0_readregs_A   : work.util.slv32_array_t(63 downto 0);
    signal pcie0_readregs_B   : work.util.slv32_array_t(63 downto 0);
    signal pcie0_readregs_C   : work.util.slv32_array_t(63 downto 0);

    -- pcie read / write memory
    signal readmem_writedata    : std_logic_vector(31 downto 0);
    signal readmem_writeaddr    : std_logic_vector(15 downto 0);
    signal readmem_wren         : std_logic;
    signal writememreadaddr     : std_logic_vector(15 downto 0);
    signal writememreaddata     : std_logic_vector(31 downto 0);

    -- pcie dma
    signal dma_data_wren, dmamem_endofevent, pcie0_dma0_hfull : std_logic;
    signal dma_data : std_logic_vector(255 downto 0);

    signal rx_data_raw, rx_data, tx_data    : work.util.slv32_array_t(15 downto 0);
    signal rx_datak_raw, rx_datak, tx_datak : work.util.slv4_array_t(15 downto 0);
    
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


    --! A10 block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    a10_block : entity work.a10_block
    generic map (
        g_XCVR0_CHANNELS => 0,
        g_XCVR0_N => 0,
        g_XCVR1_CHANNELS => 16,
        g_XCVR1_N => 4,
        g_PCIE0_X => 8,
        g_FARM    => 1,
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

        -- LED / BUTTONS
        o_LED(1)                        => LED(0),
        o_LED_BRACKET                   => LED_BRACKET,

        -- XCVR1 (10000 Mbps @ 250 MHz)
        i_xcvr1_rx( 3 downto  0)        => QSFPA_RX_p,
        i_xcvr1_rx( 7 downto  4)        => QSFPB_RX_p,
        i_xcvr1_rx(11 downto  8)        => QSFPC_RX_p,
        i_xcvr1_rx(15 downto 12)        => QSFPD_RX_p,
        o_xcvr1_tx( 3 downto  0)        => QSFPA_TX_p,
        o_xcvr1_tx( 7 downto  4)        => QSFPB_TX_p,
        o_xcvr1_tx(11 downto  8)        => QSFPC_TX_p,
        o_xcvr1_tx(15 downto 12)        => QSFPD_TX_p,
        i_xcvr1_refclk                  => (others => clk_125),

        o_xcvr1_rx_data                 => rx_data_raw,
        o_xcvr1_rx_datak                => rx_datak_raw,
        i_xcvr1_tx_data                 => tx_data,
        i_xcvr1_tx_datak                => tx_datak,
        i_xcvr1_clk                     => pcie_fastclk_out,

        -- PCIe0
        i_pcie0_rx                      => PCIE_RX_p,
        o_pcie0_tx                      => PCIE_TX_p,
        i_pcie0_perst_n                 => PCIE_PERST_n,
        i_pcie0_refclk                  => PCIE_REFCLK_p,
        o_pcie0_clk                     => pcie_fastclk_out,
        o_pcie0_clk_hz                  => LED(3),

        -- PCIe0 read interface to writable memory
        i_pcie0_wmem_addr               => writememreadaddr,
        o_pcie0_wmem_rdata              => writememreaddata,
        i_pcie0_wmem_clk                => clk_250,

        -- PCIe0 write interface to readable memory
        i_pcie0_rmem_addr               => readmem_writeaddr,
        i_pcie0_rmem_wdata              => readmem_writedata,
        i_pcie0_rmem_we                 => readmem_wren,
        i_pcie0_rmem_clk                => clk_250,

        -- PCIe0 DMA0
        i_pcie0_dma0_wdata              => dma_data,
        i_pcie0_dma0_we                 => dma_data_wren,
        i_pcie0_dma0_eoe                => dmamem_endofevent,
        o_pcie0_dma0_hfull              => pcie0_dma0_hfull,
        i_pcie0_dma0_clk                => pcie_fastclk_out,

        -- PCIe0 update interface for readable registers
        i_pcie0_rregs_A                 => pcie0_readregs_A,
        i_pcie0_rregs_B                 => pcie0_readregs_B,
        i_pcie0_rregs_C                 => pcie0_readregs_C,

        -- PCIe0 read interface for writable registers
        o_pcie0_wregs_A                 => pcie0_writeregs_A,
        i_pcie0_wregs_A_clk             => pcie_fastclk_out,
        o_pcie0_regwritten_A            => pcie0_regwritten_A,
        o_pcie0_wregs_B                 => pcie0_writeregs_B,
        i_pcie0_wregs_B_clk             => clk_250,
        o_pcie0_regwritten_B            => pcie0_regwritten_B,
        o_pcie0_regwritten_C            => pcie0_regwritten_C,
        o_pcie0_resets_n_A              => pcie0_resets_n_A,
        o_pcie0_resets_n_B              => pcie0_resets_n_B,
        o_pcie0_resets_n_C              => pcie0_resets_n_C,

        -- resets clk
        top_pll_locked                  => locked_50to125,

        o_reset_250_n                   => reset_250_n,
        o_clk_250                       => clk_250,
        o_clk_250_hz                    => LED(2),

        i_reset_125_n                   => reset_125_n,
        i_clk_125                       => clk_125,
        o_clk_125_hz                    => LED(1),

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


    --! Farm Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    farm_block : entity work.farm_block
    generic map (
        g_DDR4         => true,
        g_NLINKS_TOTL  => 16,
        g_NLINKS_PIXEL => 8,
        g_NLINKS_SCIFI => 8--,
    )
    port map (

        --! links to/from FEBs
        i_rx            => rx_data_raw,
        i_rx_k          => rx_datak_raw,
        o_tx            => tx_data,
        o_tx_k          => tx_datak,

        --! PCIe registers / memory
        i_writeregs     => pcie0_writeregs_A,
        i_regwritten    => pcie0_regwritten_A,

        o_readregs      => pcie0_readregs_A,

        i_resets_n      => pcie0_resets_n_A,

        -- TODO: write status readout entity with ADDR to PCIe REGS and mapping to one counter REG
        o_counter       => open,

        i_dmamemhalffull=> pcie0_dma0_hfull,
        o_dma_wren      => dma_data_wren,
        o_endofevent    => dmamem_endofevent,
        o_dma_data      => dma_data,

        --! 250 MHz clock pice / reset_n
        i_reset_n       => reset_pcie0_n,
        i_clk           => pcie_fastclk_out,

        -- Interface to memory bank A
        o_A_mem_ck           => DDR4A_CK,
        o_A_mem_ck_n         => DDR4A_CK_n,
        o_A_mem_a            => DDR4A_A,
        o_A_mem_act_n        => DDR4A_ACT_n,
        o_A_mem_ba           => DDR4A_BA,
        o_A_mem_bg           => DDR4A_BG,
        o_A_mem_cke          => DDR4A_CKE,
        o_A_mem_cs_n         => DDR4A_CS_n,
        o_A_mem_odt          => DDR4A_ODT,
        o_A_mem_reset_n(0)   => DDR4A_RESET_n,
        i_A_mem_alert_n      => DDR4A_ALERT_n,
        o_A_mem_we_n(0)      => DDR4A_WE_n,
        o_A_mem_ras_n(0)     => DDR4A_RAS_n,
        o_A_mem_cas_n(0)     => DDR4A_CAS_n,
        io_A_mem_dqs         => DDR4A_DQS,
        io_A_mem_dqs_n       => DDR4A_DQS_n,
        io_A_mem_dq          => DDR4A_DQ,
        o_A_mem_dm           => DDR4A_DM,
        io_A_mem_dbi_n       => DDR4A_DBI_n,
        i_A_oct_rzqin        => RZQ_DDR4_A,
        i_A_pll_ref_clk      => DDR4A_REFCLK_p,

        -- Interface to memory bank B
        o_B_mem_ck           => DDR4B_CK,
        o_B_mem_ck_n         => DDR4B_CK_n,
        o_B_mem_a            => DDR4B_A,
        o_B_mem_act_n        => DDR4B_ACT_n,
        o_B_mem_ba           => DDR4B_BA,
        o_B_mem_bg           => DDR4B_BG,
        o_B_mem_cke          => DDR4B_CKE,
        o_B_mem_cs_n         => DDR4B_CS_n,
        o_B_mem_odt          => DDR4B_ODT,
        o_B_mem_reset_n(0)   => DDR4B_RESET_n,
        i_B_mem_alert_n      => DDR4A_ALERT_n,
        o_B_mem_we_n(0)      => DDR4B_WE_n,
        o_B_mem_ras_n(0)     => DDR4B_RAS_n,
        o_B_mem_cas_n(0)     => DDR4B_CAS_n,
        io_B_mem_dqs         => DDR4B_DQS,
        io_B_mem_dqs_n       => DDR4B_DQS_n,
        io_B_mem_dq          => DDR4B_DQ,
        o_B_mem_dm           => DDR4B_DM,
        io_B_mem_dbi_n       => DDR4B_DBI_n,
        i_B_oct_rzqin        => RZQ_DDR4_B,
        i_B_pll_ref_clk      => DDR4B_REFCLK_p--,
    );

    DDR4A_SDA   <= 'Z';
    DDR4B_SDA   <= 'Z';

end architecture;
