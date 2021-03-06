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
    A10_LED_3C_1                        : out   std_logic_vector(2 downto 0);
    A10_LED_3C_2                        : out   std_logic_vector(2 downto 0);
    A10_LED_3C_3                        : out   std_logic_vector(2 downto 0);
    A10_LED_3C_4                        : out   std_logic_vector(2 downto 0);



    -- rx/tx PODs 0-7 (6 links per POD)
    rx_gbt                              : IN    STD_LOGIC_VECTOR(47 DOWNTO 0);
    tx_gbt                              : OUT   STD_LOGIC_VECTOR(47 DOWNTO 0);
    A10_REFCLK_GBT_P_0                  : IN    STD_LOGIC; -- <- SI5345_1/2[IN2] <- SI53340_1[CLK1] <- SMA1
    A10_REFCLK_GBT_P_1                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_2                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_3                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_4                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_5                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_6                  : IN    STD_LOGIC;
    A10_REFCLK_GBT_P_7                  : IN    STD_LOGIC;

    -- SFP
    A10_SFP1_TFC_RX_P                   : in    std_logic;
    A10_SFP1_TFC_TX_P                   : out   std_logic;
    A10_SFP2_TFC_RX_P                   : in    std_logic;
    A10_SFP2_TFC_TX_P                   : out   std_logic;
    A10_REFCLK_TFC_CMU_P                : in    std_logic;



    -- PCIe 0
    i_pcie0_rx                          : in    std_logic_vector(g_PCIE0_X-1 downto 0);
    o_pcie0_tx                          : out   std_logic_vector(g_PCIE0_X-1 downto 0);
    i_pcie0_perst_n                     : in    std_logic;
    i_pcie0_refclk                      : in    std_logic;

    -- PCIe 1
--    i_pcie1_rx                          : in    std_logic_vector(g_PCIE1_X-1 downto 0);
--    o_pcie1_tx                          : out   std_logic_vector(g_PCIE1_X-1 downto 0);
--    i_pcie1_perst_n                     : in    std_logic;
--    i_pcie1_refclk                      : in    std_logic;



    -- SMA
--    A10_SMA_CLK_IN_P                    : in    std_logic;
    A10_SMA_CLK_OUT_P                   : out   std_logic;

    -- SI53344
    A10_SI53344_FANOUT_CLK_P            : out   std_logic; -- -> SI53344[CLK0]
    -- - CLK_SEL = SWITCH_6[0]
    -- - SI53340_1[CLK1] <- SMA1
    A10_CUSTOM_CLK_P                    : in    std_logic; -- <- SI53344 <- SI53340_1[CLK_SEL]

    -- SI5345_1
    -- - IN_SEL = SWITCH_6[1-2]
    -- - SI5345_1[IN2] <- SI53340_1
    A10_SI5345_1_SMB_SCL                : inout std_logic;
    A10_SI5345_1_SMB_SDA                : inout std_logic;
    A10_SI5345_1_JITTER_CLOCK_P         : out   std_logic; -- -> SI5345_1[IN1]

    -- SI5345_2
    -- - IN_SEL = SWITCH_6[3-4]
    -- - SI5345_2[IN2] <- SI53340_1
    A10_SI5345_2_SMB_SCL                : inout std_logic;
    A10_SI5345_2_SMB_SDA                : inout std_logic;
    A10_SI5345_2_JITTER_CLOCK_P         : out   std_logic; -- -> SI5345_2[IN1]



    -- reset from push button through Max V
    A10_M5FL_CPU_RESET_N                : IN    STD_LOGIC;

    -- general purpose internal clock (100 MHz oscillator)
    CLK_A10_100MHZ_P                    : IN    STD_LOGIC--;
);
end entity;

architecture arch of top is

    -- constants
    constant SWB_ID : std_logic_vector(7 downto 0) := x"01";
    constant g_NLINKS_FEB_TOTL   : positive := 12;
    constant g_NLINKS_FARM_TOTL  : positive := 3;
    constant g_NLINKS_FARM_PIXEL : positive := 2;
    constant g_NLINKS_DATA_PIXEL_US : positive := 5;
    constant g_NLINKS_DATA_PIXEL_DS : positive := 5;
    constant g_NLINK_DATA_TRIGGER : positive := 1;
    constant g_NLINKS_FARM_SCIFI : positive := 1;
    constant g_NLINKS_DATA_SCIFI : positive := 2;
    constant g_NLINKS_FARM_TILE  : positive := 8;
    constant g_NLINKS_DATA_TILE  : positive := 12;

    signal led : std_logic_vector(7 downto 0) := (others => '0');
    signal reset_n : std_logic;


    
    -- local 100 MHz clock
    signal clk_100, reset_100_n : std_logic;

    signal pll_125 : std_logic;

    -- global 125 MHz clock
    signal clk_125, reset_125_n : std_logic;

    signal clk_250, reset_250_n : std_logic;

    -- 250 MHz pcie clock
    signal pcie0_clk : std_logic;
    signal pcie0_reset_n : std_logic;

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

    signal feb_rx_data, feb_tx_data : work.util.slv32_array_t(23 downto 0) := (others => x"000000BC");
    signal feb_rx_datak, feb_tx_datak : work.util.slv4_array_t(23 downto 0) := (others => "0001");
    signal feb_rx, feb_tx : work.mu3e.link_array_t(g_NLINKS_FEB_TOTL+g_NLINK_DATA_TRIGGER-1 downto 0) := (others => work.mu3e.LINK_IDLE);

    signal farm_rx_data, farm_tx_data : work.util.slv32_array_t(23 downto 0) := (others => X"000000BC");
    signal farm_rx_datak, farm_tx_datak : work.util.slv4_array_t(23 downto 0) := (others => "0001");
    signal farm_rx, farm_tx : work.mu3e.link_array_t(g_NLINKS_FARM_TOTL-1 downto 0) := (others => work.mu3e.LINK_IDLE);

    -- pll locked signal top
    signal locked_100to125 : std_logic;

begin

    A10_LED <= not led;
    reset_n <= A10_M5FL_CPU_RESET_N;



    clk_100 <= CLK_A10_100MHZ_P;

    e_reset_100_n : entity work.reset_sync
    port map ( o_reset_n => reset_100_n, i_reset_n => reset_n, i_clk => clk_100 );



    --! generate and route 125 MHz clock to SMA output
    --! (can be connected to SMA input as global clock)
    e_pll_100to125 : component work.cmp.ip_pll_100to125
    port map (
        locked => locked_100to125,
        outclk_0 => pll_125,
        refclk => clk_100,
        rst => not reset_100_n--,
    );

    A10_SMA_CLK_OUT_P <= pll_125;
    A10_SI53344_FANOUT_CLK_P <= pll_125;

    e_clk_125 : work.cmp.ip_clkctrl
    port map (
        inclk => A10_CUSTOM_CLK_P,
        outclk => clk_125--,
    );

    e_reset_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_125_n, i_reset_n => reset_n, i_clk => clk_125 );

    A10_SI5345_1_JITTER_CLOCK_P <= clk_125;
    A10_SI5345_2_JITTER_CLOCK_P <= clk_125;

    e_pcieref_clk_hz : entity work.clkdiv
    generic map ( P => 100000000 )
    port map ( o_clk => led(4), i_reset_n => reset_n, i_clk => i_pcie0_refclk );

    --! A10 block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    a10_block : entity work.a10_block
    generic map (
        g_XCVR0_CHANNELS => 24,
        g_XCVR0_N => 4,
        g_XCVR0_RX_P => (
            -- default
--             0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, -- CON0
--            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23  -- CON2
            -- schematic
--             5,  6,  4,  7,  3,  8,  0,  9,  2, 10,  1, 11, -- CON0
--            17, 18, 16, 20, 15, 19, 14, 21, 13, 22, 12, 23  -- CON1
            -- 1 is 12
            11,  1, 10,  2,  9,  0,  8,  3,  7,  4,  6,  5, -- CON0
            23, 12, 22, 13, 21, 14, 19, 15, 20, 16, 18, 17  -- CON1
            -- 1 is 1
        ),
        g_XCVR0_TX_P => (
            -- default
--             0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, -- CON1
--            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23  -- CON3
            -- schematic
--             5,  6,  4,  7,  3,  8,  0,  9,  2, 10,  1, 11,
--            17, 18, 16, 20, 15, 19, 14, 21, 13, 22, 12, 23
            -- reverse tx channels (0-11 -> 11-0)
            11,  1, 10,  2,  9,  0,  8,  3,  7,  4,  6,  5,
            23, 12, 22, 13, 21, 14, 19, 15, 20, 16, 18, 17
        ),
        g_XCVR1_CHANNELS => 24,
        g_XCVR1_N => 4,
        g_XCVR1_RX_P => (
            -- default
--             0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, -- CON4
--            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23  -- CON6
            -- schematic
            29-24, 30-24, 28-24, 31-24, 27-24, 32-24, 26-24, 33-24, 25-24, 34-24, 24-24, 35-24,
            36-24, 42-24, 37-24, 43-24, 38-24, 44-24, 39-24, 45-24, 40-24, 46-24, 41-24, 47-24
        ),
        g_XCVR1_TX_P => (
            -- default
--             0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, -- CON5
--            12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23  -- CON7
            --  schematic
--            29-24, 30-24, 28-24, 31-24, 27-24, 32-24, 26-24, 33-24, 25-24, 34-24, 24-24, 35-24,
--            36-24, 42-24, 37-24, 43-24, 38-24, 44-24, 39-24, 45-24, 40-24, 46-24, 41-24, 47-24
            -- reverse tx channels (0-11 -> 11-0)
            35-24, 24-24, 34-24, 25-24, 33-24, 26-24, 32-24, 27-24, 31-24, 28-24, 30-24, 29-24,
            47-24, 41-24, 46-24, 40-24, 45-24, 39-24, 44-24, 38-24, 43-24, 37-24, 42-24, 36-24
        ),
        g_SFP_CHANNELS => 2,
        g_PCIE0_X => g_PCIE0_X,
        g_FARM    => 0,
        g_CLK_MHZ => 100.0--,
    )
    port map (
        -- I2C
        io_i2c_scl(1)                   => A10_SI5345_1_SMB_SCL,
        io_i2c_sda(1)                   => A10_SI5345_1_SMB_SDA,
        io_i2c_scl(2)                   => A10_SI5345_2_SMB_SCL,
        io_i2c_sda(2)                   => A10_SI5345_2_SMB_SDA,

        -- XCVR0 (6250 Mbps @ 156.25 MHz)
        i_xcvr0_rx                      => rx_gbt(23 downto 0),
        o_xcvr0_tx                      => tx_gbt(23 downto 0),
        i_xcvr0_refclk                  => A10_REFCLK_GBT_P_3 & A10_REFCLK_GBT_P_2 & A10_REFCLK_GBT_P_1 & A10_REFCLK_GBT_P_0,

        o_xcvr0_rx_data                 => feb_rx_data,
        o_xcvr0_rx_datak                => feb_rx_datak,
        i_xcvr0_tx_data                 => feb_tx_data,
        i_xcvr0_tx_datak                => feb_tx_datak,
        i_xcvr0_clk                     => pcie0_clk,

        -- XCVR1 (10000 Mbps @ 250 MHz)
        i_xcvr1_rx                      => rx_gbt(47 downto 24),
        o_xcvr1_tx                      => tx_gbt(47 downto 24),
        i_xcvr1_refclk                  => A10_REFCLK_GBT_P_7 & A10_REFCLK_GBT_P_6 & A10_REFCLK_GBT_P_5 & A10_REFCLK_GBT_P_4,

        o_xcvr1_rx_data                 => farm_rx_data,
        o_xcvr1_rx_datak                => farm_rx_datak,
        i_xcvr1_tx_data                 => farm_tx_data,
        i_xcvr1_tx_datak                => farm_tx_datak,
        i_xcvr1_clk                     => pcie0_clk,

        -- SFP
        i_sfp_rx(0)                     => A10_SFP1_TFC_RX_P,
        i_sfp_rx(1)                     => A10_SFP2_TFC_RX_P,
        o_sfp_tx(0)                     => A10_SFP1_TFC_TX_P,
        o_sfp_tx(1)                     => A10_SFP2_TFC_TX_P,
        i_sfp_refclk                    => A10_REFCLK_TFC_CMU_P,

        -- PCIe0
        i_pcie0_rx                      => i_pcie0_rx,
        o_pcie0_tx                      => o_pcie0_tx,
        i_pcie0_perst_n                 => i_pcie0_perst_n,
        i_pcie0_refclk                  => i_pcie0_refclk,
        o_pcie0_reset_n                 => pcie0_reset_n,
        o_pcie0_clk                     => pcie0_clk,
        o_pcie0_clk_hz                  => led(3),

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

        o_clk_156_hz                    => led(2),

        i_reset_125_n                   => reset_125_n,
        i_clk_125                       => clk_125,
        o_clk_125_hz                    => led(1),

        i_reset_n                       => reset_100_n,
        i_clk                           => clk_100--,
    );

    --! map links comsicRun22
    gen_pixel_feb_links : for i in 0 to g_NLINKS_DATA_PIXEL_US + g_NLINKS_DATA_PIXEL_DS + g_NLINKS_DATA_SCIFI - 1 generate
        feb_rx(i).data     <= feb_rx_data(i);
        feb_rx(i).datak    <= feb_rx_datak(i);
        feb_tx_data(i)     <= feb_tx(i).data;
        feb_tx_datak(i)    <= feb_tx(i).datak;
    end generate;

    --! trigger data for cosmicRun22
    feb_rx(12).data     <= feb_rx_data(14);
    feb_rx(12).datak    <= feb_rx_datak(14);

    generate_farm_links : for i in 0 to g_NLINKS_FARM_TOTL - 1 generate
        farm_rx(i).data     <= farm_rx_data(i);
        farm_rx(i).datak    <= farm_rx_datak(i);
        farm_tx_data(i)     <= farm_tx(i).data;
        farm_tx_datak(i)    <= farm_tx(i).datak;
    end generate;


    --! SWB Block
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    e_swb_block : entity work.swb_block
    generic map (
        g_NLINKS_FEB_TOTL       => g_NLINKS_FEB_TOTL + g_NLINK_DATA_TRIGGER,
        g_NLINKS_FARM_TOTL      => g_NLINKS_FARM_TOTL,
        g_NLINKS_FARM_PIXEL     => g_NLINKS_FARM_PIXEL,
        g_NLINKS_DATA_PIXEL     => g_NLINKS_DATA_PIXEL_US + g_NLINKS_DATA_PIXEL_DS + g_NLINK_DATA_TRIGGER,
        g_NLINKS_DATA_PIXEL_US  => g_NLINKS_DATA_PIXEL_US,
        g_NLINKS_DATA_PIXEL_DS  => g_NLINKS_DATA_PIXEL_DS + g_NLINK_DATA_TRIGGER,
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

        o_farm_tx       => farm_tx,

        i_reset_n       => pcie0_reset_n,
        i_clk           => pcie0_clk--,
    );

end architecture;
