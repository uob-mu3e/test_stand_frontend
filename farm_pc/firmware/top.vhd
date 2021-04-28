library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.pcie_components.all;
use work.mudaq_registers.all;
use work.dataflow_components.all;

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
    FAN_I2C_SCL         : out   std_logic;
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
    POWER_MONITOR_I2C_SCL   : out   std_logic;
    POWER_MONITOR_I2C_SDA   : inout std_logic;

    -- //////// TEMP ////////
    TEMP_I2C_SCL        : out   std_logic;
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
    
    --//// DDR3 A /////////////
    DDR3A_A             : out std_logic_vector(15 downto 0);
    DDR3A_BA            : out std_logic_vector(2 downto 0);
    DDR3A_CAS_n         : out std_logic;
    DDR3A_CK            : out std_logic_vector(0 downto 0);
    DDR3A_CKE           : out std_logic_vector(0 downto 0);
    DDR3A_CK_n          : out std_logic_vector(0 downto 0);
    DDR3A_CS_n          : out std_logic_vector(0 downto 0);
    DDR3A_DM            : out std_logic_vector(7 downto 0);
    DDR3A_DQ            : inout std_logic_vector(63 downto 0);
    DDR3A_DQS           : inout std_logic_vector(7 downto 0);
    DDR3A_DQS_n         : inout std_logic_vector(7 downto 0);
    DDR3A_EVENT_n       : in std_logic;
    DDR3A_ODT           : out std_logic_vector(0 downto 0);
    DDR3A_RAS_n         : out std_logic;
    DDR3A_REFCLK_p      : in std_logic;
    DDR3A_RESET_n       : out std_logic;
    DDR3A_SCL           : out std_logic;
    DDR3A_SDA           : inout std_logic;
    DDR3A_WE_n          : out std_logic;
    RZQ_DDR3_A          : in std_logic;

    --//// DDR3 B/////////////
    DDR3B_A             : out std_logic_vector(15 downto 0);
    DDR3B_BA            : out std_logic_vector(2 downto 0);
    DDR3B_CAS_n         : out std_logic;
    DDR3B_CK            : out std_logic_vector(0 downto 0);
    DDR3B_CKE           : out std_logic_vector(0 downto 0);
    DDR3B_CK_n          : out std_logic_vector(0 downto 0);
    DDR3B_CS_n          : out std_logic_vector(0 downto 0);
    DDR3B_DM            : out std_logic_vector(7 downto 0);
    DDR3B_DQ            : inout std_logic_vector(63 downto 0);
    DDR3B_DQS           : inout std_logic_vector(7 downto 0);
    DDR3B_DQS_n         : inout std_logic_vector(7 downto 0);
    DDR3B_EVENT_n       : in std_logic;
    DDR3B_ODT           : out std_logic_vector(0 downto 0);
    DDR3B_RAS_n         : out std_logic;
    DDR3B_REFCLK_p      : in std_logic;
    DDR3B_RESET_n       : out std_logic;
    DDR3B_SCL           : out std_logic;
    DDR3B_SDA           : inout std_logic;
    DDR3B_WE_n          : out std_logic;
    RZQ_DDR3_B          : in std_logic--;
    
);
end entity;

architecture rtl of top is

    -- free running clock (used as nios clock)
    signal clk_50 : std_logic;
    signal reset_50_n : std_logic;
    signal clk_50_cnt : unsigned(31 downto 0);

    -- global 125 MHz clock
    signal clk_125 : std_logic;
    signal reset_125_n : std_logic;
    signal clk_125_cnt : unsigned(31 downto 0);

    -- 156.25 MHz data clock (derived from global 125 MHz clock)
    signal clk_156 : std_logic;
    signal reset_156_n : std_logic;

    -- PCIe clock
    signal pcie_clk : std_logic;
    signal pcie_reset_n : std_logic;
    
    -- DDR3 clock
    signal A_ddr3clk_cnt : unsigned(31 downto 0);

    signal nios_clk : std_logic;
    signal nios_reset_n : std_logic;
    signal flash_rst_n : std_logic;
    signal flash_ce_n_i : std_logic;

    constant NLINKS_DATA : integer := 3;
    constant NLINKS_TOTL : integer := 16;
    constant LINK_FIFO_ADDR_WIDTH : integer := 10;

    signal reset : std_logic;
    signal reset_n : std_logic;

    signal resets : std_logic_vector(31 downto 0);
    signal resets_n: std_logic_vector(31 downto 0);
    
    signal resets_fast : std_logic_vector(31 downto 0);
    signal resets_n_fast: std_logic_vector(31 downto 0);
    signal resets_ddr3 : std_logic_vector(31 downto 0);
    signal resets_n_ddr3: std_logic_vector(31 downto 0);

    ------------------ Signal declaration ------------------------
    
    -- pcie read / write regs
    signal writeregs				: reg32array;
    signal writeregs_slow		: reg32array;
    signal writeregs_ddr3		: reg32array;
    signal regwritten				: std_logic_vector(63 downto 0);
    signal regwritten_B				: std_logic_vector(63 downto 0);
    signal regwritten_C				: std_logic_vector(63 downto 0);
    signal pb_in : std_logic_vector(2 downto 0);
    
    signal readregs				: reg32array;
    signal readregs_slow			: reg32array;
    signal readregs_ddr3			: reg32array;
    
    -- pcie read / write memory
    signal readmem_writedata 	: std_logic_vector(31 downto 0);
    signal readmem_writeaddr 	: std_logic_vector(63 downto 0);
    signal readmem_writeaddr_finished: std_logic_vector(15 downto 0);
    signal readmem_writeaddr_lowbits : std_logic_vector(15 downto 0);
    signal readmem_wren	 		: std_logic;
    signal readmem_endofevent 	: std_logic;
    signal writememreadaddr 	: std_logic_vector(15 downto 0);
    signal writememreaddata 	: std_logic_vector (31 downto 0);
    
    -- pcie dma
    signal dmamem_writedata 	: std_logic_vector(255 downto 0);
    signal dmamem_wren	 		: std_logic;
    signal dmamem_endofevent 	: std_logic;
    signal dmamemhalffull 		: std_logic;
    signal dmamemhalffull_counter : std_logic_vector(31 downto 0);
    signal dmamemnothalffull_counter : std_logic_vector(31 downto 0);
    signal endofevent_counter : std_logic_vector(31 downto 0);
    signal notendofevent_counter : std_logic_vector(31 downto 0);
    signal dmamemhalffull_tx : std_logic;
    signal sync_chain_halffull : std_logic_vector(1 downto 0);
    
    -- pcie dma2
    signal dma2mem_writedata 	: std_logic_vector(255 downto 0);
    signal dma2mem_wren	 		: std_logic;
    signal dma2mem_endofevent 	: std_logic;
    signal dma2memhalffull 		: std_logic;
    
    -- pcie fast clock
    signal pcie_fastclk_out		: std_logic;
    
    -- pcie debug signals
    signal pcie_testout         : std_logic_vector(127 downto 0);
    signal counter_256          : std_logic_vector(31 downto 0);
    signal counter_ddr3          : std_logic_vector(31 downto 0);
    
    -- Clocksync stuff
    signal clk_sync : std_logic;
    signal clk_last : std_logic;
    signal clk_sync_ddr3 : std_logic;
    signal clk_last_ddr3 : std_logic;

    -- debouncer
    signal push_button0_db : std_logic;
    signal push_button1_db : std_logic;
    signal push_button2_db : std_logic;
    signal push_button3_db : std_logic;

    -- NIOS
    signal i2c_scl_in   : std_logic;
    signal i2c_scl_oe   : std_logic;
    signal i2c_sda_in   : std_logic;
    signal i2c_sda_oe   : std_logic;
    signal flash_tcm_address_out : std_logic_vector(27 downto 0);
    signal wd_rst_n     : std_logic;
    signal cpu_pio_i : std_logic_vector(31 downto 0);
    signal debug_nios : std_logic_vector(31 downto 0);
    signal av_qsfp : work.util.avalon_array_t(3 downto 0);

    -- https://www.altera.com/support/support-resources/knowledge-base/solutions/rd01262015_264.html
    signal ZERO : std_logic := '0';
    attribute keep : boolean;
    attribute keep of ZERO : signal is true;

    -- tranciever ip signals
    signal tx_clk : std_logic_vector(15 downto 0);
    signal rx_clk : std_logic_vector(15 downto 0);
    type fifo_out_array_type is array (3 downto 0) of std_logic_vector(35 downto 0);

    --all signals from QSFP plugs
    signal QSFP_TX : std_logic_vector(15 downto 0);
    signal QSFP_RX : std_logic_vector(15 downto 0);
    --data behind QSFP tranceivers
    type data_array_type is array (NLINKS_TOTL-1 downto 0) of std_logic_vector(31 downto 0);
    type datak_array_type is array (NLINKS_TOTL-1 downto 0) of std_logic_vector(3 downto 0);
    signal rx_data : data_array_type;
--        signal tx_data : data_array_type;
    signal rx_datak : datak_array_type;
--        signal tx_datak : datak_array_type;

    signal rx_data_v:       std_logic_vector(NLINKS_TOTL*32-1 downto 0);
    signal rx_datak_v:      std_logic_vector(NLINKS_TOTL*4-1 downto 0);
    signal rx_data_v_raw:   std_logic_vector(NLINKS_TOTL*32-1 downto 0);
    signal rx_datak_v_raw:  std_logic_vector(NLINKS_TOTL*4-1 downto 0);
    signal rx_sc_v:         std_logic_vector(NLINKS_TOTL*32-1 downto 0);
    signal rx_sck_v:        std_logic_vector(NLINKS_TOTL*4-1 downto 0);
    signal rx_rc_v:         std_logic_vector(NLINKS_TOTL*32-1 downto 0);
    signal rx_rck_v:        std_logic_vector(NLINKS_TOTL*4-1 downto 0);
    
    signal tx_data_v:       std_logic_vector(NLINKS_TOTL*32-1 downto 0);
    signal tx_datak_v:      std_logic_vector(NLINKS_TOTL*4-1 downto 0);

    type mapping_t is array(natural range <>) of integer;
    --mapping as follows: fiber link_mapping(0)=1 - Fiber QSFPA.1 is mapped to first(0) link
    constant link_mapping:          mapping_t(NLINKS_DATA-1 downto 0):=(1,2,4);
    signal rx_mapped_data_v:        std_logic_vector(NLINKS_DATA*32-1 downto 0);
    signal rx_mapped_datak_v:       std_logic_vector(NLINKS_DATA*4-1 downto 0);
    signal rx_mapped_linkmask:      std_logic_vector(NLINKS_DATA-1  downto 0); --writeregs_slow(FEB_ENABLE_REGISTER_W)(NLINKS_-1 downto 0),
        
    -- Slow Control
    signal mem_data_out : std_logic_vector(63 downto 0);
    signal mem_datak_out : std_logic_vector(7 downto 0);
    signal mem_add_sc : std_logic_vector(15 downto 0);
    signal mem_data_sc : std_logic_vector(31 downto 0);
    signal mem_wen_sc : std_logic;
    
    -- Link test
    signal mem_add_link_test : std_logic_vector(2 downto 0);
    signal mem_data_link_test : std_logic_vector(31 downto 0);
    signal mem_wen_link_test : std_logic;
        
    -- event counter
    signal state_out_eventcounter : std_logic_vector(3 downto 0);
    signal state_out_datagen : std_logic_vector(3 downto 0);
    signal data_pix_generated : std_logic_vector(31 downto 0);
    signal datak_pix_generated : std_logic_vector(3 downto 0);
    signal data_pix_ready : std_logic;
    signal event_length : std_logic_vector(11 downto 0);
    signal dma_data_wren : std_logic;
    signal dma_data : std_logic_vector(255 downto 0);
    signal dma_data_test : std_logic_vector(159 downto 0);
    signal dma_event_data : std_logic_vector(255 downto 0);
    signal dma_wren_cnt : std_logic;
    signal dma_wren_test : std_logic;
    signal dma_end_event_cnt : std_logic;
    signal dma_end_event_test : std_logic;
    signal data_counter : std_logic_vector(32*NLINKS_TOTL-1 downto 0);
    signal datak_counter : std_logic_vector(4*NLINKS_TOTL-1 downto 0);
    signal feb_merger_timeouts : std_logic_vector(NLINKS_TOTL-1 downto 0);
    
    -- A interface
    signal A_ddr3clk            : std_logic;
    signal A_ddr3calibrated     : std_logic;
    signal A_ddr3ready          : std_logic;
    signal A_ddr3addr           : std_logic_vector(25 downto 0);
    signal A_ddr3datain         : std_logic_vector(511 downto 0);
    signal A_ddr3dataout        : std_logic_vector(511 downto 0);
    signal A_ddr3_write         : std_logic;
    signal A_ddr3_read          : std_logic;
    signal A_ddr3_read_valid    : std_logic;

    -- B interface
    signal B_ddr3clk            : std_logic;
    signal B_ddr3calibrated     : std_logic;
    signal B_ddr3ready          : std_logic;
    signal B_ddr3addr           : std_logic_vector(25 downto 0);
    signal B_ddr3datain         : std_logic_vector(511 downto 0);
    signal B_ddr3dataout        : std_logic_vector(511 downto 0);
    signal B_ddr3_write         : std_logic;
    signal B_ddr3_read          : std_logic;
    signal B_ddr3_read_valid    : std_logic;

begin

    --! local 50 MHz clock (oscillator)
    clk_50 <= CLK_50_B2J;

    --! generate reset for 50 MHz
    e_reset_50_n : entity work.reset_sync
    port map ( o_reset_n => reset_50_n, i_reset_n => CPU_RESET_n, i_clk => clk_50 );

    --! generate reset for 125 MHz
    e_reset_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_125_n, i_reset_n => CPU_RESET_n, i_clk => clk_125 );
    
    --! generate reset for pcie_fastclk_out
    e_reset_pcie_n : entity work.reset_sync
    port map ( o_reset_n => reset_pcie_n, i_reset_n => CPU_RESET_n, i_clk => pcie_fastclk_out );

    --! generate and route 125 MHz clock to SMA output
    --! (can be connected to SMA input as global clock)
    e_pll_50to125 : component work.cmp.ip_pll_50to125
    port map (
        outclk_0 => SMA_CLKOUT,
        refclk => clk_50,
        rst => not reset_50_n
    );

    --! 125 MHz global clock (from SMA input)
    e_clk_125 : work.cmp.ip_clk_ctrl
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
        g_XCVR0_CHANNELS => 16,
        g_XCVR0_N => 4,
        g_XCVR1_CHANNELS => 0,
        g_XCVR1_N => 0,
        g_PCIE0_X => 8,
        g_PCIE1_X => 0,
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
        i_BUTTON                        => BUTTON,

        -- XCVR0 (6250 Mbps @ 156.25 MHz)
        i_xcvr0_rx( 3 downto  0)        => QSFPA_RX_p,
        i_xcvr0_rx( 7 downto  4)        => QSFPB_RX_p,
        i_xcvr0_rx(11 downto  8)        => QSFPC_RX_p,
        i_xcvr0_rx(15 downto 12)        => QSFPD_RX_p,
        o_xcvr0_tx( 3 downto  0)        => QSFPA_TX_p,
        o_xcvr0_tx( 7 downto  4)        => QSFPB_TX_p,
        o_xcvr0_tx(11 downto  8)        => QSFPC_TX_p,
        o_xcvr0_tx(15 downto 12)        => QSFPD_TX_p,
        i_xcvr0_refclk                  => (others => clk_125),

        o_xcvr0_rx_data                 => rx_data_raw,
        o_xcvr0_rx_datak                => rx_datak_raw,
        i_xcvr0_tx_data                 => tx_data,
        i_xcvr0_tx_datak                => tx_datak,

        -- PCIe0
        i_pcie0_rx                      => PCIE_RX_p,
        o_pcie0_tx                      => PCIE_TX_p,
        i_pcie0_perst_n                 => PCIE_PERST_n,
        i_pcie0_refclk                  => PCIE_REFCLK_p,
        o_pcie0_clk                     => pcie_fastclk_out,
        o_pcie0_clk_hz                  => LED(3),

        -- PCIe0 DMA0
        i_pcie0_dma0_wdata              => dma_data,
        i_pcie0_dma0_we                 => dma_data_wren,
        i_pcie0_dma0_eoe                => dmamem_endofevent,
        o_pcie0_dma0_hfull              => pcie0_dma0_hfull,
        i_pcie0_dma0_clk                => pcie_fastclk_out,

        -- PCIe0 read interface to writable memory
        i_pcie0_wmem_addr               => writememreadaddr,
        o_pcie0_wmem_rdata              => writememreaddata,
        i_pcie0_wmem_clk                => clk_156,

        -- PCIe0 write interface to readable memory
        i_pcie0_rmem_addr               => readmem_writeaddr,
        i_pcie0_rmem_wdata              => readmem_writedata,
        i_pcie0_rmem_we                 => readmem_wren,
        i_pcie0_rmem_clk                => clk_156,

        -- PCIe0 update interface for readable registers
        i_pcie0_rregs_156               => pcie0_readregs_156,
        i_pcie0_rregs_250               => pcie0_readregs_250,

        -- PCIe0 read interface for writable registers
        o_pcie0_wregs_A                 => pcie0_writeregs_250,
        i_pcie0_wregs_A_clk             => pcie_fastclk_out,
        o_pcie0_wregs_B                 => pcie0_writeregs_156,
        i_pcie0_wregs_B_clk             => clk_156,
        o_pcie0_resets_n_156            => pcie0_resets_n_156,
        o_pcie0_resets_n_250            => pcie0_resets_n_250,

        -- resets clk
        o_reset_156_n                   => reset_156_n,
        o_clk_156                       => clk_156,
        o_clk_156_hz                    => LED(2),

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

    -------- Demerge Data --------
    g_demerge: for i in NLINKS_TOTL-1 downto 0 generate
        e_data_demerge : entity work.data_demerge
        port map(
            i_clk               => clk_156,
            i_reset             => not resets_n(RESET_BIT_EVENT_COUNTER),
            i_aligned           => '1',
            i_data              => rx_data_v_raw(31+i*32 downto i*32),
            i_datak             => rx_datak_v_raw(3+i* 4 downto i* 4),
            i_fifo_almost_full  => '0',--link_fifo_almost_full(i),
            o_data              => rx_data_v(31+i*32 downto i*32),
            o_datak             => rx_datak_v(3+i* 4 downto i* 4),
            o_sc                => rx_sc_v(31+i*32 downto i*32),
            o_sck               => rx_sck_v(3+i* 4 downto i* 4),
            o_rc                => rx_rc_v(31+i*32 downto i*32),
            o_rck               => rx_rck_v(3+i* 4 downto i* 4),
            o_fpga_id           => open--,
        );
    end generate;
    

    -------- MIDAS RUN control --------

    e_run_control : entity work.run_control
    generic map (
        N_LINKS_g                           => NLINKS_TOTL--,
    )
    port map (
        i_reset_ack_seen_n                  => resets_n(RESET_BIT_RUN_START_ACK),
        i_reset_run_end_n                   => resets_n(RESET_BIT_RUN_END_ACK),
        i_buffers_empty                     => (others => '1'), -- TODO: connect buffers emtpy from dma here
        o_feb_merger_timeout                => readregs_slow(CNT_FEB_MERGE_TIMEOUT_R),
        i_aligned                           => (others => '1'),
        i_data                              => rx_rc_v,
        i_datak                             => rx_rck_v,
        i_link_enable                       => writeregs_slow(FEB_ENABLE_REGISTER_W),
        i_addr                              => writeregs_slow(RUN_NR_ADDR_REGISTER_W), -- ask for run number of FEB with this addr.
        i_run_number                        => writeregs_slow(RUN_NR_REGISTER_W)(23 downto 0),
        o_run_number                        => readregs_slow(RUN_NR_REGISTER_R), -- run number of i_addr
        o_runNr_ack                         => readregs_slow(RUN_NR_ACK_REGISTER_R), -- which FEBs have responded with run number in i_run_number
        o_run_stop_ack                      => readregs_slow(RUN_STOP_ACK_REGISTER_R),
        i_clk                               => clk_156--,
    );

    -------- Event Builder --------

    e_data_gen : entity work.data_generator_a10
    generic map(
        go_to_trailer => 4,
        go_to_sh => 3--,
    )
    port map (
        reset               => resets(RESET_BIT_DATAGEN),
        enable_pix          => writeregs_slow(DATAGENERATOR_REGISTER_W)(DATAGENERATOR_BIT_ENABLE),
        i_dma_half_full     => dmamemhalffull_tx,
        random_seed         => (others => '1'),
        data_pix_generated  => data_pix_generated,
        datak_pix_generated => datak_pix_generated,
        data_pix_ready      => data_pix_ready,
        start_global_time   => (others => '0'),
        slow_down           => writeregs_slow(DATAGENERATOR_DIVIDER_REGISTER_W),
        state_out           => state_out_datagen,
        clk                 => clk_156--,
    );
    
    -- sync halffull 2 flip-flops
    sync_halffull : process(clk_156, reset_156_n)
    begin
    if ( reset_156_n = '0' ) then
        sync_chain_halffull <= (others => '0');
    elsif rising_edge(clk_156) then
        sync_chain_halffull <= sync_chain_halffull(sync_chain_halffull'high-1 downto 0) & dmamemhalffull;
    end if;
    end process;

    dmamemhalffull_tx <= sync_chain_halffull(sync_chain_halffull'high);

    process(clk_156, reset_156_n)
    begin
    if ( reset_156_n = '0' ) then
        data_counter    <= (others => '0');
        datak_counter   <= (others => '0');
    elsif rising_edge(clk_156) then
        if (writeregs_slow(DATAGENERATOR_REGISTER_W)(DATAGENERATOR_BIT_ENABLE) = '1') then
            set_gen_data : FOR i in 0 to NLINKS_TOTL - 1 LOOP
                data_counter(31 + i * 32 downto i * 32) <= data_pix_generated;
                datak_counter(3 + i * 4 downto i * 4) <= datak_pix_generated;
            END LOOP set_gen_data;
        else
            set_link_data : FOR i in 0 to NLINKS_TOTL - 1 LOOP
                data_counter(31 + i * 32 downto i * 32) <= rx_data(i);
                datak_counter(3 + i * 4 downto i * 4) <= rx_datak(i);
            END LOOP set_link_data;
        end if;
    end if;
    end process;
    
--    e_link_merger : entity work.link_merger
--    generic map(
--        NLINKS_TOTL             => 4,--NLINKS_TOTL,
--        LINK_FIFO_ADDR_WIDTH    => 8--,
--    )
--    port map(
--        i_reset_data_n      => resets_n(RESET_BIT_LINK_MERGER),
--        i_reset_mem_n       => resets_n_fast(RESET_BIT_LINK_MERGER),--resets_n_ddr3(RESET_BIT_LINK_MERGER),
--        i_dataclk           => clk_156,
--        i_memclk            => pcie_fastclk_out,--A_ddr3clk,--clk_156,
--
--        i_link_data         => data_counter(31 + 0 * 32 downto 0 * 32) & data_counter(31 + 1 * 32 downto 1 * 32) & data_counter(31 + 2 * 32 downto 2 * 32) & data_counter(31 + 3 * 32 downto 3 * 32), --& data_pix_generated & data_pix_generated,
--        i_link_datak        => datak_counter(3 + 0 * 4 downto 0 * 4) & datak_counter(3 + 1 * 4 downto 1 * 4) & datak_counter(3 + 2 * 4 downto 2 * 4) & datak_counter(3 + 3 * 4 downto 3 * 4), -- & datak_pix_generated & datak_pix_generated,
--        i_link_valid        => 1,
--        i_link_mask_n       => writeregs(DATA_LINK_MASK_REGISTER_W)(3 downto 0),--(others => '1'),--writeregs(DATA_LINK_MASK_REGISTER_W)(NLINKS_TOTL - 1 downto 0), -- if 1 the link is active
--        
--        o_stream_rdata(0)   => LED_BRACKET(0),
--        o_hit               => dma_data,
--        o_stream_rempty     => open,
--        i_stream_rack       => '1'--,
--    );
--    dma_data_wren <= '1';
    
--     e_midas_event_builder : entity work.midas_event_builder
--     generic map (
--         NLINKS => NLINKS_TOTL--;
--     )
--     port map (
-- 			i_clk_data          => clk_156,
-- 			i_clk_dma           => pcie_fastclk_out,
-- 			i_reset_data_n      => resets_n(RESET_BIT_EVENT_COUNTER),
-- 			i_reset_dma_n       => resets_n_fast(RESET_BIT_EVENT_COUNTER),
-- 			i_rx_data           => data_counter,
-- 			i_rx_datak          => datak_counter,
-- 			i_wen_reg           => writeregs(DMA_REGISTER_W)(DMA_BIT_ENABLE),
-- 			i_link_mask_n       => writeregs(DATA_LINK_MASK_REGISTER_W)(NLINKS_TOTL - 1 downto 0), -- if 1 the link is active
-- 			i_get_n_words       => writeregs(GET_N_DMA_WORDS_REGISTER_W),
-- 			i_dmamemhalffull    => dmamemhalffull,
-- 			o_fifos_full	    => open,--readregs(EVENT_BUILD_STATUS_REGISTER_R)(31 downto 31 - NLINKS_TOTL),
-- 			o_done              => readregs(EVENT_BUILD_STATUS_REGISTER_R)(EVENT_BUILD_DONE),
-- 			o_event_wren        => dma_wren_cnt,
-- 			o_endofevent        => dma_end_event_cnt,
-- 			o_event_data        => dma_event_data,
-- 			o_state_out         => state_out_eventcounter,
--             error cnt signals
-- 			o_fifo_almost_full  => open,--link_fifo_almost_full,
--             o_fifo_almost_full          => open,
--             o_cnt_link_fifo_almost_full => readregs_slow(CNT_FIFO_ALMOST_FULL_R),
--             o_cnt_tag_fifo_full         => readregs(CNT_TAG_FIFO_FULL_R),
--             o_cnt_ram_full              => readregs(CNT_RAM_FULL_R),
--             o_cnt_stream_fifo_full      => readregs(CNT_STREAM_FIFO_FULL_R),
--             o_cnt_dma_halffull          => readregs(CNT_DMA_HALFFULL_R),
--             o_cnt_dc_link_fifo_full     => readregs_slow(CNT_DC_LINK_FIFO_FULL_R),
--             o_cnt_skip_link_data        => readregs_slow(CNT_SKIP_EVENT_LINK_FIFO_R),
--             o_cnt_skip_event_dma        => readregs(CNT_SKIP_EVENT_DMA_RAM_R),
--             o_cnt_idle_not_header       => readregs(CNT_IDLE_NOT_HEADER_R)--,
--     );
--     
--     dma_data <= dma_event_data;
--     dma_data_wren <= dma_wren_cnt;
--     dmamem_endofevent <= dma_end_event_cnt;
    
    -------- Slow Control --------
    
    e_sc_main : work.sc_main
    generic map (
        NLINKS => NLINKS_TOTL
    )
    port map (
        i_clk           => clk_156,
        i_reset_n       => resets_n(RESET_BIT_SC_MAIN),
        i_length_we     => writeregs_slow(SC_MAIN_ENABLE_REGISTER_W)(0),
        i_length        => writeregs_slow(SC_MAIN_LENGTH_REGISTER_W)(15 downto 0),
        i_mem_data      => writememreaddata,
        o_mem_addr      => writememreadaddr,
        o_mem_data      => tx_data_v,
        o_mem_datak     => tx_datak_v,
        o_done          => readregs_slow(SC_MAIN_STATUS_REGISTER_R)(SC_MAIN_DONE),
        o_state         => open--,
    );
    
    e_sc_secondary : work.sc_secondary
    generic map (
        NLINKS => NLINKS_TOTL
    )
    port map (
        reset_n                 => resets_n(RESET_BIT_SC_SECONDARY),
        i_link_enable           => writeregs_slow(FEB_ENABLE_REGISTER_W)(NLINKS_TOTL-1 downto 0),
        link_data_in            => rx_sc_v,
        link_data_in_k          => rx_sck_v,
        mem_addr_out            => mem_add_sc,
        mem_addr_finished_out   => readmem_writeaddr_finished,
        mem_data_out            => mem_data_sc,
        mem_wren                => mem_wen_sc,
        stateout(0)             => open,--LED_BRACKET(1),
        clk                     => clk_156--,
    );
    
    -------- Link Test --------
    
    e_link_observer : entity work.link_observer
    generic map (
        g_m     => 32,
        g_poly  => "10000000001000000000000000000110"
    )
    port map (
        reset_n     => resets_n(RESET_BIT_LINK_TEST),
        rx_data     => rx_data(1),
        rx_datak    => rx_datak(1),
        mem_add     => mem_add_link_test,
        mem_data    => mem_data_link_test,
        mem_wen     => mem_wen_link_test,
        clk         => clk_156--,
    );

    process(clk_156, reset_156_n)
    begin
    if ( reset_156_n = '0' ) then
        readmem_writeaddr <= (others => '0');
        readmem_writedata <= (others => '0');
        readmem_wren <= '0';
    elsif rising_edge(clk_156) then
        readmem_writeaddr <= (others => '0');
        if (writeregs_slow(LINK_TEST_REGISTER_W)(LINK_TEST_BIT_ENABLE) = '1') then
            readmem_writeaddr(2 downto 0)   <= mem_add_link_test;
            readmem_writedata               <= mem_data_link_test;
            readmem_wren                    <= mem_wen_link_test;
        else
            readmem_writeaddr(15 downto 0)  <= mem_add_sc;
            readmem_writedata               <= mem_data_sc;
            readmem_wren                    <= mem_wen_sc;
        end if;
    end if;
    end process;

    -------- PCIe --------

    -- reset regs
    e_reset_logic : entity work.reset_logic
    port map (
        rst_n                   => push_button0_db,
        reset_register          => writeregs_slow(RESET_REGISTER_W),
        resets                  => resets,
        resets_n                => resets_n,
        clk                     => clk_156--,
    );
    
    e_reset_logic_fast : entity work.reset_logic
    port map (
        rst_n                   => push_button0_db,
        reset_register          => writeregs(RESET_REGISTER_W),
        resets                  => resets_fast,
        resets_n                => resets_n_fast,
        clk                     => pcie_fastclk_out--,
    );
    
    e_reset_logic_ddr3 : entity work.reset_logic
    port map (
        rst_n                   => push_button0_db,
        reset_register          => writeregs_ddr3(RESET_REGISTER_W),
        resets                  => resets_ddr3,
        resets_n                => resets_n_ddr3,
        clk                     => DDR3A_REFCLK_p--A_ddr3clk--,
    );

    e_version_reg : entity work.version_reg
    port map (
        data_out  => readregs_slow(VERSION_REGISTER_R)(27 downto 0)
    );

    --Sync read regs from slow (156.25 MHz) to fast (250 MHz) clock
    process(pcie_fastclk_out)
    begin
    if rising_edge(pcie_fastclk_out) then
        clk_sync <= clk_156;
        clk_last <= clk_sync;
        
        clk_sync_ddr3 <= DDR3A_REFCLK_p;
        clk_last_ddr3 <= clk_sync_ddr3;
        
        if(clk_sync = '1' and clk_last = '0') then
            readregs(PLL_REGISTER_R)                <= readregs_slow(PLL_REGISTER_R);
            readregs(VERSION_REGISTER_R)            <= readregs_slow(VERSION_REGISTER_R);
            readregs(RUN_NR_REGISTER_R)             <= readregs_slow(RUN_NR_REGISTER_R);
            readregs(RUN_NR_ACK_REGISTER_R)         <= readregs_slow(RUN_NR_ACK_REGISTER_R);
            readregs(RUN_STOP_ACK_REGISTER_R)       <= readregs_slow(RUN_STOP_ACK_REGISTER_R);
            readregs(CNT_FEB_MERGE_TIMEOUT_R)       <= readregs_slow(CNT_FEB_MERGE_TIMEOUT_R);
            readregs(MEM_WRITEADDR_HIGH_REGISTER_R) <= (others => '0');
            readregs(MEM_WRITEADDR_LOW_REGISTER_R)  <= (X"0000" & readmem_writeaddr_finished);
        end if;
        
        if(clk_sync_ddr3 = '1' and clk_last_ddr3 = '0') then
            readregs(DDR3_STATUS_R) <= readregs_ddr3(DDR3_STATUS_R);
            readregs(DDR3_ERR_R) <= readregs_ddr3(DDR3_ERR_R);
            readregs(DDR3_CLK_CNT_R) <= readregs_ddr3(DDR3_CLK_CNT_R);
        end if;
        
        readregs(DMA_STATUS_R)(DMA_DATA_WEN)        <= dma_data_wren;
        readregs(DMA_HALFFUL_REGISTER_R)            <= dmamemhalffull_counter;
        readregs(DMA_NOTHALFFUL_REGISTER_R)         <= dmamemnothalffull_counter;
        readregs(DMA_ENDEVENT_REGISTER_R)           <= endofevent_counter;
        readregs(DMA_NOTENDEVENT_REGISTER_R)        <= notendofevent_counter;
    end if;
    end process;

    -- DMA status stuff
    e_dma_evaluation : entity work.dma_evaluation
    port map (
        reset_n                 => resets_n_fast(RESET_BIT_DMA_EVAL),
        dmamemhalffull          => dmamemhalffull,
        dmamem_endofevent       => dmamem_endofevent,
        halffull_counter        => dmamemhalffull_counter,
        nothalffull_counter     => dmamemnothalffull_counter,
        endofevent_counter      => endofevent_counter,
        notendofevent_counter   => notendofevent_counter,
        clk                     => pcie_fastclk_out--,
    );
       
    readmem_writeaddr_lowbits   <= readmem_writeaddr(15 downto 0);
    pb_in                       <= push_button0_db & push_button1_db & push_button2_db;
        
    counter256 : entity work.counter
    generic map (
        W => 32--,
    )
    port map (
        o_cnt       => counter_256,
        i_ena       => '1',
        
        i_clk       => pcie_fastclk_out,
        i_reset_n   => resets_n(RESET_BIT_DATAGEN)--,
    );
        
    counterddr3 : entity work.counter
    generic map (
        W => 32--,
    )
    port map (
        o_cnt       => counter_ddr3,
        i_ena       => '1',
        
        i_clk       => A_ddr3clk,
        i_reset_n   => resets_n_ddr3(RESET_BIT_DATAGEN)--,
    );

    e_pcie_block : entity work.pcie_block
    generic map (
        DMAMEMWRITEADDRSIZE     => 11,
        DMAMEMREADADDRSIZE      => 11,
        DMAMEMWRITEWIDTH        => 256--,
    )
    port map (
        local_rstn              => '1',
        appl_rstn               => '1',
        refclk                  => PCIE_REFCLK_p,
        pcie_fastclk_out        => pcie_fastclk_out,
        
        --//PCI-Express--------------------------//25 pins //--------------------------
        pcie_rx_p               => PCIE_RX_p,
        pcie_tx_p               => PCIE_TX_p,
        pcie_refclk_p           => PCIE_REFCLK_p,
        pcie_led_g2             => open,
        pcie_led_x1             => open,
        pcie_led_x4             => open,
        pcie_led_x8             => open,
        pcie_perstn             => PCIE_PERST_n,
        pcie_smbclk             => PCIE_SMBCLK,
        pcie_smbdat             => PCIE_SMBDAT,
        pcie_waken              => PCIE_WAKE_n,
        
        -- LEDs
        alive_led               => open,
        comp_led                => open,
        L0_led                  => open,
        
        -- pcie registers (write / read register, readonly, read write, in tools/dmatest/rw) -Sync read regs
        writeregs               => writeregs,
        o_writeregs_B           => writeregs_slow,
        o_writeregs_C           => writeregs_ddr3,
        
        regwritten              => regwritten,
        o_regwritten_B          => regwritten_B,
        o_regwritten_C          => regwritten_C,
        
        i_clk_B                 => clk_156,
        i_clk_C                 => DDR3A_REFCLK_p,--A_ddr3clk,
        readregs                => readregs,

        -- pcie writeable memory
        writememclk             => clk_156,
        writememreadaddr        => writememreadaddr,
        writememreaddata        => writememreaddata,

        -- pcie readable memory
        readmem_data            => readmem_writedata,
        readmem_addr            => readmem_writeaddr_lowbits,
        readmemclk              => clk_156,
        readmem_wren            => readmem_wren,
        readmem_endofevent      => readmem_endofevent,
        
        -- dma memory
        dma_data                => dma_data,
        dmamemclk               => pcie_fastclk_out,
        dmamem_wren             => dma_data_wren,
        dmamem_endofevent       => dmamem_endofevent,
        dmamemhalffull          => dmamemhalffull,
        
        -- dma memory
        dma2_data               => dma2mem_writedata,
        dma2memclk              => pcie_fastclk_out,
        dma2mem_wren            => dma2mem_wren,
        dma2mem_endofevent      => dma2mem_endofevent,
        dma2memhalffull         => dma2memhalffull,
        
        -- test ports
        testout                 => pcie_testout,
        testout_ena             => open,
        pb_in                   => pb_in,
        inaddr32_r              => readregs(inaddr32_r),
        inaddr32_w              => readregs(inaddr32_w)--,
    );
    
    
    ddr3_b : entity work.ddr3_block 
    port map(
        reset_n             => resets_n_ddr3(RESET_BIT_DDR3),
        
        -- Control and status registers
        ddr3control         => writeregs_ddr3(DDR3_CONTROL_W),
        ddr3status          => readregs_ddr3(DDR3_STATUS_R),

        -- A interface
        A_ddr3clk           => A_ddr3clk,
        A_ddr3calibrated    => A_ddr3calibrated,
        A_ddr3ready         => A_ddr3ready,
        A_ddr3addr          => A_ddr3addr,
        A_ddr3datain        => A_ddr3datain,
        A_ddr3dataout       => A_ddr3dataout,
        A_ddr3_write        => A_ddr3_write,
        A_ddr3_read         => A_ddr3_read,
        A_ddr3_read_valid   => A_ddr3_read_valid,
        
        -- B interface
        B_ddr3clk           => B_ddr3clk,
        B_ddr3calibrated    => B_ddr3calibrated,
        B_ddr3ready         => B_ddr3ready,
        B_ddr3addr          => B_ddr3addr,
        B_ddr3datain        => B_ddr3datain,
        B_ddr3dataout       => B_ddr3dataout,
        B_ddr3_write        => B_ddr3_write,
        B_ddr3_read         => B_ddr3_read,
        B_ddr3_read_valid   => B_ddr3_read_valid,
        
        -- Error counters
        errout              => readregs_ddr3(DDR3_ERR_R),

        -- Interface to memory bank A
        A_mem_ck            => DDR3A_CK,
        A_mem_ck_n          => DDR3A_CK_n,
        A_mem_a             => DDR3A_A,
        A_mem_ba            => DDR3A_BA,
        A_mem_cke           => DDR3A_CKE,
        A_mem_cs_n          => DDR3A_CS_n,
        A_mem_odt           => DDR3A_ODT,
        A_mem_reset_n(0)    => DDR3A_RESET_n,      
        A_mem_we_n(0)       => DDR3A_WE_n,
        A_mem_ras_n(0)      => DDR3A_RAS_n,
        A_mem_cas_n(0)      => DDR3A_CAS_n,
        A_mem_dqs           => DDR3A_DQS,
        A_mem_dqs_n         => DDR3A_DQS_n,
        A_mem_dq            => DDR3A_DQ,
        A_mem_dm            => DDR3A_DM,
        A_oct_rzqin         => RZQ_DDR3_A,
        A_pll_ref_clk       => DDR3A_REFCLK_p,
        
        -- Interface to memory bank B
        B_mem_ck            => DDR3B_CK,
        B_mem_ck_n          => DDR3B_CK_n,
        B_mem_a             => DDR3B_A,
        B_mem_ba            => DDR3B_BA,
        B_mem_cke           => DDR3B_CKE,
        B_mem_cs_n          => DDR3B_CS_n,
        B_mem_odt           => DDR3B_ODT,
        B_mem_reset_n(0)    => DDR3B_RESET_n,      
        B_mem_we_n(0)       => DDR3B_WE_n,
        B_mem_ras_n(0)      => DDR3B_RAS_n,
        B_mem_cas_n(0)      => DDR3B_CAS_n,
        B_mem_dqs           => DDR3B_DQS,
        B_mem_dqs_n         => DDR3B_DQS_n,
        B_mem_dq            => DDR3B_DQ,
        B_mem_dm            => DDR3B_DM,
        B_oct_rzqin         => RZQ_DDR3_B,
        B_pll_ref_clk       => DDR3B_REFCLK_p--,
     );
 
     DDR3A_SDA   <= 'Z';
     DDR3B_SDA   <= 'Z';
   
     dataflow : entity work.data_flow 
     port map(
         reset_n                 => resets_n_fast(RESET_BIT_DDR3),
         reset_n_ddr3            => resets_n_ddr3(RESET_BIT_DDR3),
 
--         Input from merging (first board) or links (subsequent boards)
         dataclk                 => A_ddr3clk,--pcie_fastclk_out,
         data_en                 => '1',--link_fifo_ren,
--         31 downto 0 -> data, 35 downto 32 -> datak, 37 downto 36 -> sop/eop
         data_in                 => (others => '0'),
         ts_in                   => counter_ddr3,--counter_256(31 downto 0),
 
--         Input from PCIe demanding events
         pcieclk                 => pcie_fastclk_out,
         ts_req_A                => writeregs_ddr3(DATA_REQ_A_W),
         req_en_A                => regwritten_C(DATA_REQ_A_W),
         ts_req_B                => writeregs_ddr3(DATA_REQ_B_W),
         req_en_B                => regwritten_C(DATA_REQ_A_W),
         tsblock_done            => writeregs_ddr3(DATA_TSBLOCK_DONE_W)(15 downto 0),
         tsblocks                => readregs_ddr3(DATA_TSBLOCKS_R),
 
--         Output to DMA
         dma_data_out            => open,--dma_data,
         dma_data_en             => open,--dma_data_wren,
         dma_eoe                 => open,--dmamem_endofevent,
 
--         Output to links -- with dataclk
         link_data_out           => open,
         link_ts_out             => open,
         link_data_en            => open,
 
--         Interface to memory bank A
         A_mem_clk               => A_ddr3clk,
         A_mem_ready             => A_ddr3ready,
         A_mem_calibrated        => A_ddr3calibrated,
         A_mem_addr              => A_ddr3addr,
         A_mem_data              => A_ddr3datain(255 downto 0),
         A_mem_write             => A_ddr3_write,
         A_mem_read              => A_ddr3_read,
         A_mem_q                 => A_ddr3dataout(255 downto 0),
         A_mem_q_valid           => A_ddr3_read_valid,
 
--         Interface to memory bank B
         B_mem_clk               => B_ddr3clk,
         B_mem_ready             => B_ddr3ready,
         B_mem_calibrated        => B_ddr3calibrated,
         B_mem_addr              => B_ddr3addr,
         B_mem_data              => B_ddr3datain(255 downto 0),
         B_mem_write             => B_ddr3_write,
         B_mem_read              => B_ddr3_read,
         B_mem_q                 => B_ddr3dataout(255 downto 0),
         B_mem_q_valid           => B_ddr3_read_valid--,
         
     );

end architecture;
