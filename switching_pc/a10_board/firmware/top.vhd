library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.pcie_components.all;
use work.mudaq_registers.all;

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
    CLK_50_B2J          : in    std_logic--;
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

    signal flash_cs_n : std_logic;


        constant NLINKS_ALIGNMENT : integer := 36;
        constant NLINKS_DATA : integer := 3;
        constant NLINKS_TOTL : integer := 16;

        signal resets : std_logic_vector(31 downto 0);
        signal pcie0_resets_n_156: std_logic_vector(31 downto 0);
        signal pcie0_resets_n_250: std_logic_vector(31 downto 0);
        
        signal resets_fast : std_logic_vector(31 downto 0);
        signal resets_n_fast: std_logic_vector(31 downto 0);

        ------------------ Signal declaration ------------------------
        
        -- pcie read / write regs
        signal writeregs				: reg32array;
        signal writeregs_slow		: reg32array;
        signal pb_in : std_logic_vector(2 downto 0);
        
        signal readregs				: reg32array;
        signal readregs_slow			: reg32array;
        
        -- pcie read / write memory
        signal readmem_writedata 	: std_logic_vector(31 downto 0);
        signal readmem_writeaddr 	: std_logic_vector(63 downto 0);
        signal readmem_writeaddr_finished: std_logic_vector(15 downto 0);
        signal readmem_writeaddr_lowbits : std_logic_vector(15 downto 0);
        signal readmem_wren	 		: std_logic;
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

        -- //pcie fast clock
        signal pcie_fastclk_out		: std_logic;

        -- Clocksync stuff
        signal clk_sync : std_logic;
        signal clk_last : std_logic;
        
        -- debouncer
        signal push_button0_db : std_logic;
        signal push_button1_db : std_logic;
        signal push_button2_db : std_logic;
        signal push_button3_db : std_logic;

        type fifo_out_array_type is array (3 downto 0) of std_logic_vector(35 downto 0);

    -- data behind QSFP tranceivers
    signal rx_data_raw, rx_data, tx_data : work.util.slv32_array_t(NLINKS_TOTL-1 downto 0);
    signal rx_datak_raw, rx_datak, tx_datak : work.util.slv4_array_t(NLINKS_TOTL-1 downto 0);

    signal rx_sc_v:         std_logic_vector(NLINKS_TOTL*32-1 downto 0);
    signal rx_sck_v:        std_logic_vector(NLINKS_TOTL*4-1 downto 0);
    signal rx_rc_v:         std_logic_vector(NLINKS_TOTL*32-1 downto 0);
    signal rx_rck_v:        std_logic_vector(NLINKS_TOTL*4-1 downto 0);

    type mapping_t is array(natural range <>) of integer;
    --mapping as follows: fiber link_mapping(0)=1 - Fiber QSFPA.1 is mapped to first(0) link
    constant link_mapping:          mapping_t(NLINKS_DATA-1 downto 0):=(1,2,4);
    signal rx_mapped_data_v:        std_logic_vector(NLINKS_DATA*32-1 downto 0);
    signal rx_mapped_datak_v:       std_logic_vector(NLINKS_DATA*4-1 downto 0);
    signal rx_mapped_linkmask:      std_logic_vector(NLINKS_DATA-1  downto 0); --writeregs_slow(FEB_ENABLE_REGISTER_W)(NLINKS_-1 downto 0),
    signal link_fifo_almost_full:   std_logic_vector(NLINKS_TOTL-1 downto 0);

--        signal idle_ch : std_logic_vector(3 downto 0);
--        
--        signal sc_data : data_array_type;
--        signal sc_datak : datak_array_type;
--        signal sc_ready : std_logic_vector(3 downto 0);
--        signal fifo_data : data_array_type;
--        signal fifo_datak : datak_array_type;
--        signal fifo_wren : std_logic_vector(3 downto 0);
--        signal fifo_out : fifo_out_array_type;
--        
--        signal fifo_read : std_logic;
--        signal fifo_empty : std_logic_vector(3 downto 0);
        
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
        signal data_bank : std_logic_vector(32*64-1 downto 0);
        signal datak_bank : std_logic_vector(4*64-1 downto 0);
        signal feb_merger_timeouts : std_logic_vector(NLINKS_TOTL-1 downto 0);

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
        o_LED                           => LED,
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

        -- PCIe0 DMA0
        i_pcie0_dma0_wdata              => dma_data,
        i_pcie0_dma0_we                 => dma_data_wren,
        i_pcie0_dma0_eoe                => dmamem_endofevent,
        o_pcie0_dma0_hfull              => dmamemhalffull,
        i_pcie0_dma0_clk                => pcie_fastclk_out,

        -- PCIe0 read interface to writable memory
        i_pcie0_wmem_addr               => writememreadaddr,
        o_pcie0_wmem_rdata              => writememreaddata,
        i_pcie0_wmem_clk                => clk_156,

        -- PCIe0 write interface to readable memory
        i_pcie0_rmem_addr               => readmem_writeaddr_lowbits,
        i_pcie0_rmem_wdata              => readmem_writedata,
        i_pcie0_rmem_we                 => readmem_wren,
        i_pcie0_rmem_clk                => clk_156,

        -- PCIe0 update interface for readable registers
        i_pcie0_rregs                   => readregs,

        -- PCIe0 read interface for writable registers
        o_pcie0_wregs_A                 => writeregs,
        i_pcie0_wregs_A_clk             => pcie_fastclk_out,
        o_pcie0_wregs_B                 => writeregs_slow,
        i_pcie0_wregs_B_clk             => clk_156,
        o_pcie0_resets_n_156            => pcie0_resets_n_156,
        o_pcie0_resets_n_250            => pcie0_resets_n_250,

        -- resets clk
        o_reset_156_n                   => reset_156_n,
        o_clk_156                       => clk_156,

        i_reset_125_n                   => reset_125_n,
        i_clk_125                       => clk_125,

        i_reset_n                       => reset_50_n,
        i_clk                           => clk_50--,
    );


    --! A10 development board setups
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


    -------- Receiving Data and word aligning --------


    
    
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
    
    data_bank <=    x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"        &
                    x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"        &
                    x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"        &
                    x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"               & x"00000000"        &
                    data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & 
                    data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) &
                    data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) &
                    data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) &
                    data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0) & data_counter(31 downto 0);
   datak_bank <=    x"0"               & x"0"               & x"0"               & x"0"               &
                    x"0"               & x"0"               & x"0"               & x"0"               & x"0"               & x"0"               & x"0"               & x"0"        &
                    x"0"               & x"0"               & x"0"               & x"0"               & x"0"               & x"0"               & x"0"               & x"0"        &
                    x"0"               & x"0"               & x"0"               & x"0"               & x"0"               & x"0"               & x"0"               & x"0"        &
                    datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & 
                    datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) &
                    datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) &
                    datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) &
                    datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) & datak_counter(3 downto 0) ;
   
    e_midas_event_builder : entity work.midas_event_builder
    generic map (
        NLINKS => 64,
        USE_ALIGNMENT => 1,
        LINK_FIFO_ADDR_WIDTH => 10--,
    )
    port map (
        i_clk_data          => clk_156,
        i_clk_dma           => pcie_fastclk_out,
        i_reset_data_n      => resets_n(RESET_BIT_EVENT_COUNTER),
        i_reset_dma_n       => resets_n_fast(RESET_BIT_EVENT_COUNTER),
        i_link_data         => data_bank,
        i_link_datak        => datak_bank,
        i_wen_reg           => writeregs(DMA_REGISTER_W)(DMA_BIT_ENABLE),
        i_link_mask_n       => x"0000000" & writeregs(DATA_LINK_MASK_REGISTER_2_W)(3 downto 0) & writeregs(DATA_LINK_MASK_REGISTER_W), -- if 1 the link is active
        i_get_n_words       => writeregs(GET_N_DMA_WORDS_REGISTER_W),
        i_dmamemhalffull    => dmamemhalffull,
        o_fifos_full        => open,--readregs(EVENT_BUILD_STATUS_REGISTER_R)(31 downto 31 - NLINKS_TOTL),
        o_done              => readregs(EVENT_BUILD_STATUS_REGISTER_R)(EVENT_BUILD_DONE),
        o_event_wren        => dma_wren_cnt,
        o_endofevent        => dma_end_event_cnt,
        o_event_data        => dma_event_data,
        o_state_out         => state_out_eventcounter,
        -- error cnt signals
        o_fifo_almost_full          => open,--link_fifo_almost_full,
        o_cnt_link_fifo_almost_full => readregs_slow(CNT_FIFO_ALMOST_FULL_R),
        o_cnt_tag_fifo_full         => readregs(CNT_TAG_FIFO_FULL_R),
        o_cnt_ram_full              => readregs(CNT_RAM_FULL_R),
        o_cnt_stream_fifo_full      => readregs(CNT_STREAM_FIFO_FULL_R),
        o_cnt_dma_halffull          => readregs(CNT_DMA_HALFFULL_R),
        o_cnt_dc_link_fifo_full     => readregs_slow(CNT_DC_LINK_FIFO_FULL_R),
        o_cnt_skip_link_data        => open, --readregs_slow(CNT_SKIP_EVENT_LINK_FIFO_R),
        o_cnt_skip_event_dma        => readregs(CNT_SKIP_EVENT_DMA_RAM_R),
        o_cnt_idle_not_header       => readregs(CNT_IDLE_NOT_HEADER_R)--,
    );

    dma_data <= dma_event_data;
    dma_data_wren <= dma_wren_cnt;
    dmamem_endofevent <= dma_end_event_cnt;

    -------- Slow Control --------

 

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

 

    -- sync read regs from slow (156.25 MHz) to fast (250 MHz) clock
    process(pcie_fastclk_out)
    begin
    if rising_edge(pcie_fastclk_out) then
        clk_sync <= clk_156;
        clk_last <= clk_sync;
        
        if(clk_sync = '1' and clk_last = '0') then
            readregs(PLL_REGISTER_R)                <= readregs_slow(PLL_REGISTER_R);
            readregs(VERSION_REGISTER_R)            <= readregs_slow(VERSION_REGISTER_R);
            readregs(RUN_NR_REGISTER_R)             <= readregs_slow(RUN_NR_REGISTER_R);
            readregs(RUN_NR_ACK_REGISTER_R)         <= readregs_slow(RUN_NR_ACK_REGISTER_R);
            readregs(RUN_STOP_ACK_REGISTER_R)       <= readregs_slow(RUN_STOP_ACK_REGISTER_R);
            readregs(CNT_FEB_MERGE_TIMEOUT_R)       <= readregs_slow(CNT_FEB_MERGE_TIMEOUT_R);
            readregs(CNT_FIFO_ALMOST_FULL_R)        <= readregs_slow(CNT_FIFO_ALMOST_FULL_R);
            readregs(CNT_DC_LINK_FIFO_FULL_R)       <= readregs_slow(CNT_DC_LINK_FIFO_FULL_R);
            readregs(CNT_SKIP_EVENT_LINK_FIFO_R)    <= readregs_slow(CNT_SKIP_EVENT_LINK_FIFO_R);
            readregs(SC_MAIN_STATUS_REGISTER_R)     <= readregs_slow(SC_MAIN_STATUS_REGISTER_R);
            readregs(MEM_WRITEADDR_HIGH_REGISTER_R) <= (others => '0');
            readregs(MEM_WRITEADDR_LOW_REGISTER_R)  <= (X"0000" & readmem_writeaddr_finished);
        end if;
        
        readregs(DMA_STATUS_R)(DMA_DATA_WEN)        <= dma_data_wren;
        readregs(DMA_HALFFUL_REGISTER_R)            <= dmamemhalffull_counter;
        readregs(DMA_NOTHALFFUL_REGISTER_R)         <= dmamemnothalffull_counter;
        readregs(DMA_ENDEVENT_REGISTER_R)           <= endofevent_counter;
        readregs(DMA_NOTENDEVENT_REGISTER_R)        <= notendofevent_counter;
    end if;
    end process;



    readmem_writeaddr_lowbits   <= readmem_writeaddr(15 downto 0);
    pb_in                       <= push_button0_db & push_button1_db & push_button2_db;

end architecture;

