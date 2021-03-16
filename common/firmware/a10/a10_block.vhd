library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.a10_pcie_registers.all;


entity a10_block is
generic (
    g_XCVR0_CHANNELS    : integer := 16;
    g_XCVR0_N           : integer := 4;
    g_XCVR1_CHANNELS    : integer := 0;
    g_XCVR1_N           : integer := 0;
    g_PCIE0_X           : integer := 8;
    g_PCIE1_X           : integer := 0;
    g_CLK_MHZ           : real := 50.0--;
);
port (
    -- flash interface
    o_flash_address     : out   std_logic_vector(31 downto 0);
    io_flash_data       : inout std_logic_vector(31 downto 0);
    o_flash_read_n      : out   std_logic;
    o_flash_write_n     : out   std_logic;
    o_flash_cs_n        : out   std_logic;
    o_flash_reset_n     : out   std_logic;

    -- I2C
    io_i2c_scl          : inout std_logic_vector(31 downto 0);
    io_i2c_sda          : inout std_logic_vector(31 downto 0);

    -- SPI
    i_spi_miso          : in    std_logic_vector(31 downto 0) := (others => '0');
    o_spi_mosi          : out   std_logic_vector(31 downto 0);
    o_spi_sclk          : out   std_logic_vector(31 downto 0);
    o_spi_ss_n          : out   std_logic_vector(31 downto 0);

    -- LED / BUTTONS
    o_LED               : out   std_logic_vector(3 downto 0);
    o_LED_BRACKET       : out   std_logic_vector(3 downto 0);
    i_BUTTON            : in    std_logic_vector(3 downto 0);
    o_HEX0_D            : out   std_logic_vector(6 downto 0);
    o_HEX1_D            : out   std_logic_vector(6 downto 0);

    -- XCVR0 (6250 Mbps @ 156.25 MHz)
    i_xcvr0_rx          : in    std_logic_vector(g_XCVR0_CHANNELS-1 downto 0) := (others => '0');
    o_xcvr0_tx          : out   std_logic_vector(g_XCVR0_CHANNELS-1 downto 0);
    i_xcvr0_refclk      : in    std_logic_vector(g_XCVR0_N-1 downto 0) := (others => '0');

    o_xcvr0_rx_data     : out   work.util.slv32_array_t(g_XCVR0_CHANNELS-1 downto 0);
    o_xcvr0_rx_datak    : out   work.util.slv4_array_t(g_XCVR0_CHANNELS-1 downto 0);
    i_xcvr0_tx_data     : in    work.util.slv32_array_t(g_XCVR0_CHANNELS-1 downto 0) := (others => (others => '0'));
    i_xcvr0_tx_datak    : in    work.util.slv4_array_t(g_XCVR0_CHANNELS-1 downto 0) := (others => (others => '0'));



    -- XCVR1 (10000 Mbps @ 250 MHz)
    i_xcvr1_rx          : in    std_logic_vector(g_XCVR1_CHANNELS-1 downto 0) := (others => '0');
    o_xcvr1_tx          : out   std_logic_vector(g_XCVR1_CHANNELS-1 downto 0);
    i_xcvr1_refclk      : in    std_logic_vector(g_XCVR1_N-1 downto 0) := (others => '0');

    o_xcvr1_rx_data     : out   work.util.slv32_array_t(g_XCVR1_CHANNELS-1 downto 0);
    o_xcvr1_rx_datak    : out   work.util.slv4_array_t(g_XCVR1_CHANNELS-1 downto 0);
    i_xcvr1_tx_data     : in    work.util.slv32_array_t(g_XCVR1_CHANNELS-1 downto 0) := (others => (others => '0'));
    i_xcvr1_tx_datak    : in    work.util.slv4_array_t(g_XCVR1_CHANNELS-1 downto 0) := (others => (others => '0'));



    -- PCIe0
    i_pcie0_rx          : in    std_logic_vector(g_PCIE0_X-1 downto 0) := (others => '0');
    o_pcie0_tx          : out   std_logic_vector(g_PCIE0_X-1 downto 0);
    i_pcie0_perst_n     : in    std_logic := '0';
    i_pcie0_refclk      : in    std_logic := '0'; -- ref 100 MHz clock
    o_pcie0_clk         : out   std_logic;

    -- PCIe0 DMA0
    i_pcie0_dma0_wdata  : in    std_logic_vector(255 downto 0) := (others => '0');
    i_pcie0_dma0_we     : in    std_logic := '0'; -- write enable
    i_pcie0_dma0_eoe    : in    std_logic := '0'; -- end of event
    o_pcie0_dma0_hfull  : out   std_logic; -- half full
    i_pcie0_dma0_clk    : in    std_logic := '0';

    -- PCIe0 read interface to writable memory
    i_pcie0_wmem_addr   : in    std_logic_vector(15 downto 0) := (others => '0');
    o_pcie0_wmem_rdata  : out   std_logic_vector(31 downto 0);
    i_pcie0_wmem_clk    : in    std_logic := '0';

    -- PCIe0 write interface to readable memory
    i_pcie0_rmem_addr   : in    std_logic_vector(15 downto 0) := (others => '0');
    i_pcie0_rmem_wdata  : in    std_logic_vector(31 downto 0) := (others => '0');
    i_pcie0_rmem_we     : in    std_logic := '0';
    i_pcie0_rmem_clk    : in    std_logic := '0';

    -- PCIe0 update interface for readable registers
    i_pcie0_rregs       : in    reg32array := (others => (others => '0'));

    -- PCIe0 read interface for writable registers
    o_pcie0_wregs_A     : out   reg32array;
    i_pcie0_wregs_A_clk : in    std_logic := '0';
    o_pcie0_wregs_B     : out   reg32array;
    i_pcie0_wregs_B_clk : in    std_logic := '0';
    o_pcie0_resets_n_156: out   std_logic_vector(31 downto 0);
    o_pcie0_resets_n_250: out   std_logic_vector(31 downto 0);


    -- PCIe1
    i_pcie1_rx          : in    std_logic_vector(g_PCIE1_X-1 downto 0) := (others => '0');
    o_pcie1_tx          : out   std_logic_vector(g_PCIE1_X-1 downto 0);
    i_pcie1_perst_n     : in    std_logic := '0';
    i_pcie1_refclk      : in    std_logic := '0'; -- ref 100 MHz clock
    o_pcie1_clk         : out   std_logic;

    -- PCIe1 DMA0
    i_pcie1_dma0_wdata  : in    std_logic_vector(255 downto 0) := (others => '0');
    i_pcie1_dma0_we     : in    std_logic := '0'; -- write enable
    i_pcie1_dma0_eoe    : in    std_logic := '0'; -- end of event
    o_pcie1_dma0_hfull  : out   std_logic; -- half full
    i_pcie1_dma0_clk    : in    std_logic := '0';

    -- PCIe1 read interface to writable memory
    i_pcie1_wmem_addr   : in    std_logic_vector(15 downto 0) := (others => '0');
    o_pcie1_wmem_rdata  : out   std_logic_vector(31 downto 0);
    i_pcie1_wmem_clk    : in    std_logic := '0';

    -- PCIe1 write interface to readable memory
    i_pcie1_rmem_addr   : in    std_logic_vector(15 downto 0) := (others => '0');
    i_pcie1_rmem_wdata  : in    std_logic_vector(31 downto 0) := (others => '0');
    i_pcie1_rmem_we     : in    std_logic := '0';
    i_pcie1_rmem_clk    : in    std_logic := '0';

    -- PCIe1 update interface for readable registers
    i_pcie1_rregs_156   : in    reg32array := (others => (others => '0'));
    i_pcie1_rregs_250   : in    reg32array := (others => (others => '0'));

    -- PCIe1 read interface for writable registers
    o_pcie1_wregs_A     : out   reg32array;
    i_pcie1_wregs_A_clk : in    std_logic := '0';
    o_pcie1_wregs_B     : out   reg32array;
    i_pcie1_wregs_B_clk : in    std_logic := '0';
    o_pcie1_resets_n_156: out   std_logic_vector(31 downto 0);
    o_pcie1_resets_n_250: out   std_logic_vector(31 downto 0);


    o_reset_156_n       : out   std_logic;
    o_clk_156           : out   std_logic;
    o_clk_156_hz        : out   std_logic;

    o_reset_250_n       : out   std_logic;
    o_clk_250           : out   std_logic;
    o_clk_250_hz        : out   std_logic;

    -- global 125 MHz clock
    i_reset_125_n       : in    std_logic;
    i_clk_125           : in    std_logic;

    -- local clock
    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of a10_block is

    signal flash_reset_n    : std_logic;
    signal nios_reset_n     : std_logic;

    signal reset_156_n      : std_logic;
    signal clk_156          : std_logic;
    signal reset_250_n      : std_logic;
    signal clk_250          : std_logic;

    signal flash_address    : std_logic_vector(31 downto 0) := (others => '0');

    signal nios_i2c_scl     : std_logic;
    signal nios_i2c_scl_oe  : std_logic;
    signal nios_i2c_sda     : std_logic;
    signal nios_i2c_sda_oe  : std_logic;
    signal nios_i2c_mask    : std_logic_vector(31 downto 0);

    signal nios_spi_miso    : std_logic;
    signal nios_spi_mosi    : std_logic;
    signal nios_spi_sclk    : std_logic;
    signal nios_spi_ss_n    : std_logic_vector(31 downto 0);

    signal nios_pio         : std_logic_vector(31 downto 0);

    signal av_xcvr0         : work.util.avalon_t;
    signal av_xcvr1         : work.util.avalon_t;

    signal pcie0_clk        : std_logic;
    signal pcie1_clk        : std_logic;

    signal pcie0_rregs      : reg32array;
    signal pcie1_rregs      : reg32array;

    --! debouncer signals
    signal push_button0_db : std_logic;
    signal push_button1_db : std_logic;
    signal push_button2_db : std_logic;
    signal push_button3_db : std_logic;

    --! counter
    signal clk_50_cnt   : unsigned(31 downto 0);
    signal clk_125_cnt  : unsigned(31 downto 0);

begin

    o_flash_reset_n <= flash_reset_n;

    o_flash_address <= flash_address;

    -- 156.25 MHz data clock (reference is 125 MHz global clock)
    e_clk_156 : component work.cmp.ip_pll_125to156
    port map (
        outclk_0 => clk_156,
        refclk => i_clk_125,
        rst => not i_reset_125_n--,
    );
    o_clk_156 <= clk_156;

    e_reset_156_n : entity work.reset_sync
    port map ( o_reset_n => reset_156_n, i_reset_n => i_reset_125_n, i_clk => clk_156 );
    o_reset_156_n <= reset_156_n;

    e_clk_156_hz : entity work.clkdiv
    generic map ( P => 156250000 )
    port map ( o_clk => o_clk_156_hz, i_reset_n => reset_156_n, i_clk => clk_156 );

    -- 250 MHz data clock (reference is 125 MHz global clock)
    e_clk_250 : component work.cmp.ip_pll_125to250
    port map (
        outclk_0 => clk_250,
        refclk => i_clk_125,
        rst => not i_reset_125_n--,
    );
    o_clk_250 <= clk_250;

    e_reset_250_n : entity work.reset_sync
    port map ( o_reset_n => reset_250_n, i_reset_n => i_reset_125_n, i_clk => clk_250 );
    o_reset_250_n <= reset_250_n;

    e_clk_250_hz : entity work.clkdiv
    generic map ( P => 250000000 )
    port map ( o_clk => o_clk_250_hz, i_reset_n => reset_250_n, i_clk => clk_250 );

    o_pcie0_clk <= pcie0_clk;

    --! save git version to version register
    e_version_reg : entity work.version_reg
    port map (
        data_out  => pcie0_rregs(VERSION_REGISTER_R)(27 downto 0)--,
    );

    --! generate reset regs for 156 MHz clk for pcie0
    e_reset_logic : entity work.reset_logic
    port map (
        rst_n          => reset_156_n,
        reset_register => o_pcie0_wregs_B(RESET_REGISTER_W),
        resets         => open,
        resets_n       => o_pcie0_resets_n_156,
        clk            => clk_156--,
    );

    --! generate reset regs for 250 MHz clk for pcie0
    e_reset_logic_fast : entity work.reset_logic
    port map (
        rst_n          => reset_250_n,
        reset_register => o_pcie0_wregs_A(RESET_REGISTER_W),
        resets         => open,
        resets_n       => o_pcie0_resets_n_250,
        clk            => o_pcie0_clk--,
    );

    --! generate reset regs for 156 MHz clk for pcie1
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------

    --! generate reset regs for 250 MHz clk for pcie1
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------


    --! 100 MHz blinky to check the pcie0 refclk
    e_pcie_clk_hz : entity work.clkdiv
    generic map ( P => 100000000 )
    port map ( o_clk => o_LED(3), i_reset_n => i_reset_n, i_clk => pcie0_clk );

    --! 100 MHz blinky to check the pcie1 refclk
    e_pcie_clk_hz : entity work.clkdiv
    generic map ( P => 100000000 )
    port map ( o_clk => o_LED(2), i_reset_n => i_reset_n, i_clk => pcie1_clk );

    --! blinky leds to check the wregs
    o_LED_BRACKET(1 downto 0) <= o_pcie0_wregs_A(LED_REGISTER_W)(1 downto 0);
    o_LED_BRACKET(3 downto 2) <= o_pcie0_wregs_B(LED_REGISTER_W)(3 downto 2);

    --! blinky leds
    o_LED(1) <= nios_pio(7);

    --! debouncer for push_button
    e_debouncer : entity work.debouncer
    generic map (
        W => 4,
        N => 125 * 10**3 -- 1ms
    )
    port map (
        i_d         => i_BUTTON,
        o_q(0)      => push_button0_db,
        o_q(1)      => push_button1_db,
        o_q(2)      => push_button2_db,
        o_q(3)      => push_button3_db,
        i_reset_n   => i_reset_n,
        i_clk       => i_clk--,
    );

    --! counters for 50 MHz clk
    process(i_clk)
    begin
    if rising_edge(i_clk) then
        clk_50_cnt <= clk_50_cnt + 1;
    end if;
    end process;

    --! counters for 125 MHz clk
    process(i_clk_125)
    begin
    if rising_edge(i_clk_125) then
        clk_125_cnt <= clk_125_cnt + 1;
    end if;
    end process;

    --! monitor 50 MHz clock with seg7 
    --! this works not on the LHCb Board
    e_hex2seg7_50 : entity work.hex2seg7
    port map (
        i_hex => std_logic_vector(clk_50_cnt)(27 downto 24),
        o_seg => o_HEX0_D--,
    );

    --! monitor 125 MHz clock with seg7 
    --! this works not on the LHCb Board
    --! so it will be just be not used there
    e_hex2seg7_125 : entity work.hex2seg7
    port map (
        i_hex => std_logic_vector(clk_125_cnt)(27 downto 24),
        o_seg => o_HEX1_D--,
    );


    -- nios reset sequence
    e_nios_reset_n : entity work.debouncer
    generic map (
        W => 2,
        N => integer(g_CLK_MHZ * 1000000.0 * 0.200) -- 200ms
    )
    port map (
        i_d(0) => '1',
        o_q(0) => flash_reset_n,

        i_d(1) => flash_reset_n,
        o_q(1) => nios_reset_n,

        i_reset_n => i_reset_n,
        i_clk => i_clk--,
    );

    -- nios
    e_nios : work.cmp.nios
    port map (
        flash_tcm_address_out           => flash_address(27 downto 0),
        flash_tcm_data_out              => io_flash_data,
        flash_tcm_read_n_out(0)         => o_flash_read_n,
        flash_tcm_write_n_out(0)        => o_flash_write_n,
        flash_tcm_chipselect_n_out(0)   => o_flash_cs_n,

        i2c_scl_in                      => nios_i2c_scl,
        i2c_scl_oe                      => nios_i2c_scl_oe,
        i2c_sda_in                      => nios_i2c_sda,
        i2c_sda_oe                      => nios_i2c_sda_oe,
        i2c_mask_export                 => nios_i2c_mask,

        spi_MISO                        => nios_spi_miso,
        spi_MOSI                        => nios_spi_mosi,
        spi_SCLK                        => nios_spi_sclk,
        spi_SS_n                        => nios_spi_ss_n,

        pio_export                      => nios_pio,

        avm_xcvr0_address               => av_xcvr0.address(17 downto 0),
        avm_xcvr0_read                  => av_xcvr0.read,
        avm_xcvr0_readdata              => av_xcvr0.readdata,
        avm_xcvr0_write                 => av_xcvr0.write,
        avm_xcvr0_writedata             => av_xcvr0.writedata,
        avm_xcvr0_waitrequest           => av_xcvr0.waitrequest,

        avm_xcvr0_reset_reset_n         => i_reset_125_n,
        avm_xcvr0_clock_clk             => i_clk_125,

        avm_xcvr1_address               => av_xcvr1.address(17 downto 0),
        avm_xcvr1_read                  => av_xcvr1.read,
        avm_xcvr1_readdata              => av_xcvr1.readdata,
        avm_xcvr1_write                 => av_xcvr1.write,
        avm_xcvr1_writedata             => av_xcvr1.writedata,
        avm_xcvr1_waitrequest           => av_xcvr1.waitrequest,

        avm_xcvr1_reset_reset_n         => i_reset_125_n,
        avm_xcvr1_clock_clk             => i_clk_125,

        rst_reset_n                     => nios_reset_n,
        clk_clk                         => i_clk--,
    );



    -- i2c mux
    e_i2c_mux : entity work.i2c_mux
    port map (
        io_scl      => io_i2c_scl,
        io_sda      => io_i2c_sda,

        o_scl       => nios_i2c_scl,
        i_scl_oe    => nios_i2c_scl_oe,
        o_sda       => nios_i2c_sda,
        i_sda_oe    => nios_i2c_sda_oe,

        i_mask      => nios_i2c_mask--,
    );



    -- spi mux
    o_spi_mosi <= (others => nios_spi_mosi);
    o_spi_sclk <= (others => nios_spi_sclk);
    o_spi_ss_n <= nios_spi_ss_n;
    -- TODO: implement mux



    -- xcvr_block 6250 Mbps @ 156.25 MHz
    generate_xcvr0_block : if ( g_XCVR0_CHANNELS > 0 ) generate
    e_xcvr0_block : entity work.xcvr_block
    generic map (
        g_XCVR_NAME => "xcvr_a10",
        g_XCVR_N => g_XCVR0_N,
        g_CHANNELS => g_XCVR0_CHANNELS / g_XCVR0_N,
        g_REFCLK_MHZ => 125.0,
        g_CLK_MHZ => 125.0--,
    )
    port map (
        o_rx_data           => o_xcvr0_rx_data,
        o_rx_datak          => o_xcvr0_rx_datak,

        i_tx_data           => i_xcvr0_tx_data,
        i_tx_datak          => i_xcvr0_tx_datak,

        i_tx_clk            => (others => clk_156),
        i_rx_clk            => (others => clk_156),

        i_rx_serial         => i_xcvr0_rx,
        o_tx_serial         => o_xcvr0_tx,

        i_refclk            => i_xcvr0_refclk,

        i_avs_address       => av_xcvr0.address(17 downto 0),
        i_avs_read          => av_xcvr0.read,
        o_avs_readdata      => av_xcvr0.readdata,
        i_avs_write         => av_xcvr0.write,
        i_avs_writedata     => av_xcvr0.writedata,
        o_avs_waitrequest   => av_xcvr0.waitrequest,

        i_reset_n           => i_reset_125_n,
        i_clk               => i_clk_125--, -- TODO: use clk_156
    );
    end generate;



    -- xcvr_block 10000 Mbps @ 250 MHz
    generate_xcvr1_block : if ( g_XCVR1_CHANNELS > 0 ) generate
    e_xcvr1_block : entity work.xcvr_block
    generic map (
        g_XCVR_NAME => "xcvr_enh",
        g_XCVR_N => g_XCVR1_N,
        g_CHANNELS => g_XCVR1_CHANNELS / g_XCVR1_N,
        g_REFCLK_MHZ => 125.0,
        g_CLK_MHZ => 125.0--,
    )
    port map (
        o_rx_data           => o_xcvr1_rx_data,
        o_rx_datak          => o_xcvr1_rx_datak,

        i_tx_data           => i_xcvr1_tx_data,
        i_tx_datak          => i_xcvr1_tx_datak,

        i_rx_clk            => (others => clk_250),
        i_tx_clk            => (others => clk_250),

        i_rx_serial         => i_xcvr1_rx,
        o_tx_serial         => o_xcvr1_tx,

        i_refclk            => i_xcvr1_refclk,

        i_avs_address       => av_xcvr1.address(17 downto 0),
        i_avs_read          => av_xcvr1.read,
        o_avs_readdata      => av_xcvr1.readdata,
        i_avs_write         => av_xcvr1.write,
        i_avs_writedata     => av_xcvr1.writedata,
        o_avs_waitrequest   => av_xcvr1.waitrequest,

        i_reset_n           => i_reset_125_n,
        i_clk               => i_clk_125--,
    );
    end generate;

    --! PCIe register mapping
    e_register_mapping : entity work.pcie_register_mapping
    port map(
        i_pcie0_rregs_156   => i_pcie0_rregs_156,
        i_pcie0_rregs_250   => i_pcie0_rregs_250,

        i_pcie1_rregs_156   => i_pcie1_rregs_156,
        i_pcie1_rregs_250   => i_pcie1_rregs_250,

        o_pcie0_rregs       => pcie0_rregs,
        o_pcie1_rregs       => pcie1_rregs,
    
        i_clk_156           => clk_156,

        i_pcie0_clk         => pcie0_clk,
        i_pcie1_clk         => pcie1_clk--,
    );

    -- PCIe0
    generate_pcie0 : if ( g_PCIE0_X > 0 ) generate
    e_pcie0_block : entity work.pcie_block
    generic map (
        DMAMEMWRITEADDRSIZE     => 11,
        DMAMEMREADADDRSIZE      => 11,
        DMAMEMWRITEWIDTH        => 256,
        g_PCIE_X => g_PCIE0_X--,
    )
    port map (
        local_rstn              => '1',
        appl_rstn               => '1',
        refclk                  => i_pcie0_refclk,
        pcie_fastclk_out        => pcie0_clk,

        pcie_rx_p               => i_pcie0_rx,
        pcie_tx_p               => o_pcie0_tx,
        pcie_refclk_p           => i_pcie0_refclk,
        pcie_perstn             => i_pcie0_perst_n,
        pcie_smbclk             => '0',
        pcie_smbdat             => '0',
        pcie_waken              => open,

        readregs                => pcie0_rregs,
        writeregs               => o_pcie0_wregs_A,

        i_clk_B                 => i_pcie0_wregs_B_clk,
        o_writeregs_B           => o_pcie0_wregs_B,

        writememreadaddr        => i_pcie0_wmem_addr,
        writememreaddata        => o_pcie0_wmem_rdata,
        writememclk             => i_pcie0_wmem_clk,

        readmem_addr            => i_pcie0_rmem_addr,
        readmem_data            => i_pcie0_rmem_wdata,
        readmem_wren            => i_pcie0_rmem_we,
        readmemclk              => i_pcie0_rmem_clk,
        readmem_endofevent      => '0',

        dma_data                => i_pcie0_dma0_wdata,
        dmamem_wren             => i_pcie0_dma0_we,
        dmamem_endofevent       => i_pcie0_dma0_eoe,
        dmamemhalffull          => o_pcie0_dma0_hfull,
        dmamemclk               => i_pcie0_dma0_clk,

        dma2memclk              => i_pcie0_dma0_clk--,
    );
    end generate;

    --! DMA evaluationg / monitoring for PCIe 0
    e_dma_evaluation_pcie0 : entity work.dma_evaluation
    port map (
        reset_n                 => o_pcie0_resets_n_250(RESET_BIT_DMA_EVAL),
        dmamemhalffull          => o_pcie0_dma0_hfull,
        dmamem_endofevent       => i_pcie0_dma0_eoe,
        halffull_counter        => pcie0_rregs(DMA_HALFFUL_REGISTER_R),
        nothalffull_counter     => pcie0_rregs(DMA_NOTHALFFUL_REGISTER_R),
        endofevent_counter      => pcie0_rregs(DMA_ENDEVENT_REGISTER_R),
        notendofevent_counter   => pcie0_rregs(DMA_NOTENDEVENT_REGISTER_R),
        clk                     => pcie0_clk--,
    );
    pcie0_rregs(DMA_STATUS_R)(DMA_DATA_WEN) <= i_pcie0_dma0_we;

    -- PCIe1
    generate_pcie1 : if ( g_PCIE1_X > 0 ) generate
    e_pcie1_block : entity work.pcie_block
    generic map (
        DMAMEMWRITEADDRSIZE     => 11,
        DMAMEMREADADDRSIZE      => 11,
        DMAMEMWRITEWIDTH        => 256,
        g_PCIE_X => g_PCIE1_X--,
    )
    port map (
        local_rstn              => '1',
        appl_rstn               => '1',
        refclk                  => i_pcie1_refclk,
        pcie_fastclk_out        => pcie1_clk,

        pcie_rx_p               => i_pcie1_rx,
        pcie_tx_p               => o_pcie1_tx,
        pcie_refclk_p           => i_pcie1_refclk,
        pcie_perstn             => i_pcie1_perst_n,
        pcie_smbclk             => '0',
        pcie_smbdat             => '0',
        pcie_waken              => open,

        readregs                => pcie1_rregs,
        writeregs               => o_pcie1_wregs_A,

        i_clk_B                 => i_pcie1_wregs_B_clk,
        o_writeregs_B           => o_pcie1_wregs_B,

        writememreadaddr        => i_pcie1_wmem_addr,
        writememreaddata        => o_pcie1_wmem_rdata,
        writememclk             => i_pcie1_wmem_clk,

        readmem_addr            => i_pcie1_rmem_addr,
        readmem_data            => i_pcie1_rmem_wdata,
        readmem_wren            => i_pcie1_rmem_we,
        readmemclk              => i_pcie1_rmem_clk,
        readmem_endofevent      => '0',

        dma_data                => i_pcie1_dma0_wdata,
        dmamem_wren             => i_pcie1_dma0_we,
        dmamem_endofevent       => i_pcie1_dma0_eoe,
        dmamemhalffull          => o_pcie1_dma0_hfull,
        dmamemclk               => i_pcie1_dma0_clk,

        dma2memclk              => i_pcie1_dma0_clk--,
    );
    end generate;

    --! DMA evaluationg / monitoring for PCIe 1
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------
    --! ------------------------------------------------------------------------

end architecture;
