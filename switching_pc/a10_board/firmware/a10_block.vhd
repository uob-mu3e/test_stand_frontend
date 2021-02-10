library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

use work.pcie_components.all;

entity a10_block is
generic (
    g_XCVR0_CHANNELS    : integer := 16;
    g_XCVR0_N           : integer := 4;
    g_XCVR1_CHANNELS    : integer := 0;
    g_XCVR1_N           : integer := 0;
    g_PCIE0_X           : integer := 8;
    g_PCIE1_X           : integer := 0--;
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

    o_nios_hz           : out   std_logic;



    -- XCVR0 (6250 Mbps @ 156.25 MHz)
    i_xcvr0_rx          : in    std_logic_vector(g_XCVR0_CHANNELS-1 downto 0) := (others => '0');
    o_xcvr0_tx          : out   std_logic_vector(g_XCVR0_CHANNELS-1 downto 0);

    o_xcvr0_rx_data     : out   work.util.slv32_array_t(g_XCVR0_CHANNELS-1 downto 0);
    o_xcvr0_rx_datak    : out   work.util.slv4_array_t(g_XCVR0_CHANNELS-1 downto 0);
    i_xcvr0_tx_data     : in    work.util.slv32_array_t(g_XCVR0_CHANNELS-1 downto 0) := (others => (others => '0'));
    i_xcvr0_tx_datak    : in    work.util.slv4_array_t(g_XCVR0_CHANNELS-1 downto 0) := (others => (others => '0'));



    -- XCVR1 (10000 Mbps @ 250 MHz)
    i_xcvr1_rx          : in    std_logic_vector(g_XCVR1_CHANNELS-1 downto 0) := (others => '0');
    o_xcvr1_tx          : out   std_logic_vector(g_XCVR1_CHANNELS-1 downto 0);

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



    -- PCIe1
    i_pcie1_refclk      : in    std_logic := '0'; -- ref 100 MHz clock



    i_reset_156_n       : in    std_logic;
    i_clk_156           : in    std_logic;

    i_clk_250           : in    std_logic;

    -- global 125 MHz clock
    i_reset_125_n       : in    std_logic;
    i_clk_125           : in    std_logic;

    -- local 50 MHz clock
    i_reset_50_n        : in    std_logic;
    i_clk_50            : in    std_logic--;
);
end entity;

architecture arch of a10_block is

    signal flash_reset_n    : std_logic;

    signal pcie0_clk        : std_logic;

    signal nios_reset_n     : std_logic;

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

begin

    o_flash_reset_n <= flash_reset_n;

    o_nios_hz <= nios_pio(7);

    o_pcie0_clk <= pcie0_clk;



    -- nios reset sequence
    e_nios_reset_n : entity work.debouncer
    generic map (
        W => 2,
        N => integer(50e6 * 0.200) -- 200ms
    )
    port map (
        i_d(0) => '1',
        o_q(0) => flash_reset_n,

        i_d(1) => flash_reset_n,
        o_q(1) => nios_reset_n,

        i_reset_n => i_reset_50_n,
        i_clk => i_clk_50--,
    );

    -- nios
    e_nios : work.cmp.nios
    port map (
        avm_reset_reset_n               => i_reset_125_n,
        avm_clock_clk                   => i_clk_125, -- TODO: use clk_156

        avm_xcvr_address                => av_xcvr0.address(15 downto 0),
        avm_xcvr_read                   => av_xcvr0.read,
        avm_xcvr_readdata               => av_xcvr0.readdata,
        avm_xcvr_write                  => av_xcvr0.write,
        avm_xcvr_writedata              => av_xcvr0.writedata,
        avm_xcvr_waitrequest            => av_xcvr0.waitrequest,

        flash_tcm_address_out           => o_flash_address(27 downto 0),
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

        rst_reset_n                     => i_reset_50_n,
        clk_clk                         => i_clk_50--,
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
        N_XCVR_g => g_XCVR0_N,
        N_CHANNELS_g => g_XCVR0_CHANNELS / g_XCVR0_N--,
    )
    port map (
        o_rx_data           => o_xcvr0_rx_data,
        o_rx_datak          => o_xcvr0_rx_datak,

        i_tx_data           => i_xcvr0_tx_data,
        i_tx_datak          => i_xcvr0_tx_datak,

        i_tx_clk            => (others => i_clk_156),
        i_rx_clk            => (others => i_clk_156),

        i_rx_serial         => i_xcvr0_rx,
        o_tx_serial         => o_xcvr0_tx,

        i_refclk            => (others => i_clk_125),

        i_avs_address       => av_xcvr0.address(15 downto 0),
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
        N_XCVR_g => g_XCVR1_N,
        N_CHANNELS_g => g_XCVR1_CHANNELS / g_XCVR1_N--,
    )
    port map (
        o_rx_data           => o_xcvr1_rx_data,
        o_rx_datak          => o_xcvr1_rx_datak,

        i_tx_data           => i_xcvr1_tx_data,
        i_tx_datak          => i_xcvr1_tx_datak,

        i_tx_clk            => (others => i_clk_156),
        i_rx_clk            => (others => i_clk_156),

        i_rx_serial         => i_xcvr1_rx,
        o_tx_serial         => o_xcvr1_tx,

        i_refclk            => (others => i_clk_125),

        i_avs_address       => av_xcvr1.address(15 downto 0),
        i_avs_read          => av_xcvr1.read,
        o_avs_readdata      => av_xcvr1.readdata,
        i_avs_write         => av_xcvr1.write,
        i_avs_writedata     => av_xcvr1.writedata,
        o_avs_waitrequest   => av_xcvr1.waitrequest,

        i_reset_n           => i_reset_125_n,
        i_clk               => i_clk_125--,
    );
    end generate;



    -- PCIe0
    generate_pcie0_x8 : if ( g_PCIE0_X = 8 ) generate
    e_pcie0_block : entity work.pcie_block
    generic map (
        DMAMEMWRITEADDRSIZE     => 11,
        DMAMEMREADADDRSIZE      => 11,
        DMAMEMWRITEWIDTH        => 256,
        PCIE_X_g => g_PCIE0_X--,
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

        readregs                => i_pcie0_rregs,
        writeregs               => o_pcie0_wregs_A,

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



    -- PCIe1
    generate_pcie1_x8 : if ( g_PCIE1_X = 8 ) generate
    end generate;

end architecture;
