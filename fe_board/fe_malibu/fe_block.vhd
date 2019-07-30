library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

entity fe_block is
generic (
    FPGA_ID_g : std_logic_vector(15 downto 0) := X"0000"--;
);
port (
    -- 125 MHz
    i_nios_clk      : in    std_logic;
    i_nios_reset_n  : in    std_logic;

    i_i2c_scl       : in    std_logic;
    o_i2c_scl_oe    : out   std_logic;
    i_i2c_sda       : in    std_logic;
    o_i2c_sda_oe    : out   std_logic;

    i_spi_miso      : in    std_logic;
    o_spi_mosi      : out   std_logic;
    o_spi_sclk      : out   std_logic;
    o_spi_ss_n      : out   std_logic_vector(15 downto 0);



    -- MSCB interface
    i_mscb_data     : in    std_logic;
    o_mscb_data     : out   std_logic;
    o_mscb_oe       : out   std_logic;



    -- QSFP links
    i_qsfp_rx       : in    std_logic_vector(3 downto 0);
    o_qsfp_tx       : out   std_logic_vector(3 downto 0);
    i_qsfp_refclk   : in    std_logic;

    i_fifo_data     : in    std_logic_vector(35 downto 0);
    i_fifo_empty    : in    std_logic;
    o_fifo_rack     : out   std_logic;



    -- POD links (reset system)
    i_pod_rx        : in    std_logic_vector(3 downto 0);
    o_pod_tx        : out   std_logic_vector(3 downto 0);
    i_pod_refclk    : in    std_logic;



    -- avalon master
    -- address units - words
    -- read latency - 1
    o_avm_address       : out   std_logic_vector(13 downto 0);
    o_avm_read          : out   std_logic;
    i_avm_readdata      : in    std_logic_vector(31 downto 0);
    o_avm_write         : out   std_logic;
    o_avm_writedata     : out   std_logic_vector(31 downto 0);
    i_avm_waitrequest   : in    std_logic;

    -- SC RAM
    i_sc_ram_address    : in    std_logic_vector(15 downto 0);
    o_sc_ram_rdata      : out   std_logic_vector(31 downto 0);
    i_sc_ram_wdata      : in    std_logic_vector(31 downto 0);
    i_sc_ram_we         : in    std_logic;

    i_reset_n       : in    std_logic;
    -- 156.25 MHz
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of fe_block is

    signal nios_pio : std_logic_vector(31 downto 0);

    signal av_sc : work.util.avalon_t;



    signal av_qsfp, av_pod : work.util.avalon_t;

    signal qsfp_tx_data : std_logic_vector(127 downto 0) :=
          X"03CAFE" & work.util.D28_5
        & X"02BABE" & work.util.D28_5
        & X"01DEAD" & work.util.D28_5
        & X"00BEEF" & work.util.D28_5;
    signal qsfp_tx_datak : std_logic_vector(15 downto 0) :=
          "0001"
        & "0001"
        & "0001"
        & "0001";
    signal pod_tx_data : std_logic_vector(31 downto 0) :=
          work.util.D28_5
        & work.util.D28_5
        & work.util.D28_5
        & work.util.D28_5;
    signal pod_tx_datak : std_logic_vector(3 downto 0) :=
          "1"
        & "1"
        & "1"
        & "1";

    signal qsfp_rx_data : std_logic_vector(127 downto 0);
    signal qsfp_rx_datak : std_logic_vector(15 downto 0);
    signal pod_rx_data : std_logic_vector(31 downto 0);
    signal pod_rx_datak : std_logic_vector(3 downto 0);



    signal mscb_to_nios_parallel_in : std_logic_vector(11 downto 0);
    signal mscb_from_nios_parallel_out : std_logic_vector(11 downto 0);
    signal mscb_counter_in : unsigned(15 downto 0);

    signal reset_bypass : std_logic_vector(11 downto 0);

    signal run_state_125 : run_state_t;
    signal run_state_156 : run_state_t;
    signal terminated : std_logic;

begin

    e_nios : component work.cmp.nios
    port map (
        avm_address         => o_avm_address,
        avm_read            => o_avm_read,
        avm_readdata        => i_avm_readdata,
        avm_write           => o_avm_write,
        avm_writedata       => o_avm_writedata,
        avm_waitrequest     => i_avm_waitrequest,

        avm_sc_address      => av_sc.address(17 downto 0),
        avm_sc_read         => av_sc.read,
        avm_sc_readdata     => av_sc.readdata,
        avm_sc_write        => av_sc.write,
        avm_sc_writedata    => av_sc.writedata,
        avm_sc_waitrequest  => av_sc.waitrequest,

        avm_clk_clk         => i_clk,
        avm_reset_reset_n   => i_reset_n,



        avm_qsfp_address        => av_qsfp.address(15 downto 0),
        avm_qsfp_read           => av_qsfp.read,
        avm_qsfp_readdata       => av_qsfp.readdata,
        avm_qsfp_write          => av_qsfp.write,
        avm_qsfp_writedata      => av_qsfp.writedata,
        avm_qsfp_waitrequest    => av_qsfp.waitrequest,

        avm_pod_address         => av_pod.address(15 downto 0),
        avm_pod_read            => av_pod.read,
        avm_pod_readdata        => av_pod.readdata,
        avm_pod_write           => av_pod.write,
        avm_pod_writedata       => av_pod.writedata,
        avm_pod_waitrequest     => av_pod.waitrequest,

        --
        -- nios base
        --

        i2c_scl_in => i_i2c_scl,
        i2c_scl_oe => o_i2c_scl_oe,
        i2c_sda_in => i_i2c_sda,
        i2c_sda_oe => o_i2c_sda_oe,

        spi_miso => i_spi_miso,
        spi_mosi => o_spi_mosi,
        spi_sclk => o_spi_sclk,
        spi_ss_n => o_spi_ss_n,

        pio_export => nios_pio,

        -- mscb
        parallel_mscb_in_export => mscb_to_nios_parallel_in,
        parallel_mscb_out_export => mscb_from_nios_parallel_out,
        counter_in_export => std_logic_vector(mscb_counter_in),

        -- reset bypass
        reset_bypass_out_export => reset_bypass,

        rst_reset_n => i_nios_reset_n,
        clk_clk => i_nios_clk--,
    );



    e_data_sc_path : entity work.data_sc_path
    port map (
        i_avs_address       => av_sc.address(17 downto 2),
        i_avs_read          => av_sc.read,
        o_avs_readdata      => av_sc.readdata,
        i_avs_write         => av_sc.write,
        i_avs_writedata     => av_sc.writedata,
        o_avs_waitrequest   => av_sc.waitrequest,

        i_fifo_data         => i_fifo_data,
        i_fifo_data_empty   => i_fifo_empty,
        o_fifo_data_read    => o_fifo_rack,

        i_link_data         => qsfp_rx_data(31 downto 0),
        i_link_datak        => qsfp_rx_datak(3 downto 0),

        o_link_data         => qsfp_tx_data(31 downto 0),
        o_link_datak        => qsfp_tx_datak(3 downto 0),

        o_terminated        => terminated,
        i_run_state         => run_state_156,

        i_reset             => not i_reset_n,
        i_clk               => i_clk--,
    );



    e_reset_system : entity work.resetsys
    port map (
        clk_reset_rx_125=> i_pod_refclk,
        clk_global_125  => i_nios_clk,
        clk_156         => i_clk,
        clk_free        => i_clk,
        state_out_156   => run_state_156,
        state_out_125   => run_state_125,
        reset_in        => i_reset_n,
        resets_out      => open,
        phase_out       => open,
        data_in         => pod_rx_data(7 downto 0),
        reset_bypass    => reset_bypass,
        run_number_out  => open,
        fpga_id         => FPGA_ID_g,
        terminated      => terminated,
        testout         => open--,
    );



    e_mscb : entity work.mscb
    port map (
        mscb_to_nios_parallel_in    => mscb_to_nios_parallel_in,
        mscb_from_nios_parallel_out => mscb_from_nios_parallel_out,
        mscb_data_in                => i_mscb_data,
        mscb_data_out               => o_mscb_data,
        mscb_oe                     => o_mscb_oe,
        mscb_counter_in             => mscb_counter_in,

        reset                       => not i_nios_reset_n,
        nios_clk                    => i_nios_clk--,
    );



    e_qsfp : entity work.xcvr_s4
    generic map (
        NUMBER_OF_CHANNELS_g => 4,
        CHANNEL_WIDTH_g => 32,
        INPUT_CLOCK_FREQUENCY_g => 156250000,
        DATA_RATE_g => 6250,
        CLK_MHZ_g => 125--,
    )
    port map (
        i_tx_data   => qsfp_tx_data,
        i_tx_datak  => qsfp_tx_datak,

        o_rx_data   => qsfp_rx_data,
        o_rx_datak  => qsfp_rx_datak,

        o_tx_clkout => open,
        i_tx_clkin  => (others => i_qsfp_refclk),
        o_rx_clkout => open,
        i_rx_clkin  => (others => i_qsfp_refclk),

        o_tx_serial => o_qsfp_tx,
        i_rx_serial => i_qsfp_rx,

        i_pll_clk   => i_qsfp_refclk,
        i_cdr_clk   => i_qsfp_refclk,

        i_avs_address       => av_qsfp.address(15 downto 2),
        i_avs_read          => av_qsfp.read,
        o_avs_readdata      => av_qsfp.readdata,
        i_avs_write         => av_qsfp.write,
        i_avs_writedata     => av_qsfp.writedata,
        o_avs_waitrequest   => av_qsfp.waitrequest,

        i_reset => not i_nios_reset_n,
        i_clk   => i_nios_clk--,
    );



    e_pod : entity work.xcvr_s4
    generic map (
        NUMBER_OF_CHANNELS_g => 4,
        CHANNEL_WIDTH_g => 8,
        INPUT_CLOCK_FREQUENCY_g => 125000000,
        DATA_RATE_g => 1250,
        CLK_MHZ_g => 125--,
    )
    port map (
        i_tx_data   => pod_tx_data,
        i_tx_datak  => pod_tx_datak,

        o_rx_data   => pod_rx_data,
        o_rx_datak  => pod_rx_datak,

        o_tx_clkout => open,
        i_tx_clkin  => (others => i_pod_refclk),
        o_rx_clkout => open,
        i_rx_clkin  => (others => i_pod_refclk),

        o_tx_serial => o_pod_tx,
        i_rx_serial => i_pod_rx,

        i_pll_clk   => i_pod_refclk,
        i_cdr_clk   => i_pod_refclk,

        i_avs_address       => av_pod.address(15 downto 2),
        i_avs_read          => av_pod.read,
        o_avs_readdata      => av_pod.readdata,
        i_avs_write         => av_pod.write,
        i_avs_writedata     => av_pod.writedata,
        o_avs_waitrequest   => av_pod.waitrequest,

        i_reset => not i_nios_reset_n,
        i_clk   => i_nios_clk--,
    );

end architecture;
