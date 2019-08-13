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



    --
    i_fifo_rempty   : in    std_logic;
    o_fifo_rack     : out   std_logic;
    i_fifo_rdata    : in    std_logic_vector(35 downto 0);



    -- QSFP links
    i_qsfp_rx       : in    std_logic_vector(3 downto 0);
    o_qsfp_tx       : out   std_logic_vector(3 downto 0);
    i_qsfp_refclk   : in    std_logic;

    -- POD links (reset system)
    i_pod_rx        : in    std_logic_vector(3 downto 0);
    o_pod_tx        : out   std_logic_vector(3 downto 0);
    i_pod_refclk    : in    std_logic;



    -- slow control registers
    o_sc_reg_addr   : out   std_logic_vector(7 downto 0);
    o_sc_reg_re     : out   std_logic;
    i_sc_reg_rdata  : in    std_logic_vector(31 downto 0);
    o_sc_reg_we     : out   std_logic;
    o_sc_reg_wdata  : out   std_logic_vector(31 downto 0);



    i_reset_n       : in    std_logic;
    -- 156.25 MHz
    i_clk           : in    std_logic--;
);
end entity;

architecture arch of fe_block is

    signal nios_pio : std_logic_vector(31 downto 0);



    signal av_sc : work.util.avalon_t;

    signal sc_fifo_rempty : std_logic;
    signal sc_fifo_rack : std_logic;
    signal sc_fifo_rdata : std_logic_vector(35 downto 0);

    signal sc_ram, sc_reg : work.util.rw_t;
    signal fe_reg : work.util.rw_t;

    signal reg_cmdlen : std_logic_vector(31 downto 0);
    signal reg_offset : std_logic_vector(31 downto 0);



    signal mscb_to_nios_parallel_in : std_logic_vector(11 downto 0);
    signal mscb_from_nios_parallel_out : std_logic_vector(11 downto 0);
    signal mscb_counter_in : unsigned(15 downto 0);

    signal reg_reset_bypass : std_logic_vector(31 downto 0);

    signal run_state_125 : run_state_t;
    signal run_state_156 : run_state_t;
    signal terminated : std_logic;



    signal av_qsfp, av_pod : work.util.avalon_t;

    signal qsfp_rx_data : std_logic_vector(127 downto 0);
    signal qsfp_rx_datak : std_logic_vector(15 downto 0);
    signal pod_rx_data : std_logic_vector(31 downto 0);
    signal pod_rx_datak : std_logic_vector(3 downto 0);

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

begin

    -- local regs : 0xF0-0xFF
    fe_reg.addr <= sc_reg.addr;
    fe_reg.re <= sc_reg.re when ( sc_reg.addr(7 downto 4) = X"F" ) else '0';
    fe_reg.we <= sc_reg.we when ( sc_reg.addr(7 downto 4) = X"F" ) else '0';
    fe_reg.wdata <= sc_reg.wdata;

    -- external regs : 0x00-0xEF
    o_sc_reg_addr <= sc_reg.addr(7 downto 0);
    o_sc_reg_re <= sc_reg.re when ( sc_reg.addr(7 downto 4) /= X"F" ) else '0';
    o_sc_reg_we <= sc_reg.we when ( sc_reg.addr(7 downto 4) /= X"F" ) else '0';
    o_sc_reg_wdata <= sc_reg.wdata;

    -- use fe_reg.rdata if prev cycle was fe_reg read
    sc_reg.rdata <=
        fe_reg.rdata when ( fe_reg.rvalid = '1' ) else
        i_sc_reg_rdata;

    process(i_clk)
    begin
    if rising_edge(i_clk) then
        fe_reg.rvalid <= fe_reg.re;

        -- cmdlen
        if ( fe_reg.addr(3 downto 0) = X"0" and fe_reg.re = '1' ) then
            fe_reg.rdata <= reg_cmdlen;
        end if;
        if ( fe_reg.addr(3 downto 0) = X"0" and fe_reg.we = '1' ) then
            reg_cmdlen <= fe_reg.wdata;
        end if;

        -- offset
        if ( fe_reg.addr(3 downto 0) = X"1" and fe_reg.re = '1' ) then
            fe_reg.rdata <= reg_offset;
        end if;
        if ( fe_reg.addr(3 downto 0) = X"1" and fe_reg.we = '1' ) then
            reg_offset <= fe_reg.wdata;
        end if;

        -- reset bypass
        if ( fe_reg.addr(3 downto 0) = X"4" and fe_reg.re = '1' ) then
            fe_reg.rdata <= reg_reset_bypass;
        end if;
        if ( fe_reg.addr(3 downto 0) = X"4" and fe_reg.we = '1' ) then
            reg_reset_bypass <= fe_reg.wdata;
        end if;

        --
    end if;
    end process;



    e_nios : component work.cmp.nios
    port map (
        avm_sc_address      => av_sc.address(15 downto 0),
        avm_sc_read         => av_sc.read,
        avm_sc_readdata     => av_sc.readdata,
        avm_sc_write        => av_sc.write,
        avm_sc_writedata    => av_sc.writedata,
        avm_sc_waitrequest  => av_sc.waitrequest,

        avm_clk_clk         => i_clk,
        avm_reset_reset_n   => i_reset_n,



        -- mscb
        parallel_mscb_in_export => mscb_to_nios_parallel_in,
        parallel_mscb_out_export => mscb_from_nios_parallel_out,
        counter_in_export => std_logic_vector(mscb_counter_in),



        avm_qsfp_address        => av_qsfp.address(13 downto 0),
        avm_qsfp_read           => av_qsfp.read,
        avm_qsfp_readdata       => av_qsfp.readdata,
        avm_qsfp_write          => av_qsfp.write,
        avm_qsfp_writedata      => av_qsfp.writedata,
        avm_qsfp_waitrequest    => av_qsfp.waitrequest,

        avm_pod_address         => av_pod.address(13 downto 0),
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

        rst_reset_n => i_nios_reset_n,
        clk_clk => i_nios_clk--,
    );



    e_sc_ram : entity work.sc_ram
    port map (
        i_ram_addr          => sc_ram.addr(15 downto 0),
        i_ram_re            => sc_ram.re,
        o_ram_rvalid        => sc_ram.rvalid,
        o_ram_rdata         => sc_ram.rdata,
        i_ram_we            => sc_ram.we,
        i_ram_wdata         => sc_ram.wdata,

        i_avs_address       => av_sc.address(15 downto 0),
        i_avs_read          => av_sc.read,
        o_avs_readdata      => av_sc.readdata,
        i_avs_write         => av_sc.write,
        i_avs_writedata     => av_sc.writedata,
        o_avs_waitrequest   => av_sc.waitrequest,

        o_reg_addr          => sc_reg.addr(7 downto 0),
        o_reg_re            => sc_reg.re,
        i_reg_rdata         => sc_reg.rdata,
        o_reg_we            => sc_reg.we,
        o_reg_wdata         => sc_reg.wdata,

        i_reset_n           => i_reset_n,
        i_clk               => i_clk--;
    );

    e_sc_rx : entity work.sc_rx
    port map (
        i_link_data     => qsfp_rx_data(31 downto 0),
        i_link_datak    => qsfp_rx_datak(3 downto 0),

        o_fifo_rempty   => sc_fifo_rempty,
        i_fifo_rack     => sc_fifo_rack,
        o_fifo_rdata    => sc_fifo_rdata,

        o_ram_addr      => sc_ram.addr,
        o_ram_re        => sc_ram.re,
        i_ram_rvalid    => sc_ram.rvalid,
        i_ram_rdata     => sc_ram.rdata,
        o_ram_we        => sc_ram.we,
        o_ram_wdata     => sc_ram.wdata,

        i_reset_n       => i_reset_n,
        i_clk           => i_clk--,
    );



    e_merger : entity work.data_merger
    port map (
        fpga_ID_in              => (5=>'1',others => '0'),
        FEB_type_in             => "111010",
        run_state               => run_state_156,

        data_out                => qsfp_tx_data(31 downto 0),
        data_is_k               => qsfp_tx_datak(3 downto 0),

        slowcontrol_fifo_empty  => sc_fifo_rempty,
        slowcontrol_read_req    => sc_fifo_rack,
        data_in_slowcontrol     => sc_fifo_rdata,

        data_fifo_empty         => i_fifo_rempty,
        data_read_req           => o_fifo_rack,
        data_in                 => i_fifo_rdata,

        override_data_in        => (others => '0'),
        override_data_is_k_in   => (others => '0'),
        override_req            => '0',
        override_granted        => open,

        terminated              => terminated,
        data_priority           => '0',

        leds                    => open,

        reset                   => not i_reset_n,
        clk                     => i_clk--,
    );



    e_reset_system : entity work.resetsys
    port map (
        clk_reset_rx_125=> i_pod_refclk,
        clk_global_125  => i_nios_clk,
        clk_156         => i_clk,
        clk_free        => i_clk,
        state_out_156   => run_state_156,
        state_out_125   => run_state_125,
        reset_in        => not i_nios_reset_n,
        resets_out      => open,
        phase_out       => open,
        data_in         => pod_rx_data(7 downto 0),
        reset_bypass    => reg_reset_bypass(11 downto 0),
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

        i_avs_address       => av_qsfp.address(13 downto 0),
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

        i_avs_address       => av_pod.address(13 downto 0),
        i_avs_read          => av_pod.read,
        o_avs_readdata      => av_pod.readdata,
        i_avs_write         => av_pod.write,
        i_avs_writedata     => av_pod.writedata,
        o_avs_waitrequest   => av_pod.waitrequest,

        i_reset => not i_nios_reset_n,
        i_clk   => i_nios_clk--,
    );

end architecture;
