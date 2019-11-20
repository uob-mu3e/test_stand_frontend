library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

entity fe_block is
generic (
    FPGA_ID_g : std_logic_vector(15 downto 0) := X"0000";
    -- frontend board type
    -- - 111010 : mupix
    -- - 111000 : mutrig
    -- - 000111 and 000000 : reserved (DO NOT USE)
    FEB_type_in : std_logic_vector(5 downto 0);
    NIOS_CLK_HZ_g : positive := 125000000--;
);
port (
    i_i2c_scl       : in    std_logic;
    o_i2c_scl_oe    : out   std_logic;
    i_i2c_sda       : in    std_logic;
    o_i2c_sda_oe    : out   std_logic;

    -- spi interface to si chip
    i_spi_si_miso   : in    std_logic;
    o_spi_si_mosi   : out   std_logic;
    o_spi_si_sclk   : out   std_logic;
    o_spi_si_ss_n   : out   std_logic;

    -- spi interface to asics
    i_spi_miso      : in    std_logic;
    o_spi_mosi      : out   std_logic;
    o_spi_sclk      : out   std_logic;
    o_spi_ss_n      : out   std_logic_vector(15 downto 0);



    -- QSFP links
    i_qsfp_rx       : in    std_logic_vector(3 downto 0);
    o_qsfp_tx       : out   std_logic_vector(3 downto 0);

    -- POD links (reset system)
    i_pod_rx        : in    std_logic_vector(3 downto 0);
    o_pod_tx        : out   std_logic_vector(3 downto 0);



    --
    i_fifo_rempty   : in    std_logic;
    o_fifo_rack     : out   std_logic;
    i_fifo_rdata    : in    std_logic_vector(35 downto 0);

    -- MSCB interface
    i_mscb_data     : in    std_logic;
    o_mscb_data     : out   std_logic;
    o_mscb_oe       : out   std_logic;

    -- slow control registers
    o_malibu_reg_addr   : out   std_logic_vector(7 downto 0);
    o_malibu_reg_re     : out   std_logic;
    i_malibu_reg_rdata  : in    std_logic_vector(31 downto 0) := X"CCCCCCCC";
    o_malibu_reg_we     : out   std_logic;
    o_malibu_reg_wdata  : out   std_logic_vector(31 downto 0);
    o_scifi_reg_addr    : out   std_logic_vector(7 downto 0);
    o_scifi_reg_re      : out   std_logic;
    i_scifi_reg_rdata   : in    std_logic_vector(31 downto 0) := X"CCCCCCCC";
    o_scifi_reg_we      : out   std_logic;
    o_scifi_reg_wdata   : out   std_logic_vector(31 downto 0);
    o_mupix_reg_addr    : out   std_logic_vector(7 downto 0);
    o_mupix_reg_re      : out   std_logic;
    i_mupix_reg_rdata   : in    std_logic_vector(31 downto 0) := X"CCCCCCCC";
    o_mupix_reg_we      : out   std_logic;
    o_mupix_reg_wdata   : out   std_logic_vector(31 downto 0);



    -- reset system
    o_run_state_125 : out   run_state_t;



    -- nios clock (async)
    i_nios_clk      : in    std_logic;
    o_nios_clk_mon  : out   std_logic;
    -- 156.25 MHz (data, QSFP)
    i_clk_156       : in    std_logic;
    o_clk_156_mon   : out   std_logic;
    -- 125 MHz (global clock, POD)
    i_clk_125       : in    std_logic;
    o_clk_125_mon   : out   std_logic;

    i_areset_n      : in    std_logic--;
);
end entity;

architecture arch of fe_block is

    signal nios_reset_n, reset_156_n, reset_125_n : std_logic;

    signal nios_pio : std_logic_vector(31 downto 0);
    signal nios_irq : std_logic_vector(3 downto 0) := (others => '0');

    signal av_sc : work.util.avalon_t;

    signal sc_fifo_rempty : std_logic;
    signal sc_fifo_rack : std_logic;
    signal sc_fifo_rdata : std_logic_vector(35 downto 0);

    signal sc_ram, sc_reg : work.util.rw_t;
    signal fe_reg : work.util.rw_t;
    signal malibu_reg, scifi_reg, mupix_reg : work.util.rw_t;

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

    signal pod_rx_clk : std_logic_vector(3 downto 0);
    signal pod_rx_reset_n : std_logic_vector(3 downto 0);

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

    -- generate resets

    e_nios_reset_n : entity work.reset_sync
    port map ( o_reset_n => nios_reset_n, i_reset_n => i_areset_n, i_clk => i_nios_clk );

    e_reset_156_n : entity work.reset_sync
    port map ( o_reset_n => reset_156_n, i_reset_n => i_areset_n, i_clk => i_clk_156 );

    e_reset_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_125_n, i_reset_n => i_areset_n, i_clk => i_clk_125 );



    -- generate 1 Hz clock monitor clocks

    -- NIOS_CLK_HZ_g -> 1 Hz
    e_nios_clk_hz : entity work.clkdiv
    generic map ( P => NIOS_CLK_HZ_g )
    port map ( o_clk => o_nios_clk_mon, i_reset_n => nios_reset_n, i_clk => i_nios_clk );

    -- 156.25 MHz -> 1 Hz
    e_clk_156_hz : entity work.clkdiv
    generic map ( P => 156250000 )
    port map ( o_clk => o_clk_156_mon, i_reset_n => reset_156_n, i_clk => i_clk_156 );

    -- 125 MHz -> 1 Hz
    e_clk_125_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( o_clk => o_clk_125_mon, i_reset_n => reset_125_n, i_clk => i_clk_125 );



    -- map slow control address space

    -- malibu regs : 0x40-0x5F
    o_malibu_reg_addr <= sc_reg.addr(7 downto 0);
    o_malibu_reg_re <= malibu_reg.re;
      malibu_reg.re <= sc_reg.re when ( sc_reg.addr(7 downto 5) = "010" ) else '0';
    o_malibu_reg_we <= sc_reg.we when ( sc_reg.addr(7 downto 5) = "010" ) else '0';
    o_malibu_reg_wdata <= sc_reg.wdata;

    -- scifi regs : 0x60-0x7F
    o_scifi_reg_addr <= sc_reg.addr(7 downto 0);
    o_scifi_reg_re <= scifi_reg.re;
      scifi_reg.re <= sc_reg.re when ( sc_reg.addr(7 downto 5) = "011" ) else '0';
    o_scifi_reg_we <= sc_reg.we when ( sc_reg.addr(7 downto 5) = "011" ) else '0';
    o_scifi_reg_wdata <= sc_reg.wdata;

    -- mupix regs : 0x80-0x9F
    o_mupix_reg_addr <= sc_reg.addr(7 downto 0);
    o_mupix_reg_re <= mupix_reg.re;
      mupix_reg.re <= sc_reg.re when ( sc_reg.addr(7 downto 5) = "100" ) else '0';
    o_mupix_reg_we <= sc_reg.we when ( sc_reg.addr(7 downto 5) = "100" ) else '0';
    o_mupix_reg_wdata <= sc_reg.wdata;

    -- local regs : 0xF0-0xFF
    fe_reg.addr <= sc_reg.addr;
    fe_reg.re <= sc_reg.re when ( sc_reg.addr(7 downto 4) = X"F" ) else '0';
    fe_reg.we <= sc_reg.we when ( sc_reg.addr(7 downto 4) = X"F" ) else '0';
    fe_reg.wdata <= sc_reg.wdata;

    -- select valid rdata
    sc_reg.rdata <=
        i_malibu_reg_rdata when ( malibu_reg.rvalid = '1' ) else
        i_scifi_reg_rdata when ( scifi_reg.rvalid = '1' ) else
        i_mupix_reg_rdata when ( mupix_reg.rvalid = '1' ) else
        fe_reg.rdata when ( fe_reg.rvalid = '1' ) else
        X"CCCCCCCC";

    process(i_clk_156)
    begin
    if rising_edge(i_clk_156) then
        malibu_reg.rvalid <= malibu_reg.re;
        scifi_reg.rvalid <= scifi_reg.re;
        mupix_reg.rvalid <= mupix_reg.re;
        fe_reg.rvalid <= fe_reg.re;

        fe_reg.rdata <= X"CCCCCCCC";

        -- cmdlen
        if ( fe_reg.addr(7 downto 0) = X"F0" and fe_reg.re = '1' ) then
            fe_reg.rdata <= reg_cmdlen;
        end if;
        if ( fe_reg.addr(7 downto 0) = X"F0" and fe_reg.we = '1' ) then
            reg_cmdlen <= fe_reg.wdata;
        end if;

        -- offset
        if ( fe_reg.addr(7 downto 0) = X"F1" and fe_reg.re = '1' ) then
            fe_reg.rdata <= reg_offset;
        end if;
        if ( fe_reg.addr(7 downto 0) = X"F1" and fe_reg.we = '1' ) then
            reg_offset <= fe_reg.wdata;
        end if;

        -- reset bypass
        if ( fe_reg.addr(7 downto 0) = X"F4" and fe_reg.re = '1' ) then
            fe_reg.rdata <= reg_reset_bypass;
        end if;
        if ( fe_reg.addr(7 downto 0) = X"F4" and fe_reg.we = '1' ) then
            reg_reset_bypass <= fe_reg.wdata;
        end if;

        -- mscb

        --
    end if;
    end process;



    -- nios system
    nios_irq(0) <= '1' when ( reg_cmdlen(31 downto 16) /= (31 downto 16 => '0') ) else '0';



    e_nios : component work.cmp.nios
    port map (
        -- SC, QSFP and irq
        clk_156_reset_reset_n   => reset_156_n,
        clk_156_clock_clk       => i_clk_156,

        -- POD
        clk_125_reset_reset_n   => reset_125_n,
        clk_125_clock_clk       => i_clk_125,

        -- mscb
        parallel_mscb_in_export     => mscb_to_nios_parallel_in,
        parallel_mscb_out_export    => mscb_from_nios_parallel_out,
        counter_in_export           => std_logic_vector(mscb_counter_in),

        irq_bridge_irq          => nios_irq,

        avm_sc_address          => av_sc.address(15 downto 0),
        avm_sc_read             => av_sc.read,
        avm_sc_readdata         => av_sc.readdata,
        avm_sc_write            => av_sc.write,
        avm_sc_writedata        => av_sc.writedata,
        avm_sc_waitrequest      => av_sc.waitrequest,

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

        spi_si_miso => i_spi_si_miso,
        spi_si_mosi => o_spi_si_mosi,
        spi_si_sclk => o_spi_si_sclk,
        spi_si_ss_n => o_spi_si_ss_n,

        pio_export => nios_pio,

        rst_reset_n => nios_reset_n,
        clk_clk => i_nios_clk--,
    );



    e_sc_ram : entity work.sc_ram
    generic map (
        RAM_ADDR_WIDTH_g => 14--,
    )
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

        i_reset_n           => reset_156_n,
        i_clk               => i_clk_156--;
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

        i_reset_n       => reset_156_n,
        i_clk           => i_clk_156--,
    );



    e_merger : entity work.data_merger
    port map (
        fpga_ID_in              => FPGA_ID_g,
        FEB_type_in             => FEB_type_in,
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

        reset                   => not reset_156_n,
        clk                     => i_clk_156--,
    );



    e_link_test : entity work.linear_shift_link
    generic map (
        g_m => 32,
        g_poly => "10000000001000000000000000000110"--,
    )
    port map (
        i_sync_reset    => '0',
        i_seed          => (others => '1'),
        i_en            => work.util.to_std_logic(run_state_156 = work.daq_constants.RUN_STATE_LINK_TEST),
        o_lsfr          => qsfp_tx_data(63 downto 32),
        o_datak         => qsfp_tx_datak(7 downto 4),
        reset_n         => reset_156_n,
        i_clk           => i_clk_156--,
    );



    e_reset_system : entity work.resetsys
    port map (
        i_data_125_rx           => pod_rx_data(7 downto 0),
        i_reset_125_rx_n        => pod_rx_reset_n(0),
        i_clk_125_rx            => pod_rx_clk(0),

        o_state_125             => run_state_125,
        i_reset_125_n           => reset_125_n,
        i_clk_125               => i_clk_125,

        o_state_156             => run_state_156,
        i_reset_156_n           => reset_156_n,
        i_clk_156               => i_clk_156,

        resets_out              => open,
        reset_bypass            => reg_reset_bypass(11 downto 0),
        run_number_out          => open,
        fpga_id                 => FPGA_ID_g,
        terminated              => terminated,
        testout                 => open,

        o_phase                 => open,
        i_reset_n               => nios_reset_n,
        i_clk                   => i_nios_clk--,
    );

    o_run_state_125 <= run_state_125;



    e_mscb : entity work.mscb
    generic map (
        CLK_HZ_g => NIOS_CLK_HZ_g--,
    )
    port map (
        mscb_to_nios_parallel_in    => mscb_to_nios_parallel_in,
        mscb_from_nios_parallel_out => mscb_from_nios_parallel_out,
        mscb_data_in                => i_mscb_data,
        mscb_data_out               => o_mscb_data,
        mscb_oe                     => o_mscb_oe,
        mscb_counter_in             => mscb_counter_in,

        o_mscb_irq                  => nios_irq(1),
        i_mscb_address              => X"ACA0",

        reset                       => not nios_reset_n,
        nios_clk                    => i_nios_clk--,
    );



    e_qsfp : entity work.xcvr_s4
    generic map (
        NUMBER_OF_CHANNELS_g => 4,
        CHANNEL_WIDTH_g => 32,
        INPUT_CLOCK_FREQUENCY_g => 156250000,
        DATA_RATE_g => 6250,
        CLK_HZ_g => 156250000--,
    )
    port map (
        i_tx_data   => qsfp_tx_data,
        i_tx_datak  => qsfp_tx_datak,

        o_rx_data   => qsfp_rx_data,
        o_rx_datak  => qsfp_rx_datak,

        o_tx_clkout => open,
        i_tx_clkin  => (others => i_clk_156),
        o_rx_clkout => open,
        i_rx_clkin  => (others => i_clk_156),

        o_tx_serial => o_qsfp_tx,
        i_rx_serial => i_qsfp_rx,

        i_pll_clk   => i_clk_156,
        i_cdr_clk   => i_clk_156,

        i_avs_address       => av_qsfp.address(13 downto 0),
        i_avs_read          => av_qsfp.read,
        o_avs_readdata      => av_qsfp.readdata,
        i_avs_write         => av_qsfp.write,
        i_avs_writedata     => av_qsfp.writedata,
        o_avs_waitrequest   => av_qsfp.waitrequest,

        i_reset     => not reset_156_n,
        i_clk       => i_clk_156--,
    );



    g_pod_rx_reset_n : for i in pod_rx_reset_n'range generate
    begin
        e_pod_rx_reset_n : entity work.reset_sync
        port map ( o_reset_n => pod_rx_reset_n(i), i_reset_n => i_areset_n, i_clk => pod_rx_clk(i) );
    end generate;

    e_pod : entity work.xcvr_s4
    generic map (
        NUMBER_OF_CHANNELS_g => 4,
        CHANNEL_WIDTH_g => 8,
        INPUT_CLOCK_FREQUENCY_g => 125000000,
        DATA_RATE_g => 1250,
        CLK_HZ_g => 125000000--,
    )
    port map (
        i_tx_data   => pod_tx_data,
        i_tx_datak  => pod_tx_datak,

        o_rx_data   => pod_rx_data,
        o_rx_datak  => pod_rx_datak,

        o_tx_clkout => open,
        i_tx_clkin  => (others => i_clk_125),
        o_rx_clkout => pod_rx_clk,
        i_rx_clkin  => pod_rx_clk,

        o_tx_serial => o_pod_tx,
        i_rx_serial => i_pod_rx,

        i_pll_clk   => i_clk_125,
        i_cdr_clk   => i_clk_125,

        i_avs_address       => av_pod.address(13 downto 0),
        i_avs_read          => av_pod.read,
        o_avs_readdata      => av_pod.readdata,
        i_avs_write         => av_pod.write,
        i_avs_writedata     => av_pod.writedata,
        o_avs_waitrequest   => av_pod.waitrequest,

        i_reset     => not reset_125_n,
        i_clk       => i_clk_125--,
    );

end architecture;
