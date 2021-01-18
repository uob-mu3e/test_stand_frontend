library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_misc.all;
use work.daq_constants.all;
use work.feb_sc_registers.all;

entity fe_block is
generic (
    feb_mapping : natural_array_t(3 downto 0) := 3&2&1&0;
    PHASE_WIDTH_g : positive := 16;
    NIOS_CLK_MHZ_g : real;
    N_LINKS : positive := 1--;
);
port (
    i_fpga_id       : in    std_logic_vector(N_LINKS*16 - 1 downto 0);
    -- frontend board type
    -- - 111010 : mupix
    -- - 111000 : mutrig
    -- - 000111 and 000000 : reserved (DO NOT USE)
    i_fpga_type     : in    std_logic_vector(5 downto 0);

    i_i2c_scl       : in    std_logic;
    o_i2c_scl_oe    : out   std_logic;
    i_i2c_sda       : in    std_logic;
    o_i2c_sda_oe    : out   std_logic;

    -- spi interface to si chip
    i_spi_si_miso   : in    std_logic_vector(1 downto 0) := (others => '0');
    o_spi_si_mosi   : out   std_logic_vector(1 downto 0);
    o_spi_si_sclk   : out   std_logic_vector(1 downto 0);
    o_spi_si_ss_n   : out   std_logic_vector(1 downto 0);

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



    i_can_terminate           : in std_logic := '0';
    i_ack_run_prep_permission : in std_logic := '1';

    --main fiber data fifo
    i_fifo_we       : in    std_logic_vector(N_LINKS-1 downto 0);
    i_fifo_wdata    : in    std_logic_vector(36*(N_LINKS-1)+35 downto 0);

    o_fifos_almost_full         : out   std_logic_vector(N_LINKS-1 downto 0);

    -- slow control fifo
    o_scfifo_write     : out   std_logic;
    o_scfifo_wdata     : in    std_logic_vector(35 downto 0):=(others =>'-');

    -- MSCB interface
    i_mscb_data     : in    std_logic;
    o_mscb_data     : out   std_logic;
    o_mscb_oe       : out   std_logic;

    -- slow control registers
    -- malibu regs : 0x40-0x5F
    o_malibu_reg_addr   : out   std_logic_vector(7 downto 0);
    o_malibu_reg_re     : out   std_logic;
    i_malibu_reg_rdata  : in    std_logic_vector(31 downto 0) := X"CCCCCCCC";
    o_malibu_reg_we     : out   std_logic;
    o_malibu_reg_wdata  : out   std_logic_vector(31 downto 0);
    -- scifi regs : 0x60-0x7F
    o_scifi_reg_addr    : out   std_logic_vector(7 downto 0);
    o_scifi_reg_re      : out   std_logic;
    i_scifi_reg_rdata   : in    std_logic_vector(31 downto 0) := X"CCCCCCCC";
    o_scifi_reg_we      : out   std_logic;
    o_scifi_reg_wdata   : out   std_logic_vector(31 downto 0);
    -- mupix regs : 0x80-0x9F
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
    
    signal reset_pod : std_logic := '0'; -- not in reset for default
    signal reset_pod_125_n : std_logic := '1';
    signal reset_qsfp : std_logic := '0'; -- not in reset for default

    signal nios_pio : std_logic_vector(31 downto 0);
    signal nios_irq : std_logic_vector(3 downto 0) := (others => '0');

    signal spi_si_miso, spi_si_mosi, spi_si_sclk : std_logic;
    signal spi_si_ss_n : std_logic_vector(o_spi_si_ss_n'range);

    signal av_sc : work.util.avalon_t;

    signal sc_fifo_we : std_logic;
    signal sc_fifo_wdata : std_logic_vector(35 downto 0);

    signal sc_ram, sc_reg : work.util.rw_t;
    signal fe_reg : work.util.rw_t;
    signal malibu_reg, scifi_reg, mupix_reg : work.util.rw_t;

    signal reg_cmdlen : std_logic_vector(31 downto 0);
    signal reg_offset : std_logic_vector(31 downto 0);

    signal av_nios : work.util.avalon_t;



    signal linktest_data    : std_logic_vector(31 downto 0);
    signal linktest_datak   : std_logic_vector(3 downto 0);
    signal linktest_granted : std_logic_vector(N_LINKS-1 downto 0);

    signal av_mscb : work.util.avalon_t;

    signal reg_reset_bypass : std_logic_vector(31 downto 0);
    signal reg_reset_bypass_payload : std_logic_vector(31 downto 0);

    signal run_state_125 : run_state_t;
    signal run_state_156 : run_state_t;
    signal terminated : std_logic;
    signal reset_phase : std_logic_vector(PHASE_WIDTH_g - 1 downto 0);

    signal run_number : std_logic_vector(31 downto 0);
    signal merger_rate_count : std_logic_vector(31 downto 0);

    signal reconfig_clk : std_logic;

    signal av_qsfp, av_pod : work.util.avalon_t;

    signal pod_rx_clk : std_logic_vector(3 downto 0);
    signal pod_rx_reset_n : std_logic_vector(3 downto 0);

    signal qsfp_rx_data : std_logic_vector(127 downto 0);
    signal qsfp_rx_datak : std_logic_vector(15 downto 0);
    signal pod_rx_data : std_logic_vector(31 downto 0);
    signal pod_rx_datak : std_logic_vector(3 downto 0);
    
    signal i_fpga_id_reg : std_logic_vector(N_LINKS*16-1 downto 0) := i_fpga_id;

    signal qsfp_tx_data : std_logic_vector(127 downto 0) :=
          X"000000" & work.util.D28_5
        & X"000000" & work.util.D28_5
        & X"000000" & work.util.D28_5
        & X"000000" & work.util.D28_5;
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
    
    e_reset_pod_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_pod_125_n, i_reset_n => not reset_pod, i_clk => i_clk_125 );



    -- generate 1 Hz clock monitor clocks

    -- NIOS_CLK_MHZ_g -> 1 Hz
    e_nios_clk_hz : entity work.clkdiv
    generic map ( P => integer(NIOS_CLK_MHZ_g * 1000000.0) )
    port map ( o_clk => o_nios_clk_mon, i_reset_n => nios_reset_n, i_clk => i_nios_clk );

    -- 156.25 MHz -> 1 Hz
    e_clk_156_hz : entity work.clkdiv
    generic map ( P => 156250000 )
    port map ( o_clk => o_clk_156_mon, i_reset_n => reset_156_n, i_clk => i_clk_156 );

    -- 125 MHz -> 1 Hz
    e_clk_125_hz : entity work.clkdiv
    generic map ( P => 125000000 )
    port map ( o_clk => o_clk_125_mon, i_reset_n => reset_125_n, i_clk => i_clk_125 );



    -- SPI
    spi_si_miso <= '1' when ( (i_spi_si_miso or spi_si_ss_n) = (spi_si_ss_n'range => '1') ) else '0';
    o_spi_si_mosi <= (o_spi_si_mosi'range => spi_si_mosi);
    o_spi_si_sclk <= (o_spi_si_sclk'range => spi_si_sclk);
    o_spi_si_ss_n <= spi_si_ss_n;



    -- map slow control address space
	 -- TODO: We only have one subdetector per FEB and the frimware is specific:
	 -- All registers above 0x40 can be used for the specific subdetecto

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
    fe_reg.re <= sc_reg.re when ( sc_reg.addr(REG_AREA_RANGE) = REG_AREA_GENERIC ) else '0';
    fe_reg.we <= sc_reg.we when ( sc_reg.addr(REG_AREA_RANGE) = REG_AREA_GENERIC ) else '0';
    fe_reg.wdata <= sc_reg.wdata;

    -- select valid rdata
    sc_reg.rdata <=
        i_malibu_reg_rdata when ( malibu_reg.rvalid = '1' ) else
        i_scifi_reg_rdata when ( scifi_reg.rvalid = '1' ) else
        i_mupix_reg_rdata when ( mupix_reg.rvalid = '1' ) else
        fe_reg.rdata when ( fe_reg.rvalid = '1' ) else
        X"CCCCCCCC";

    process(i_clk_156, reset_156_n)
	
	 variable regaddr : integer;
	 
    begin
    if ( reset_156_n = '0' ) then
        malibu_reg.rvalid <= '0';
        scifi_reg.rvalid <= '0';
        mupix_reg.rvalid <= '0';
        fe_reg.rvalid <= '0';

        fe_reg.rdata <= X"CCCCCCCC";

        av_nios.address <= (others => '0');
        av_nios.read <= '0';
        av_nios.write <= '0';
    elsif rising_edge(i_clk_156) then
        regaddr := TO_INTEGER(UNSIGNED(fe_reg.addr(7 downto 0)));
        malibu_reg.rvalid <= malibu_reg.re;
        scifi_reg.rvalid <= scifi_reg.re;
        mupix_reg.rvalid <= mupix_reg.re;
        fe_reg.rvalid <= fe_reg.re;

        fe_reg.rdata <= X"CCCCCCCC";
        
        -- only valid for one cycle
        reset_qsfp <= '0';
        --reset_pod <= '0';

        -- cmdlen
        if ( regaddr = CMD_LEN_REGISTER_RW and fe_reg.re = '1' ) then
            fe_reg.rdata <= reg_cmdlen;
        end if;
        if ( regaddr = CMD_LEN_REGISTER_RW and fe_reg.we = '1' ) then
            reg_cmdlen <= fe_reg.wdata;
        end if;

        -- offset
        if ( regaddr = CMD_OFFSET_REGISTER_RW and fe_reg.re = '1' ) then
            fe_reg.rdata <= reg_offset;
        end if;
        if ( regaddr = CMD_OFFSET_REGISTER_RW and fe_reg.we = '1' ) then
            reg_offset <= fe_reg.wdata;
        end if;

        -- reset bypass
        if ( regaddr = RUN_STATE_RESET_BYPASS_REGISTER_RW and fe_reg.re = '1' ) then
            fe_reg.rdata(15 downto 0) <= reg_reset_bypass(15 downto 0);
            fe_reg.rdata(16+9 downto 16) <= run_state_156;
        end if;
        if ( regaddr = RUN_STATE_RESET_BYPASS_REGISTER_RW and fe_reg.we = '1' ) then
            reg_reset_bypass(15 downto 0) <= fe_reg.wdata(15 downto 0); -- upper bits are read-only status
        end if;

        -- reset payload
        if ( regaddr = RESET_PAYLOAD_RGEISTER_RW and fe_reg.re = '1' ) then
            fe_reg.rdata <= reg_reset_bypass_payload;
        end if;
        if ( regaddr = RESET_PAYLOAD_RGEISTER_RW and fe_reg.we = '1' ) then
            reg_reset_bypass_payload <= fe_reg.wdata;
        end if;

        -- rate measurement
        if ( regaddr = MERGER_RATE_REGISTER_R and fe_reg.re = '1' ) then
            fe_reg.rdata <= merger_rate_count;
        end if;

        -- reset phase
        if ( regaddr = RESET_PHASE_REGISTER_R and fe_reg.re = '1' ) then
            fe_reg.rdata(PHASE_WIDTH_g - 1 downto 0) <= reset_phase;
        end if;
        
        -- reset qsfp / pod
        if ( regaddr = RESET_OPTICAL_LINKS_REGISTER_RW and fe_reg.re = '1' ) then
            fe_reg.rdata(0) <= reset_pod;
            fe_reg.rdata(1) <= reset_qsfp;
        end if;
        if ( regaddr = RESET_OPTICAL_LINKS_REGISTER_RW and fe_reg.we = '1' ) then
            reset_pod <= fe_reg.wdata(0);
            reset_qsfp <= fe_reg.wdata(1);
        end if;

        -- mscb

        -- git head hash
        if ( regaddr = GIT_HASH_REGISTER_R and fe_reg.re = '1' ) then
            fe_reg.rdata <= (others => '0');
            fe_reg.rdata <= work.cmp.GIT_HEAD(0 to 31);
        end if;
        -- fpga id
        if ( regaddr = FPGA_ID_REGISTER_RW and fe_reg.re = '1' ) then
            fe_reg.rdata <= (others => '0');
            fe_reg.rdata(i_fpga_id_reg'range) <= i_fpga_id_reg;
        end if;
        if ( regaddr = FPGA_ID_REGISTER_RW and fe_reg.we = '1' ) then
            fe_reg.rdata <= (others => '0');
            i_fpga_id_reg(N_LINKS*16-1 downto 0) <= fe_reg.wdata(N_LINKS*16-1 downto 0);
        end if;
        -- fpga type
        if ( regaddr = FPGA_TYPE_REGISTER_R and fe_reg.re = '1' ) then
            fe_reg.rdata <= (others => '0');
            fe_reg.rdata(i_fpga_type'range) <= i_fpga_type;
        end if;

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
        avm_mscb_address        => av_mscb.address(3 downto 0),
        avm_mscb_read           => av_mscb.read,
        avm_mscb_readdata       => av_mscb.readdata,
        avm_mscb_write          => av_mscb.write,
        avm_mscb_writedata      => av_mscb.writedata,
        avm_mscb_waitrequest    => av_mscb.waitrequest,

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

        spi_si_miso => spi_si_miso,
        spi_si_mosi => spi_si_mosi,
        spi_si_sclk => spi_si_sclk,
        spi_si_ss_n => spi_si_ss_n,

        pio_export => nios_pio,

        rst_reset_n => nios_reset_n,
        clk_clk => i_nios_clk--,
    );



    e_sc_ram : entity work.sc_ram
    generic map (
        RAM_ADDR_WIDTH_g => 14--,
    )
    port map (
        i_ram_addr              => sc_ram.addr(15 downto 0),
        i_ram_re                => sc_ram.re,
        o_ram_rvalid            => sc_ram.rvalid,
        o_ram_rdata             => sc_ram.rdata,
        i_ram_we                => sc_ram.we,
        i_ram_wdata             => sc_ram.wdata,

        i_avs_address           => av_sc.address(15 downto 0),
        i_avs_read              => av_sc.read,
        o_avs_readdata          => av_sc.readdata,
        i_avs_write             => av_sc.write,
        i_avs_writedata         => av_sc.writedata,
        o_avs_waitrequest       => av_sc.waitrequest,

        o_reg_addr              => sc_reg.addr(7 downto 0),
        o_reg_re                => sc_reg.re,
        i_reg_rdata             => sc_reg.rdata,
        o_reg_we                => sc_reg.we,
        o_reg_wdata             => sc_reg.wdata,

        i_reset_n               => reset_156_n,
        i_clk                   => i_clk_156--;
    );

    e_sc_rx : entity work.sc_rx
    port map (
        i_link_data     => qsfp_rx_data(32*(feb_mapping(0)+1)-1 downto 32*feb_mapping(0)),
        i_link_datak    => qsfp_rx_datak(4*(feb_mapping(0)+1)-1 downto 4*feb_mapping(0)),

        o_fifo_we       => sc_fifo_we,
        o_fifo_wdata    => sc_fifo_wdata,

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
    generic map(
        N_LINKS                    => N_LINKS,
        feb_mapping                => feb_mapping--,
    )
    port map (
        fpga_ID_in                 => i_fpga_id_reg,
        FEB_type_in                => i_fpga_type,

        run_state                  => run_state_156,
        run_number                 => run_number,

        o_data_out                 => qsfp_tx_data,
        o_data_is_k                => qsfp_tx_datak,

        slowcontrol_write_req      => sc_fifo_we,
        i_data_in_slowcontrol      => sc_fifo_wdata,

        data_write_req             => i_fifo_we,
        i_data_in                  => i_fifo_wdata,
        o_fifos_almost_full        => o_fifos_almost_full,

        override_data_in           => linktest_data,
        override_data_is_k_in      => linktest_datak,
        override_req               => work.util.to_std_logic(run_state_156 = work.daq_constants.RUN_STATE_LINK_TEST),   --TODO test and find better way to connect this
        override_granted           => linktest_granted,

        can_terminate              => i_can_terminate,
        o_terminated               => terminated,
        i_ack_run_prep_permission  => i_ack_run_prep_permission,
        data_priority              => '0',
        o_rate_count               => merger_rate_count,

        reset                      => not reset_156_n,
        clk                        => i_clk_156--,
    );


    --TODO: do we need two independent link test modules for both fibers?
    e_link_test : entity work.linear_shift_link
    generic map (
        g_m => 32,
        g_poly => "10000000001000000000000000000110"--,
    )
    port map (
        i_sync_reset    => not work.util.and_reduce(linktest_granted),
        i_seed          => (others => '1'),
        i_en            => work.util.to_std_logic(run_state_156 = work.daq_constants.RUN_STATE_LINK_TEST),
        o_lsfr          => linktest_data,
        o_datak         => linktest_datak,
        reset_n         => reset_156_n,
        i_clk           => i_clk_156--,
    );



    e_reset_system : entity work.resetsys
    generic map (
         PHASE_WIDTH_g => PHASE_WIDTH_g--,
    )
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
        reset_bypass_payload    => reg_reset_bypass_payload,
        run_number_out          => run_number,
        fpga_id                 => i_fpga_id_reg(15 downto 0),
        terminated              => terminated, --TODO: test with two datamergers
        testout                 => open,

        o_phase                 => reset_phase,
        i_reset_n               => nios_reset_n,
        i_clk                   => i_nios_clk--,
    );

    o_run_state_125 <= run_state_125;



    e_mscb : entity work.mscb
    generic map (
        CLK_MHZ_g => 156.25--,
    )
    port map (
        i_avs_address           => av_mscb.address(3 downto 0),
        i_avs_read              => av_mscb.read,
        o_avs_readdata          => av_mscb.readdata,
        i_avs_write             => av_mscb.write,
        i_avs_writedata         => av_mscb.writedata,
        o_avs_waitrequest       => av_mscb.waitrequest,

        i_rx_data               => i_mscb_data,
        o_tx_data               => o_mscb_data,
        o_tx_data_oe            => o_mscb_oe,

        o_irq                   => nios_irq(1),
        i_mscb_address              => X"ACA0",

        i_reset_n               => reset_156_n,
        i_clk                   => i_clk_156--,
    );



    g_reconfig_clk : if ( NIOS_CLK_MHZ_g <= 50.0 ) generate
        reconfig_clk <= i_nios_clk; -- Frequency Range : 37.5 to 50 MHz
    end generate;

    -- generate reconfig_clk = 50 MHz
    g_reconfig_clk_altpll : if ( NIOS_CLK_MHZ_g > 50.0 ) generate
        e_reconfig_clk : entity work.ip_altpll
        generic map (
            INCLK0_MHZ => NIOS_CLK_MHZ_g,
            DIV => integer(NIOS_CLK_MHZ_g * 1000000.0) / work.util.gcd(integer(NIOS_CLK_MHZ_g * 1000000.0), 50000000),
            MUL => 50000000 / work.util.gcd(integer(NIOS_CLK_MHZ_g * 1000000.0), 50000000)--,
        )
        port map (
            c0 => reconfig_clk,
            locked => open,
            areset => not nios_reset_n,
            inclk0 => i_nios_clk--,
        );
    end generate;



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

        i_reconfig_clk  => reconfig_clk, -- 37.5 to 50 MHz

        i_reset     => not reset_156_n or reset_qsfp,
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

        i_reconfig_clk  => reconfig_clk,

        i_reset     => (not reset_125_n) or (not reset_pod_125_n),
        i_clk       => i_clk_125--,
    );

end architecture;
