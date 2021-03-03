library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;
use work.feb_sc_registers.all;

LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

entity fe_block_v2 is
generic (
    feb_mapping : natural_array_t(3 downto 0) := 3&2&1&0;
    PHASE_WIDTH_g : positive := 16;
    NIOS_CLK_MHZ_g : real;
    N_LINKS : positive := 1--;
);
port (
    i_fpga_id       : in    std_logic_vector(7 downto 0);
    -- frontend board type
    -- - 111010 : mupix
    -- - 111000 : mutrig
    -- - 000111 and 000000 : reserved (DO NOT USE)
    i_fpga_type     : in    std_logic_vector(5 downto 0);

    -- i2c firefly interface
    io_i2c_ffly_scl     : inout std_logic;
    io_i2c_ffly_sda     : inout std_logic;
    o_i2c_ffly_ModSel_n : out   std_logic_vector(1 downto 0);
    o_ffly_Rst_n        : out   std_logic_vector(1 downto 0);
    i_ffly_Int_n        : in    std_logic_vector(1 downto 0);
    i_ffly_ModPrs_n     : in    std_logic_vector(1 downto 0);

    -- si chip
    i_spi_si_miso       : in    std_logic_vector(1 downto 0) := (others => '0');
    o_spi_si_mosi       : out   std_logic_vector(1 downto 0);
    o_spi_si_sclk       : out   std_logic_vector(1 downto 0);
    o_spi_si_ss_n       : out   std_logic_vector(1 downto 0);

    o_si45_oe_n         : out   std_logic_vector(1 downto 0) := (others => '0');
    i_si45_intr_n       : in    std_logic_vector(1 downto 0) := (others => '1');
    i_si45_lol_n        : in    std_logic_vector(1 downto 0) := (others => '0');
    o_si45_rst_n        : out   std_logic_vector(1 downto 0) := (others => '1');
    o_si45_fdec         : out   std_logic_vector(1 downto 0) := (others => '0');
    o_si45_finc         : out   std_logic_vector(1 downto 0) := (others => '0');

    -- spi interface to asics
    i_spi_miso          : in    std_logic;
    o_spi_mosi          : out   std_logic;
    o_spi_sclk          : out   std_logic;
    o_spi_ss_n          : out   std_logic_vector(15 downto 0);

    -- Fireflies
    o_ffly1_tx          : out   std_logic_vector(3 downto 0);
    o_ffly2_tx          : out   std_logic_vector(3 downto 0);
    i_ffly1_rx          : in    std_logic;
    i_ffly2_rx          : in    std_logic_vector(2 downto 0);

    i_ffly1_lvds_rx     : in    std_logic;
    i_ffly2_lvds_rx     : in    std_logic;

    -- run control flags from detector-block
    i_can_terminate           : in std_logic:='0';
    i_ack_run_prep_permission : in std_logic:='1';

    --main fiber data fifo
    i_fifo_write        : in    std_logic_vector(N_LINKS-1 downto 0);
    i_fifo_wdata        : in    std_logic_vector(36*(N_LINKS-1)+35 downto 0);

    o_fifos_almost_full : out   std_logic_vector(N_LINKS-1 downto 0);

    -- slow control fifo
    o_scfifo_write      : out   std_logic;
    o_scfifo_wdata      : in    std_logic_vector(35 downto 0):=(others =>'-');

    -- MSCB interface
    i_mscb_data         : in    std_logic;
    o_mscb_data         : out   std_logic;
    o_mscb_oe           : out   std_logic;

    -- MAX10 IF
    o_max10_spi_sclk    : out   std_logic;
    io_max10_spi_mosi   : inout std_logic;
    io_max10_spi_miso   : inout std_logic;
    io_max10_spi_D1     : inout std_logic := 'Z';
    io_max10_spi_D2     : inout std_logic := 'Z';
    io_max10_spi_D3     : inout std_logic := 'Z';
    o_max10_spi_csn     : out   std_logic := '1';

    -- slow control registers
    -- subdetector regs : 0x40-0xFF
    o_subdet_reg_addr   : out   std_logic_vector(7 downto 0);
    o_subdet_reg_re     : out   std_logic;
    i_subdet_reg_rdata  : in    std_logic_vector(31 downto 0) := X"CCCCCCCC";
    o_subdet_reg_we     : out   std_logic;
    o_subdet_reg_wdata  : out   std_logic_vector(31 downto 0);

    -- reset system
    o_run_state_125 : out   run_state_t;
    o_run_state_156 : out   run_state_t;

    -- nios clock (async)
    i_nios_clk      : in    std_logic;
    o_nios_clk_mon  : out   std_logic;
    -- 156.25 MHz (data)
    i_clk_156       : in    std_logic;
    o_clk_156_mon   : out   std_logic;
    -- 125 MHz (global clock)
    i_clk_125       : in    std_logic;
    o_clk_125_mon   : out   std_logic;

    i_areset_n      : in    std_logic;
    
    i_testin        : in    std_logic--;
);
end entity;

architecture arch of fe_block_v2 is

    signal nios_reset_n             : std_logic;
    signal reset_156_n              : std_logic;
    signal reset_125_n              : std_logic;
    signal reset_125_RRX_n          : std_logic;

    signal nios_pio                 : std_logic_vector(31 downto 0);
    signal nios_irq                 : std_logic_vector(3 downto 0) := (others => '0');

    signal spi_si_miso              : std_logic;
    signal spi_si_mosi              : std_logic;
    signal spi_si_sclk              : std_logic;
    signal spi_si_ss_n              : std_logic_vector(o_spi_si_ss_n'range);

    signal av_sc                    : work.util.avalon_t;

    signal sc_fifo_write            : std_logic;
    signal sc_fifo_wdata            : std_logic_vector(35 downto 0);

    signal sc_ram, sc_reg           : work.util.rw_t;
    signal fe_reg                   : work.util.rw_t;
    signal subdet_reg               : work.util.rw_t;

    signal reg_cmdlen               : std_logic_vector(31 downto 0);
    signal reg_offset               : std_logic_vector(31 downto 0);

    signal linktest_data            : std_logic_vector(31 downto 0);
    signal linktest_datak           : std_logic_vector(3 downto 0);
    signal linktest_granted         : std_logic_vector(N_LINKS-1 downto 0);

    signal av_mscb                  : work.util.avalon_t;

    signal reg_reset_bypass         : std_logic_vector(15 downto 0);
    signal reg_reset_bypass_payload : std_logic_vector(31 downto 0);

    signal run_state_125            : run_state_t;
    signal run_state_156            : run_state_t;
    signal run_state_156_resetsys   : run_state_t;

    signal terminated               : std_logic;
    signal reset_phase              : std_logic_vector(PHASE_WIDTH_g - 1 downto 0);

    signal run_number               : std_logic_vector(31 downto 0);
    signal merger_rate_count        : std_logic_vector(31 downto 0);

    signal av_ffly                  : work.util.avalon_t;

    signal ffly_rx_data             : std_logic_vector(127 downto 0);
    signal ffly_rx_datak            : std_logic_vector(15 downto 0);

    signal fpga_id_reg              : std_logic_vector(N_LINKS*16-1 downto 0);

    signal ffly_tx_data             : std_logic_vector(127 downto 0) :=
                                          X"000000" & work.util.D28_5
                                        & X"000000" & work.util.D28_5
                                        & X"000000" & work.util.D28_5
                                        & X"000000" & work.util.D28_5;
    signal ffly_tx_datak            : std_logic_vector(15 downto 0) :=
                                          "0001"
                                        & "0001"
                                        & "0001"
                                        & "0001";

    signal reset_link_rx            : std_logic_vector(7 downto 0);
    signal reset_link_rx_clk        : std_logic;

    signal arriaV_temperature       : std_logic_vector(7 downto 0);
    signal arriaV_temperature_clr   : std_logic;
    signal arriaV_temperature_ce    : std_logic;
    
    -- Max 10 SPI 
    signal adc_reg                  : reg32array_t(4 downto 0);
    signal i_adc_data_o             : std_logic_vector(31 downto 0);
    signal SPI_addr_o               : std_logic_vector(6 downto 0);
    
    signal SPI_command              : std_logic_vector (15 downto 0); -- [15-1] SPI inst [0] aktiv
    signal SPI_aktiv                : std_logic := '0';
    signal SPI_Adc_cnt              : unsigned(12 downto 0) := (others => '0');
    signal SPI_inst                 : std_logic_vector(14 downto 0) := X"00" & "000100" & '1'; --"[14-7] free [6-1] word cnt [0] R/W
    signal SPI_done                 : std_logic;
    signal SPI_rw                   : std_logic;


    signal max_spi_strobe           : std_logic;
    signal max_spi_addr             : std_logic_vector(6 downto 0);
    signal max_spi_rw               : std_logic;    
    signal max_spi_data_to_max      : std_logic_vector(31 downto 0);
    signal max_spi_numbytes         : std_logic_vector(8 downto 0);
    signal max_spi_next_data        : std_logic;
    signal max_spi_word_from_max    : std_logic_vector(31 downto 0);
    signal max_spi_word_en          : std_logic;
    signal max_spi_byte_from_max    : std_logic_vector(7 downto 0);
    signal max_spi_byte_en          : std_logic;
    signal max_spi_busy             : std_logic;

    signal max_spi_counter          : integer;
    signal max10_version            : reg32;
    signal max10_status             : reg32;	 
    signal max10_spiflash_cmdaddr   : reg32;

    type max_spi_state_t is (idle, programming, fifocopy, control, controlwait, programmingaddrwait, programmingaddr,
    flashwait, busyread, programmingend, maxversion, statuswait, maxstatus, adcwait, maxadc, endwait);
    signal max_spi_state :   max_spi_state_t;  
    signal program_req :   std_logic;
    signal flash_busy :    std_logic;
	 signal busy_last	: std_logic;

    signal programming_status   : reg32;
    signal programming_ctrl     : reg32;
    signal programming_data     : reg32;
    signal programming_data_ena : std_logic;
    signal programming_addr     : reg32;
    signal programming_addr_ena : std_logic;
    
    signal read_programming_fifo : std_logic;
    signal programming_data_from_fifo : reg32;


    signal wordcounter : integer;

begin

    --v_reg: version_reg 
    --    PORT MAP(
    --        data_out    => version_out(27 downto 0)
    --    );

    process(i_clk_156)
    begin
    if rising_edge(i_clk_156) then
        o_run_state_156 <= run_state_156_resetsys;
        run_state_156   <= run_state_156_resetsys;
    end if;
    end process;

    -- generate resets
    e_nios_reset_n : entity work.reset_sync
    port map ( o_reset_n => nios_reset_n, i_reset_n => i_areset_n, i_clk => i_nios_clk );

    e_reset_156_n : entity work.reset_sync
    port map ( o_reset_n => reset_156_n, i_reset_n => i_areset_n, i_clk => i_clk_156 );

    e_reset_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_125_n, i_reset_n => i_areset_n, i_clk => i_clk_125 );

    e_reset_line_125_n : entity work.reset_sync
    port map ( o_reset_n => reset_125_RRX_n, i_reset_n => i_areset_n, i_clk => reset_link_rx_clk);



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

    -- subdetector regs : 0x40-0xFF
    o_subdet_reg_addr <= sc_reg.addr(7 downto 0);
    o_subdet_reg_re <= subdet_reg.re;
      subdet_reg.re <= sc_reg.re when ( sc_reg.addr(REG_AREA_RANGE) /= REG_AREA_GENERIC ) else '0';
    o_subdet_reg_we <= sc_reg.we when ( sc_reg.addr(REG_AREA_RANGE) /= REG_AREA_GENERIC ) else '0';
    o_subdet_reg_wdata <= sc_reg.wdata;

    -- local regs 
    fe_reg.addr <= sc_reg.addr;
    fe_reg.re <= sc_reg.re when ( sc_reg.addr(REG_AREA_RANGE) = REG_AREA_GENERIC ) else '0';
    fe_reg.we <= sc_reg.we when ( sc_reg.addr(REG_AREA_RANGE) = REG_AREA_GENERIC ) else '0';
    fe_reg.wdata <= sc_reg.wdata;

    -- select valid rdata
    sc_reg.rdata <=
        i_subdet_reg_rdata when ( subdet_reg.rvalid = '1' ) else
        fe_reg.rdata when ( fe_reg.rvalid = '1' ) else
        X"CCCCCCCC";

    process(i_clk_156)
    begin
    if rising_edge(i_clk_156) then
        subdet_reg.rvalid   <= subdet_reg.re;
        fe_reg.rvalid       <= fe_reg.re;
    end if;
    end process;


    e_reg_mapping : entity work.feb_reg_mapping
    port map (
        i_clk_156                   => i_clk_156,
        i_reset_n                   => reset_156_n,

        i_reg_add                   => fe_reg.addr(7 downto 0),
        i_reg_re                    => fe_reg.re,
        o_reg_rdata                 => fe_reg.rdata,
        i_reg_we                    => fe_reg.we,
        i_reg_wdata                 => fe_reg.wdata,

        -- inputs  156--------------------------------------------
        -- ALL INPUTS DEFAULT TO (n*4-1 downto 0 => x"CCC..", others => '1')
        i_run_state_156             => run_state_156,
        i_merger_rate_count         => merger_rate_count,
        i_reset_phase               => reset_phase,
        i_arriaV_temperature        => arriaV_temperature,
        i_fpga_type                 => i_fpga_type,
        i_adc_reg                   => adc_reg,
        i_max10_version             => max10_version,
        i_max10_status              => max10_status,
        i_programming_status        => programming_status,

        -- outputs 156--------------------------------------------
        o_reg_cmdlen                => reg_cmdlen,
        o_reg_offset                => reg_offset,
        o_reg_reset_bypass          => reg_reset_bypass,
        o_reg_reset_bypass_payload  => reg_reset_bypass_payload,
        o_arriaV_temperature_clr    => arriaV_temperature_clr,
        o_arriaV_temperature_ce     => arriaV_temperature_ce,
        o_fpga_id_reg               => fpga_id_reg,
        o_programming_ctrl          => programming_ctrl,
        o_programming_data          => programming_data,
        o_programming_data_ena      => programming_data_ena,
        o_programming_addr          => programming_addr,
        o_programming_addr_ena      => programming_addr_ena--,
    );



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

        avm_qsfp_address        => av_ffly.address(13 downto 0),
        avm_qsfp_read           => av_ffly.read,
        avm_qsfp_readdata       => av_ffly.readdata,
        avm_qsfp_write          => av_ffly.write,
        avm_qsfp_writedata      => av_ffly.writedata,
        avm_qsfp_waitrequest    => av_ffly.waitrequest,

        --
        -- nios base
        --

        --i2c_scl_in => i_i2c_scl, -- not in use
        --i2c_scl_oe => o_i2c_scl_oe,
        --i2c_sda_in => i_i2c_sda,
        --i2c_sda_oe => o_i2c_sda_oe,

        spi_miso        => i_spi_miso,
        spi_mosi        => o_spi_mosi,
        spi_sclk        => o_spi_sclk,
        spi_ss_n        => o_spi_ss_n,

        spi_si_miso     => spi_si_miso,
        spi_si_mosi     => spi_si_mosi,
        spi_si_sclk     => spi_si_sclk,
        spi_si_ss_n     => spi_si_ss_n,

        pio_export      => nios_pio,

        temp_tsdcalo            => arriaV_temperature,
        temp_ce_ce              => arriaV_temperature_ce,
        temp_clr_reset          => arriaV_temperature_clr,
        temp_done_tsdcaldone    => open,

        rst_reset_n     => nios_reset_n,
        clk_clk         => i_nios_clk--,
    );



    e_sc_ram : entity work.sc_ram
    generic map (
        RAM_ADDR_WIDTH_g => to_integer(unsigned(FEB_SC_RAM_SIZE))--,--14--,
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
        i_link_data     => ffly_rx_data(32*(feb_mapping(0)+1)-1 downto 32*feb_mapping(0)),
        i_link_datak    => ffly_rx_datak(4*(feb_mapping(0)+1)-1 downto 4*feb_mapping(0)),

        o_fifo_we       => sc_fifo_write,
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
        fpga_ID_in                 => fpga_id_reg,
        FEB_type_in                => i_fpga_type,

        run_state                  => run_state_156,
        run_number                 => run_number,

        o_data_out                 => ffly_tx_data,
        o_data_is_k                => ffly_tx_datak,

        slowcontrol_write_req      => sc_fifo_write,
        i_data_in_slowcontrol      => sc_fifo_wdata,

        data_write_req             => i_fifo_write,
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
        i_data_125_rx           => reset_link_rx(7 downto 0),
        i_reset_125_rx_n        => reset_125_RRX_n,
        i_clk_125_rx            => reset_link_rx_clk,

        o_state_125             => run_state_125,
        i_reset_125_n           => reset_125_n,
        i_clk_125               => i_clk_125,

        o_state_156             => run_state_156_resetsys,
        i_reset_156_n           => reset_156_n,
        i_clk_156               => i_clk_156,

        resets_out              => open,
        reset_bypass            => reg_reset_bypass(11 downto 0),
        reset_bypass_payload    => reg_reset_bypass_payload,
        run_number_out          => run_number,
        fpga_id                 => fpga_id_reg(15 downto 0),
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

    firefly: entity work.firefly
    port map(
        i_clk                           => i_clk_156,
        i_sysclk                        => i_nios_clk,
        i_clk_i2c                       => i_nios_clk,
        o_clk_reco                      => reset_link_rx_clk,
        i_clk_lvds                      => i_clk_125,
        i_reset_n                       => nios_reset_n,
        i_reset_156_n                   => reset_156_n,
        i_reset_125_rx_n                => reset_125_RRX_n,
        i_lvds_align_reset_n            => i_testin,

        --rx
        i_data_fast_serial              => i_ffly2_rx & i_ffly1_rx,
        o_data_fast_parallel            => ffly_rx_data,
        o_datak                         => ffly_rx_datak,
        --tx
        o_data_fast_serial(3 downto 0)  => o_ffly1_tx,
        o_data_fast_serial(7 downto 4)  => o_ffly2_tx,
        i_data_fast_parallel            => ffly_tx_data & ffly_tx_data,
        i_datak                         => ffly_tx_datak & ffly_tx_datak,

        --lvds rx
        i_data_lvds_serial              => i_ffly2_lvds_rx & i_ffly1_lvds_rx,
        o_data_lvds_parallel(7 downto 0)=> reset_link_rx,
        o_data_lvds_parallel(15 downto 8)=>open,

        --I2C
        i_i2c_enable                    => '1',
        o_Mod_Sel_n                     => o_i2c_ffly_ModSel_n,
        o_Rst_n                         => o_ffly_Rst_n,
        io_scl                          => io_i2c_ffly_scl,
        io_sda                          => io_i2c_ffly_sda,
        i_int_n                         => i_ffly_Int_n,
        i_modPrs_n                      => i_ffly_ModPrs_n,

        --Avalon
        i_avs_address                   => av_ffly.address(13 downto 0),
        i_avs_read                      => av_ffly.read,
        o_avs_readdata                  => av_ffly.readdata,
        i_avs_write                     => av_ffly.write,
        i_avs_writedata                 => av_ffly.writedata,
        o_avs_waitrequest               => av_ffly.waitrequest,

        o_testclkout                    => open,
        o_testout                       => open--,
    );


    e_max10_spi : entity work.max10_spi
        port map(
        -- Max10 SPI
        o_SPI_csn           => o_max10_spi_csn,
        o_SPI_clk           => o_max10_spi_sclk,
        io_SPI_mosi         => io_max10_spi_mosi,
        io_SPI_miso         => io_max10_spi_miso,
        io_SPI_D1           => io_max10_spi_D1,
        io_SPI_D2           => io_max10_spi_D2,
        io_SPI_D3           => io_max10_spi_D3,
    
        -- Interface to Arria
        clk50               => i_nios_clk,
        reset_n             => nios_reset_n,
        strobe              => max_spi_strobe,
        addr                => max_spi_addr,
        rw                  => max_spi_rw, 
        data_to_max         => max_spi_data_to_max, 
        numbytes            => max_spi_numbytes,
        next_data           => max_spi_next_data,
        word_from_max       => max_spi_word_from_max,
        word_en             => max_spi_word_en,
        byte_from_max       => max_spi_byte_from_max,
        byte_en             => max_spi_byte_en,
        busy                => max_spi_busy
        );



    program_req <= programming_ctrl(0);

    process(i_nios_clk, nios_reset_n)
    begin
    if(nios_reset_n = '0')then
        max_spi_strobe 	 <= '0';
        max_spi_counter  <= 0;
        max_spi_state    <= idle;
		max_spi_numbytes <= "000000000";
		max_spi_rw		 <= '0';
        read_programming_fifo   <= '0';
		  busy_last	<= '0';
		  programming_status(0) <= '0';
		  programming_status(1) <= '0';
		  flash_busy 				<= '0';
		  
    elsif(i_nios_clk'event and i_nios_clk = '1')then
        max_spi_counter <= max_spi_counter + 1;
        max_spi_strobe <= '0';
		

        programming_status(0) <= '0';
		  programming_status(31 downto 4) <= (others => '0');

        read_programming_fifo <= '0';
		  
		  busy_last		<= max_spi_busy;
		  
        case max_spi_state is
        when idle =>
				programming_status(4) <= '1';
				max_spi_rw		 <= '0';
            if(program_req = '1') then
                max_spi_state <= programming;
            elsif(max_spi_counter > 5000000) then
                max_spi_state <= maxversion;
                max_spi_counter <= 0;
            end if;  
				programming_status(0) <= '0';
				programming_status(1) <= '0';
		
        when programming =>
				programming_status(5) <= '1';
            -- Idea for programming:
            -- - go to programming state by setting programming_ctrl(0) = 1
            -- - Fill at least 256 byte into the programming FIFO
            -- - Set the address - this will copy the FIFO to the MAX, then trigger
            --   flash programming
            -- - Status is polled and reflected in MAX10 status until busy goes low
            -- - Rinse and repeat

            programming_status(0) <= '1';
				
            if(program_req = '0') then
                max_spi_state <= idle;
            end if;    

            if(programming_addr_ena = '1') then
                max_spi_state <= fifocopy;
					 programming_status(1) <= '1';
            end if;    

        when fifocopy =>
				programming_status(6) <= '1';
            max_spi_addr    <= FEBSPI_ADDR_PROGRAMMING_WFIFO;
            max_spi_numbytes <= "100000000";
            max_spi_strobe   <= '1';
            max_spi_rw       <= '1';
            max_spi_data_to_max     <= programming_data_from_fifo;
            read_programming_fifo   <= max_spi_next_data;
            max_spi_state    			<= programmingaddrwait;
							
			when programmingaddrwait =>
				programming_status(7) <= '1';
				max_spi_data_to_max     <= programming_data_from_fifo;
            read_programming_fifo   <= max_spi_next_data;
            if(max_spi_busy = '0' and busy_last = '1') then
                max_spi_rw       <= '0';
                max_spi_strobe   <= '0';
                max_spi_state    <= programmingaddr;
					 read_programming_fifo   <= '0';
            end if;	
				
        when programmingaddr =>
				programming_status(8) <= '1';
            max_spi_addr    <= FEBSPI_ADDR_PROGRAMMING_ADDR;
            max_spi_numbytes <= "000000100";
            max_spi_strobe   <= '1';
            max_spi_rw       <= '1';
            max_spi_data_to_max     <= programming_addr;
            max_spi_state    			<= controlwait;
            				
		 when controlwait =>
				programming_status(9) <= '1';
            if(max_spi_busy = '0' and busy_last = '1') then
                max_spi_rw       <= '0';
                max_spi_strobe   <= '0';
                max_spi_state    <= control;
            end if;
			when control =>
				programming_status(10) <= '1';
				max_spi_addr    <= FEBSPI_ADDR_PROGRAMMING_CTRL;
            max_spi_numbytes <= "000000100";
            max_spi_strobe   <= '1';
            max_spi_rw       <= '1';
            max_spi_data_to_max     <= X"00000001";
            max_spi_state    <= flashwait;		
				flash_busy       <= '1';
				
        when flashwait =>
				programming_status(11) <= '1';
            if(max_spi_busy = '0' and busy_last = '1') then
                max_spi_rw       <= '0';
                max_spi_strobe   <= '0';
                if(flash_busy = '1') then
                    max_spi_state    <= busyread;
                else
                    max_spi_state    <= programmingend;
                end if;
            end if;
        when busyread =>
				programming_status(12) <= '1';
            max_spi_addr    <= FEBSPI_ADDR_PROGRAMMING_STATUS;
            max_spi_numbytes <= "000000100";
            max_spi_strobe   <= '1';
            if(max_spi_word_en = '1') then
                flash_busy   <= max_spi_word_from_max(0);
                max_spi_strobe   <= '0';
                max_spi_state    <= flashwait;
            end if;
        when programmingend =>
				programming_status(13) <= '1';
            max_spi_addr    <= FEBSPI_ADDR_PROGRAMMING_CTRL;
            max_spi_numbytes <= "000000100";
            max_spi_strobe   <= '1';
            max_spi_rw       <= '1';
            max_spi_data_to_max     <= (others => '0');
				if(max_spi_busy = '0' and busy_last = '1') then
                max_spi_rw       <= '0';
                max_spi_strobe   <= '0';
                max_spi_state    <= programming;
					 programming_status(1) <= '0';
            end if;	
				
				
        when maxversion =>
				programming_status(14) <= '1';
            max_spi_addr    <= FEBSPI_ADDR_GITHASH;
            max_spi_numbytes <= "000000100";
            max_spi_strobe   <= '1';
            if(max_spi_word_en = '1') then
                max10_version   <= max_spi_word_from_max;
                max_spi_strobe   <= '0';
                max_spi_state    <= statuswait;
            end if;
		  when statuswait =>
				programming_status(15) <= '1';
				if(max_spi_busy = '0')then
					max_spi_state    <= maxstatus;
				end if;
        when maxstatus =>
				programming_status(16) <= '1';
            max_spi_addr    <= FEBSPI_ADDR_STATUS;
            max_spi_numbytes <= "000000100";
            max_spi_strobe   <= '1';
            if(max_spi_word_en = '1') then
                max10_status   <= max_spi_word_from_max;
                max_spi_strobe   <= '0';
                max_spi_state    <= adcwait;
                wordcounter      <= 0;
            end if;  
		  when adcwait =>
				programming_status(17) <= '1';
				if(max_spi_busy = '0')then
					max_spi_state    <= maxadc;
				end if;
        when maxadc =>
				programming_status(18) <= '1';
            max_spi_addr    <= FEBSPI_ADDR_ADCDATA;
            max_spi_numbytes <= "000010100";
            max_spi_strobe   <= '1';   
            if(max_spi_word_en = '1') then
                wordcounter <= wordcounter + 1;
                if(wordcounter = 0) then
                    adc_reg(0)   <= max_spi_word_from_max;
                elsif(wordcounter = 1) then
                    adc_reg(1)   <= max_spi_word_from_max;
                elsif(wordcounter = 2) then
                    adc_reg(2)   <= max_spi_word_from_max; 
				elsif(wordcounter = 3) then
                    adc_reg(4)   <= max_spi_word_from_max; 
                    max_spi_strobe   <= '0';
                    max_spi_state    <= endwait;
                end if;
            end if;
			when endwait =>
				programming_status(19) <= '1';
				if(max_spi_busy = '0')then
					max_spi_state    <= idle;
				end if;	
        when others =>
				programming_status(20) <= '1';
            max_spi_state <= idle;     
            max_spi_strobe <= '0';
        end case;
    end if;
    end process;

    scfifo_component : altera_mf.altera_mf_components.dcfifo
    GENERIC MAP (
            add_ram_output_register => "ON",
            intended_device_family => "Arria V",
            lpm_numwords => 512,
            lpm_showahead => "ON",
            lpm_type => "dcfifo",
            lpm_width => 32,
            lpm_widthu => 9,
            overflow_checking => "ON",
            underflow_checking => "ON",
            use_eab => "ON"
    )
    PORT MAP (
            wrclk => i_clk_156,
            rdclk => i_nios_clk,
            data => programming_data, 
            rdreq => read_programming_fifo,
            aclr => programming_ctrl(1),
            wrreq => programming_data_ena,
            rdempty => programming_status(2),
            rdfull  => programming_status(3),
            q => programming_data_from_fifo
    );

end architecture;
