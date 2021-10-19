library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;

entity tb_sc_new is
end entity;

architecture rtl of tb_sc_new is
signal clk          : std_logic := '1';
signal reset_n      : std_logic;
signal data_in      : std_logic_vector(31 downto 0) := (others => '0');
signal datak_in     : std_logic_vector( 3 downto 0) := (others => '0');

signal data_out     : std_logic_vector(35 downto 0) := (others => '0');
signal data_out_we  : std_logic := '0';

signal sc_ram       : work.util.rw_t;
signal fe_reg       : work.util.rw_t;
signal sc_reg       : work.util.rw_t;
signal subdet_reg   : work.util.rw_t;

signal av_sc_address    : std_logic_vector(15 downto 0) := (others => '0');
signal av_sc_read       : std_logic := '0';
signal av_sc_readdata   : std_logic_vector(31 downto 0) := (others => '0');
signal av_sc_write      : std_logic := '0';
signal av_sc_writedata  : std_logic_vector(31 downto 0) := (others => '0');
signal av_sc_waitrequest: std_logic := '0';

begin

    clk     <= not clk after (500 ns / 50);
    reset_n <= '0', '1' after 32 ns;

    e_sc_rx : entity work.sc_rx
    port map (
        i_link_data     => data_in,
        i_link_datak    => datak_in,

        o_fifo_we       => data_out_we,
        o_fifo_wdata    => data_out,

        o_ram_addr      => sc_ram.addr,
        o_ram_re        => sc_ram.re,
        i_ram_rvalid    => sc_ram.rvalid,
        i_ram_rdata     => sc_ram.rdata,
        o_ram_we        => sc_ram.we,
        o_ram_wdata     => sc_ram.wdata,

        i_reset_n       => reset_n,
        i_clk           => clk--,
    );

    e_sc_ram : entity work.sc_ram_new
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

        i_avs_address           => av_sc_address(15 downto 0),
        i_avs_read              => av_sc_read,
        o_avs_readdata          => av_sc_readdata,
        i_avs_write             => av_sc_write,
        i_avs_writedata         => av_sc_writedata,
        o_avs_waitrequest       => av_sc_waitrequest,

        o_reg_addr              => sc_reg.addr(7 downto 0),
        o_reg_re                => sc_reg.re,
        i_reg_rdata             => sc_reg.rdata,
        o_reg_we                => sc_reg.we,
        o_reg_wdata             => sc_reg.wdata,

        i_reset_n               => reset_n,
        i_clk                   => clk--;
    );

    e_sc_node: entity work.sc_node
    generic map(
        ADD_SLAVE0_DELAY_g  => 1,
        ADD_SLAVE1_DELAY_g  => 1,
        N_REPLY_CYCLES_g    => 2,
        SLAVE0_ADDR_MATCH_g => "00------"
    )
    port map(
        i_clk           => clk,
        i_reset_n       => reset_n,

        -- to upper SC nodes 0x00-0xFF
        i_master_addr   => sc_reg.addr(7 downto 0),
        i_master_re     => sc_reg.re,
        o_master_rdata  => sc_reg.rdata,
        i_master_we     => sc_reg.we,
        i_master_wdata  => sc_reg.wdata,

        -- to feb common regs 0x00-0x3F
        o_slave0_addr   => fe_reg.addr(7 downto 0),
        o_slave0_re     => fe_reg.re,
        i_slave0_rdata  => fe_reg.rdata,
        o_slave0_we     => fe_reg.we,
        o_slave0_wdata  => fe_reg.wdata,

        -- to subdetector regs 0x40-0xFF
        o_slave1_addr   => subdet_reg.addr(7 downto 0),
        o_slave1_re     => subdet_reg.re,
        i_slave1_rdata  => subdet_reg.rdata,
        o_slave1_we     => subdet_reg.we,
        o_slave1_wdata  => subdet_reg.wdata
    );

    e_reg_mapping : entity work.feb_reg_mapping
    port map (
        i_clk_156                   => clk,
        i_reset_n                   => reset_n,

        i_reg_add                   => fe_reg.addr(7 downto 0),
        i_reg_re                    => fe_reg.re,
        o_reg_rdata                 => fe_reg.rdata,
        i_reg_we                    => fe_reg.we,
        i_reg_wdata                 => fe_reg.wdata,

        -- inputs  156--------------------------------------------
        -- ALL INPUTS DEFAULT TO (n*4-1 downto 0 => x"CCC..", others => '1')
        i_run_state_156             => (others => '0'),
        i_merger_rate_count         => (others => '0'),
        i_reset_phase               => (others => '0'),
        i_arriaV_temperature        => (others => '0'),
        i_fpga_type                 => (others => '0'),
        i_max10_status              => (others => '0'),
        i_programming_status        => (others => '0'),

        i_ffly_pwr                  => (others => '0'),
        i_ffly_temp                 => (others => '0'),
        i_ffly_alarm                => (others => '0'),
        i_ffly_vcc                  => (others => '0'),
        
        i_si45_intr_n               => (others => '0'),
        i_si45_lol_n                => (others => '0')--,
    );

    process
    begin
        wait;
    end process;

end architecture;
