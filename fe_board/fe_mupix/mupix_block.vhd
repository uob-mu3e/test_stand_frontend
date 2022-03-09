-- mupix block of FEB firmware 
-- M. Mueller

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mupix_registers.all;
use work.mudaq.all;
use work.mupix.all;

entity mupix_block is
    generic(
        IS_TELESCOPE_g : std_logic := '0';
        LINK_ORDER_g : mp_link_order_t := MP_LINK_ORDER--;
    );
port (
    i_fpga_id               : in  std_logic_vector(7 downto 0);

    -- config signals to mupix
    o_clock                 : out std_logic_vector( 3 downto 0);
    o_SIN                   : out std_logic_vector( 3 downto 0);
    o_mosi                  : out std_logic_vector( 3 downto 0);
    o_csn                   : out std_logic_vector(11 downto 0);

    i_run_state_125           : in  run_state_t;
    i_run_state_156           : in  run_state_t;
    o_ack_run_prep_permission : out std_logic :='1';

    -- mupix dac regs
    i_reg_add               : in  std_logic_vector(15 downto 0);
    i_reg_re                : in  std_logic;
    o_reg_rdata             : out std_logic_vector(31 downto 0);
    i_reg_we                : in  std_logic;
    i_reg_wdata             : in  std_logic_vector(31 downto 0);

    -- data 
    o_fifo_wdata            : out std_logic_vector(35 downto 0);
    o_fifo_write            : out std_logic;

    o_data_bypass           : out std_logic_vector(31 downto 0);
    o_data_bypass_we        : out std_logic;

    i_lvds_data_in          : in  std_logic_vector(35 downto 0);

    i_reset                 : in  std_logic;
    -- 156.25 MHz
    i_clk156                   : in  std_logic;
    i_clk125                : in  std_logic;
    i_lvds_rx_inclock_A     : in  std_logic;
    i_lvds_rx_inclock_B     : in  std_logic;
    i_sync_reset_cnt        : in  std_logic;

    i_trigger_in0           : in  std_logic := '0';
    i_trigger_in1           : in  std_logic := '0';
    i_trigger_in0_timestamp : in  std_logic_vector(31 downto 0) := (others => '0');
    i_trigger_in1_timestamp : in  std_logic_vector(31 downto 0) := (others => '0')--;
);
end entity;

architecture arch of mupix_block is

    signal datapath_reset_n : std_logic;

    signal mp_ctrl_reg      : work.util.rw_t;
    signal mp_datapath_reg  : work.util.rw_t;

    signal iram_addr    : std_logic_vector(15 downto 0) := (others => '0');
    signal iram_we      : std_logic := '0';
    signal iram_rdata   : std_logic_vector(31 downto 0) := (others => '0'); 
    signal iram_wdata   : std_logic_vector(31 downto 0) := (others => '0');

begin

    datapath_reset_n <= '0' when (i_reset='1' or i_run_state_156=RUN_STATE_SYNC) else '1';

    e_mupix_ctrl : work.mupix_ctrl
    port map (
        i_clk                       => i_clk156,
        i_reset_n                   => not i_reset,

        i_reg_add                   => mp_ctrl_reg.addr(15 downto 0),
        i_reg_re                    => mp_ctrl_reg.re,
        o_reg_rdata                 => mp_ctrl_reg.rdata,
        i_reg_we                    => mp_ctrl_reg.we,
        i_reg_wdata                 => mp_ctrl_reg.wdata,

        o_clock                     => o_clock,
        o_SIN                       => o_sin,
        o_mosi                      => o_mosi,
        o_csn                       => o_csn--,
    );

    e_mupix_datapath : work.mupix_datapath
    generic map (
        IS_TELESCOPE_g  => IS_TELESCOPE_g,
        LINK_ORDER_g => LINK_ORDER_g--,
    )
    port map (
        i_reset_n           => datapath_reset_n,
        i_reset_n_regs      => not i_reset,

        i_clk156            => i_clk156,
        i_clk125            => i_clk125,

        i_lvds_rx_inclock_A => i_lvds_rx_inclock_A,
        i_lvds_rx_inclock_B => i_lvds_rx_inclock_B,

        lvds_data_in        => i_lvds_data_in,

        i_reg_add           => mp_datapath_reg.addr(15 downto 0),
        i_reg_re            => mp_datapath_reg.re,
        o_reg_rdata         => mp_datapath_reg.rdata,
        i_reg_we            => mp_datapath_reg.we,
        i_reg_wdata         => mp_datapath_reg.wdata,

        o_fifo_wdata        => o_fifo_wdata,
        o_fifo_write        => o_fifo_write,

        o_data_bypass       => o_data_bypass,
        o_data_bypass_we    => o_data_bypass_we,

        i_sync_reset_cnt    => i_sync_reset_cnt,
        i_fpga_id           => i_fpga_id,
        i_run_state_125     => i_run_state_125,
        i_run_state_156     => i_run_state_156,
        
        i_trigger_in0       => i_trigger_in0,
        i_trigger_in1       => i_trigger_in1,
        i_trigger_in0_timestamp => i_trigger_in0_timestamp,
        i_trigger_in1_timestamp => i_trigger_in1_timestamp--,
    );

    e_lvl1_sc_node: entity work.sc_node
    generic map (
        SLAVE1_ADDR_MATCH_g => "000000----------",
        SLAVE2_ADDR_MATCH_g => "0000------------",
        ADD_SLAVE1_DELAY_g  => 3,
        ADD_SLAVE2_DELAY_g  => 3,
        N_REPLY_CYCLES_g    => 4--,
    )   
    port map (
        i_clk          => i_clk156,
        i_reset_n      => not i_reset,

        i_master_addr  => i_reg_add,
        i_master_re    => i_reg_re,
        o_master_rdata => o_reg_rdata,
        i_master_we    => i_reg_we,
        i_master_wdata => i_reg_wdata,

        o_slave0_addr  => mp_datapath_reg.addr(15 downto 0),
        o_slave0_re    => mp_datapath_reg.re,
        i_slave0_rdata => mp_datapath_reg.rdata,
        o_slave0_we    => mp_datapath_reg.we,
        o_slave0_wdata => mp_datapath_reg.wdata,

        o_slave1_addr  => iram_addr,
        o_slave1_re    => open,
        i_slave1_rdata => iram_rdata,
        o_slave1_we    => iram_we,
        o_slave1_wdata => iram_wdata,

        o_slave2_addr  => mp_ctrl_reg.addr(15 downto 0),
        o_slave2_re    => mp_ctrl_reg.re,
        i_slave2_rdata => mp_ctrl_reg.rdata,
        o_slave2_we    => mp_ctrl_reg.we,
        o_slave2_wdata => mp_ctrl_reg.wdata--,
    );

    e_iram : entity work.ram_1r1w
    generic map (
        g_DATA_WIDTH => 32,
        g_ADDR_WIDTH => 10--,
    )
    port map (
        i_raddr => iram_addr(10-1 downto 0),
        o_rdata => iram_rdata,
        i_rclk  => i_clk156,

        i_waddr => iram_addr(10-1 downto 0),
        i_wdata => iram_wdata,
        i_we    => iram_we,
        i_wclk  => i_clk156--,
    );

end architecture;
