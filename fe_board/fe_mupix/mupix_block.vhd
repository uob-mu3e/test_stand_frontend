-- last change: M.Mueller, November 2020 (muellem@uni-mainz.de)

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;

use work.mupix_registers.all;
use work.mudaq.all;
use work.mupix.all;

entity mupix_block is
port (
    i_fpga_id               : in  std_logic_vector(7 downto 0);

    -- config signals to mupix
    o_clock                 : out std_logic_vector( 3 downto 0);
    o_SIN                   : out std_logic_vector( 3 downto 0);
    o_mosi                  : out std_logic_vector( 3 downto 0);
    o_csn                   : out std_logic_vector(11 downto 0);

    i_run_state_125           : in  run_state_t;
    i_run_state_156           : in  run_state_t;
    o_ack_run_prep_permission : out std_logic;

    -- mupix dac regs
    i_reg_add               : in  std_logic_vector(7 downto 0);
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
    o_lvds_invert_mon       : out std_logic;

    i_reset                 : in  std_logic;
    -- 156.25 MHz
    i_clk156                   : in  std_logic;
    i_clk125                : in  std_logic;
    i_lvds_rx_inclock_A     : in  std_logic;
    i_lvds_rx_inclock_B     : in  std_logic;
    i_sync_reset_cnt        : in  std_logic--;
);
end entity;

architecture arch of mupix_block is

    signal datapath_reset_n             : std_logic;
    signal reg_valid                    : std_logic := '0';
    signal reg_rdata                    : std_logic_vector(31 downto 0);
    signal reg_rdata_datapath           : std_logic_vector(31 downto 0);

    signal spi_clock        : std_logic;
    signal spi_mosi         : std_logic;
    signal spi_csn          : std_logic;
    signal hotfix : work.util.slv32_array_t(35 downto 0);
    signal hotfix_back      : std_logic;

begin

    datapath_reset_n <= '0' when (i_reset='1' or i_run_state_156=RUN_STATE_SYNC) else '1';
    o_lvds_invert_mon <= hotfix_back;

    process(i_clk156,i_reset)
    begin
        if(i_reset = '1') then 
            reg_valid <= '0';
        elsif(rising_edge(i_clk156)) then
            reg_valid <= i_reg_re; -- reg_rdata from datapath is valid after 1 cycle
        end if;
    end process;

    --o_reg_rdata <= reg_rdata_datapath when (unsigned(i_reg_add) >= MUPIX_DATAPATH_ADDR_START and reg_valid = '1') else reg_rdata;
    o_reg_rdata <= reg_rdata;


    e_mupix_ctrl : work.mupix_ctrl
    port map (
        i_clk                       => i_clk156,
        i_reset_n                   => not i_reset,

        i_reg_add                   => i_reg_add,
        i_reg_re                    => i_reg_re,
        o_reg_rdata                 => reg_rdata,
        i_reg_we                    => i_reg_we,
        i_reg_wdata                 => i_reg_wdata,

        i_hotfix_reroute            => hotfix,
        o_hotfix_backroute          => hotfix_back,
        o_clock                     => o_clock,
        o_SIN                       => o_SIN,
        o_mosi                      => o_mosi,
        o_csn                       => o_csn--,
    );

    e_mupix_datapath : work.mupix_datapath
    port map (
        i_reset_n           => datapath_reset_n,
        i_reset_n_regs      => not i_reset,
        i_reset_n_lvds      => not i_reset,--'1',--reset_n_lvds,todo: reg

        i_clk156            => i_clk156,
        i_clk125            => i_clk125,

        i_lvds_rx_inclock_A => i_lvds_rx_inclock_A,
        i_lvds_rx_inclock_B => i_lvds_rx_inclock_B,

        lvds_data_in        => i_lvds_data_in,

        i_reg_add           => i_reg_add,
        i_reg_re            => i_reg_re,
        o_reg_rdata         => reg_rdata_datapath,
        i_reg_we            => i_reg_we,
        i_reg_wdata         => i_reg_wdata,

        o_fifo_wdata        => o_fifo_wdata,
        o_fifo_write        => o_fifo_write,

        o_data_bypass       => o_data_bypass,
        o_data_bypass_we    => o_data_bypass_we,

        i_sync_reset_cnt    => i_sync_reset_cnt,
        i_fpga_id           => i_fpga_id,
        i_run_state_125     => i_run_state_125,
        i_run_state_156     => i_run_state_156,
        o_hotfix_reroute    => hotfix,
        i_hotfix_backroute  => hotfix_back--,
    );

end architecture;
