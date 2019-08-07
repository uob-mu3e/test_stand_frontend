library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

entity data_sc_path is
generic (
    SC_RAM_WIDTH_g : positive := 14--;
);
port (
    i_avs_address       : in    std_logic_vector(15 downto 0);
    i_avs_read          : in    std_logic;
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic;
    i_avs_writedata     : in    std_logic_vector(31 downto 0);
    o_avs_waitrequest   : out   std_logic;

    i_fifo_data         : in    std_logic_vector(35 downto 0);
    i_fifo_empty        : in    std_logic;
    o_fifo_rack         : out   std_logic;

    i_link_data         : in    std_logic_vector(31 downto 0);
    i_link_datak        : in    std_logic_vector(3 downto 0);

    o_link_data         : out   std_logic_vector(31 downto 0);
    o_link_datak        : out   std_logic_vector(3 downto 0);

    o_terminated        : out   std_logic;
    i_run_state         : in    run_state_t;

    i_reset             : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of data_sc_path is

    signal ram_addr : std_logic_vector(31 downto 0);
    signal ram_re : std_logic;
    signal ram_rdata : std_logic_vector(31 downto 0);
    signal ram_rvalid : std_logic;
    signal ram_we : std_logic;
    signal ram_wdata : std_logic_vector(31 downto 0);

    signal sc_fifo_rempty : std_logic;
    signal sc_fifo_rdata : std_logic_vector(35 downto 0);
    signal sc_fifo_rack : std_logic;

begin

    ----------------------------------------------------------------------------
    -- SLOW CONTROL

    e_sc_ram : entity work.sc_ram
    port map (
        i_ram_addr          => ram_addr(15 downto 0),
        i_ram_re            => ram_re,
        o_ram_rdata         => ram_rdata,
        o_ram_rvalid        => ram_rvalid,
        i_ram_we            => ram_we,
        i_ram_wdata         => ram_wdata,

        i_avs_address       => i_avs_address,
        i_avs_read          => i_avs_read,
        o_avs_readdata      => o_avs_readdata,
        i_avs_write         => i_avs_write,
        i_avs_writedata     => i_avs_writedata,
        o_avs_waitrequest   => o_avs_waitrequest,

--        o_reg_addr          => o_sc_reg_addr,
--        o_reg_re            => open,
--        i_reg_rdata         => i_sc_reg_rdata,
--        o_reg_we            => o_sc_reg_we,
--        o_reg_wdata         => o_sc_reg_wdata,

        i_reset_n           => not i_reset,
        i_clk               => i_clk--;
    );



    e_sc : entity work.sc_rx
    port map (
        i_link_data => i_link_data,
        i_link_datak => i_link_datak,

        o_fifo_rempty   => sc_fifo_rempty,
        o_fifo_rdata    => sc_fifo_rdata,
        i_fifo_rack     => sc_fifo_rack,

        o_ram_addr      => ram_addr,
        o_ram_re        => ram_re,
        i_ram_rdata     => ram_rdata,
        i_ram_rvalid    => ram_rvalid,
        o_ram_we        => ram_we,
        o_ram_wdata     => ram_wdata,

        i_reset_n => not i_reset,
        i_clk => i_clk--,
    );

    ----------------------------------------------------------------------------

    e_merger : entity work.data_merger
    port map (
        fpga_ID_in              => (5=>'1',others => '0'),
        FEB_type_in             => "111010",
        run_state               => i_run_state,

        data_out                => o_link_data(31 downto 0),
        data_is_k               => o_link_datak(3 downto 0),

        slowcontrol_fifo_empty  => sc_fifo_rempty,
        data_in_slowcontrol     => sc_fifo_rdata,
        slowcontrol_read_req    => sc_fifo_rack,

        data_in                 => i_fifo_data,
        data_fifo_empty         => i_fifo_empty,
        data_read_req           => o_fifo_rack,

        override_data_in        => (others => '0'),
        override_data_is_k_in   => (others => '0'),
        override_req            => '0',
        override_granted        => open,

        terminated              => o_terminated,
        data_priority           => '0',

        leds                    => open,

        reset                   => i_reset,
        clk                     => i_clk--,
    );

    ----------------------------------------------------------------------------

end architecture;
