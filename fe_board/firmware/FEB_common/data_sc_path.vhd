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

    signal ram_addr_a : std_logic_vector(31 downto 0);
    signal ram_rdata_a : std_logic_vector(31 downto 0);
    signal ram_wdata_a : std_logic_vector(31 downto 0);
    signal ram_we_a : std_logic;

    signal ram_re_a, ram_rvalid_a : std_logic;

    signal data_to_fifo : std_logic_vector(35 downto 0);
    signal data_to_fifo_we : std_logic;
    signal data_from_fifo : std_logic_vector(35 downto 0);
    signal data_from_fifo_re : std_logic;
    signal data_from_fifo_empty : std_logic;

    signal sc_to_fifo : std_logic_vector(35 downto 0);
    signal sc_to_fifo_we : std_logic;
    signal sc_from_fifo : std_logic_vector(35 downto 0);
    signal sc_from_fifo_re : std_logic;
    signal sc_from_fifo_empty : std_logic;

begin

    ----------------------------------------------------------------------------
    -- SLOW CONTROL

    e_sc_ram : entity work.ip_ram
    generic map (
        ADDR_WIDTH => SC_RAM_WIDTH_g,
        DATA_WIDTH => 32--,
    )
    port map (
        address_b   => i_avs_address(SC_RAM_WIDTH_g-1 downto 0),
        q_b         => o_avs_readdata,
        wren_b      => i_avs_write,
        data_b      => i_avs_writedata,
        clock_b     => i_clk,

        address_a   => ram_addr_a(SC_RAM_WIDTH_g-1 downto 0),
        q_a         => ram_rdata_a,
        wren_a      => ram_we_a,
        data_a      => ram_wdata_a,
        clock_a     => i_clk--,
    );
    o_avs_waitrequest <= '0';

    process(i_clk)
    begin
    if rising_edge(i_clk) then
        ram_rvalid_a <= ram_re_a;
    end if;
    end process;

    e_sc : entity work.sc_rx
    port map (
        i_link_data => i_link_data,
        i_link_datak => i_link_datak,

        o_fifo_we => sc_to_fifo_we,
        o_fifo_wdata => sc_to_fifo,

        o_ram_addr      => ram_addr_a,
        o_ram_re        => ram_re_a,
        i_ram_rdata     => ram_rdata_a,
        i_ram_rvalid    => ram_rvalid_a,
        o_ram_we        => ram_we_a,
        o_ram_wdata     => ram_wdata_a,

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

        data_in_slowcontrol     => sc_from_fifo,
        slowcontrol_fifo_empty  => sc_from_fifo_empty,
        slowcontrol_read_req    => sc_from_fifo_re,

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

    e_sc_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 10,
        DATA_WIDTH => 36--,
    )
    port map (
        clock           => i_clk,
        data            => sc_to_fifo,
        rdreq           => sc_from_fifo_re,
        sclr            => i_reset,
        wrreq           => sc_to_fifo_we,
        almost_empty    => open,
        almost_full     => open,
        empty           => sc_from_fifo_empty,
        full            => open,
        q               => sc_from_fifo,
        usedw           => open--,
    );

    ----------------------------------------------------------------------------

end architecture;
