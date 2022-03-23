library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.a10_counters.all;
use work.a10_pcie_registers.all;

entity swb_readout_counters is
generic (
    g_A_CNT             : positive := 4;
    g_NLINKS_DATA_SCIFI : positive := 4;
    g_NLINKS_DATA_PIXEL : positive := 10--;
);
port (
    --! register inputs for pcie0
    i_wregs_add         : in    std_logic_vector(31 downto 0);

    --! counters
    i_counter           : in    work.util.slv32_array_t(g_A_CNT - 1 downto 0);

    --! register outputs for pcie0
    o_pcie_data         : out   std_logic_vector(31 downto 0);
    o_pcie_addr         : out   std_logic_vector(31 downto 0);

    --! i_reset
    i_reset_n           : in    std_logic;

    --! clocks
    i_clk               : in    std_logic--;

);
end entity;

--! @brief arch definition of the a10_readout_counters
--! @details The arch of the a10_readout_counters sync
--! the three clk domains used in the A10 board and outputs
--! the counters for a given input addr
architecture arch of swb_readout_counters is

    signal swb_counter_addr, link_id, link_counter_addr : integer := 0;

begin

    swb_counter_addr <= to_integer(unsigned(i_wregs_add(SWB_COUNTER_ADDR_RANGE)));
    link_id <= to_integer(unsigned(i_wregs_add(SWB_LINK_RANGE)));
    link_counter_addr <= swb_counter_addr + link_id * 5;

    --! map counters pixel
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        o_pcie_data <= (others => '0');
        o_pcie_addr <= (others => '0');
        --
    elsif ( rising_edge(i_clk) ) then
        o_pcie_addr <= i_wregs_add;
        case swb_counter_addr is
        when SWB_STREAM_FIFO_FULL_PIXEL_CNT | SWB_BANK_BUILDER_IDLE_NOT_HEADER_PIXEL_CNT | SWB_BANK_BUILDER_RAM_FULL_PIXEL_CNT | SWB_BANK_BUILDER_TAG_FIFO_FULL_PIXEL_CNT =>
            o_pcie_data <= i_counter(swb_counter_addr);
        when SWB_LINK_FIFO_ALMOST_FULL_PIXEL_CNT | SWB_LINK_FIFO_FULL_PIXEL_CNT | SWB_SKIP_EVENT_PIXEL_CNT | SWB_EVENT_PIXEL_CNT | SWB_SUB_HEADER_PIXEL_CNT =>
            o_pcie_data <= i_counter(link_counter_addr);
        when others =>
            null;
        end case;

    end if;
    end process;

end architecture;
