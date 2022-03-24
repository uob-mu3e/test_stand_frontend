library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.a10_counters.all;
use work.a10_pcie_registers.all;

entity swb_readout_counters is
generic (
    g_A_CNT                 : positive := 4;
    g_NLINKS_DATA_PIXEL_US  : positive := 5;
    g_NLINKS_DATA_PIXEL_DS  : positive := 5;
    g_NLINKS_DATA_SCIFI     : positive := 5--;
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

    signal swb_counter_addr, link_id, link_counter_addr, link_addr, datapath_counter_addr : integer := 0;

begin

    link_id <= to_integer(unsigned(i_wregs_add(SWB_LINK_RANGE)));
    swb_counter_addr <= to_integer(unsigned(i_wregs_add(SWB_COUNTER_ADDR_RANGE)));

    link_addr <= NDATAPATH_CNTS + swb_counter_addr + link_id * NLINK_CNTS;
    datapath_counter_addr <= swb_counter_addr                                                                                                             when i_wregs_add(SWB_DETECTOR_RANGE) = "00" else
                             swb_counter_addr + (NDATAPATH_CNTS+g_NLINKS_DATA_PIXEL_US*NLINK_CNTS)                                                        when i_wregs_add(SWB_DETECTOR_RANGE) = "01" else
                             swb_counter_addr + (NDATAPATH_CNTS+g_NLINKS_DATA_PIXEL_US*NLINK_CNTS) + (NDATAPATH_CNTS+g_NLINKS_DATA_PIXEL_DS*NLINK_CNTS)   when i_wregs_add(SWB_DETECTOR_RANGE) = "10" else
                             0;
    link_counter_addr     <= link_addr                                                                                                             when i_wregs_add(SWB_DETECTOR_RANGE) = "00" else
                             link_addr + (NDATAPATH_CNTS+g_NLINKS_DATA_PIXEL_US*NLINK_CNTS)                                                        when i_wregs_add(SWB_DETECTOR_RANGE) = "01" else
                             link_addr + (NDATAPATH_CNTS+g_NLINKS_DATA_PIXEL_US*NLINK_CNTS) + (NDATAPATH_CNTS+g_NLINKS_DATA_PIXEL_DS*NLINK_CNTS)   when i_wregs_add(SWB_DETECTOR_RANGE) = "10" else
                             0;

    --! map counters pixel
    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        o_pcie_data <= (others => '0');
        o_pcie_addr <= (others => '0');
        --
    elsif ( rising_edge(i_clk) ) then
        o_pcie_addr <= i_wregs_add;
        if ( i_wregs_add(SWB_COUNTER_TYPE) = '0' ) then
            o_pcie_data <= i_counter(link_counter_addr);
        else
            o_pcie_data <= i_counter(swb_counter_addr);
        end if;

    end if;
    end process;

end architecture;
