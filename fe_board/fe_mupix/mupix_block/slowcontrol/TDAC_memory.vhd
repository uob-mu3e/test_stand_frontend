----------------------------------------------------------------------------
-- storage for Mupix TDACs
-- M. Mueller, Feb 2022
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;

use work.mupix.all;
use work.mudaq.all;


entity tdac_memory is
    generic( 
        N_CHIPS_g                 : positive := 4;
        PAGE_ADDR_WIDTH_g         : positive := 6;
        ADDR_WIDTH_g              : positive := 8--;
    );
    port(
        i_clk               : in  std_logic;
        i_reset_n           : in  std_logic;

        o_tdac_dpf_we       : out std_logic_vector(N_CHIPS_g-1 downto 0);
        o_tdac_dpf_wdata    : out reg32array(N_CHIPS_g-1 downto 0);
        i_tdac_dpf_empty    : in  std_logic_vector(N_CHIPS_g-1 downto 0);

        i_data              : in  std_logic_vector(31 downto 0);
        i_we                : in  std_logic;
        i_chip              : in  integer range 0 to N_CHIPS_g-1--;
    );
end entity tdac_memory;

architecture RTL of tdac_memory is

    type TDAC_page_type is record
        start_addr      :   std_logic_vector(ADDR_WIDTH_g-1 downto 0);
        addr            :   std_logic_vector(PAGE_ADDR_WIDTH_g-1 downto 0);
        in_use          :   boolean;
        full            :   boolean;
        chip            :   integer range 0 to N_CHIPS_g-1;
        --chip_page_ID    :   integer range 0 to TODO : max pages per chip;
    end record;

    type TDAC_page_array_type   is array( natural range <> ) of TDAC_page_type;
    --signal TDAC_page_array      : TDAC_page_array_type(TODO: n pages-1 downto 0);

    signal ram_we               : std_logic;
    signal ram_wdata            : std_logic;

begin
    -- TODO
    o_tdac_dpf_wdata    <= (others => (others => '0'));
    o_tdac_dpf_we       <= (others => '0');

    -- todo assign start addresses to TDAC pages
    -- loop
    -- tdac_pagearray(I).startaddr <= ...

    write_process: process (i_clk, i_reset_n) is
    begin
        if(i_reset_n = '0') then

        elsif(rising_edge(i_clk)) then

        end if;
    end process;

    ram_1r1w_inst: entity work.ram_1r1w -- better split into multiple RAM IP's each with size of 1 page ?
      generic map (
        g_DATA_WIDTH       => 32,
        g_ADDR_WIDTH       => ADDR_WIDTH_g--,
      )
      port map (
        i_raddr => x"00",
        o_rdata => open,
        i_rclk  => i_clk,
        i_waddr => x"00",
        i_wdata => x"00000000",
        i_we    => '0',
        i_wclk  => i_clk
      );
end RTL;
