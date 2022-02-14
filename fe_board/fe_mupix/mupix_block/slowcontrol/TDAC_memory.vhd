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
        N_CHIPS_g                 : positive := 4
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

begin
    -- TODO
    o_tdac_dpf_wdata    <= (others => (others => '0'));
    o_tdac_dpf_we       <= (others => '0');
end RTL;
