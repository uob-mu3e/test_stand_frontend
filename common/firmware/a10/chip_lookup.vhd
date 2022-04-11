--
-- Marius Koeppel, November 2021
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;


entity chip_lookup is
generic (
    g_LOOPUP_NAME : string := "intRun2021"--;
);
port (
    i_fpgaID    : in   std_logic_vector (5 downto 0);
    i_chipID    : in   std_logic_vector (3 downto 0);
    o_chipID    : out  std_logic_vector (6 downto 0)--;
);
end entity;

architecture arch of chip_lookup is

    signal chipID : std_logic_vector (6 downto 0);

begin

    o_chipID <= chipID;

    generate_intRun2021 : if ( g_LOOPUP_NAME = "intRun2021" ) generate
        e_intRun2021 : entity work.chip_lookup_int_2021
        port map ( i_fpgaID => i_fpgaID(3 downto 0), i_chipID => i_chipID, o_chipID => chipID );
    end generate;

    generate_edmRun2021 : if ( g_LOOPUP_NAME = "edmRun2021" ) generate
        e_edmRun2021 : entity work.chip_lookup_edm_2021
        port map ( i_fpgaID => i_fpgaID(3 downto 0), i_chipID => i_chipID, o_chipID => chipID );
    end generate;

end architecture;
