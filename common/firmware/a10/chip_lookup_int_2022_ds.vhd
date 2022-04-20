--
-- Marius Koeppel, July 2021
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

entity chip_lookup_int_2022_ds is
port (
    i_fpgaID    : in   std_logic_vector (3 downto 0);
    i_chipID    : in   std_logic_vector (3 downto 0);
    o_chipID    : out  std_logic_vector (6 downto 0)--;
);
end entity;

architecture arch of chip_lookup_int_2022_ds is

begin

    o_chipID <=
        "0000000" when i_fpgaID = x"0" and i_chipID = x"0" else
        "0000001" when i_fpgaID = x"0" and i_chipID = x"1" else
        "0000010" when i_fpgaID = x"0" and i_chipID = x"2" else
        "0000011" when i_fpgaID = x"0" and i_chipID = x"3" else
        "0000100" when i_fpgaID = x"0" and i_chipID = x"4" else
        "0000101" when i_fpgaID = x"0" and i_chipID = x"5" else
        "0000110" when i_fpgaID = x"0" and i_chipID = x"6" else
        "0000111" when i_fpgaID = x"0" and i_chipID = x"7" else
        "0001000" when i_fpgaID = x"0" and i_chipID = x"8" else
        "0001001" when i_fpgaID = x"0" and i_chipID = x"9" else
        "0001010" when i_fpgaID = x"0" and i_chipID = x"a" else
        "0001011" when i_fpgaID = x"0" and i_chipID = x"b" else
        "0001100" when i_fpgaID = x"1" and i_chipID = x"0" else
        "0001101" when i_fpgaID = x"1" and i_chipID = x"1" else
        "0001110" when i_fpgaID = x"1" and i_chipID = x"2" else
        "0001111" when i_fpgaID = x"1" and i_chipID = x"3" else
        "0010000" when i_fpgaID = x"1" and i_chipID = x"4" else
        "0010001" when i_fpgaID = x"1" and i_chipID = x"5" else
        "0010010" when i_fpgaID = x"1" and i_chipID = x"6" else
        "0010011" when i_fpgaID = x"1" and i_chipID = x"7" else
        "0010100" when i_fpgaID = x"1" and i_chipID = x"8" else
        "0010101" when i_fpgaID = x"1" and i_chipID = x"9" else
        "0010110" when i_fpgaID = x"1" and i_chipID = x"a" else
        "0010111" when i_fpgaID = x"1" and i_chipID = x"b" else
        "0011000" when i_fpgaID = x"2" and i_chipID = x"0" else
        "0011001" when i_fpgaID = x"2" and i_chipID = x"1" else
        "0011010" when i_fpgaID = x"2" and i_chipID = x"2" else
        "0011011" when i_fpgaID = x"2" and i_chipID = x"3" else
        "0011100" when i_fpgaID = x"2" and i_chipID = x"4" else
        "0011101" when i_fpgaID = x"2" and i_chipID = x"5" else
        "0011110" when i_fpgaID = x"2" and i_chipID = x"6" else
        "0011111" when i_fpgaID = x"2" and i_chipID = x"7" else
        "0100000" when i_fpgaID = x"2" and i_chipID = x"8" else
        "0100001" when i_fpgaID = x"2" and i_chipID = x"9" else
        "0100010" when i_fpgaID = x"2" and i_chipID = x"a" else
        "0100011" when i_fpgaID = x"2" and i_chipID = x"b" else
        "0100100" when i_fpgaID = x"3" and i_chipID = x"0" else
        "0100101" when i_fpgaID = x"3" and i_chipID = x"1" else
        "0100110" when i_fpgaID = x"3" and i_chipID = x"2" else
        "0100111" when i_fpgaID = x"3" and i_chipID = x"3" else
        "0101000" when i_fpgaID = x"3" and i_chipID = x"4" else
        "0101001" when i_fpgaID = x"3" and i_chipID = x"5" else
        "0101010" when i_fpgaID = x"3" and i_chipID = x"6" else
        "0101011" when i_fpgaID = x"3" and i_chipID = x"7" else
        "0101100" when i_fpgaID = x"3" and i_chipID = x"8" else
        "0101101" when i_fpgaID = x"3" and i_chipID = x"9" else
        "0101110" when i_fpgaID = x"3" and i_chipID = x"a" else
        "0101111" when i_fpgaID = x"3" and i_chipID = x"b" else
        "0110000" when i_fpgaID = x"4" and i_chipID = x"0" else
        "0110001" when i_fpgaID = x"4" and i_chipID = x"1" else
        "0110010" when i_fpgaID = x"4" and i_chipID = x"2" else
        "0110011" when i_fpgaID = x"4" and i_chipID = x"3" else
        "0110100" when i_fpgaID = x"4" and i_chipID = x"4" else
        "0110101" when i_fpgaID = x"4" and i_chipID = x"5" else
        "0110110" when i_fpgaID = x"4" and i_chipID = x"6" else
        "0110111" when i_fpgaID = x"4" and i_chipID = x"7" else
        "0111000" when i_fpgaID = x"4" and i_chipID = x"8" else
        "0111001" when i_fpgaID = x"4" and i_chipID = x"9" else
        "0111010" when i_fpgaID = x"4" and i_chipID = x"a" else
        "0111011" when i_fpgaID = x"4" and i_chipID = x"b" else
        "1111111";

end architecture;
