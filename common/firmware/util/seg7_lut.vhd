library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity seg7_lut is
    port (
        hex : in  std_logic_vector(3 downto 0);
        seg : out std_logic_vector(6 downto 0)--;
    );
end entity seg7_lut;

architecture rtl of seg7_lut is
begin

    with hex select
    seg <=
        "1000000" when X"0",
        "1111001" when X"1",
        "0100100" when X"2",
        "0110000" when X"3",
        "0011001" when X"4",
        "0010010" when X"5",
        "0000010" when X"6",
        "1111000" when X"7",
        "0000000" when X"8",
        "0011000" when X"9",
        "0001000" when X"A",
        "0000011" when X"B",
        "1000110" when X"C",
        "0100001" when X"D",
        "0000110" when X"E",
        "0001110" when X"F",
        "1111111" when others;

end architecture rtl;
