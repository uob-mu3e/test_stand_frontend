library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity seg7_lut is
    port (
        clk : in  std_logic;
        hex : in  std_logic_vector(3 downto 0);
        seg : out std_logic_vector(6 downto 0)--;
    );
end entity seg7_lut;

architecture rtl of seg7_lut is

begin

    seg_p : process(clk)
    begin
    if rising_edge(clk) then
        case hex is
			  when X"0" => seg <= "1000000";
			  when X"1" => seg <= "1111001";
			  when X"2" => seg <= "0100100";
			  when X"3" => seg <= "0110000";
			  when X"4" => seg <= "0011001";
			  when X"5" => seg <= "0010010";
			  when X"6" => seg <= "0000010";
			  when X"7" => seg <= "1111000";
			  when X"8" => seg <= "0000000";
			  when X"9" => seg <= "0011000";
			  when X"A" => seg <= "0001000";
			  when X"B" => seg <= "0000011";
			  when X"C" => seg <= "1000110";
			  when X"D" => seg <= "0100001";
			  when X"E" => seg <= "0000110";
			  when X"F" => seg <= "0001110";
			  when others => --idle
        end case;
    end if; -- rising_edge
    end process seg_p;

end architecture rtl;