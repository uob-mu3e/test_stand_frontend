--
-- 8b10b decoder
-- https://en.wikipedia.org/wiki/8b/10b_encoding
--
-- author : Alexandr Kozlinskiy
-- date : 2018-07-01
--

library ieee;
use ieee.std_logic_1164.all;

entity dec_8b10b is
port (
    -- input 10-bit data (8b10b encoded)
    i_data      : in    std_logic_vector(9 downto 0);
    -- input disparity
    i_disp      : in    std_logic;
    -- output data (K bit & 8-bit data)
    o_data      : out   std_logic_vector(8 downto 0);
    -- output disparity
    o_disp      : out   std_logic;
    -- disparity error
    o_disperr   : out   std_logic;
    -- error if invalid code
    o_err       : out   std_logic--;
);
end entity;

architecture arch of dec_8b10b is

    -- disp & 6-bit group
    signal G6sel : std_logic_vector(6 downto 0);
    -- err & disperr & disp & 5-bit group
    signal G5 : std_logic_vector(7 downto 0);
    signal A7 : std_logic;

    signal K28, Kx7 : std_logic;

    -- disp & 4-bit group
    signal G4sel : std_logic_vector(4 downto 0);
    -- err & disperr & disp & 3-bit group
    signal G3 : std_logic_vector(5 downto 0);

begin

    Kx7 <= work.util.to_std_logic( ( i_data(9 downto 6) = "0001" or i_data(9 downto 6) = "1110" ) and (
        G5(4 downto 0) = "10111" or -- D.23
        G5(4 downto 0) = "11011" or -- D.27
        G5(4 downto 0) = "11101" or -- D.29
        G5(4 downto 0) = "11110" ) -- D.30
    );

    -- disp & 6 bits in
    G6sel <= i_disp & i_data(5 downto 0);
    -- err & disperr & disp & 5 bits out
    process(G6sel)
    begin
        A7 <= '0';
        K28 <= '0';

        case G6sel is
        when '0' & "111001" => G5 <= '0' & '0' & '1' & "00000";
        when '1' & "000110" => G5 <= '0' & '0' & '0' & "00000";
        when '0' & "101110" => G5 <= '0' & '0' & '1' & "00001";
        when '1' & "010001" => G5 <= '0' & '0' & '0' & "00001";
        when '0' & "101101" => G5 <= '0' & '0' & '1' & "00010";
        when '1' & "010010" => G5 <= '0' & '0' & '0' & "00010";
        when '0' & "100011" => G5 <= '0' & '0' & '0' & "00011";
        when '1' & "100011" => G5 <= '0' & '0' & '1' & "00011";
        when '0' & "101011" => G5 <= '0' & '0' & '1' & "00100";
        when '1' & "010100" => G5 <= '0' & '0' & '0' & "00100";
        when '0' & "100101" => G5 <= '0' & '0' & '0' & "00101";
        when '1' & "100101" => G5 <= '0' & '0' & '1' & "00101";
        when '0' & "100110" => G5 <= '0' & '0' & '0' & "00110";
        when '1' & "100110" => G5 <= '0' & '0' & '1' & "00110";
        when '0' & "000111" => G5 <= '0' & '0' & '0' & "00111"; -- D.07
        when '1' & "111000" => G5 <= '0' & '0' & '1' & "00111"; -- D.07
        when '0' & "100111" => G5 <= '0' & '0' & '1' & "01000";
        when '1' & "011000" => G5 <= '0' & '0' & '0' & "01000";
        when '0' & "101001" => G5 <= '0' & '0' & '0' & "01001";
        when '1' & "101001" => G5 <= '0' & '0' & '1' & "01001";
        when '0' & "101010" => G5 <= '0' & '0' & '0' & "01010";
        when '1' & "101010" => G5 <= '0' & '0' & '1' & "01010";
        when '0' & "001011" => G5 <= '0' & '0' & '0' & "01011";
        when '1' & "001011" => G5 <= '0' & '0' & '1' & "01011"; A7 <= '1'; -- D.11.A7
        when '0' & "101100" => G5 <= '0' & '0' & '0' & "01100";
        when '1' & "101100" => G5 <= '0' & '0' & '1' & "01100";
        when '0' & "001101" => G5 <= '0' & '0' & '0' & "01101";
        when '1' & "001101" => G5 <= '0' & '0' & '1' & "01101"; A7 <= '1'; -- D.13.A7
        when '0' & "001110" => G5 <= '0' & '0' & '0' & "01110";
        when '1' & "001110" => G5 <= '0' & '0' & '1' & "01110"; A7 <= '1'; -- D.14.A7
        when '0' & "111010" => G5 <= '0' & '0' & '1' & "01111";
        when '1' & "000101" => G5 <= '0' & '0' & '0' & "01111";
        when '0' & "110110" => G5 <= '0' & '0' & '1' & "10000";
        when '1' & "001001" => G5 <= '0' & '0' & '0' & "10000";
        when '0' & "110001" => G5 <= '0' & '0' & '0' & "10001"; A7 <= '1'; -- D.17.A7
        when '1' & "110001" => G5 <= '0' & '0' & '1' & "10001";
        when '0' & "110010" => G5 <= '0' & '0' & '0' & "10010"; A7 <= '1'; -- D.18.A7
        when '1' & "110010" => G5 <= '0' & '0' & '1' & "10010";
        when '0' & "010011" => G5 <= '0' & '0' & '0' & "10011";
        when '1' & "010011" => G5 <= '0' & '0' & '1' & "10011";
        when '0' & "110100" => G5 <= '0' & '0' & '0' & "10100"; A7 <= '1'; -- D.20.A7
        when '1' & "110100" => G5 <= '0' & '0' & '1' & "10100";
        when '0' & "010101" => G5 <= '0' & '0' & '0' & "10101";
        when '1' & "010101" => G5 <= '0' & '0' & '1' & "10101";
        when '0' & "010110" => G5 <= '0' & '0' & '0' & "10110";
        when '1' & "010110" => G5 <= '0' & '0' & '1' & "10110";
        when '0' & "010111" => G5 <= '0' & '0' & '1' & "10111";
        when '1' & "101000" => G5 <= '0' & '0' & '0' & "10111";
        when '0' & "110011" => G5 <= '0' & '0' & '1' & "11000";
        when '1' & "001100" => G5 <= '0' & '0' & '0' & "11000";
        when '0' & "011001" => G5 <= '0' & '0' & '0' & "11001";
        when '1' & "011001" => G5 <= '0' & '0' & '1' & "11001";
        when '0' & "011010" => G5 <= '0' & '0' & '0' & "11010";
        when '1' & "011010" => G5 <= '0' & '0' & '1' & "11010";
        when '0' & "011011" => G5 <= '0' & '0' & '1' & "11011";
        when '1' & "100100" => G5 <= '0' & '0' & '0' & "11011";
        when '0' & "011100" => G5 <= '0' & '0' & '0' & "11100"; -- D.28
        when '1' & "011100" => G5 <= '0' & '0' & '1' & "11100"; -- D.28
        when '0' & "111100" => G5 <= '0' & '0' & '1' & "11100"; K28 <= '1'; -- K.28
        when '1' & "000011" => G5 <= '0' & '0' & '0' & "11100"; K28 <= '1'; -- K.28
        when '0' & "011101" => G5 <= '0' & '0' & '1' & "11101";
        when '1' & "100010" => G5 <= '0' & '0' & '0' & "11101";
        when '0' & "011110" => G5 <= '0' & '0' & '1' & "11110";
        when '1' & "100001" => G5 <= '0' & '0' & '0' & "11110";
        when '0' & "110101" => G5 <= '0' & '0' & '1' & "11111";
        when '1' & "001010" => G5 <= '0' & '0' & '0' & "11111";
        -- invalid disparity
        when '1' & "111001" => G5 <= '0' & '1' & '1' & "00000";
        when '0' & "000110" => G5 <= '0' & '1' & '0' & "00000";
        when '1' & "101110" => G5 <= '0' & '1' & '1' & "00001";
        when '0' & "010001" => G5 <= '0' & '1' & '0' & "00001";
        when '1' & "101101" => G5 <= '0' & '1' & '1' & "00010";
        when '0' & "010010" => G5 <= '0' & '1' & '0' & "00010";
        when '1' & "101011" => G5 <= '0' & '1' & '1' & "00100";
        when '0' & "010100" => G5 <= '0' & '1' & '0' & "00100";
        when '1' & "100111" => G5 <= '0' & '1' & '1' & "01000";
        when '0' & "011000" => G5 <= '0' & '1' & '0' & "01000";
        when '1' & "111010" => G5 <= '0' & '1' & '1' & "01111";
        when '0' & "000101" => G5 <= '0' & '1' & '0' & "01111";
        when '1' & "110110" => G5 <= '0' & '1' & '1' & "10000";
        when '0' & "001001" => G5 <= '0' & '1' & '0' & "10000";
        when '1' & "010111" => G5 <= '0' & '1' & '1' & "10111";
        when '0' & "101000" => G5 <= '0' & '1' & '0' & "10111";
        when '1' & "110011" => G5 <= '0' & '1' & '1' & "11000";
        when '0' & "001100" => G5 <= '0' & '1' & '0' & "11000";
        when '1' & "011011" => G5 <= '0' & '1' & '1' & "11011";
        when '0' & "100100" => G5 <= '0' & '1' & '0' & "11011";
        when '1' & "111100" => G5 <= '0' & '1' & '1' & "11100"; K28 <= '1'; -- K.28
        when '0' & "000011" => G5 <= '0' & '1' & '0' & "11100"; K28 <= '1'; -- K.28
        when '1' & "011101" => G5 <= '0' & '1' & '1' & "11101";
        when '0' & "100010" => G5 <= '0' & '1' & '0' & "11101";
        when '1' & "011110" => G5 <= '0' & '1' & '1' & "11110";
        when '0' & "100001" => G5 <= '0' & '1' & '0' & "11110";
        when '1' & "110101" => G5 <= '0' & '1' & '1' & "11111";
        when '0' & "001010" => G5 <= '0' & '1' & '0' & "11111";
        --
        when '1' & "000111" => G5 <= '1' & '1' & '1' & "00111"; -- D.07
        when '0' & "111000" => G5 <= '1' & '1' & '0' & "00111"; -- D.07
        -- invalid code
        when '0' & "001111" => G5 <= '1' & '0' & '1' & "XXXXX";
        when '1' & "110000" => G5 <= '1' & '0' & '0' & "XXXXX";
        -- invalid code
        when '1' & "001111" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '0' & "110000" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '0' & "111111" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '1' & "000000" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '1' & "111111" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '0' & "000000" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '0' & "111110" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '1' & "000001" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '1' & "111110" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '0' & "000001" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '0' & "111101" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '1' & "000010" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '1' & "111101" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '0' & "000010" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '0' & "111011" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '1' & "000100" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '1' & "111011" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '0' & "000100" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '0' & "110111" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '1' & "001000" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '1' & "110111" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '0' & "001000" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '0' & "101111" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '1' & "010000" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '1' & "101111" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '0' & "010000" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '0' & "011111" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '1' & "100000" => G5 <= '1' & '1' & '0' & "XXXXX";
        when '1' & "011111" => G5 <= '1' & '1' & '1' & "XXXXX";
        when '0' & "100000" => G5 <= '1' & '1' & '0' & "XXXXX";
        when others => G5 <= (others => 'X');
        end case;
    end process;

    -- disp & 4 bits in
    G4sel <= G5(5) & i_data(9 downto 6);
    -- err & disperr & disp & 3 bits out
    process(G4sel, A7, K28, Kx7)
    begin
        case G4sel is
        when '0' & "1101" => G3 <= '0' & '0' & '1' & "000"; -- D.x.0
        when '1' & "0010" => G3 <= '0' & '0' & '0' & "000"; -- D.x.0
        when '0' & "1001" => G3 <= '0' & '0' & '0' & "001"; -- D.x.1
            if ( K28 = '1' ) then G3(2 downto 0) <= "110"; end if; -- K.28.6
        when '1' & "1001" => G3 <= '0' & '0' & '1' & "001"; -- D.x.1
        when '0' & "1010" => G3 <= '0' & '0' & '0' & "010"; -- D.x.2
            if ( K28 = '1' ) then G3(2 downto 0) <= "101"; end if; -- K.28.5
        when '1' & "1010" => G3 <= '0' & '0' & '1' & "010"; -- D.x.2
        when '0' & "0011" => G3 <= '0' & '0' & '0' & "011"; -- D.x.3
        when '1' & "1100" => G3 <= '0' & '0' & '1' & "011"; -- D.x.3
        when '0' & "1011" => G3 <= '0' & '0' & '1' & "100"; -- D.x.4
        when '1' & "0100" => G3 <= '0' & '0' & '0' & "100"; -- D.x.4
        when '0' & "0101" => G3 <= '0' & '0' & '0' & "101"; -- D.x.5
            if ( K28 = '1' ) then G3(2 downto 0) <= "010"; end if; -- K.28.2
        when '1' & "0101" => G3 <= '0' & '0' & '1' & "101"; -- D.x.5
        when '0' & "0110" => G3 <= '0' & '0' & '0' & "110"; -- D.x.6
            if ( K28 = '1' ) then G3(2 downto 0) <= "001"; end if; -- K.28.1
        when '1' & "0110" => G3 <= '0' & '0' & '1' & "110"; -- D.x.6
        when '0' & "0111" => G3 <= '0' & '0' & '1' & "111"; -- D.x.P7
            G3(5) <= (A7 or K28 or Kx7);
        when '1' & "1000" => G3 <= '0' & '0' & '0' & "111"; -- D.x.P7
            G3(5) <= (A7 or K28 or Kx7);
        when '0' & "1110" => G3 <= '0' & '0' & '1' & "111"; -- D.x.A7, K.x.7
            G3(5) <= not (A7 or K28 or Kx7);
        when '1' & "0001" => G3 <= '0' & '0' & '0' & "111"; -- D.x.A7, K.x.7
            G3(5) <= not (A7 or K28 or Kx7);
        -- invalid disparity
        when '1' & "1101" => G3 <= '0' & '1' & '1' & "000"; -- D.x.0
        when '0' & "0010" => G3 <= '0' & '1' & '0' & "000"; -- D.x.0
        when '1' & "1011" => G3 <= '0' & '1' & '1' & "100"; -- D.x.4
        when '0' & "0100" => G3 <= '0' & '1' & '0' & "100"; -- D.x.4
        when '1' & "0111" => G3 <= '0' & '1' & '1' & "111"; -- D.x.P7
            G3(5) <= (A7 or K28 or Kx7);
        when '0' & "1000" => G3 <= '0' & '1' & '0' & "111"; -- D.x.P7
            G3(5) <= (A7 or K28 or Kx7);
        when '1' & "1110" => G3 <= '0' & '1' & '1' & "111"; -- D.x.A7, K.x.7
            G3(5) <= not (A7 or K28 or Kx7);
        when '0' & "0001" => G3 <= '0' & '1' & '0' & "111"; -- D.x.A7, K.x.7
            G3(5) <= not (A7 or K28 or Kx7);
        --
        when '1' & "0011" => G3 <= '0' & '1' & '1' & "011"; -- D.x.3
        when '0' & "1100" => G3 <= '0' & '1' & '0' & "011"; -- D.x.3
        -- invalid code
        when '0' & "1111" => G3 <= '1' & '1' & '1' & "XXX";
        when '1' & "0000" => G3 <= '1' & '1' & '0' & "XXX";
        when '1' & "1111" => G3 <= '1' & '1' & '1' & "XXX";
        when '0' & "0000" => G3 <= '1' & '1' & '0' & "XXX";
        when others => G3 <= (others => 'X');
        end case;
    end process;

    o_data(7 downto 0) <= G3(2 downto 0) & G5(4 downto 0);
    o_data(8) <= K28 or Kx7; -- TODO : not err
    o_disp <= G3(3);
    o_disperr <= G5(6) or G3(4);
    o_err <= G5(7) or G3(5) or G3(4);

end architecture;
