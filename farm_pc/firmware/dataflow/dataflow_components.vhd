    library ieee;
    use ieee.std_logic_1164.all;

    package dataflow_components is

    subtype tsrange_type is std_logic_vector(15 downto 0);
    subtype tsupper is natural range 15 downto 8;-- 31 downto 16;
    subtype tslower is natural range 7 downto 0; --15 downto 0;

    constant tsone : tsrange_type := (others => '1');
    constant tszero : tsrange_type := (others => '1');
    constant tree_padding : std_logic_vector(37 downto 0) := "11" & x"FFFFFFFFF";
    constant tree_paddingk : std_logic_vector(37 downto 0) := "11" & x"EEEEEEEEE";
    constant tree_zero : std_logic_vector(37 downto 0) := "00" & x"000000000";

    subtype dataplusts_type is std_logic_vector(271 downto 0);

    type data_array is array (natural range <>) of std_logic_vector(37 downto 0);
    type hit_array_t is array (7 downto 0) of std_logic_vector(31 downto 0);

    type fifo_array_2 is array(natural range <>) of std_logic_vector(1 downto 0);
    type fifo_array_4 is array(natural range <>) of std_logic_vector(3 downto 0);
    type fifo_array_8 is array(natural range <>) of std_logic_vector(7 downto 0);
    type fifo_array_32 is array(natural range <>) of std_logic_vector(31 downto 0);
    type fifo_array_38 is array(natural range <>) of std_logic_vector(37 downto 0);
    type fifo_array_64 is array(natural range <>) of std_logic_vector(63 downto 0);
    type fifo_array_76 is array(natural range <>) of std_logic_vector(75 downto 0);
    type fifo_array_66 is array(natural range <>) of std_logic_vector(65 downto 0);
    type fifo_array_78 is array(natural range <>) of std_logic_vector(77 downto 0);
    type fifo_array_152 is array(natural range <>) of std_logic_vector(151 downto 0);

    type offset is array(natural range <>) of integer range 64 downto 0;
    
    function link_36_to_std (
        i : in integer--;
    ) return std_logic_vector;
    
end package dataflow_components;
    
package body dataflow_components is

    function link_36_to_std (
        i : in  integer--;
    ) return std_logic_vector is
    
    begin
        case i is
        when  0 => return "000000";
        when  1 => return "000001";
        when  2 => return "000010";
        when  3 => return "000011";
        when  4 => return "000100";
        when  5 => return "000101";
        when  6 => return "000110";
        when  7 => return "000111";
        when  8 => return "001000";
        when  9 => return "001001";
        when 10 => return "001010";
        when 11 => return "001011";
        when 12 => return "001100";
        when 13 => return "001101";
        when 14 => return "001110";
        when 15 => return "001111";
        when 16 => return "010000";
        when 17 => return "010001";
        when 18 => return "010010";
        when 19 => return "010011";
        when 20 => return "010100";
        when 21 => return "010101";
        when 22 => return "010110";
        when 23 => return "010111";
        when 24 => return "011000";
        when 25 => return "011001";
        when 26 => return "011010";
        when 27 => return "011011";
        when 28 => return "011100";
        when 29 => return "011101";
        when 30 => return "011110";
        when 31 => return "011111";
        when 32 => return "100000";
        when 33 => return "100001";
        when 34 => return "100010";
        when 35 => return "100011";
        when others =>
            return "111111";
        end case;
    end function;
    
end package body;
