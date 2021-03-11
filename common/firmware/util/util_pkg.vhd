--
-- author : Alexandr Kozlinskiy
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

use std.textio.all;
use ieee.std_logic_textio.all;

package util is


    --! basic array types    
    subtype slv2_t is std_logic_vector(1 downto 0);
    type slv2_array_t is array ( natural range <> ) of slv2_t;
    subtype slv4_t is std_logic_vector(3 downto 0);
    type slv4_array_t is array ( natural range <> ) of slv4_t;
    subtype slv6_t is std_logic_vector(5 downto 0);
    type slv6_array_t is array ( natural range <> ) of slv6_t;
    subtype slv8_t is std_logic_vector(7 downto 0);
    type slv8_array_t is array ( natural range <> ) of slv8_t;
    subtype slv16_t is std_logic_vector(15 downto 0);
    type slv16_array_t is array ( natural range <> ) of slv16_t;
    subtype slv32_t is std_logic_vector(31 downto 0);
    type slv32_array_t is array ( natural range <> ) of slv32_t;
    subtype slv37_t is std_logic_vector(31 downto 0);
    type slv37_array_t is array ( natural range <> ) of slv37_t;
    subtype slv38_t is std_logic_vector(37 downto 0);
    type slv38_array_t is array ( natural range <> ) of slv38_t;
    subtype slv64_t is std_logic_vector(65 downto 0);
    type slv64_array_t is array ( natural range <> ) of slv64_t;
    subtype slv66_t is std_logic_vector(65 downto 0);
    type slv6_array_t is array ( natural range <> ) of slv66_t;
    subtype slv76_t is std_logic_vector(75 downto 0);
    type slv76_array_t is array ( natural range <> ) of slv76_t;
    subtype slv78_t is std_logic_vector(77 downto 0);
    type slv78_array_t is array ( natural range <> ) of slv78_t;
    subtype slv152_t is std_logic_vector(151 downto 0);
    type slv152_array_t is array ( natural range <> ) of slv152_t;
    subtype slv256_t is std_logic_vector(255 downto 0);
    type slv256_array_t is array ( natural range <> ) of slv256_t;


    --! 8b/10b words
    constant D16_2 : std_logic_vector(7 downto 0) := X"50";
    constant D21_4 : std_logic_vector(7 downto 0) := x"95";
    constant D02_5 : std_logic_vector(7 downto 0) := X"A2";
    constant D21_5 : std_logic_vector(7 downto 0) := x"B5";
    constant D28_4 : std_logic_vector(7 downto 0) := x"9C";
    constant D28_5 : std_logic_vector(7 downto 0) := X"BC";
    constant D28_7 : std_logic_vector(7 downto 0) := X"FC";
    constant D05_6 : std_logic_vector(7 downto 0) := X"C5";
    constant K28_2 : std_logic_vector(7 downto 0) := x"5C";
    constant K28_3 : std_logic_vector(7 downto 0) := x"7C";
    constant K28_4 : std_logic_vector(7 downto 0) := x"9C";
    constant K28_5 : std_logic_vector(7 downto 0) := x"BC";
    constant K28_6 : std_logic_vector(7 downto 0) := x"DC";
    

    --! data path farm types
    subtype dataplusts_type is std_logic_vector(271 downto 0);
    type offset is array(natural range <>) of integer range 64 downto 0;
    subtype tsrange_type is std_logic_vector(15 downto 0);
    subtype tsupper is natural range 15 downto 8;
    subtype tslower is natural range 7 downto 0;
    constant tsone : tsrange_type := (others => '1');
    constant tszero : tsrange_type := (others => '0');


    --! data path swb types
    constant tree_padding : std_logic_vector(37 downto 0) := "11" & x"FFFFFFFFF";
    constant tree_paddingk : std_logic_vector(37 downto 0) := "11" & x"EEEEEEEEE";
    constant tree_zero : std_logic_vector(37 downto 0) := "00" & x"000000000";
    constant pre_marker : std_logic_vector(5 downto 0) := "110000";
    constant sh_marker : std_logic_vector(5 downto 0)  := "110001";
    constant tr_marker : std_logic_vector(5 downto 0)  := "110010";
    constant ts1_marker : std_logic_vector(5 downto 0) := "110011";
    constant ts2_marker : std_logic_vector(5 downto 0) := "110100";
    constant err_marker : std_logic_vector(5 downto 0) := "110101";


    --! FEB - SWB protocol
    constant HEADER_K:    std_logic_vector(31 downto 0) := x"bcbcbcbc";
    constant DATA_HEADER_ID:    std_logic_vector(5 downto 0) := "111010";
    constant DATA_SUB_HEADER_ID:    std_logic_vector(5 downto 0) := "111111";
    constant ACTIVE_SIGNAL_HEADER_ID:    std_logic_vector(5 downto 0) := "111101";
    constant RUN_TAIL_HEADER_ID:    std_logic_vector(5 downto 0) := "111110";
    constant TIMING_MEAS_HEADER_ID:    std_logic_vector(5 downto 0) := "111100";
    constant SC_HEADER_ID:    std_logic_vector(5 downto 0) := "111011";


    --! PCIe types
    subtype reg32 is std_logic_vector(31 downto 0);
    constant NREGISTERS :  integer := 64;
    type reg32array is array (NREGISTERS-1 downto 0) of reg32;


    type avalon_t is record
        address         :   std_logic_vector(31 downto 0);
        read            :   std_logic;
        readdata        :   std_logic_vector(31 downto 0);
        write           :   std_logic;
        writedata       :   std_logic_vector(31 downto 0);
        waitrequest     :   std_logic;
        readdatavalid   :   std_logic;
    end record;
    type avalon_array_t is array(natural range <>) of avalon_t;



    -- avalon memory mapped interface
    type avmm_t is record
        address         :   std_logic_vector(31 downto 0);
        read            :   std_logic;
        readdata        :   std_logic_vector(31 downto 0);
        write           :   std_logic;
        writedata       :   std_logic_vector(31 downto 0);
        waitrequest     :   std_logic;
        readdatavalid   :   std_logic;
    end record;
    type avmm_array_t is array(natural range <>) of avmm_t;

    type rw_t is record
        addr            :   std_logic_vector(31 downto 0);
        re              :   std_logic; -- read enable
        rvalid          :   std_logic; -- read valid
        rdata           :   std_logic_vector(31 downto 0);
        we              :   std_logic; -- write enable
        wdata           :   std_logic_vector(31 downto 0);
    end record;



    -- Greatest Common Divisor
    function gcd (
        p, q : positive--;
    ) return positive;

    function max (
        l, r : integer--;
    ) return integer;

    function vector_width (
        v : natural--;
    ) return positive;

    function bin2gray (
        v : std_logic_vector--;
    ) return std_logic_vector;

    function gray2bin (
        v : std_logic_vector--;
    ) return std_logic_vector;

    function gray_inc (
        v : std_logic_vector--;
    ) return std_logic_vector;

    function shift_right (
        v : std_logic_vector;
        n : natural--;
    ) return std_logic_vector;

    function shift_left (
        v : std_logic_vector;
        n : natural--;
    ) return std_logic_vector;

    function resize (
        v : std_logic_vector;
        n : positive--;
    ) return std_logic_vector;

    function and_reduce (
        v : std_logic_vector--;
    ) return std_logic;

    function or_reduce (
        v : std_logic_vector--;
    ) return std_logic;

    function xor_reduce (
        v : std_logic_vector--;
    ) return std_logic;

    function to_std_logic (
        b : in boolean--;
    ) return std_logic;

    function reverse (
        v : std_logic_vector--;
    ) return std_logic_vector;

    procedure char_to_hex (
        c : in character;
        v : out std_logic_vector(3 downto 0);
        good : out boolean--;
    );

    procedure string_to_hex (
        s : in string;
        v : out std_logic_vector;
        good : out boolean--;
    );

    procedure read_hex (
        l : inout line;
        value : out std_logic_vector;
        good : out boolean--;
    );

    function hex_to_ascii (
        h : in std_logic_vector--;
    ) return std_logic_vector;

    function link_36_to_std (
        i : in integer--;
    ) return std_logic_vector;


    -- LFSR 32
    -- src: http://www.xilinx.com/support/documentation/application_notes/xapp052.pdf
    --
    -- taps: 31, 21, 1, 0
    function lfsr_32 (
        data : std_logic_vector(31 downto 0)--;
    ) return std_logic_vector;

    -- CRC-32C (Castagnoli) 0x1.1EDC6F41
    -- src: http://www.easics.com/services/freesics/crctool.html
    -- polynomial: x^32 + x^28 + x^27 + x^26 + x^25 + x^23 + x^22 + x^20 + x^19 + x^18 + x^14 + x^13 + x^11 + x^10 + x^9 + x^8 + x^6 + 1
    -- data width: 32
    -- convention: the first serial bit is D[31] -- TODO: D[0]
    function crc32 (
        data : std_logic_vector(31 downto 0);
        crc  : std_logic_vector(31 downto 0)--;
    ) return std_logic_vector;



    function count_bits_4 (
        data : std_logic_vector(3 downto 0)--;
    ) return natural;

    function count_bits_32 (
        data : std_logic_vector(31 downto 0)--;
    ) return natural;

    function count_bits (
        data : std_logic_vector--;
    ) return natural;

    impure
    function read_hex (
        fname : in string;
        N : in positive;
        W : in positive--;
    ) return std_logic_vector;

    function to_string (
        v : in std_logic--;
    ) return string;

    function to_string (
        v : in std_logic_vector--;
    ) return string;

    function to_string (
        v : in unsigned--;
    ) return string;

    function to_hstring (
        v : std_logic_vector--;
    ) return string;

    function to_hstring (
        v : unsigned--;
    ) return string;

    -- Select Graphic Rendition
    function sgr (
        n : natural--;
    ) return string;

end package;

package body util is

    function gcd (
        p, q : positive--;
    ) return positive is
        variable p_v : positive := p;
        variable q_v : positive := q;
    begin
        while ( p_v /= q_v ) loop
            if ( p_v > q_v ) then
                p_v := p_v - q_v;
            else
                q_v := q_v - p_v;
            end if;
        end loop;
        return p_v;
    end function;

    function max (
        l, r : integer
    ) return integer is
    begin
        if l > r then
            return l;
        else
            return r;
        end if;
    end function;

    function vector_width (
        v : natural--;
    ) return positive is
    begin
        if ( v = 0 or v = 1 ) then
            return 1;
        end if;
        return positive(ceil(log2(real(v))));
    end function;

    function bin2gray (
        v : std_logic_vector--;
    ) return std_logic_vector is
    begin
        return v xor shift_right(v, 1);
    end function;

    function gray2bin (
        v : std_logic_vector--;
    ) return std_logic_vector is
        variable b : std_logic := '0';
        variable r : std_logic_vector(v'range);
    begin
        for i in v'range loop
            b := b xor v(i);
            r(i) := b;
        end loop;
        return r;
    end function;

    function gray_inc (
        v : std_logic_vector--;
    ) return std_logic_vector is
        variable r : std_logic_vector(v'range) := (others => '0');
    begin
        r := gray2bin(v);
        r := std_logic_vector(unsigned(r) + 1);
        return bin2gray(r);
    end function;

    function shift_right (
        v : std_logic_vector;
        n : natural--;
    ) return std_logic_vector is
    begin
        return std_logic_vector(shift_right(unsigned(v), n));
    end function;

    function shift_left (
        v : std_logic_vector;
        n : natural--;
    ) return std_logic_vector is
    begin
        return std_logic_vector(shift_left(unsigned(v), n));
    end function;

    function resize (
        v : std_logic_vector;
        n : positive--;
    ) return std_logic_vector is
    begin
        return std_logic_vector(resize(unsigned(v), n));
    end function;

    function and_reduce (
        v : std_logic_vector--;
    ) return std_logic is
    begin
        return to_std_logic(v = (v'range => '1'));
    end function;

    function or_reduce (
        v : std_logic_vector--;
    ) return std_logic is
    begin
        return to_std_logic(v /= (v'range => '0'));
    end function;

    function xor_reduce (
        v : std_logic_vector--;
    ) return std_logic is
        alias a : std_logic_vector(v'length-1 downto 0) is v;
    begin
        if ( v'length = 0 ) then
            report "(xor_reduce) v'length = 0" severity failure;
            return 'X';
        end if;
        if ( a'length = 1 ) then
            return a(0);
        end if;
        return xor_reduce(a(a'length-1 downto a'length/2)) xor xor_reduce(a(a'length/2-1 downto 0));
    end function;

    function to_std_logic (
        b : in boolean--;
    ) return std_logic is
    begin
        if b then
            return '1';
        else
            return '0';
        end if;
    end function;

    function reverse (
        v : std_logic_vector--;
    ) return std_logic_vector is
        variable r : std_logic_vector(v'range);
        alias a : std_logic_vector(v'reverse_range) is v;
    begin
        for i in a'range loop
            r(i) := a(i);
        end loop;
        return r;
    end function;

    procedure char_to_hex (
        c : in character;
        v : out std_logic_vector(3 downto 0);
        good : out boolean--;
    ) is
    begin
        good := true;
        case c is
        when '0' => v := X"0";
        when '1' => v := X"1";
        when '2' => v := X"2";
        when '3' => v := X"3";
        when '4' => v := X"4";
        when '5' => v := X"5";
        when '6' => v := X"6";
        when '7' => v := X"7";
        when '8' => v := X"8";
        when '9' => v := X"9";

        when 'a' | 'A' => v := X"A";
        when 'b' | 'B' => v := X"B";
        when 'c' | 'C' => v := X"C";
        when 'd' | 'D' => v := X"D";
        when 'e' | 'E' => v := X"E";
        when 'f' | 'F' => v := X"F";

        when others =>
           report "(char_to_hex) invalid hex character '" & c & "'" severity failure;
           good := false;
           v := "XXXX";
        end case;
    end procedure;

    function hex_to_ascii (
        h : in  std_logic_vector--;
    ) return std_logic_vector is
    
    begin
        case h is
        when x"0" => return X"30";
        when x"1" => return X"31";
        when x"2" => return X"32";
        when x"3" => return X"33";
        when x"4" => return X"34";
        when x"5" => return X"35";
        when x"6" => return X"36";
        when x"7" => return X"37";
        when x"8" => return X"38";
        when x"9" => return X"39";
        when x"A" => return X"41";
        when x"B" => return X"42";
        when x"C" => return X"43";
        when x"D" => return X"44";
        when x"E" => return X"45";
        when x"F" => return X"46";

        when others =>
            return x"3F";
        end case;
    end function;

    procedure string_to_hex (
        s : in string;
        v : out std_logic_vector;
        good : out boolean--;
    ) is
        variable ok : boolean;
        variable good_i : boolean;
    begin
        good_i := true;
        for i in 0 to s'length-1 loop
            char_to_hex(s(s'right-i), v(3+4*i+v'right downto 4*i+v'right), ok);
            good_i := good_i and ok;
        end loop;
        good := good_i;
    end procedure;

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

    procedure read_hex (
        l : inout line;
        value : out std_logic_vector;
        good : out boolean--;
    ) is
        variable v : std_logic_vector(value'range);
        variable c : character;
        variable s : string(1 to value'length/4);
        variable ok : boolean;
    begin
        good := false;

        if value'length mod 4 /= 0 then
            report "(read_hex) value'length mod 4 /= 0" severity failure;
            return;
        end if;

        -- skip spaces
        loop
            read(l, c);
            exit when ((c /= ' ') and (c /= CR) and (c /= HT));
        end loop;

        -- skip comment
        if c = '#' then
            return;
        end if;

        s(1) := c;
        read(L, s(2 to s'right), ok);
        if not ok then
            return;
        end if;

        string_to_hex(s, v, ok);
        if not ok then
            return;
        end if;

        value := v;
        good := true;
    end procedure;

    impure
    function read_hex (
        fname : in string;
        N : in positive;
        W : in positive--;
    ) return std_logic_vector is
        variable data : std_logic_vector(N*W-1 downto 0);
        variable data_i : std_logic_vector(W-1 downto 0);
        variable i : integer := 0;
        file f : text;
        variable fs : file_open_status;
        variable l : line;
        variable c : character;
        variable s : string(1 to W/4);
        variable ok : boolean;
    begin
        if fname'length = 0 then
            return data;
        end if;

        file_open(fs, f, fname, READ_MODE);
        assert ( fs = open_ok ) report "(read_hex) file_open_status = '" & FILE_OPEN_STATUS'image(fs) & "'" severity failure;

        while ( not endfile(f) ) loop
            readline(f, l);
            read(l, c, ok);
            next when ( not ok or c = '#' );
            s(1) := c;
            read(l, s(2 to s'right), ok);
            next when ( not ok );
            work.util.string_to_hex(s, data_i, ok);
            next when ( not ok );
            data(W-1+i*W downto i*W) := data_i;
            i := i + 1;
        end loop;

        file_close(f);
        return data;
    end function;



    function lfsr_32(
        data : std_logic_vector(31 downto 0)--;
    ) return std_logic_vector is
    begin
        return data(30 downto 0) &
              (data(31) xor data(21) xor data(1) xor data(0));
    end function;

    function crc32 (
        data : std_logic_vector(31 downto 0);
        crc  : std_logic_vector(31 downto 0)--;
    ) return std_logic_vector is
        variable d      : std_logic_vector(31 downto 0);
        variable c      : std_logic_vector(31 downto 0);
        variable newcrc : std_logic_vector(31 downto 0);
    begin
        d := data;
        c := crc;

        newcrc(0) := d(31) xor d(30) xor d(28) xor d(27) xor d(26) xor d(25) xor d(23) xor d(21) xor d(18) xor d(17) xor d(16) xor d(12) xor d(9) xor d(8) xor d(7) xor d(6) xor d(5) xor d(4) xor d(0) xor c(0) xor c(4) xor c(5) xor c(6) xor c(7) xor c(8) xor c(9) xor c(12) xor c(16) xor c(17) xor c(18) xor c(21) xor c(23) xor c(25) xor c(26) xor c(27) xor c(28) xor c(30) xor c(31);
        newcrc(1) := d(31) xor d(29) xor d(28) xor d(27) xor d(26) xor d(24) xor d(22) xor d(19) xor d(18) xor d(17) xor d(13) xor d(10) xor d(9) xor d(8) xor d(7) xor d(6) xor d(5) xor d(1) xor c(1) xor c(5) xor c(6) xor c(7) xor c(8) xor c(9) xor c(10) xor c(13) xor c(17) xor c(18) xor c(19) xor c(22) xor c(24) xor c(26) xor c(27) xor c(28) xor c(29) xor c(31);
        newcrc(2) := d(30) xor d(29) xor d(28) xor d(27) xor d(25) xor d(23) xor d(20) xor d(19) xor d(18) xor d(14) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(6) xor d(2) xor c(2) xor c(6) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(14) xor c(18) xor c(19) xor c(20) xor c(23) xor c(25) xor c(27) xor c(28) xor c(29) xor c(30);
        newcrc(3) := d(31) xor d(30) xor d(29) xor d(28) xor d(26) xor d(24) xor d(21) xor d(20) xor d(19) xor d(15) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(3) xor c(3) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(15) xor c(19) xor c(20) xor c(21) xor c(24) xor c(26) xor c(28) xor c(29) xor c(30) xor c(31);
        newcrc(4) := d(31) xor d(30) xor d(29) xor d(27) xor d(25) xor d(22) xor d(21) xor d(20) xor d(16) xor d(13) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(4) xor c(4) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(13) xor c(16) xor c(20) xor c(21) xor c(22) xor c(25) xor c(27) xor c(29) xor c(30) xor c(31);
        newcrc(5) := d(31) xor d(30) xor d(28) xor d(26) xor d(23) xor d(22) xor d(21) xor d(17) xor d(14) xor d(13) xor d(12) xor d(11) xor d(10) xor d(9) xor d(5) xor c(5) xor c(9) xor c(10) xor c(11) xor c(12) xor c(13) xor c(14) xor c(17) xor c(21) xor c(22) xor c(23) xor c(26) xor c(28) xor c(30) xor c(31);
        newcrc(6) := d(30) xor d(29) xor d(28) xor d(26) xor d(25) xor d(24) xor d(22) xor d(21) xor d(17) xor d(16) xor d(15) xor d(14) xor d(13) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(5) xor d(4) xor d(0) xor c(0) xor c(4) xor c(5) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(13) xor c(14) xor c(15) xor c(16) xor c(17) xor c(21) xor c(22) xor c(24) xor c(25) xor c(26) xor c(28) xor c(29) xor c(30);
        newcrc(7) := d(31) xor d(30) xor d(29) xor d(27) xor d(26) xor d(25) xor d(23) xor d(22) xor d(18) xor d(17) xor d(16) xor d(15) xor d(14) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(6) xor d(5) xor d(1) xor c(1) xor c(5) xor c(6) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(14) xor c(15) xor c(16) xor c(17) xor c(18) xor c(22) xor c(23) xor c(25) xor c(26) xor c(27) xor c(29) xor c(30) xor c(31);
        newcrc(8) := d(25) xor d(24) xor d(21) xor d(19) xor d(15) xor d(13) xor d(11) xor d(10) xor d(8) xor d(5) xor d(4) xor d(2) xor d(0) xor c(0) xor c(2) xor c(4) xor c(5) xor c(8) xor c(10) xor c(11) xor c(13) xor c(15) xor c(19) xor c(21) xor c(24) xor c(25);
        newcrc(9) := d(31) xor d(30) xor d(28) xor d(27) xor d(23) xor d(22) xor d(21) xor d(20) xor d(18) xor d(17) xor d(14) xor d(11) xor d(8) xor d(7) xor d(4) xor d(3) xor d(1) xor d(0) xor c(0) xor c(1) xor c(3) xor c(4) xor c(7) xor c(8) xor c(11) xor c(14) xor c(17) xor c(18) xor c(20) xor c(21) xor c(22) xor c(23) xor c(27) xor c(28) xor c(30) xor c(31);
        newcrc(10) := d(30) xor d(29) xor d(27) xor d(26) xor d(25) xor d(24) xor d(22) xor d(19) xor d(17) xor d(16) xor d(15) xor d(7) xor d(6) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(6) xor c(7) xor c(15) xor c(16) xor c(17) xor c(19) xor c(22) xor c(24) xor c(25) xor c(26) xor c(27) xor c(29) xor c(30);
        newcrc(11) := d(21) xor d(20) xor d(12) xor d(9) xor d(6) xor d(5) xor d(4) xor d(3) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(3) xor c(4) xor c(5) xor c(6) xor c(9) xor c(12) xor c(20) xor c(21);
        newcrc(12) := d(22) xor d(21) xor d(13) xor d(10) xor d(7) xor d(6) xor d(5) xor d(4) xor d(3) xor d(2) xor d(1) xor c(1) xor c(2) xor c(3) xor c(4) xor c(5) xor c(6) xor c(7) xor c(10) xor c(13) xor c(21) xor c(22);
        newcrc(13) := d(31) xor d(30) xor d(28) xor d(27) xor d(26) xor d(25) xor d(22) xor d(21) xor d(18) xor d(17) xor d(16) xor d(14) xor d(12) xor d(11) xor d(9) xor d(3) xor d(2) xor d(0) xor c(0) xor c(2) xor c(3) xor c(9) xor c(11) xor c(12) xor c(14) xor c(16) xor c(17) xor c(18) xor c(21) xor c(22) xor c(25) xor c(26) xor c(27) xor c(28) xor c(30) xor c(31);
        newcrc(14) := d(30) xor d(29) xor d(25) xor d(22) xor d(21) xor d(19) xor d(16) xor d(15) xor d(13) xor d(10) xor d(9) xor d(8) xor d(7) xor d(6) xor d(5) xor d(3) xor d(1) xor d(0) xor c(0) xor c(1) xor c(3) xor c(5) xor c(6) xor c(7) xor c(8) xor c(9) xor c(10) xor c(13) xor c(15) xor c(16) xor c(19) xor c(21) xor c(22) xor c(25) xor c(29) xor c(30);
        newcrc(15) := d(31) xor d(30) xor d(26) xor d(23) xor d(22) xor d(20) xor d(17) xor d(16) xor d(14) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(6) xor d(4) xor d(2) xor d(1) xor c(1) xor c(2) xor c(4) xor c(6) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(14) xor c(16) xor c(17) xor c(20) xor c(22) xor c(23) xor c(26) xor c(30) xor c(31);
        newcrc(16) := d(31) xor d(27) xor d(24) xor d(23) xor d(21) xor d(18) xor d(17) xor d(15) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(7) xor d(5) xor d(3) xor d(2) xor c(2) xor c(3) xor c(5) xor c(7) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(15) xor c(17) xor c(18) xor c(21) xor c(23) xor c(24) xor c(27) xor c(31);
        newcrc(17) := d(28) xor d(25) xor d(24) xor d(22) xor d(19) xor d(18) xor d(16) xor d(13) xor d(12) xor d(11) xor d(10) xor d(9) xor d(8) xor d(6) xor d(4) xor d(3) xor c(3) xor c(4) xor c(6) xor c(8) xor c(9) xor c(10) xor c(11) xor c(12) xor c(13) xor c(16) xor c(18) xor c(19) xor c(22) xor c(24) xor c(25) xor c(28);
        newcrc(18) := d(31) xor d(30) xor d(29) xor d(28) xor d(27) xor d(21) xor d(20) xor d(19) xor d(18) xor d(16) xor d(14) xor d(13) xor d(11) xor d(10) xor d(8) xor d(6) xor d(0) xor c(0) xor c(6) xor c(8) xor c(10) xor c(11) xor c(13) xor c(14) xor c(16) xor c(18) xor c(19) xor c(20) xor c(21) xor c(27) xor c(28) xor c(29) xor c(30) xor c(31);
        newcrc(19) := d(29) xor d(27) xor d(26) xor d(25) xor d(23) xor d(22) xor d(20) xor d(19) xor d(18) xor d(16) xor d(15) xor d(14) xor d(11) xor d(8) xor d(6) xor d(5) xor d(4) xor d(1) xor d(0) xor c(0) xor c(1) xor c(4) xor c(5) xor c(6) xor c(8) xor c(11) xor c(14) xor c(15) xor c(16) xor c(18) xor c(19) xor c(20) xor c(22) xor c(23) xor c(25) xor c(26) xor c(27) xor c(29);
        newcrc(20) := d(31) xor d(25) xor d(24) xor d(20) xor d(19) xor d(18) xor d(15) xor d(8) xor d(4) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(4) xor c(8) xor c(15) xor c(18) xor c(19) xor c(20) xor c(24) xor c(25) xor c(31);
        newcrc(21) := d(26) xor d(25) xor d(21) xor d(20) xor d(19) xor d(16) xor d(9) xor d(5) xor d(3) xor d(2) xor d(1) xor c(1) xor c(2) xor c(3) xor c(5) xor c(9) xor c(16) xor c(19) xor c(20) xor c(21) xor c(25) xor c(26);
        newcrc(22) := d(31) xor d(30) xor d(28) xor d(25) xor d(23) xor d(22) xor d(20) xor d(18) xor d(16) xor d(12) xor d(10) xor d(9) xor d(8) xor d(7) xor d(5) xor d(3) xor d(2) xor d(0) xor c(0) xor c(2) xor c(3) xor c(5) xor c(7) xor c(8) xor c(9) xor c(10) xor c(12) xor c(16) xor c(18) xor c(20) xor c(22) xor c(23) xor c(25) xor c(28) xor c(30) xor c(31);
        newcrc(23) := d(30) xor d(29) xor d(28) xor d(27) xor d(25) xor d(24) xor d(19) xor d(18) xor d(16) xor d(13) xor d(12) xor d(11) xor d(10) xor d(7) xor d(5) xor d(3) xor d(1) xor d(0) xor c(0) xor c(1) xor c(3) xor c(5) xor c(7) xor c(10) xor c(11) xor c(12) xor c(13) xor c(16) xor c(18) xor c(19) xor c(24) xor c(25) xor c(27) xor c(28) xor c(29) xor c(30);
        newcrc(24) := d(31) xor d(30) xor d(29) xor d(28) xor d(26) xor d(25) xor d(20) xor d(19) xor d(17) xor d(14) xor d(13) xor d(12) xor d(11) xor d(8) xor d(6) xor d(4) xor d(2) xor d(1) xor c(1) xor c(2) xor c(4) xor c(6) xor c(8) xor c(11) xor c(12) xor c(13) xor c(14) xor c(17) xor c(19) xor c(20) xor c(25) xor c(26) xor c(28) xor c(29) xor c(30) xor c(31);
        newcrc(25) := d(29) xor d(28) xor d(25) xor d(23) xor d(20) xor d(17) xor d(16) xor d(15) xor d(14) xor d(13) xor d(8) xor d(6) xor d(4) xor d(3) xor d(2) xor d(0) xor c(0) xor c(2) xor c(3) xor c(4) xor c(6) xor c(8) xor c(13) xor c(14) xor c(15) xor c(16) xor c(17) xor c(20) xor c(23) xor c(25) xor c(28) xor c(29);
        newcrc(26) := d(31) xor d(29) xor d(28) xor d(27) xor d(25) xor d(24) xor d(23) xor d(15) xor d(14) xor d(12) xor d(8) xor d(6) xor d(3) xor d(1) xor d(0) xor c(0) xor c(1) xor c(3) xor c(6) xor c(8) xor c(12) xor c(14) xor c(15) xor c(23) xor c(24) xor c(25) xor c(27) xor c(28) xor c(29) xor c(31);
        newcrc(27) := d(31) xor d(29) xor d(27) xor d(24) xor d(23) xor d(21) xor d(18) xor d(17) xor d(15) xor d(13) xor d(12) xor d(8) xor d(6) xor d(5) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(5) xor c(6) xor c(8) xor c(12) xor c(13) xor c(15) xor c(17) xor c(18) xor c(21) xor c(23) xor c(24) xor c(27) xor c(29) xor c(31);
        newcrc(28) := d(31) xor d(27) xor d(26) xor d(24) xor d(23) xor d(22) xor d(21) xor d(19) xor d(17) xor d(14) xor d(13) xor d(12) xor d(8) xor d(5) xor d(4) xor d(3) xor d(2) xor d(1) xor d(0) xor c(0) xor c(1) xor c(2) xor c(3) xor c(4) xor c(5) xor c(8) xor c(12) xor c(13) xor c(14) xor c(17) xor c(19) xor c(21) xor c(22) xor c(23) xor c(24) xor c(26) xor c(27) xor c(31);
        newcrc(29) := d(28) xor d(27) xor d(25) xor d(24) xor d(23) xor d(22) xor d(20) xor d(18) xor d(15) xor d(14) xor d(13) xor d(9) xor d(6) xor d(5) xor d(4) xor d(3) xor d(2) xor d(1) xor c(1) xor c(2) xor c(3) xor c(4) xor c(5) xor c(6) xor c(9) xor c(13) xor c(14) xor c(15) xor c(18) xor c(20) xor c(22) xor c(23) xor c(24) xor c(25) xor c(27) xor c(28);
        newcrc(30) := d(29) xor d(28) xor d(26) xor d(25) xor d(24) xor d(23) xor d(21) xor d(19) xor d(16) xor d(15) xor d(14) xor d(10) xor d(7) xor d(6) xor d(5) xor d(4) xor d(3) xor d(2) xor c(2) xor c(3) xor c(4) xor c(5) xor c(6) xor c(7) xor c(10) xor c(14) xor c(15) xor c(16) xor c(19) xor c(21) xor c(23) xor c(24) xor c(25) xor c(26) xor c(28) xor c(29);
        newcrc(31) := d(30) xor d(29) xor d(27) xor d(26) xor d(25) xor d(24) xor d(22) xor d(20) xor d(17) xor d(16) xor d(15) xor d(11) xor d(8) xor d(7) xor d(6) xor d(5) xor d(4) xor d(3) xor c(3) xor c(4) xor c(5) xor c(6) xor c(7) xor c(8) xor c(11) xor c(15) xor c(16) xor c(17) xor c(20) xor c(22) xor c(24) xor c(25) xor c(26) xor c(27) xor c(29) xor c(30);

        return newcrc;
    end function;



    function count_bits_4(
        data : std_logic_vector(3 downto 0)--;
    ) return natural is
    begin
        case data is
        when "0000" => return 0;
        when "0001" | "0010" | "0100" | "1000" => return 1;
        when "0111" | "1011" | "1101" | "1110" => return 3;
        when "1111" => return 4;
        when others => return 2;
        end case;
    end function;

    function count_bits_32(
        data : std_logic_vector(31 downto 0)--;
    ) return natural is
    begin
        return (
            (
                count_bits_4(data(31 downto 28)) +
                count_bits_4(data(27 downto 24))
            ) + (
                count_bits_4(data(23 downto 20)) +
                count_bits_4(data(19 downto 16))
            )
        ) + (
            (
                count_bits_4(data(15 downto 12)) +
                count_bits_4(data(11 downto  8))
            ) + (
                count_bits_4(data( 7 downto  4)) +
                count_bits_4(data( 3 downto  0))
            )
        );
    end function;

    function count_bits (
        data : std_logic_vector--;
    ) return natural is
        variable data_v : std_logic_vector(data'length-1 downto 0);
    begin
        data_v := data;
        if ( data_v'length > 1 ) then
            return count_bits(data_v(data_v'length-1 downto data_v'length/2)) + count_bits(data_v(data_v'length/2-1 downto 0));
        else
            return to_integer(unsigned(data_v));
        end if;
    end function;

    function to_string (
        v : in std_logic--;
    ) return string is
        variable s : string(1 to 1);
    begin
        s(1) := std_logic'image(v)(2);
        return s;
    end function;

    function to_string (
        v : in std_logic_vector--;
    ) return string is
        variable s : string(1 to v'length);
        variable j : integer := 1;
    begin
        for i in v'range loop
            s(j) := to_string(v(i))(1);
            j := j + 1;
        end loop;
        return s;
    end function;

    function to_string (
        v : in unsigned--;
    ) return string is
    begin
        return to_string(std_logic_vector(v));
    end function;

    function to_hstring (
        v : std_logic_vector--;
    ) return string is
        variable r : string(1 to (v'length + 3) / 4) := (others => 'X');
        variable u : unsigned(v'length+3 downto 0);
        constant lut : string(1 to 16) := "0123456789ABCDEF";
    begin
        u := resize(unsigned(v), u'length);
        for i in r'range loop
            next when ( is_x(std_logic_vector(u(4*i-1 downto 4*i-4))) );
            r(r'length-i+1) := lut(1 + to_integer(u(4*i-1 downto 4*i-4)));
        end loop;
        return r;
    end function;

    function to_hstring (
        v : unsigned--;
    ) return string is
    begin
        return to_hstring(std_logic_vector(v));
    end function;

    function sgr (
        n : natural--;
    ) return string is
    begin
        return ESC & "[" & natural'image(n) & "m";
    end function;

end package body;
