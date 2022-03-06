--
-- author : Alexandr Kozlinskiy
-- date : 2018-03-30
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package mu3e is

    constant LINK_LENGTH : positive := 32 + 4 + 1;

    type link_t is record
        data    : std_logic_vector(31 downto 0);
        datak   : std_logic_vector(3 downto 0);

        idle    : std_logic;
    end record;
    type link_array_t is array(natural range <>) of link_t;

    constant LINK_IDLE : link_t := (
        data => X"000000" & work.util.D28_5,
        datak => "0001",
        others => '0'
    );

    subtype RANGE_LINK_FPGA_ID is integer range 23 downto 8;

    function to_link (
        data : std_logic_vector(31 downto 0);
        datak : std_logic_vector(3 downto 0)--;
    ) return link_t;

    function to_link (
        slv : std_logic_vector(LINK_LENGTH-1 downto 0)--;
    ) return link_t;

    function to_slv (
        link : link_t--;
    ) return std_logic_vector;

end package;

package body mu3e is

    function to_link (
        data : std_logic_vector(31 downto 0);
        datak : std_logic_vector(3 downto 0)--;
    ) return link_t is
        variable link : link_t;
    begin
        link.data := data;
        link.datak := datak;

        link.idle := work.util.to_std_logic(
            datak = "0001" and data(7 downto 0) = work.util.D28_5
            and data(31 downto 26) = "000000"
        );

        return link;
    end function;

    function to_link (
        slv : std_logic_vector(LINK_LENGTH-1 downto 0)--;
    ) return link_t is
        variable link : link_t;
    begin
        link.data := slv(31 downto 0);
        link.datak := slv(35 downto 32);
        link.idle := slv(36);
        return link;
    end function;

    function to_slv (
        link : link_t--;
    ) return std_logic_vector is
    begin
        return link.idle & link.datak & link.data;
    end function;

end package body;
