--
-- author : Alexandr Kozlinskiy
-- date : 2018-03-30
--

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package mu3e is

    --                                data  datak  idl   sop  dthdr sbhdr  eop
    constant LINK_LENGTH : positive := 32  +  4  +  1  +  1  +  1  +  1  +  1;

    type link_t is record
        data    : std_logic_vector(31 downto 0);
        datak   : std_logic_vector(3 downto 0);
        idle    : std_logic;
        sop     : std_logic; -- start of packet (preamble)
        dthdr   : std_logic; -- DaTa HeaDeR
        sbhdr   : std_logic; -- SuB HeaDeR
        eop     : std_logic; -- end of packet (trailer)
    end record;
    type link_array_t is array(natural range <>) of link_t;

    constant LINK_ZERO : link_t := (
        data => X"00000000",
        datak => "0000",
        others => '0'
    );

    constant LINK_IDLE : link_t := (
        data => X"000000" & work.util.D28_5,
        datak => "0001",
        idle => '1',
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

    function to_link_array (
        data : work.util.slv32_array_t;
        datak : work.util.slv4_array_t--;
    ) return link_array_t;

end package;

package body mu3e is

    function to_link (
        data : std_logic_vector(31 downto 0);
        datak : std_logic_vector(3 downto 0)--;
    ) return link_t is
        variable link : link_t;
        variable length : integer := 0;
    begin
        link.data := data;
        length := length + 32;
        link.datak := datak;
        length := length + 4;

        link.idle := work.util.to_std_logic(
            datak = "0001" and data(7 downto 0) = work.util.D28_5
            and data(31 downto 26) = "000000"
        );
        length := length + 1;

        link.sop := not link.idle and work.util.to_std_logic(
            datak = "0001" and data(7 downto 0) = work.util.D28_5 -- BC
        );
        length := length + 1;

        link.dthdr := '0';
        length := length + 1;

        link.sbhdr := '0';
        length := length + 1;

        link.eop := not link.idle and work.util.to_std_logic(
            datak = "0001" and data(7 downto 0) = work.util.D28_4 -- 9C
        );
        length := length + 1;

        assert ( length = LINK_LENGTH ) severity failure;
        return link;
    end function;

    function to_link (
        slv : std_logic_vector(LINK_LENGTH-1 downto 0)--;
    ) return link_t is
        variable link : link_t;
        variable length : integer := 0;
    begin
        link.data := slv(31 downto 0);
        length := length + 32;
        link.datak := slv(35 downto 32);
        length := length + 4;

        link.idle := slv(36);
        length := length + 1;

        link.sop := slv(37);
        length := length + 1;

        link.dthdr := slv(38);
        length := length + 1;

        link.sbhdr := slv(39);
        length := length + 1;

        link.eop := slv(40);
        length := length + 1;

        assert ( length = LINK_LENGTH ) severity failure;
        return link;
    end function;

    function to_slv (
        link : link_t--;
    ) return std_logic_vector is
        variable slv : std_logic_vector(LINK_LENGTH-1 downto 0);
        variable length : integer := 0;
    begin
        slv(31 downto 0) := link.data;
        length := length + 32;
        slv(35 downto 32) := link.datak;
        length := length + 4;

        slv(36) := link.idle;
        length := length + 1;

        slv(37) := link.sop;
        length := length + 1;

        slv(38) := link.dthdr;
        length := length + 1;

        slv(39) := link.sbhdr;
        length := length + 1;

        slv(40) := link.eop;
        length := length + 1;

        assert ( length = LINK_LENGTH ) severity failure;
        return slv;
    end function;

    function to_link_array (
        data : work.util.slv32_array_t;
        datak : work.util.slv4_array_t--;
    ) return link_array_t is
        variable links : link_array_t(data'range);
    begin
        for i in data'range loop
            links(i) := to_link(data(i), datak(i));
        end loop;
        return links;
    end function;

end package body;
