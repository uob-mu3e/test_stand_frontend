library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

--
-- SC RAM
--
-- - ram port has priority
-- - map upper 256 words to reg port
--
entity sc_ram is
generic (
    RAM_ADDR_WIDTH_g : positive := 16--;
);
port (
    -- ram slave
    i_ram_addr          : in    std_logic_vector(15 downto 0) := (others => 'X');
    i_ram_re            : in    std_logic := '0';
    o_ram_rvalid        : out   std_logic;
    o_ram_rdata         : out   std_logic_vector(31 downto 0);
    i_ram_we            : in    std_logic := '0';
    i_ram_wdata         : in    std_logic_vector(31 downto 0) := (others => 'X');

    -- avalon slave
    -- address units - words
    -- read latency - 1
    i_avs_address       : in    std_logic_vector(15 downto 0) := (others => 'X');
    i_avs_read          : in    std_logic := '0';
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic := '0';
    i_avs_writedata     : in    std_logic_vector(31 downto 0) := (others => 'X');
    o_avs_waitrequest   : out   std_logic;

    -- reg master
    o_reg_addr          : out   std_logic_vector(7 downto 0);
    o_reg_re            : out   std_logic;
    i_reg_rdata         : in    std_logic_vector(31 downto 0) := (others => 'X');
    o_reg_we            : out   std_logic;
    o_reg_wdata         : out   std_logic_vector(31 downto 0);

    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture arch of sc_ram is

    signal ram_addr : std_logic_vector(15 downto 0);
    signal ram_re, ram_re_q : std_logic;
    signal ram_rdata : std_logic_vector(31 downto 0);
    signal ram_we : std_logic;
    signal ram_wdata : std_logic_vector(31 downto 0);

    signal avs_waitrequest : std_logic;

    signal reg_re, reg_re_q, reg_we : std_logic;

begin

    -- psl default clock is rising_edge(i_clk) ;

    -- psl assert always ( i_ram_re = '0' or i_ram_we = '0' ) ;
    -- psl assert always ( i_avs_read = '0' or i_avs_write = '0' ) ;
    -- psl assert always ( o_reg_re = '0' or o_reg_we = '0' ) ;

    -- psl assert always ( i_ram_re -> next o_ram_rvalid ) ;



    e_ram : entity work.ram_1r1w
    generic map (
        DATA_WIDTH_g => 32,
        ADDR_WIDTH_g => RAM_ADDR_WIDTH_g--,
    )
    port map (
        i_raddr => ram_addr(RAM_ADDR_WIDTH_g-1 downto 0),
        o_rdata => ram_rdata,
        i_rclk => i_clk,

        i_waddr => ram_addr(RAM_ADDR_WIDTH_g-1 downto 0),
        i_wdata => ram_wdata,
        i_we => ram_we,
        i_wclk => i_clk--,
    );

    ram_addr <=
        (others => '0') when ( reg_re = '1' or reg_we = '1' ) else
        i_ram_addr when ( i_ram_re = '1' or i_ram_we = '1' ) else
        i_avs_address when ( i_avs_read = '1' or i_avs_write = '1' ) else
        (others => '0');
    ram_re <=
        '0' when ( ram_addr(15 downto RAM_ADDR_WIDTH_g) /= (15 downto RAM_ADDR_WIDTH_g => '0') ) else
        '1' when ( i_ram_re = '1' and reg_re = '0' ) else
        '1' when ( i_avs_read = '1' and reg_re = '0' and avs_waitrequest = '0' ) else
        '0';
    ram_we <=
        '0' when ( ram_addr(15 downto RAM_ADDR_WIDTH_g) /= (15 downto RAM_ADDR_WIDTH_g => '0') ) else
        '1' when ( i_ram_we = '1' and reg_we = '0' ) else
        '1' when ( i_avs_write = '1' and reg_we = '0' and avs_waitrequest = '0' ) else
        '0';
    ram_wdata <=
        i_ram_wdata when ( i_ram_we = '1' and reg_we = '0' ) else
        i_avs_writedata when ( i_avs_write = '1' and reg_we = '0' ) else
        (others => '0');



    o_avs_waitrequest <= avs_waitrequest;
    avs_waitrequest <=
        '1' when ( i_ram_re = '1' or i_ram_we = '1' ) else
        '0';



    o_reg_addr <=
        i_ram_addr(7 downto 0) when ( i_ram_re = '1' or i_ram_we = '1' ) else
        i_avs_address(7 downto 0) when ( i_avs_read = '1' or i_avs_write = '1' ) else
        (others => '0');

    o_reg_re <= reg_re;
    reg_re <=
        '1' when ( i_ram_addr(15 downto 8) = X"FF" and i_ram_re = '1' ) else
        '1' when ( i_avs_address(15 downto 8) = X"FF" and i_avs_read = '1' and avs_waitrequest = '0' ) else
        '0';

    o_reg_we <= reg_we;
    reg_we <=
        '1' when ( i_ram_addr(15 downto 8) = X"FF" and i_ram_we = '1' ) else
        '1' when ( i_avs_address(15 downto 8) = X"FF" and i_avs_write = '1' and avs_waitrequest = '0' ) else
        '0';

    o_reg_wdata <=
        i_ram_wdata when ( reg_we = '1' and i_ram_we = '1' ) else
        i_avs_writedata when ( reg_we = '1' and i_avs_write = '1' ) else
        (others => '0');



    o_ram_rdata <=
        ram_rdata when ( ram_re_q = '1' ) else
        i_reg_rdata when ( reg_re_q = '1' ) else
        X"CCCCCCCC";

    o_avs_readdata <=
        ram_rdata when ( ram_re_q = '1' ) else
        i_reg_rdata when ( reg_re_q = '1' ) else
        X"CCCCCCCC";



    process(i_clk, i_reset_n)
    begin
    if ( i_reset_n = '0' ) then
        o_ram_rvalid <= '0';
        ram_re_q <= '0';
        reg_re_q <= '0';
        --
    elsif rising_edge(i_clk) then
        o_ram_rvalid <= i_ram_re;
        ram_re_q <= ram_re;
        reg_re_q <= reg_re;
        --
    end if;
    end process;

end architecture;
