library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_sc_rx is
end entity;

architecture arch of tb_sc_rx is

    constant CLK_MHZ : real := 100.0;
    signal clk, reset_n : std_logic := '0';

    signal link_data : std_logic_vector(31 downto 0);
    signal link_datak : std_logic_vector(3 downto 0);

    signal fifo_rempty : std_logic;
    signal fifo_rdata : std_logic_vector(35 downto 0);
    signal fifo_rack : std_logic;

    type ram_t is array (natural range <>) of std_logic_vector(31 downto 0);
    signal ram : ram_t(0 to 15);

    signal ram_addr : std_logic_vector(31 downto 0);
    signal ram_re : std_logic;
    signal ram_rdata : std_logic_vector(31 downto 0);
    signal ram_rvalid : std_logic;
    signal ram_we : std_logic;
    signal ram_wdata : std_logic_vector(31 downto 0);

    signal DONE : unsigned(2 downto 0) := (others => '0');

begin

    clk <= not clk after (0.5 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);

    e_sc : entity work.sc_rx
    port map (
        i_link_data => link_data,
        i_link_datak => link_datak,

        o_fifo_rempty => fifo_rempty,
        o_fifo_rdata => fifo_rdata,
        i_fifo_rack => fifo_rack,

        o_ram_addr => ram_addr,
        o_ram_re => ram_re,
        i_ram_rdata => ram_rdata,
        i_ram_rvalid => ram_rvalid,
        o_ram_we => ram_we,
        o_ram_wdata => ram_wdata,

        i_reset_n => reset_n,
        i_clk => clk--,
    );
    fifo_rack <= not fifo_rempty;

    process(clk)
        variable i : integer;
    begin
    if rising_edge(clk) then
        if ( is_x(ram_addr) ) then
            i := 0;
        else
            i := to_integer(unsigned(ram_addr));
        end if;
        if ( ram_we = '1' ) then
            ram(i) <= ram_wdata;
        end if;
        ram_rdata <= ram(i);
        ram_rvalid <= ram_re;
    end if;
    end process;

    -- link
    process
    begin
        -- idle
        link_data <= X"000000BC";
        link_datak <= "0001";

        wait until rising_edge(reset_n);

    ----------------------------------------------------------------------------
    -- write
    ----------------------------------------------------------------------------

        wait until rising_edge(clk);
        -- SC, write, ID = 0
        link_data <= "000111" & "11" & X"0000" & X"BC";
        link_datak <= "0001";

        wait until rising_edge(clk);
        -- start address
        link_data <= X"00000008";
        link_datak <= "0000";

        wait until rising_edge(clk);
        -- length
        link_data <= X"00000004";
        link_datak <= "0000";

        for i in 0 to 15 loop
            wait until rising_edge(clk);
            -- idle
            link_data <= X"000000BC";
            link_datak <= "0001";
        end loop;

        wait until rising_edge(clk);
        -- data[0]
        link_data <= X"00112233";
        link_datak <= "0000";

        wait until rising_edge(clk);
        -- data[1]
        link_data <= X"11223344";
        link_datak <= "0000";

        wait until rising_edge(clk);
        -- data[2]
        link_data <= X"22334455";
        link_datak <= "0000";

        wait until rising_edge(clk);
        -- data[3]
        link_data <= X"33445566";
        link_datak <= "0000";

        wait until rising_edge(clk);
        -- stop
        link_data <= X"0000009C";
        link_datak <= "0001";

        wait until rising_edge(clk);
        -- idle
        link_data <= X"000000BC";
        link_datak <= "0001";

        wait for 100 ns;

    ----------------------------------------------------------------------------
    -- read
    ----------------------------------------------------------------------------

        wait until rising_edge(clk);
        -- SC, read, ID = 0
        link_data <= "000111" & "10" & X"0000" & X"BC";
        link_datak <= "0001";

        wait until rising_edge(clk);
        -- start address
        link_data <= X"00000008";
        link_datak <= "0000";

        wait until rising_edge(clk);
        -- length
        link_data <= X"00000004";
        link_datak <= "0000";

        wait until rising_edge(clk);
        -- stop
        link_data <= X"0000009C";
        link_datak <= "0001";

        wait until rising_edge(clk);
        -- idle
        link_data <= X"000000BC";
        link_datak <= "0001";

        wait for 100 ns;

    ----------------------------------------------------------------------------

        DONE(2) <= '1';

        wait;
    end process;

    -- check ram input
    process
    begin

        wait until rising_edge(reset_n);

    ----------------------------------------------------------------------------
    -- write
    ----------------------------------------------------------------------------

        -- wdata[0]
        report "wait wdata[0]";
        wait until rising_edge(clk) and ram_we = '1';
        assert ( ram_addr = X"00000008" ) severity error;
        assert ( ram_wdata = X"00112233" ) severity error;

        -- wdata[1]
        report "wait wdata[1]";
        wait until rising_edge(clk) and ram_we = '1';
        assert ( ram_addr = X"00000009" ) severity error;
        assert ( ram_wdata = X"11223344" ) severity error;

        -- wdata[2]
        report "wait wdata[2]";
        wait until rising_edge(clk) and ram_we = '1';
        assert ( ram_addr = X"0000000A" ) severity error;
        assert ( ram_wdata = X"22334455" ) severity error;

        -- wdata[3]
        report "wait wdata[3]";
        wait until rising_edge(clk) and ram_we = '1';
        assert ( ram_addr = X"0000000B" ) severity error;
        assert ( ram_wdata = X"33445566" ) severity error;

        wait until rising_edge(clk);
        assert ( ram_we = '0' );

    ----------------------------------------------------------------------------
    -- read
    ----------------------------------------------------------------------------

        report "wait rdata[0]";
        wait until rising_edge(clk) and ram_re = '1';
        assert ( ram_addr = X"00000008" ) severity error;

        report "wait rdata[1]";
        wait until rising_edge(clk) and ram_re = '1';
        assert ( ram_addr = X"00000009" ) severity error;

        report "wait rdata[2]";
        wait until rising_edge(clk) and ram_re = '1';
        assert ( ram_addr = X"0000000A" ) severity error;

        report "wait rdata[3]";
        wait until rising_edge(clk) and ram_re = '1';
        assert ( ram_addr = X"0000000B" ) severity error;

        wait until rising_edge(clk);
        assert ( ram_re = '0' );

    ----------------------------------------------------------------------------

        DONE(1) <= '1';

        wait;
    end process;

    -- check fifo input
    process
    begin

        wait until rising_edge(reset_n);

    ----------------------------------------------------------------------------
    -- write
    ----------------------------------------------------------------------------

        -- ack
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "1110" & X"00000008" ) severity error;

        -- length
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "0000" & X"00010004" ) severity error;

        -- stop
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "0011" & X"00000000" ) severity error;

        wait until rising_edge(clk);
        assert ( fifo_rempty = '1' ) severity error;

    ----------------------------------------------------------------------------
    -- read
    ----------------------------------------------------------------------------

        -- ack
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "1010" & X"00000008" ) severity error;

        -- length
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "0000" & X"00010004" ) severity error;

        -- data[0]
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "0000" & X"00112233" ) severity error;

        -- data[1]
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "0000" & X"11223344" ) severity error;

        -- data[2]
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "0000" & X"22334455" ) severity error;

        -- data[3]
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "0000" & X"33445566" ) severity error;

        -- stop
        wait until rising_edge(clk) and fifo_rempty = '0';
        assert ( fifo_rdata = "0011" & X"00000000" ) severity error;

        wait until rising_edge(clk);
        assert ( fifo_rempty = '1' ) severity error;

    ----------------------------------------------------------------------------

        DONE(0) <= '1';

        wait;
    end process;

    process
    begin
        wait for 1000 ns;
        assert ( DONE = (DONE'range => '1') ) severity error;
        wait;
    end process;

end architecture;
