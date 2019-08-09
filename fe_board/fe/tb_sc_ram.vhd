library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;

entity tb_sc_ram is
end entity;

architecture arch of tb_sc_ram is

    constant CLK_MHZ : real := 100.0;
    signal clk, reset_n : std_logic := '0';

    signal ram_addr : std_logic_vector(15 downto 0);
    signal ram_re : std_logic;
    signal ram_rvalid : std_logic;
    signal ram_rdata : std_logic_vector(31 downto 0);
    signal ram_we : std_logic;
    signal ram_wdata : std_logic_vector(31 downto 0);

    signal avs_address : std_logic_vector(15 downto 0);
    signal avs_read : std_logic;
    signal avs_readdata : std_logic_vector(31 downto 0);
    signal avs_write : std_logic;
    signal avs_writedata : std_logic_vector(31 downto 0);
    signal avs_waitrequest : std_logic;

    signal reg_addr : std_logic_vector(7 downto 0);
    signal reg_re : std_logic;
    signal reg_rdata : std_logic_vector(31 downto 0);
    signal reg_we : std_logic;
    signal reg_wdata : std_logic_vector(31 downto 0);

    signal DONE : unsigned(2 downto 0) := (others => '0');

begin

    clk <= not clk after (0.5 us / CLK_MHZ);
    reset_n <= '0', '1' after (1.0 us / CLK_MHZ);

    e_sc_ram : entity work.sc_ram
    generic map (
        RAM_ADDR_WIDTH_g => 8--,
    )
    port map (
        i_ram_addr          => ram_addr,
        i_ram_re            => ram_re,
        o_ram_rvalid        => ram_rvalid,
        o_ram_rdata         => ram_rdata,
        i_ram_we            => ram_we,
        i_ram_wdata         => ram_wdata,

        i_avs_address       => avs_address,
        i_avs_read          => avs_read,
        o_avs_readdata      => avs_readdata,
        i_avs_write         => avs_write,
        i_avs_writedata     => avs_writedata,
        o_avs_waitrequest   => avs_waitrequest,

        o_reg_addr          => reg_addr,
        o_reg_re            => reg_re,
        i_reg_rdata         => reg_rdata,
        o_reg_we            => reg_we,
        o_reg_wdata         => reg_wdata,

        i_reset_n => reset_n,
        i_clk => clk--,
    );

    process(clk)
    begin
    if ( reset_n = '0' ) then
        --
    elsif rising_edge(clk) then
        reg_rdata <= X"EEEE0000" + reg_addr;

        assert ( ram_re = '0' or ram_we = '0' ) severity failure;
        assert ( avs_read = '0' or avs_write = '0' ) severity failure;

        if ( ram_re /= ram_we and ram_addr >= X"FF00" ) then
            assert ( reg_addr = ram_addr(7 downto 0) ) severity error;
            assert ( reg_re = ram_re and reg_we = ram_we ) severity error;
        end if;

        if ( ram_re /= ram_we and ram_addr < X"FF00" ) then
            assert ( reg_re = '0' and reg_we = '0' ) severity error;
        end if;

        if ( ram_re /= ram_we ) then
            assert ( avs_waitrequest = '1' ) severity error;
        end if;

        if ( avs_read /= avs_write and avs_address >= X"FF00" and avs_waitrequest = '0' ) then
            assert ( reg_addr = avs_address(7 downto 0) ) severity error;
            assert ( reg_re = avs_read and reg_we = avs_write ) severity error;
        end if;

        if ( avs_read /= avs_write and avs_address < X"FF00" and avs_waitrequest = '0' ) then
            assert ( reg_re = '0' and reg_we = '0' ) severity error;
        end if;
        --
    end if;
    end process;

    process
    begin
        ram_re <= '0';
        ram_we <= '0';
        avs_read <= '0';
        avs_write <= '0';

        wait until rising_edge(reset_n);



        -- write to ram (ram port)
        for i in 0 to 15 loop
            wait until rising_edge(clk);
            ram_addr <= X"0000" + i;
            ram_we <= '1';
            ram_wdata <= X"DDDD0000" + i;

            wait until rising_edge(clk);
            ram_we <= '0';
        end loop;

        -- read from ram (ram port)
        for i in 0 to 15 loop
            wait until rising_edge(clk);
            ram_addr <= X"0000" + i;
            ram_re <= '1';

            wait until rising_edge(clk);
            ram_re <= '0';

            wait until rising_edge(clk);
            assert ( ram_rvalid = '1' and ram_rdata = X"DDDD0000" + i ) severity error;
        end loop;

        -- read from reg (ram port)
        for i in 0 to 15 loop
            wait until rising_edge(clk);
            ram_addr <= X"FF00" + i;
            ram_re <= '1';

            wait until rising_edge(clk);
            ram_re <= '0';

            wait until rising_edge(clk);
            assert ( ram_rvalid = '1' and ram_rdata = X"EEEE0000" + i ) severity error;
        end loop;



        -- write to ram (avalon port)
        for i in 0 to 15 loop
            wait until rising_edge(clk);
            avs_address <= X"0010" + i;
            avs_write <= '1';
            avs_writedata <= X"DDDD0010" + i;

            wait until rising_edge(clk);
            avs_write <= '0';
        end loop;

        -- read from ram (avalon port)
        for i in 0 to 15 loop
            wait until rising_edge(clk);
            avs_address <= X"0010" + i;
            avs_read <= '1';

            wait until rising_edge(clk);
            avs_read <= '0';

            wait until rising_edge(clk);
            assert ( avs_waitrequest = '0' and avs_readdata = X"DDDD0010" + i ) severity error;
        end loop;

        -- read from reg (avalon port)
        for i in 0 to 15 loop
            wait until rising_edge(clk);
            avs_address <= X"FF10" + i;
            avs_read <= '1';

            wait until rising_edge(clk);
            avs_read <= '0';

            wait until rising_edge(clk);
            assert ( avs_waitrequest = '0' and avs_readdata = X"EEEE0010" + i ) severity error;
        end loop;



        -- write to ram (ram and avalon ports)
        for i in 0 to 15 loop
            wait until rising_edge(clk);
            ram_addr <= X"0020" + i;
            ram_we <= '1';
            ram_wdata <= X"DDDD0020" + i;
            avs_address <= X"0030" + i;
            avs_write <= '1';
            avs_writedata <= X"DDDD0030" + i;

            wait until rising_edge(clk);
            ram_we <= '0';
            assert ( avs_waitrequest = '1' ) severity error;

            wait until rising_edge(clk);
            assert ( avs_waitrequest = '0' ) severity error;
            avs_write <= '0';
        end loop;

        -- read from ram (ram and avalon ports)
        for i in 0 to 15 loop
            wait until rising_edge(clk);
            ram_addr <= X"0020" + i;
            ram_re <= '1';
            avs_address <= X"0030" + i;
            avs_read <= '1';

            wait until rising_edge(clk);
            ram_re <= '0';
            assert ( avs_waitrequest = '1' ) severity error;

            wait until rising_edge(clk);
            assert ( ram_rvalid = '1' and ram_rdata = X"DDDD0020" + i ) severity error;
            assert ( avs_waitrequest = '0' ) severity error;
            avs_read <= '0';

            wait until rising_edge(clk);
            assert ( avs_waitrequest = '0' and avs_readdata = X"DDDD0030" + i ) severity error;
        end loop;



        wait;
    end process;

end architecture;
