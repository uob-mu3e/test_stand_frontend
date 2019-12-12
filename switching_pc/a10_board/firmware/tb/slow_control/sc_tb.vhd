library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

--  A testbench has no ports.
entity sc_tb is
end sc_tb;

architecture behav of sc_tb is
  --  Declaration of the component that will be instantiated.

  --  Specifies which entity is bound with the component.
  		
  		signal clk : std_logic;
  		signal reset_n : std_logic := '1';
  		signal writememdata : std_logic_vector(31 downto 0);
  		signal writememdata_out : std_logic_vector(31 downto 0);
  		signal writememaddr : std_logic_vector(15 downto 0);
  		signal memaddr : std_logic_vector(15 downto 0);
  		signal mem_data_out : std_logic_vector(63 downto 0);
  		signal mem_datak_out : std_logic_vector(7 downto 0);
  		signal mem_data_out_slave : std_logic_vector(31 downto 0);
  		signal mem_datak_out_slave : std_logic_vector(3 downto 0);
  		signal mem_addr_out_slave : std_logic_vector(15 downto 0);
  		signal mem_wren_slave : std_logic;

  		constant ckTime: 		time	:= 10 ns;

  		constant CODE_START : std_logic_vector(11 downto 0) := x"BAD";
  		constant CODE_STOP : std_logic_vector(31 downto 0) 	:= x"0000009C";

  		signal writememwren : std_logic;
		
begin
  --  Component instantiation.
    sc_master_0 : entity work.sc_master 
  	generic map (
		NLINKS => 2
	)
	port map(
		clk					=> clk,
		reset_n				=> reset_n,
		enable				=> '1',
		mem_data_in			=> writememdata_out,
		mem_addr			=> memaddr,
		mem_data_out		=> mem_data_out,
		mem_data_out_k		=> mem_datak_out,
		done				=> open,
		stateout			=> open
  );

  sc_slave_0 : entity work.sc_slave
    generic map (
		NLINKS => 2
	)
	port map(
		clk					=> clk,
		reset_n				=> reset_n,
		enable				=> '1',
		link_data_in		=> mem_data_out,
		link_data_in_k		=> mem_datak_out,
		mem_data_out		=> mem_data_out_slave,
		mem_addr_out		=> mem_addr_out_slave,
		mem_wren 			=> mem_wren_slave,
		mem_addr_finished_out				=> open,
		stateout			=> open
  );

	wram : entity work.ip_ram
  	port map (
        address_a => writememaddr(7 downto 0),
        address_b => (others => '0'),
        clock_a => clk,
        clock_b => '0',
        data_a => writememdata,
        data_b => (others => '0'),
        wren_a => writememwren,
        wren_b => '0',
        q_a => writememdata_out,
        q_b => open--,
  	);

     rram : entity work.ip_ram
     port map (
           address_a => mem_addr_out_slave(7 downto 0),
           address_b => (others => '0'),
           clock_a => clk,
           clock_b => '0',
           data_a => mem_data_out_slave,
           data_b => (others => '0'),
           wren_a => mem_wren_slave,
           wren_b => '0',
           q_a => open,
           q_b => open--, 
    );

  	-- generate the clock
	ckProc: process
	begin
		clk <= '0';
		wait for ckTime/2;
		clk <= '1';
		wait for ckTime/2;
	end process;

	inita : process
	begin
		reset_n	 <= '0';
		wait for 8 ns;
		reset_n	 <= '1';

		wait for 10 us;
		reset_n  <= '0';

		wait for 8 ns;
		reset_n  <= '1';
		
		wait;
	end process inita;

	memory : process(reset_n, clk)
	begin
	if(reset_n = '0')then
		writememdata <= (others => '0');
		writememaddr <= x"FFFF";
		writememwren <= '0';
		
	elsif(rising_edge(clk))then
		writememwren <= '0';
		writememaddr <= writememaddr + 1;
		if(((writememaddr > x"FFFE") or (writememaddr < x"000F")))then
			if(writememaddr(3 downto 0) = x"F")then
				writememdata(31 downto 20) <= CODE_START;
				writememwren <= '1';
			elsif(writememaddr(3 downto 0) = x"0")then
				writememdata <= x"1f0000bc";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"1")then
				writememdata <= x"0000000a";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"2")then
				writememdata <= x"00000001";
				writememwren <= '1';	
         elsif(writememaddr(3 downto 0)  = x"3")then
				writememdata <= x"0000000b";
				writememwren <= '1';	
			elsif(writememaddr(3 downto 0)  = x"4")then
				writememdata <= CODE_STOP;
				writememwren <= '1';
			end if;
		end if;

	end if;
	end process memory;
end behav;
