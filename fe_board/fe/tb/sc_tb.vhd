library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;

--  A testbench has no ports.
entity sc_tb is
end sc_tb;

architecture behav of sc_tb is
  --  Declaration of the component that will be instantiated.
	component sc_master is
		generic(
			NLINKS : integer :=4
		);
		port(
			clk:					in std_logic;
			reset_n:				in std_logic;
			enable:					in std_logic;
			mem_data_in:			in std_logic_vector(31 downto 0);
			mem_addr:				out std_logic_vector(15 downto 0);
			mem_data_out:			out std_logic_vector(NLINKS * 32 - 1 downto 0);
			mem_data_out_k:			out std_logic_vector(NLINKS * 4 - 1 downto 0);
			done:					out std_logic;
			stateout:				out std_logic_vector(27 downto 0)
			);		
	end component sc_master;

	component sc_slave is
		port(
			clk:				in std_logic;
			reset_n:			in std_logic;
			enable:				in std_logic;
			link_data_in:		in std_logic_vector(31 downto 0);
			link_data_in_k:		in std_logic_vector(3 downto 0);
			mem_data_out:		out std_logic_vector(31 downto 0);
			mem_addr_out:		out std_logic_vector(15 downto 0);
			mem_wren:			out std_logic;			
			done:				out std_logic;
			stateout:			out std_logic_vector(27 downto 0)
			);		
	end component sc_slave;

	component sram is
  		port (
    		clock   : in  std_logic;
    		reset_n	: in  std_logic;
    		we      : in  std_logic;
    		read_address   : in  std_logic_vector(15 downto 0);
    		write_address  : in  std_logic_vector(15 downto 0);
    		datain  : in  std_logic_vector(31 downto 0);
    		dataout : out std_logic_vector(31 downto 0)
  		);
	end component sram;


  --  Specifies which entity is bound with the component.
  		for sc_master_0: sc_master use entity work.sc_master;
  		
  		signal clk : std_logic;
  		signal reset_n : std_logic := '1';
  		signal writememdata : std_logic_vector(31 downto 0);
  		signal writememdata_out : std_logic_vector(31 downto 0);
  		signal writememaddr : unsigned(15 downto 0);
  		signal memaddr : std_logic_vector(15 downto 0);
  		signal mem_data_out : std_logic_vector(31 downto 0);
  		signal mem_datak_out : std_logic_vector(3 downto 0);
  		signal mem_data_out_slave : std_logic_vector(31 downto 0);
  		signal mem_datak_out_slave : std_logic_vector(3 downto 0);
  		signal mem_addr_out_slave : std_logic_vector(15 downto 0);
  		signal mem_wren_slave : std_logic;
  		signal dataout_ram : std_logic_vector(31 downto 0);
  		signal mem_addr_read_sc_s4 : std_logic_vector(15 downto 0);
  		signal mem_addr_write_sc_s4 : std_logic_vector(15 downto 0);
  		signal mem_wren_sc_s4 : std_logic;
  		signal mem_data_out_sc_s4 : std_logic_vector(31 downto 0);
  		signal mem_addr_out : std_logic_vector(31 downto 0);

  		constant ckTime: 		time	:= 10 ns;

  		constant CODE_START : std_logic_vector(11 downto 0) := x"BAD";
  		constant CODE_STOP : std_logic_vector(31 downto 0) 	:= x"0000009C";

  		signal writememwren : std_logic;

    signal ram_re, ram_rvalid : std_logic;

begin
  --  Component instantiation.

	wram : sram
		port map (
		clock   		=> clk,
		reset_n			=> reset_n,
		we      	  	=> writememwren,
		read_address  	=> memaddr,
		write_address 	=> std_logic_vector(writememaddr),
		datain 			=> writememdata,
		dataout 		=> writememdata_out
	);

	sc_master_0: sc_master
		generic map (
		NLINKS => 1
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

    process(clk)
    begin
    if rising_edge(clk) then
        ram_rvalid <= ram_re;
    end if;
    end process;

    e_sc : entity work.sc_rx
    port map (
        i_link_data => mem_data_out,
        i_link_datak => mem_datak_out,

        o_fifo_we => open,
        o_fifo_wdata => open,

        o_ram_addr => mem_addr_out,
        o_ram_re => ram_re,
        i_ram_rdata => dataout_ram,
        i_ram_rvalid => ram_rvalid,
        o_ram_we => mem_wren_sc_s4,
        o_ram_wdata => mem_data_out_sc_s4,

        i_reset_n => reset_n,
        i_clk => clk--,
    );


	mem_addr_write_sc_s4 <= mem_addr_out(15 downto 0);
	mem_addr_read_sc_s4 <= mem_addr_out(15 downto 0);

	rram : sram
  	port map (
		clock   		=> clk,
		reset_n			=> reset_n,
		we      	  	=> mem_wren_sc_s4,
		read_address  	=> mem_addr_read_sc_s4,
		write_address 	=> mem_addr_write_sc_s4,
		datain 			=> mem_data_out_sc_s4,
		dataout 		=> dataout_ram
  	);


	--sc_slave_0: sc_slave
	--port map(
	--	clk					=> clk,
	--	reset_n				=> reset_n,
	--	enable				=> '1',
	--	link_data_in		=> mem_data_out,
	--	link_data_in_k		=> mem_datak_out,
	--	mem_data_out		=> mem_data_out_slave,
	--	mem_addr_out		=> mem_addr_out_slave,
	--	mem_wren 			=> mem_wren_slave,
	--	done				=> open,
	--	stateout			=> open
	--);








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
				writememdata(7 downto 0) <= x"BC"; -- K28.5
				writememdata(23 downto 8) <= (others => '0'); --FPGA ID
				writememdata(25 downto 24) <= "11"; -- SC Type write
				writememdata(31 downto 26) <= "000111";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"1")then
				writememdata <= x"00000001";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"2")then
				writememdata <= x"00000005";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"3")then
				writememdata <= x"CAFEBABE";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"4")then
				writememdata <= x"BABEBABE";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"5")then
				writememdata <= x"AAAAAAAA";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"6")then
				writememdata <= x"CCCCCCCC";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"7")then
				writememdata <= x"DDDDDDDD";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"8")then
				writememdata <= CODE_STOP;
				writememwren <= '1';
			elsif(writememaddr(3 downto 0) = x"9")then
				writememdata(31 downto 20) <= CODE_START;
				writememwren <= '1';
			elsif(writememaddr(3 downto 0) = x"A")then
				writememdata(7 downto 0) <= x"BC"; -- K28.5
				writememdata(23 downto 8) <= (others => '0'); --FPGA ID
				writememdata(25 downto 24) <= "10"; -- SC Type read
				writememdata(31 downto 26) <= "000111";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"B")then
				writememdata <= x"00000001";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"C")then
				writememdata <= x"00000005";
				writememwren <= '1';
			elsif(writememaddr(3 downto 0)  = x"D")then
				writememdata <= CODE_STOP;
				writememwren <= '1';
			end if;
		end if;

	end if;
	end process memory;
end behav;
