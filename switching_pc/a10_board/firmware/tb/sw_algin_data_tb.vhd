library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

--  A testbench has no ports.
entity sw_algin_data_tb is
end entity;

architecture behav of sw_algin_data_tb is
  --  Declaration of the component that will be instantiated.
		component sw_algin_data is
			generic(
				NLINKS : integer := 4
			);
			port (
				clks_read 		: in  std_logic_vector(NLINKS - 1 downto 0); -- 312,50 MHZ
				clks_write     : in  std_logic_vector(NLINKS - 1 downto 0); -- 156,25 MHZ
				
				clk_node_write : in  std_logic; -- 312,50 MHZ
				clk_node_read  : in  std_logic;

				reset_n			: in  std_logic;
				
				data_in 			: in std_logic_vector(NLINKS * 32 - 1 downto 0);
				fpga_id_in		: in std_logic_vector(NLINKS * 16 - 1 downto 0);
						
				enables_in		: in std_logic_vector(NLINKS - 1 downto 0);
				
				node_rdreq		: in std_logic;
				
				data_out	      : out std_logic_vector((NLINKS / 4) * 64 - 1 downto 0);
				state_out		: out std_logic_vector(3 downto 0);
				node_full_out  : out std_logic_vector(NLINKS / 4 - 1 downto 0);
				node_empty_out	: out std_logic_vector(NLINKS / 4 - 1 downto 0)
		);		
		end component sw_algin_data;

  		signal clk_156 : std_logic;
		signal clk_312 : std_logic;
		
  		signal reset_n : std_logic;
		signal enable : std_logic;
		
		signal link_0  : std_logic_vector(31 downto 0);
		signal link_1  : std_logic_vector(31 downto 0);
		signal link_2  : std_logic_vector(31 downto 0);
		signal link_3  : std_logic_vector(31 downto 0);

		signal link_0_k : std_logic;
		signal link_1_k : std_logic;
		signal link_2_k : std_logic;
		signal link_3_k : std_logic;
		
		signal global_time_0  : std_logic_vector(47 downto 0);
		signal global_time_1  : std_logic_vector(47 downto 0);
		signal global_time_2  : std_logic_vector(47 downto 0);
		signal global_time_3  : std_logic_vector(47 downto 0);

  		constant time156: 		time	:= 6.4 ns; -- 156,25 MHZ
		
		signal counter : std_logic_vector(31 downto 0);
		
		signal clks_read : std_logic_vector(4 - 1 downto 0);
		signal clks_write : std_logic_vector(4 - 1 downto 0);
		signal data_in : std_logic_vector(4 * 32 - 1 downto 0);
		signal fpga_id_in : std_logic_vector(4 * 16 - 1 downto 0);
		signal enables_in : std_logic_vector(4 - 1 downto 0);
		
		signal data_out	      : std_logic_vector((4 / 4) * 64 - 1 downto 0);
		signal state_out		: std_logic_vector(3 downto 0);
		signal node_full_out  : std_logic_vector(4 / 4 - 1 downto 0);
		signal node_empty_out	: std_logic_vector(4 / 4 - 1 downto 0);
		
begin
  --  Component instantiation.
  algining_data : sw_algin_data
	generic map(
		NLINKS => 4
	)
	port map(
		clks_read         	 => clks_read, -- 156,25 MHZ
		clks_write			    => clks_write, -- 312,50 MHZ

		clk_node_write      	 => clk_312,--: in  std_logic; -- 156,25 MHZ
		clk_node_read     	 => clk_312,--: in  std_logic; -- To be defined

		reset_n					 => reset_n,--: in  std_logic;
		
		data_in					 => data_in,
		fpga_id_in			    => fpga_id_in, -- FPGA-ID
		
		enables_in				 => enables_in,
		
		node_rdreq				 => '1',
		
		data_out					 => data_out,
		state_out				 => state_out,
		node_full_out			 => node_full_out,
		node_empty_out			 => node_empty_out
	);
	
	clks_read <= clk_312 & clk_312 & clk_312 & clk_312;
	clks_write <= clk_156 & clk_156 & clk_156 & clk_156;
	data_in <= link_0 & link_1 & link_2 & link_3;
	fpga_id_in <= "0000000000000001" & "0000000000000011" & "0000000000000111" & "0000000000001111";
	enables_in <= link_0_k & link_1_k & link_2_k & link_3_k;
	
	
  	-- generate the clock
	ck156: process
	begin
		clk_156 <= '0';
		wait for time156/2;
		clk_156 <= '1';
		wait for time156/2;
	end process;
	
	ck312: process
	begin
		clk_312 <= '0';
		wait for time156/4;
		clk_312 <= '1';
		wait for time156/4;
	end process;

	inita : process
	begin
		reset_n	 <= '0';
		enable	<= '0';
		wait for 20 ns;
		reset_n	 <= '1';

		wait for 20 ns;
		reset_n  <= '0';

		wait for 20 ns;
		reset_n  <= '1';
		
		reset_n	 <= '0';
		wait for 20 ns;
		reset_n	 <= '1';

		wait for 20 ns;
		reset_n  <= '0';

		wait for 20 ns;
		reset_n  <= '1';
		enable	<= '1';
		
		wait;
	end process inita;
	
	global_time : process(reset_n, clk_156)
	begin
	if(reset_n = '0')then
		global_time_0(47 downto 4) <= (others => '0');
		global_time_1(47 downto 4) <= (others => '0');
		global_time_2(47 downto 4) <= (others => '0');
		global_time_3(47 downto 4) <= (others => '0');
		global_time_0(3 downto 0) <= "0001";
		global_time_1(3 downto 0) <= "0000";
		global_time_2(3 downto 0) <= "0010";
		global_time_3(3 downto 0) <= "0011";
		
	elsif(rising_edge(clk_156))then
		if (enable = '1') then
			global_time_0 <= global_time_0 + '1';
			global_time_1 <= global_time_1 + '1';
			global_time_2 <= global_time_2 + '1';
			global_time_3 <= global_time_3 + '1';
		end if;
	end if;
	end process global_time;

	data : process(reset_n, clk_156)
	begin
	if(reset_n = '0')then
		link_0 <= (others => '0');
		link_1 <= (others => '0');
		link_2 <= (others => '0');
		link_3 <= (others => '0');
		link_0_k <= '0';
		link_1_k <= '0';
		link_2_k <= '0';
		link_3_k <= '0';
		counter <= (others => '0');
		
	elsif(rising_edge(clk_156))then
		if (enable = '1') then
			link_0_k <= '0';
			link_1_k <= '0';
			link_2_k <= '0';
			link_3_k <= '0';
			link_0 <= (others => '0');
			link_1 <= (others => '0');
			link_2 <= (others => '0');
			link_3 <= (others => '0');
			counter <= counter + '1';
			
			if(conv_integer(counter) = 0)then
				link_0_k <= '1';
				link_1_k <= '1';
				link_2_k <= '1';
				link_3_k <= '1';
				link_0(15 downto 0) <= "0000001110100000";
				link_1(15 downto 0) <= "0000001110100000";
				link_2(15 downto 0) <= "0000001110100000";
				--link_3(15 downto 0) <= "0000001110100000";
				link_3 <= x"ABCDE00D";
				link_0(31 downto 16) <= global_time_0(47 downto 32);
				link_1(31 downto 16) <= global_time_1(47 downto 32);
				link_2(31 downto 16) <= global_time_2(47 downto 32);
				link_3(31 downto 16) <= global_time_3(47 downto 32);
				
			elsif(conv_integer(counter) = 1)then
				link_0_k <= '1';
				link_1_k <= '1';
				link_2_k <= '1';
				link_3_k <= '1';
				link_0 <= global_time_0(31 downto 0);
				link_1 <= global_time_1(31 downto 0);
				link_2 <= global_time_2(31 downto 0);
				--link_3 <= global_time_3(31 downto 0);
				link_3 <= x"ABCDEBAD";
				
			elsif(conv_integer(counter) = 2)then
				link_0_k <= '1';
				link_1_k <= '1';
				link_2_k <= '1';
				link_3_k <= '1';
				link_0(9 downto 4) <= "111111";
				link_1(9 downto 4) <= "111111";
				link_2(9 downto 4) <= "111111";
				--link_3(9 downto 4) <= "111111";
				link_3(15 downto 0) <= "0000001110100000";
				link_0 <= (others => '0');
				link_1 <= (others => '0');
				link_2 <= (others => '0');
				link_3(31 downto 16) <= global_time_3(47 downto 32);
							
			elsif(conv_integer(counter) = 3)then
				link_0_k <= '1';
				link_1_k <= '1';
				link_2_k <= '1';
				link_3_k <= '1';
				link_0 <= x"AAAAAAAA";
				link_1 <= x"BBBBBBBB";
				link_2 <= x"CCCCCCCC";
				link_3 <= global_time_3(31 downto 0);

			elsif(conv_integer(counter) = 4)then
				link_0_k <= '1';
				link_1_k <= '1';
				link_2_k <= '1';
				link_3_k <= '1';
				link_0 <= x"AAAAAAAA";
				link_1 <= x"BBBBBBBB";
				link_2 <= x"CCCCCCCC";
				link_3(9 downto 4) <= "111111";
				link_3 <= (others => '0');
				
			elsif(conv_integer(counter) = 5)then
				link_0_k <= '1';
				link_1_k <= '1';
				link_2_k <= '1';
				link_3_k <= '1';
				link_0 <= x"AAAAAAAA";
				link_1 <= x"BBBBBBBB";
				link_2 <= x"CCCCCCCC";
				link_3 <= x"DDDDDDDD";

			elsif(conv_integer(counter) = 6)then
				link_0_k <= '1';
				link_1_k <= '1';
				link_2_k <= '1';
				link_3_k <= '1';
				link_0 <= x"AAAAAAAA";
				link_1 <= x"BBBBBBBB";
				link_2 <= x"CCCCCCCC";
				link_3 <= x"DDDDDDDD";
				
			end if;
		end if;


	end if;
	end process data;

end architecture;
