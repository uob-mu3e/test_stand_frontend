----------------------------------------------------------------------------
-- Slow Control Unit for Frontend Board
-- Marius Koeppel, Mainz University
-- makoeppe@students.uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity sc_s4 is
	port(
		clk:                in std_logic;
		reset_n:            in std_logic;
		enable:             in std_logic;
		
		mem_data_in:        in std_logic_vector(31 downto 0);
		
		link_data_in:       in std_logic_vector(31 downto 0);
		link_data_in_k:     in std_logic_vector(3 downto 0);
		
		fifo_data_out:      out std_logic_vector(35 downto 0);
		fifo_we:				  out std_logic;
		
		mem_data_out:       out std_logic_vector(31 downto 0);
		mem_addr_out:       out std_logic_vector(15 downto 0);
		mem_wren:           out std_logic;
		
		stateout:           out std_logic_vector(27 downto 0)
	);
end entity sc_s4;


architecture rtl of sc_s4 is

	component fifo is
    Generic (
           add  : natural := 8;
           width  : natural := 8
           );
    Port ( Din   : in  STD_LOGIC_VECTOR (width-1 downto 0);
           Wr    : in  STD_LOGIC;
           Dout  : out STD_LOGIC_VECTOR (width-1 downto 0);
           Rd    : in  STD_LOGIC;
           Empty : out STD_LOGIC;
           Full  : out STD_LOGIC;
           CLK   : in  STD_LOGIC;
           reset_n : in  STD_LOGIC
           );
	end component fifo;

	signal fifo_wr : std_logic;
	signal fifo_re : std_logic;
	signal fifo_data_in : std_logic_vector(35 downto 0);
	signal fifo_data_q : std_logic_vector(35 downto 0);

	signal mem_data_o : std_logic_vector(31 downto 0);
	signal mem_data_i : std_logic_vector(31 downto 0);
	signal mem_addr_o : std_logic_vector(15 downto 0);
	signal mem_wren_o : std_logic;
	
	signal start_add 	: std_logic_vector(15 downto 0);
	signal got_length : std_logic_vector(15 downto 0);
	signal add_cnt		: std_logic_vector(15 downto 0);
	signal wait_cnt	: std_logic;
	signal ram_cnt		: std_logic;
	
	signal sc_type		: std_logic_vector(1 downto 0);

	type state_type is (waiting, starting, get_length, writing, reading, trailer);
	signal state : state_type;

begin

	mem_data_out <= mem_data_o;
	mem_addr_out <= mem_addr_o;
	mem_wren     <= mem_wren_o;
	mem_data_i	 <= mem_data_in;

	i_fifo : fifo
	generic map (
    	add 	=> 16,
      	width 	=> 36
    )
    port map ( 
		Din   => fifo_data_in,
       	Wr    => fifo_wr,
       	Dout  => fifo_data_q,
       	Rd    => fifo_re,
       	Empty => open,
       	Full  => open,
       	CLK   => clk,
       	reset_n => reset_n
    );

    fifo_pro : process(reset_n, clk)
    begin
    if(reset_n = '0')then
    	fifo_data_in 	<= (others => '0');
    	fifo_wr 		<= '0';
    elsif(rising_edge(clk))then
    	if(link_data_in = x"000000BC" and link_data_in_k(0) = '1') then
			fifo_data_in	<= (others => '0');
			fifo_wr 		<= '0';
		else
			fifo_data_in	<= link_data_in_k & link_data_in;
			fifo_wr 		<= '1';
		end if;
	end if;
	end process;

	memory : process(reset_n, clk)
	begin
	if(reset_n = '0')then
		mem_data_o 		 	<= (others => '0');
		mem_addr_o 		  	<= (others => '0');
		stateout 		  	<= (others => '0');
		start_add 		  	<= (others => '0');
		got_length 		  	<= (others => '0');
		add_cnt 		  	<= (others => '0');
		fifo_data_out		<= (others => '0');
		sc_type				<= (others => '0');
		fifo_re 			<= '0';
		mem_wren_o 			<= '0';
		ram_cnt 			<= '0';
		wait_cnt 			<= '0';
		fifo_we 			<= '0';
	elsif(rising_edge(clk))then
		stateout 		<= (others => '0');
		mem_data_o 		<= (others => '0');
		fifo_data_out 	<= (others => '0');
		mem_wren_o 		<= '0';
		fifo_we 		<= '0';
			
		case state is
		
			when waiting =>
				fifo_re <= '1';
				stateout(3 downto 0) 		<= x"1";
				if (fifo_data_q(7 downto 0) = x"BC" 
					and fifo_data_q(32) = '1' 
					and fifo_data_q(31 downto 26) = "000111") then
						sc_type    				<= fifo_data_q(25 downto 24);
						state 					<= starting;
				end if;
			
			when starting =>
				stateout(3 downto 0) <= x"2";
				fifo_data_out			<= sc_type & "10" & fifo_data_q(31 downto 0);
				fifo_we					<= '1';
				start_add 				<= fifo_data_q(15 downto 0);
				state 					<= get_length;	
			
			when get_length =>
				stateout(3 downto 0) <= x"3";
				got_length 						 	<= fifo_data_q(15 downto 0);
				fifo_data_out(35 downto 17)	<= (others => '0');
				fifo_data_out(16) 			 	<= '1';
				fifo_data_out(15 downto 0)	 	<= fifo_data_q(15 downto 0);
				fifo_we							 	<= '1';
				if (sc_type = "10") then -- read
					state 				<= reading;
				elsif (sc_type = "11") then -- write
					state 				<= writing;
				end if;
		
			when writing =>
				stateout(3 downto 0) <= x"4";
				if (add_cnt = got_length) then
					state 		  <= trailer;
				else
					if (wait_cnt = '0') then
						mem_data_o <= fifo_data_q(31 downto 0);
						mem_wren_o <= '1';
						mem_addr_o <= start_add;
						wait_cnt   <= not wait_cnt;
					else
						mem_data_o <= fifo_data_q(31 downto 0);
						mem_wren_o <= '1';
						mem_addr_o <= start_add + '1';
					end if;
					add_cnt 		  <= add_cnt + '1';
				end if;
				
			when reading =>
				stateout(3 downto 0) <= x"4";
				if (add_cnt = got_length) then
					state 		  		<= trailer;
				else
					if (wait_cnt = '0') then
						mem_addr_o 		<= start_add;
						wait_cnt   		<= not wait_cnt;
						fifo_data_out	<= "0000" & mem_data_i;
					else
						ram_cnt 				<= not ram_cnt;
						if (ram_cnt = '1') then
							mem_addr_o		<= start_add + '1';
						else
							fifo_data_out 	<= "0000" & mem_data_i;
							fifo_we			<= '1';
						end if;
						
					end if;
					add_cnt <= add_cnt + '1';
				end if;
				
			when trailer =>
				stateout(3 downto 0) <= x"6";
				if (fifo_data_q(7 downto 0) = x"0000009C" and fifo_data_q(32) = '1') then
						start_add 						<= (others => '0');
						fifo_data_out(35 downto 32) 	<= "0011";
						fifo_data_out(31 downto 0)		<= (others => '0');
						fifo_we							<= '1';
						wait_cnt   						<= not wait_cnt;
						fifo_re 						<= '0';
						state 							<= waiting;
				end if;
			
			when others =>
				stateout(3 downto 0) <= x"E";
				mem_data_o <= (others => '0');
				mem_addr_o <= (others => '0');
				start_add <= (others => '0');
				mem_wren_o <= '0';
		end case;
	end if;
	end process;

end rtl;