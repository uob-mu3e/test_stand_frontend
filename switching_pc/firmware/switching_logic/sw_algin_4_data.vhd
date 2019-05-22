-- FE sorter switching board
-- fe_sorter_4_data.vhd

-- MuPix data header (96 Bit) 
-- 4 bit to be specified
-- 6 bit header ID 111010
-- 2 bit to be specified
-- 2 * 8 bit K 28.5
-- 16 bit FPGA ID
-- 48 bit timestamp
-- 6 bit to be specified

-- Example --
-- 11111011110010111100111110101111
-- 00000000000000000001111111111111
-- 11111100000000000000000000000000

-- MuPix data sub-Header (32 Bit)
-- 4 bit to be specified
-- 6 bit Header ID 111111
-- 6 bit timestamp
-- 16 bit overflow indicators

-- Example --
-- 11111111111111110000011111111111

--  (32 Bit)
-- 4 bit timestamp
-- 6 bit chip id
-- 8 bit row
-- 8 bit column
-- 6 bit ToT

-- Example --
-- 11111111111111111111111111110001

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use work.mudaq_components.all;

entity sw_algin_4_data is 
	port (
		data_in_fifo_clk_0         : in  std_logic; -- 156,25 MHZ
		data_in_fifo_clk_1         : in  std_logic; -- 156,25 MHZ
		data_in_fifo_clk_2         : in  std_logic; -- 156,25 MHZ
		data_in_fifo_clk_3         : in  std_logic; -- 156,25 MHZ
		data_out_fifo_clk        : in  std_logic; -- 312,50 MHZ

		data_in_node_clk         : in  std_logic; -- 156,25 MHZ
		data_out_node_clk        : in  std_logic; -- To be defined

		reset_n					 : in  std_logic;
		
		reset_n_fifo_0			 : in  std_logic;
		reset_n_fifo_1			 : in  std_logic;
		reset_n_fifo_2			 : in  std_logic;
		reset_n_fifo_3			 : in  std_logic;
		
		data_in_0				 : in std_logic_vector(31 downto 0); -- FPGA-ID = 0000000000000001
		data_in_1				 : in std_logic_vector(31 downto 0); -- FPGA-ID = 0000000000000011
		data_in_2				 : in std_logic_vector(31 downto 0); -- FPGA-ID = 0000000000000111
		data_in_3				 : in std_logic_vector(31 downto 0); -- FPGA-ID = 0000000000001111
		
		datak_in_0				 : in std_logic;
		datak_in_1				 : in std_logic;
		datak_in_2				 : in std_logic;
		datak_in_3				 : in std_logic;
		
		data_out	             : out std_logic_vector(63 downto 0);
		error_out				 : out std_logic
		
	);
end sw_algin_4_data;

architecture rtl of sw_algin_4_data is

	type state_type is (wait_for_sheader, read_sheader1, read_sheader2, read_sheader3, sorting);
	type sorted_output_type is (fifo_0, fifo_1, fifo_2, fifo_3, error_same_time);

	signal state 				: state_type;
	signal sorted_output		: sorted_output_type;

	signal rdreg_fifo_0      	: std_logic;
	signal rdreg_fifo_1      	: std_logic;
	signal rdreg_fifo_2      	: std_logic;
	signal rdreg_fifo_3      	: std_logic;

	signal fifo_0_data_out   	: std_logic_vector(31 downto 0);
	signal fifo_1_data_out	 	: std_logic_vector(31 downto 0);
	signal fifo_2_data_out	 	: std_logic_vector(31 downto 0);
	signal fifo_3_data_out	 	: std_logic_vector(31 downto 0);

	signal fifo_0_full		 	: std_logic;
	signal fifo_1_full		 	: std_logic;
	signal fifo_2_full		 	: std_logic;
	signal fifo_3_full		 	: std_logic;

	signal fifo_0_empty		 	: std_logic;
	signal fifo_1_empty		 	: std_logic;
	signal fifo_2_empty		 	: std_logic;
	signal fifo_3_empty		 	: std_logic;

	signal count_until_sheader	: std_logic_vector(31 downto 0);
	signal same_time			: std_logic;

	signal fifos_not_empty 		: std_logic;

	signal empty_counter 		: std_logic_vector(9 downto 0); -- length is equal to 6 bit time from header and 4 bit time from event

	signal sheader_0			: std_logic_vector(95 downto 0);
	signal sheader_1			: std_logic_vector(95 downto 0);
	signal sheader_2			: std_logic_vector(95 downto 0);
	signal sheader_3			: std_logic_vector(95 downto 0);

	signal buffer_global_time	: std_logic_vector(47 downto 0);

	signal node_data_in  		: std_logic_vector(63 downto 0);

	signal fifo_0_64_data_out	: std_logic_vector(63 downto 0);
	signal fifo_1_64_data_out	: std_logic_vector(63 downto 0);
	signal fifo_2_64_data_out	: std_logic_vector(63 downto 0);
	signal fifo_3_64_data_out	: std_logic_vector(63 downto 0);

	signal node_wreq 			: std_logic;
	signal node_rdreq           : std_logic;
	signal node_aclr            : std_logic;
	signal node_data_out 		: std_logic_vector(63 downto 0);
	signal node_full			: std_logic;
	signal node_empty			: std_logic;

	signal send_id_time_0		: std_logic;
	signal send_id_time_1		: std_logic;
	signal send_id_time_2		: std_logic;
	signal send_id_time_3		: std_logic;

	signal first_32_bit_0		: std_logic;
	signal first_32_bit_1		: std_logic;
	signal first_32_bit_2		: std_logic;
	signal first_32_bit_3		: std_logic;

	signal writing_events_0		: std_logic;
	signal writing_events_1		: std_logic;
	signal writing_events_2		: std_logic;
	signal writing_events_3		: std_logic;

	signal error_signal			: std_logic;
	
begin
	
		fifos_not_empty <= (not fifo_0_empty) and (not fifo_1_empty) and (not fifo_2_empty) and (not fifo_3_empty);

		error_out 		<= error_signal;

		node_aclr 		<= reset_n_fifo_0 or reset_n_fifo_1 or reset_n_fifo_2 or reset_n_fifo_3;
		
		data_out 		<= node_data_out;

		node_fifo : component ip_sw_tree_fifo_64
		port map (
			data  => node_data_in,
			wrreq => node_wreq,
			rdreq => node_rdreq,
			rdclk => data_in_node_clk,
			wrclk => data_out_node_clk,
			aclr  => node_aclr,
			q     => node_data_out,
			wrfull  => node_full,
			rdempty => node_empty
		);

		sw_data_0_fifo : component ip_sw_tree_fifo_32
		port map (
			data  => data_in_0,
			wrreq => datak_in_0,
			rdreq => rdreg_fifo_0,
			rdclk => data_in_fifo_clk_0,
			wrclk => data_out_fifo_clk,
			aclr  => reset_n_fifo_0,
			q     => fifo_0_data_out,
			wrfull  => fifo_0_full,
			rdempty => fifo_0_empty
		);

		sw_data_1_fifo : component ip_sw_tree_fifo_32
		port map (
			data  => data_in_1,
			wrreq => datak_in_1,
			rdreq => rdreg_fifo_1,
			rdclk => data_in_fifo_clk_1,
			wrclk => data_out_fifo_clk,
			aclr  => reset_n_fifo_1,
			q     => fifo_1_data_out,
			wrfull  => fifo_1_full,
			rdempty => fifo_1_empty
		);

		sw_data_2_fifo : component ip_sw_tree_fifo_32
		port map (
			data  => data_in_2,
			wrreq => datak_in_2,
			rdreq => rdreg_fifo_2,
			rdclk => data_in_fifo_clk_2,
			wrclk => data_out_fifo_clk,
			aclr  => reset_n_fifo_2,
			q     => fifo_2_data_out,
			wrfull  => fifo_2_full,
			rdempty => fifo_2_empty
		);

		sw_data_3_fifo : component ip_sw_tree_fifo_32
		port map (
			data  => data_in_3,
			wrreq => datak_in_3,
			rdreq => rdreg_fifo_3,
			rdclk => data_in_fifo_clk_3,
			wrclk => data_out_fifo_clk,
			aclr  => reset_n_fifo_3,
			q     => fifo_3_data_out,
			wrfull  => fifo_3_full,
			rdempty => fifo_3_empty
		);
	  
		sort : process (data_out_fifo_clk, reset_n)
			
			variable globale_time_0 : std_logic_vector(47 downto 0) := (others => '0');
			variable globale_time_1 : std_logic_vector(47 downto 0) := (others => '0');
			variable globale_time_2 : std_logic_vector(47 downto 0) := (others => '0');
			variable globale_time_3 : std_logic_vector(47 downto 0) := (others => '0');

			variable writing_events_0 : std_logic := '0';
			variable writing_events_1 : std_logic := '0';
			variable writing_events_2 : std_logic := '0';
			variable writing_events_3 : std_logic := '0';

			variable sorting_state : integer := 0; -- 1 = write_sheader_1. 2 = write_sheader_2, 3 = write_sheader_3, 4 = event_header

		begin 
			if (reset_n = '0') then
				rdreg_fifo_0 		<= '0';
				rdreg_fifo_1 		<= '0';
				rdreg_fifo_2 		<= '0';
				rdreg_fifo_3 		<= '0';
				count_until_sheader <= (others => '0');
				empty_counter 		<= (others => '0');
				state 				<= wait_for_sheader;
				sheader_0 			<= (others => '0');
				sheader_1 			<= (others => '0');
				sheader_2 			<= (others => '0');
				sheader_3 			<= (others => '0');
				node_data_in    	<= (others => '0');
				fifo_0_64_data_out  <= (others => '0');
				fifo_1_64_data_out  <= (others => '0');
				fifo_2_64_data_out  <= (others => '0');
				fifo_3_64_data_out  <= (others => '0');
				buffer_global_time  <= (others => '0');
				node_wreq 			<= '0';
				send_id_time_0		<= '0';
				send_id_time_1		<= '0';
				send_id_time_2		<= '0';
				send_id_time_3		<= '0';
				same_time			<= '0';
				first_32_bit_0		<= '1';
				first_32_bit_1		<= '1';
				first_32_bit_2		<= '1';
				first_32_bit_3		<= '1';
				error_signal		<= '0';
			elsif rising_edge(data_out_fifo_clk) then
				if (empty_counter = x"F") then
					error_signal <= '1';
				else 
					error_signal <= '0';
				end if;
				if (fifos_not_empty = '1') then
					case state is
						-- wait until every input has an sheader and throw all other data away
						when wait_for_sheader =>
							count_until_sheader <= count_until_sheader + '1';
							if fifo_0_data_out(9 downto 4) = "111010" then
								rdreg_fifo_0 <= '0';
							else
								rdreg_fifo_0 <= '1';
							end if;
							if fifo_1_data_out(9 downto 4) = "111010" then
								rdreg_fifo_1 <= '0';
							else
								rdreg_fifo_1 <= '1';
							end if;
							if fifo_2_data_out(9 downto 4) = "111010" then
								rdreg_fifo_2 <= '0';
							else
								rdreg_fifo_2 <= '1';
							end if;
							if fifo_3_data_out(9 downto 4) = "111010" then
								rdreg_fifo_3 <= '0';
							else
								rdreg_fifo_3 <= '1';
							end if;
							if (fifo_0_data_out(9 downto 4) = "111010") and (fifo_1_data_out(9 downto 4) = "111010") and (fifo_2_data_out(9 downto 4) = "111010") and (fifo_3_data_out(9 downto 4) = "111010") then
								rdreg_fifo_0 <= '1';
								rdreg_fifo_1 <= '1';
								rdreg_fifo_2 <= '1';
								rdreg_fifo_3 <= '1';
								state <= read_sheader1;
								count_until_sheader <= (others => '0');
							end if;
						when read_sheader1 =>
							sheader_0(31 downto 0)      <= fifo_0_data_out;
							sheader_1(31 downto 0)      <= fifo_1_data_out;
							sheader_2(31 downto 0)      <= fifo_2_data_out;
							sheader_3(31 downto 0)      <= fifo_3_data_out;
							state <= read_sheader2;
						when read_sheader2 =>
							sheader_0(63 downto 32)      <= fifo_0_data_out;
							sheader_1(63 downto 32)      <= fifo_1_data_out;
							sheader_2(63 downto 32)      <= fifo_2_data_out;
							sheader_3(63 downto 32)      <= fifo_3_data_out;
							globale_time_0(19 downto 0)	 :=	fifo_0_data_out(31 downto 12);
							globale_time_1(19 downto 0)	 :=	fifo_1_data_out(31 downto 12);
							globale_time_2(19 downto 0)	 :=	fifo_2_data_out(31 downto 12);
							globale_time_3(19 downto 0)	 :=	fifo_3_data_out(31 downto 12);
							state <= read_sheader3;
						when read_sheader3 =>
							sheader_0(95 downto 64)      <= fifo_0_data_out;
							sheader_1(95 downto 64)      <= fifo_1_data_out;
							sheader_2(95 downto 64)      <= fifo_2_data_out;
							sheader_3(95 downto 64)      <= fifo_3_data_out;
							globale_time_0(47 downto 20) :=	fifo_0_data_out(27 downto 0);
							globale_time_1(47 downto 20) :=	fifo_1_data_out(27 downto 0);
							globale_time_2(47 downto 20) :=	fifo_2_data_out(27 downto 0);
							globale_time_3(47 downto 20) :=	fifo_3_data_out(27 downto 0);
							rdreg_fifo_0 <= '0';
							rdreg_fifo_1 <= '0';
							rdreg_fifo_2 <= '0';
							rdreg_fifo_3 <= '0';
							state <= sorting;
						when sorting =>
							if fifo_0_data_out(9 downto 4) = "111111" then
								globale_time_0(5 downto 0) := fifo_0_data_out(15 downto 10);
								writing_events_0		   := '0';
							end if;

							if fifo_1_data_out(9 downto 4) = "111111" then
								globale_time_1(5 downto 0) := fifo_1_data_out(15 downto 10);
								writing_events_1		   := '0';
							end if;

							if fifo_2_data_out(9 downto 4) = "111111" then
								globale_time_2(5 downto 0) := fifo_2_data_out(15 downto 10);
								writing_events_2		   := '0';
							end if;

							if fifo_3_data_out(9 downto 4) = "111111" then
								globale_time_3(5 downto 0) := fifo_3_data_out(15 downto 10);
								writing_events_3		   := '0';
							end if;

							if fifo_0_data_out(9 downto 4) = "111010" then
								writing_events_0		   := '0';
							end if;

							if fifo_1_data_out(9 downto 4) = "111010" then
								writing_events_1		   := '0';
							end if;

							if fifo_2_data_out(9 downto 4) = "111010" then
								writing_events_2		   := '0';
							end if;

							if fifo_3_data_out(9 downto 4) = "111010" then
								writing_events_3		   := '0';
							end if;

							if (writing_events_0 or writing_events_1 or writing_events_2 or writing_events_3) = '1' then
								sorting_state := 4; --event_header
							else
								if (globale_time_0 < globale_time_1) and (globale_time_0 < globale_time_2) and (globale_time_0 < globale_time_3) then
									sorted_output 			<= fifo_0;
									if sorting_state = 2 then
										sorting_state 		:= 2; --write_sheader_2
									elsif (fifo_0_data_out(9 downto 4) = "111010") then
										sorting_state 		:= 1; --write_sheader_1
									else
										sorting_state 		:= 4; --event_header
										writing_events_0	:= '1';
									end if;
									same_time	  			<= '0';
								elsif (globale_time_1 < globale_time_0) and (globale_time_1 < globale_time_2) and (globale_time_1 < globale_time_3) then
									sorted_output 			<= fifo_1;
									if sorting_state = 2 then
										sorting_state 		:= 2; --write_sheader_2
									elsif (fifo_1_data_out(9 downto 4) = "111010") then
										sorting_state 		:= 1; --write_sheader_1
									else
										sorting_state 		:= 4; --event_header
										writing_events_1	:= '1';
									end if;
									same_time	  			<= '0';
								elsif (globale_time_2 < globale_time_0) and (globale_time_2 < globale_time_1) and (globale_time_2 < globale_time_3) then
									sorted_output 			<= fifo_2;
									if sorting_state = 2 then
										sorting_state 		:= 2; --write_sheader_2
									elsif (fifo_2_data_out(9 downto 4) = "111010") then
										sorting_state 		:= 1; --write_sheader_1
									else
										sorting_state 		:= 4; --event_header
										writing_events_2	:= '1';
									end if;
									same_time	  			<= '0';
								elsif (globale_time_3 < globale_time_0) and (globale_time_3 < globale_time_1) and (globale_time_3 < globale_time_2) then
									sorted_output 			<= fifo_3;
									if sorting_state = 2 then
										sorting_state 		:= 2; --write_sheader_2
									elsif (fifo_3_data_out(9 downto 4) = "111010") then
										sorting_state 		:= 1; --write_sheader_1
									else
										sorting_state 		:= 4; --event_header
										writing_events_3	:= '1';
									end if;
									same_time	  			<= '0';
								else
									same_time	  			<= '1';
									sorted_output 			<= error_same_time;
								end if;
							end if;
							case sorted_output is
								when fifo_0 =>
									node_data_in <= fifo_0_64_data_out;
									rdreg_fifo_0 <= '1';
									rdreg_fifo_1 <= '0';
									rdreg_fifo_2 <= '0';
									rdreg_fifo_3 <= '0';
									node_wreq 	 <= '1';
									if sorting_state = 1 then --write_sheader_1
										if first_32_bit_0 = '1' then
											fifo_0_64_data_out(31 downto 0) 	<= fifo_0_data_out;
										else
											fifo_0_64_data_out(63 downto 32)	<= fifo_0_data_out;
										end if;
										first_32_bit_0							<= not first_32_bit_0;
										sorting_state 							:= 2; --write_sheader_2
									elsif sorting_state = 2 then --write_sheader_2
										buffer_global_time(19 downto 0)			<= fifo_0_data_out(31 downto 12);
										if first_32_bit_0 = '1' then
											fifo_0_64_data_out(31 downto 0) 	<= fifo_0_data_out;
										else
											fifo_0_64_data_out(63 downto 32)	<= fifo_0_data_out;
										end if;
										first_32_bit_0							<= not first_32_bit_0;
										sorting_state							:= 3; --write_sheader_3;
									elsif sorting_state = 3 then --write_sheader_3
										buffer_global_time(47 downto 20)		<= fifo_0_data_out(27 downto 0);
										if first_32_bit_0 = '1' then
											fifo_0_64_data_out(31 downto 0) 	<= fifo_0_data_out;
										else
											fifo_0_64_data_out(63 downto 32)	<= fifo_0_data_out;
										end if;
										first_32_bit_0							<= not first_32_bit_0;
										globale_time_0 							:= buffer_global_time;
										sorting_state							:= 4; --event_header;
									elsif sorting_state = 4 then --event_header;
										if first_32_bit_0 = '1' then
											fifo_0_64_data_out(31 downto 0) 	<= fifo_0_data_out;
										else
											fifo_0_64_data_out(63 downto 32)	<= fifo_0_data_out;
										end if;
										first_32_bit_0							<= not first_32_bit_0;
									end if;
								when fifo_1 =>
									node_data_in <= fifo_1_64_data_out;
									rdreg_fifo_0 <= '0';
									rdreg_fifo_1 <= '1';
									rdreg_fifo_2 <= '0';
									rdreg_fifo_3 <= '0';
									node_wreq 	 <= '1';
									if sorting_state = 1 then --write_sheader_1
										if first_32_bit_1 = '1' then
											fifo_1_64_data_out(31 downto 0) 	<= fifo_1_data_out;
										else
											fifo_1_64_data_out(63 downto 32)	<= fifo_1_data_out;
										end if;
										first_32_bit_1							<= not first_32_bit_1;
										sorting_state 							:= 2; --write_sheader_2
									elsif sorting_state = 2 then --write_sheader_2
										buffer_global_time(19 downto 0)			<= fifo_1_data_out(31 downto 12);
										if first_32_bit_1 = '1' then
											fifo_1_64_data_out(31 downto 0) 	<= fifo_1_data_out;
										else
											fifo_1_64_data_out(63 downto 32)	<= fifo_1_data_out;
										end if;
										first_32_bit_1							<= not first_32_bit_1;
										sorting_state							:= 3; --write_sheader_3;
									elsif sorting_state = 3 then --write_sheader_3
										buffer_global_time(47 downto 20)		<= fifo_1_data_out(27 downto 0);
										if first_32_bit_1 = '1' then
											fifo_1_64_data_out(31 downto 0) 	<= fifo_1_data_out;
										else
											fifo_1_64_data_out(63 downto 32)	<= fifo_1_data_out;
										end if;
										first_32_bit_1							<= not first_32_bit_1;
										globale_time_1 							:= buffer_global_time;
										sorting_state							:= 4; --event_header;
									elsif sorting_state = 4 then --event_header;
										if first_32_bit_1 = '1' then
											fifo_1_64_data_out(31 downto 0) 	<= fifo_1_data_out;
										else
											fifo_1_64_data_out(63 downto 32)	<= fifo_1_data_out;
										end if;
										first_32_bit_1							<= not first_32_bit_1;
									end if;
								when fifo_2 =>
									node_data_in <= fifo_2_64_data_out;
									rdreg_fifo_0 <= '0';
									rdreg_fifo_1 <= '0';
									rdreg_fifo_2 <= '1';
									rdreg_fifo_3 <= '0';
									node_wreq 	 <= '1';
									if sorting_state = 1 then --write_sheader_1	
										if first_32_bit_2 = '1' then
											fifo_2_64_data_out(31 downto 0) 	<= fifo_2_data_out;
										else
											fifo_2_64_data_out(63 downto 32)	<= fifo_2_data_out;
										end if;
										first_32_bit_2							<= not first_32_bit_2;
										sorting_state 							:= 2; --write_sheader_2
									elsif sorting_state = 2 then --write_sheader_2
										buffer_global_time(19 downto 0)			<= fifo_2_data_out(31 downto 12);
										if first_32_bit_2 = '1' then
											fifo_2_64_data_out(31 downto 0) 	<= fifo_2_data_out;
										else
											fifo_2_64_data_out(63 downto 32)	<= fifo_2_data_out;
										end if;
										first_32_bit_2							<= not first_32_bit_2;
										sorting_state							:= 3; --write_sheader_3;
									elsif sorting_state = 3 then --write_sheader_3
										buffer_global_time(47 downto 20)		<= fifo_2_data_out(27 downto 0);
										if first_32_bit_2 = '1' then
											fifo_2_64_data_out(31 downto 0) 	<= fifo_2_data_out;
										else
											fifo_2_64_data_out(63 downto 32)	<= fifo_2_data_out;
										end if;
										first_32_bit_2							<= not first_32_bit_2;
										globale_time_2 							:= buffer_global_time;
										sorting_state							:= 4; --event_header;
									elsif sorting_state = 4 then --event_header;
										if first_32_bit_2 = '1' then
											fifo_2_64_data_out(31 downto 0) 	<= fifo_2_data_out;
										else
											fifo_2_64_data_out(63 downto 32)	<= fifo_2_data_out;
										end if;
										first_32_bit_2							<= not first_32_bit_2;
									end if;
								when fifo_3 =>
									node_data_in <= fifo_3_64_data_out;
									rdreg_fifo_0 <= '0';
									rdreg_fifo_1 <= '0';
									rdreg_fifo_2 <= '0';
									rdreg_fifo_3 <= '1';
									node_wreq 	 <= '1';
									if sorting_state = 1 then --write_sheader_1	
										if first_32_bit_3 = '1' then
											fifo_3_64_data_out(31 downto 0) 	<= fifo_3_data_out;
										else
											fifo_3_64_data_out(63 downto 32)	<= fifo_3_data_out;
										end if;
										first_32_bit_3							<= not first_32_bit_3;
										sorting_state 							:= 2; --write_sheader_2
									elsif sorting_state = 2 then --write_sheader_2
										buffer_global_time(19 downto 0)			<= fifo_3_data_out(31 downto 12);
										if first_32_bit_3 = '1' then
											fifo_3_64_data_out(31 downto 0) 	<= fifo_3_data_out;
										else
											fifo_3_64_data_out(63 downto 32)	<= fifo_3_data_out;
										end if;
										first_32_bit_3							<= not first_32_bit_0;
										sorting_state							:= 3; --write_sheader_3;
									elsif sorting_state = 3 then --write_sheader_3
										buffer_global_time(47 downto 20)		<= fifo_3_data_out(27 downto 0);
										if first_32_bit_3 = '1' then
											fifo_3_64_data_out(31 downto 0) 	<= fifo_3_data_out;
										else
											fifo_3_64_data_out(63 downto 32)	<= fifo_3_data_out;
										end if;
										first_32_bit_3							<= not first_32_bit_3;
										globale_time_3 							:= buffer_global_time;
										sorting_state							:= 4; --event_header;
									elsif sorting_state = 4 then --event_header;
										if first_32_bit_3 = '1' then
											fifo_3_64_data_out(31 downto 0) 	<= fifo_3_data_out;
										else
											fifo_3_64_data_out(63 downto 32)	<= fifo_3_data_out;
										end if;
										first_32_bit_3							<= not first_32_bit_3;
									end if;
								when error_same_time =>
									error_signal <= '1';
							end case;
					end case;
				else
					empty_counter		<= empty_counter + '1';
					node_wreq			<= '0';
					rdreg_fifo_0 <= '0';
					rdreg_fifo_1 <= '0';
					rdreg_fifo_2 <= '0';
					rdreg_fifo_3 <= '0';
				end if;
			end if;
		end process sort;
		
end architecture rtl;
