-- FE sorter switching board
-- fe_sorter_data.vhd

-- TimeStamp from decoder (48 Bit) 
-- 48 bit timestamp

-- enables_in 
-- high if data is on data_in

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
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
--use work.mudaq_components.all;

entity sw_algin_data is 
generic (
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
end entity;

architecture rtl of sw_algin_data is

	type state_type is (wait_for_ts, read_ts, algining);
	signal state 				: state_type;
	signal align_state		: integer;

	type data_array_type is array (NLINKS - 1 downto 0) of std_logic_vector(31 downto 0);
	type ts_array_type is array (NLINKS - 1 downto 0) of std_logic_vector(47 downto 0);
	type idx_type is array (NLINKS - 1 downto 0) of integer;
	type node_data_array_type is array (NLINKS / 4 - 1 downto 0) of std_logic_vector(63 downto 0);

	signal data_array : data_array_type;
	signal fifo_data_out_array : data_array_type;
	signal ts_array : ts_array_type;
	signal node_data_array : node_data_array_type;
	signal fifo_64_data_array : node_data_array_type;
	signal buffer_ts	: ts_array_type;

	signal rdreg_fifo      	: std_logic_vector(NLINKS - 1 downto 0);
	signal fifo_full			: std_logic_vector(NLINKS - 1 downto 0);
	signal fifo_empty			: std_logic_vector(NLINKS - 1 downto 0);
	signal fifos_not_empty  : integer;

	signal node_wreq			: std_logic_vector(NLINKS / 4 - 1 downto 0);

	signal low_bits         : std_logic_vector(NLINKS - 1 downto 0);

	signal empty_counter 	: std_logic_vector(9 downto 0); -- length is equal to 6 bit time from header and 4 bit time from event

	signal writing_events  	: std_logic_vector(NLINKS - 1 downto 0);

	signal reset				: std_logic;

	signal wait_cnt			: std_logic;

	signal test             : std_logic_vector(NLINKS - 1 downto 0);

begin

		reset <= not reset_n;

		set_array:
		for I in 0 to (NLINKS - 1) generate
			process(clks_write(I), reset_n)
			begin
				data_array(I) <= data_in((I+1)*32-1 downto I*32);
			end process;
		end generate set_array;

		fifos_not_empty <= conv_integer(fifo_empty);

		gen_nodes:
		for I in 0 to (NLINKS / 4 - 1) generate
			e_node_fifo : component work.cmp.ip_sw_fifo_64
			port map (
				data    => node_data_array(I),
				wrreq   => node_wreq(I),
				rdreq   => node_rdreq,
				rdclk   => clk_node_read,
				wrclk   => clk_node_write,
				aclr    => reset,
				q       => data_out((I+1)*64-1 downto I*64),
				wrfull  => node_full_out(I),
				rdempty => node_empty_out(I)--,
			);
		end generate gen_nodes;

		gen_fifo:
		for I in 0 to (NLINKS - 1) generate
			e_tree_fifo : component work.cmp.ip_sw_fifo_32
			port map (
				data    => data_array(I),
				wrreq   => enables_in(I),
				rdreq   => rdreg_fifo(I),
				rdclk   => clks_read(I),
				wrclk   => clks_write(I),
				aclr    => reset,
				q       => fifo_data_out_array(I),
				wrfull  => fifo_full(I),
				rdempty => fifo_empty(I)--,
			);
		end generate gen_fifo;

		
		process (clks_write(0), reset_n)
		
			variable tmp_array : ts_array_type;
			variable tmp_idx	 : idx_type;
			variable header_state : integer := 0; -- 1 = write_sheader_1. 2 = write_sheader_2, 3 = write_sheader_3, 4 = event_header
			variable i : integer;
			variable j : integer;
		
		begin 
			if (reset_n = '0') then
				state_out				<= x"0";
				state 					<= wait_for_ts;
				align_state				<= NLINKS + 1;
				node_data_array      <= (others => (others => '0'));
				ts_array					<= (others => (others => '0'));
				buffer_ts  				<= (others => (others => '0'));
				rdreg_fifo				<= (others => '1');
				empty_counter 			<= (others => '0');
				writing_events			<= (others => '0');
				low_bits					<= (others => '0');
				wait_cnt					<= '0';
			elsif rising_edge(clks_write(0)) then
				if (empty_counter = x"F") then
					state_out <= x"F";
				end if;
				wait_cnt <= not wait_cnt;
				if (fifos_not_empty = 0) then
					case state is
						-- wait until a ts is on every input
						when wait_for_ts =>
							state_out <= x"1";
							waiting:
							for I in 0 to (NLINKS - 1) loop
								if fifo_data_out_array(I)(9 downto 4) = "111010" then --fifo_0_data_out(9 downto 4) = "111010"
									rdreg_fifo(I) <= '0';
									ts_array(I)(47 downto 32) <= fifo_data_out_array(I)(31 downto 16);
									test(I) <= '1';
								else
									rdreg_fifo(I) <= '1';
								end if;
							end loop waiting;

							if (rdreg_fifo = 0) then
								rdreg_fifo <= (others => '1');
								state <= read_ts;
							end if;
							
						when read_ts =>
							state_out <= x"2";
							
							getts:
							for I in 0 to (NLINKS - 1) loop
								ts_array(I)(31 downto 0) <= fifo_data_out_array(I);
							end loop getts;
							
							rdreg_fifo <= (others => '0');
							state <= algining;
							
						when algining =>
							state_out <= x"3";
							
							check_header:
							for I in 0 to (NLINKS - 1) loop
								if (fifo_data_out_array(I)(9 downto 4) = "111111") then
									ts_array(I) <= ts_array(I) + "111111";
									writing_events(I) <= '0';
								end if;
								
								if (fifo_data_out_array(I)(9 downto 4)= "111010") then
									writing_events(I) <= '0';
								end if;
							end loop check_header;
							
							if (writing_events > 0) then
								header_state := 4; --event_header
							else
							
								generate_idx:
								for I in 0 to (NLINKS - 1) loop
									tmp_idx(I) := I;
								end loop;
							
								tmp_array := ts_array;
								for lvl in 0 to tmp_array'right/2 loop -- should be log2(tmp_array'right)
									for itm in 0 to tmp_array'right loop
										i := 2**(lvl+1) * itm;
										j := i + 2**lvl;
										next when ((i > tmp_array'right) or (j > tmp_array'right));
										if (tmp_array(j) < tmp_array(i)) then
											tmp_array(i) := tmp_array(j);
											tmp_idx(i) := tmp_idx(j);
										end if;
									end loop;
								end loop;
								
								align_state <= tmp_idx(0);
								
								if(header_state = 2) then
									header_state := 2; --write_sheader_2
								elsif(fifo_data_out_array(tmp_idx(0))(9 downto 4) = "111010") then
									header_state := 1; --write_sheader_1
								else
									header_state := 3; --event_header
									writing_events(tmp_idx(0)) <= '1';
								end if;
							end if;

							gen_when:
							for I in 0 to (NLINKS - 1) loop
								if(align_state = I) then
									state_out <= x"4"; 
									node_data_array(NLINKS/4 - 1 + I/4) <= fifo_64_data_array(NLINKS/4 - 1 + I/4);
									rdreg_fifo(I) <= '1';
									rdreg_fifo <= (others => '0');
									
									if(low_bits(I) = '1') then
										fifo_64_data_array(NLINKS/4 - 1 + I/4)(31 downto 0) <= fifo_data_out_array(I);
									else
										fifo_64_data_array(NLINKS/4 - 1 + I/4)(63 downto 32) <= fifo_data_out_array(I);
									end if;
									low_bits(I) <= not low_bits(I);
									
									
									if(header_state = 1) then --write_sheader_1
										header_state := 2;
										buffer_ts(I)(47 downto 32) <= fifo_data_out_array(I)(31 downto 16);
									elsif(header_state = 2) then --write_sheader_2
										buffer_ts(I)(31 downto 0) <= fifo_data_out_array(I);
										ts_array(I) <= buffer_ts(I);
										header_state := 3; --event_header
									end if;
								end if;
							end loop gen_when;	
						end case;
				else
					empty_counter		<= empty_counter + '1';
					node_wreq			<= (others => '0');
					rdreg_fifo			<= (others => '0');
				end if;
			end if;
		end process;

end architecture;
