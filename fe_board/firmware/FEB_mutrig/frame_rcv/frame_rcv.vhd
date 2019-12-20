----------------------------------------------------------------------------------
-- Company:
-- Engineer: Huangshan Chen (chen@kip.uni-heidelberg.de)
--
-- Create Date:		16.05.2016
-- Design Name:
-- Module Name:		frame receiver module
-- Project Name:
-- Target Devices:
-- Tool versions:
-- Description:
--
--		taking data from deserialzer, and form event data
--
-- Dependencies:
--
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
--
-- Revision 1.00 - adpated to MuTRiG frame structure
-- Additional Comments:
--
----------------------------------------------------------------------------------


Library IEEE;
use IEEE.STD_LOGIC_1164.all;
use IEEE.NUMERIC_STD.all;


use work.serial_comm_defs.all;
--use work.mutrig_constants.all;

entity frame_rcv is
	generic (
		EVENT_DATA_WIDTH	: positive := 48;
		N_BYTES_PER_WORD	: positive := 6;
		N_BYTES_PER_WORD_SHORT	: positive := 3
	);
	port(
		i_rst			       : in std_logic;
		i_clk			       : in std_logic;
		i_data			    : in std_logic_vector(7 downto 0);
		i_byteisk		    : in std_logic;
		i_enable		 : in std_logic; -- do not generate new frames (but may finish pending one)

		o_frame_number		 : out std_logic_vector(15 downto 0);
		o_frame_info		 : out std_logic_vector(15 downto 0);
		o_frame_info_ready : out std_logic;
		o_new_frame		    : out std_logic;
		o_word			    : out std_logic_vector(EVENT_DATA_WIDTH -1 downto 0);
		o_new_word		    : out std_logic;
		o_busy			    : out std_logic;

		o_end_of_frame	    : out std_logic;
		o_crc_error		    : out std_logic;
		o_crc_err_count 	 : out std_logic_vector(31 downto 0)
	);
end entity frame_rcv;

architecture rtl of frame_rcv is
component crc16_calc is
	port (
		i_clk : in std_logic;
		i_rst : in std_logic;
		i_d_valid: in std_logic;
		i_din : in std_logic_vector(7 downto 0);
		o_crc_reg : out std_logic_vector(15 downto 0);
		o_crc_8 : out std_logic_vector(7 downto 0)
	);
end component; -- crc16_calc;


	signal s_o_word				: std_logic_vector(EVENT_DATA_WIDTH-1 downto 0);
	signal p_word, n_word			: std_logic_vector(N_BYTES_PER_WORD*8 -1 downto 0);
	signal p_word_extra, n_word_extra	: std_logic_vector(3 downto 0);
	signal p_new_word, n_new_word		: std_logic;

	--CRC specific signals --{{{
	signal s_crc_result				: std_logic_vector(15 downto 0);
	signal p_crc_din_valid, n_crc_din_valid		: std_logic;	--Start of a new CRC calculation
	signal p_crc_rst, n_crc_rst			: std_logic;
	signal p_crc_err_count, n_crc_err_count		: std_logic_vector(31 downto 0);
	signal p_crc_error, n_crc_error			: std_logic;
	signal p_end_of_frame, n_end_of_frame		: std_logic;
	signal s_o_frame_info				: std_logic_vector(15 downto 0);
	--}}}


	-- Frame information signals --{{{
	signal p_new_frame, n_new_frame			: std_logic;	--Indicates a new frame
	signal p_frame_number, n_frame_number		: std_logic_vector(15 downto 0);
	signal p_word_cnt, n_word_cnt			: unsigned(9 downto 0);
	signal p_frame_len, n_frame_len			: std_logic_vector(9 downto 0);
	 -- frame_flags : | i_SC_gen_idle_sig | i_SC_fast_mode | i_SC_prbs_debug | i_SC_single_prbs | p_fifo_full | '0' |
	signal p_frame_flags, n_frame_flags		: std_logic_vector(5 downto 0);
	signal p_frame_info_ready, n_frame_info_ready : std_logic;
	--}}}


	-- FSM states --{{{
	type FSM_states is (
		FS_IDLE,
		FS_FRAME_COUNTER, 
		FS_EVENT_COUNTER, 
		FS_UNPACK, 
		FS_UNPACK_EXTRA,
		FS_CRC_CALC, 
		FS_CRC_CHECK, 
		FS_ERROR
	);
	signal p_state, n_state				: FSM_states;
	signal p_state_wait_cnt, n_state_wait_cnt	: natural range 7 downto 0; --cnt to wait withiin state
	--}}}

begin


	o_new_frame	       <= p_new_frame;
	o_frame_number	    <= p_frame_number;
	o_end_of_frame     <= p_end_of_frame;

	-- the latching of the frame_info is done outside
	-- the frame_flags information will be usefull if it's updated in the beginning of the frame, for the prbs_checker
	o_frame_info 	    <= p_frame_flags & p_frame_len;
	o_crc_err_count    <= p_crc_err_count;
	o_crc_error	       <= p_crc_error;

	o_new_word	       <= p_new_word;
	o_frame_info_ready <= p_frame_info_ready;

	-- taking care of the long event word and short event word
	-- if i_SC_fast_mode = 0, 48-bit word, send out s_o_word
	-- if i_SC_fast_mode = 1, 27-bit word, the data are at highest 27 bits;
	-- re-arrange the bit:
	--   put the E_flag ( s_o_word(48-27) ) to bit(0)
	--   fill the bit( 48-27 downto 1) with '0'
	o_word		<= s_o_word(EVENT_DATA_WIDTH-1 downto 0) when p_frame_flags(4) = '0' else
			s_o_word(EVENT_DATA_WIDTH-1 downto EVENT_DATA_WIDTH-26) & std_logic_vector(to_unsigned(0,21)) & s_o_word(EVENT_DATA_WIDTH-27);


	u_crc16 : crc16_calc --{{{
	port map(
		i_clk		=> i_clk,
		i_rst		=> n_crc_rst,
		i_d_valid	=> n_crc_din_valid,
		i_din		=> i_data,
		o_crc_reg	=> s_crc_result,
		o_crc_8		=> open
	); --}}}


	syn: process(i_clk)
	begin
		if rising_edge(i_clk) then
			if i_rst = '1' then
				p_state			<= FS_IDLE;
				p_crc_err_count		<= (others => '0');
				p_frame_number		<= (others => '0');
			else
				p_state            <= n_state;
				p_state_wait_cnt   <= n_state_wait_cnt;

				p_word             <= n_word;
				p_word_extra		 <= n_word_extra;
				p_new_word         <= n_new_word;

				p_new_frame        <= n_new_frame;
				p_frame_number		 <= n_frame_number;
				p_frame_len        <= n_frame_len;
				p_frame_flags		 <= n_frame_flags;

				p_word_cnt         <= n_word_cnt;
				p_crc_err_count    <= n_crc_err_count;
				p_crc_din_valid    <= n_crc_din_valid;
				p_crc_rst		    <= n_crc_rst;
				p_crc_error		    <= n_crc_error;
				p_end_of_frame	    <= n_end_of_frame;
				p_frame_info_ready <= n_frame_info_ready;


				if n_new_word = '1' then
					s_o_word	<= n_word;
				end if;

--				if n_new_frame_info = '1' then
--					s_o_frame_info <= p_frame_flags & p_frame_len;
--				end if;
			end if;
		end if;
	end process;


	comb: process(
		i_data, i_byteisk, 
		p_state, p_state_wait_cnt, p_crc_err_count, p_crc_din_valid, p_word_cnt, p_new_frame, p_frame_number, p_frame_len, 
		n_word_cnt, s_crc_result, p_word, p_word_extra, p_crc_rst, i_enable, p_frame_flags
	) --{{{
	begin
		--DEFAULT SIGNAL ASSIGNMENTS
		o_busy			<= '1';
		n_state			    <= p_state;
		n_state_wait_cnt	 <= p_state_wait_cnt;

		n_word			    <= p_word;
		n_word_extra	    <= p_word_extra;
		n_new_word		    <= '0';
		n_word_cnt		    <= p_word_cnt;

		n_crc_error		    <= '0';
		n_crc_err_count	 <= p_crc_err_count;
		n_crc_din_valid	 <= p_crc_din_valid;
		n_crc_rst		    <= p_crc_rst;
		n_end_of_frame     <= '0';

		n_new_frame		    <= '0';
		n_frame_len		    <= p_frame_len;
		n_frame_number		 <= p_frame_number;
		n_frame_flags		 <= p_frame_flags;
		n_frame_info_ready <= '0';


		case p_state is
			when FS_IDLE =>
				o_busy           <= '0';
				--Initialize the frame len meta information
				n_frame_len      <= (others => '0');
				n_frame_flags    <= (others => '0');
				n_word_cnt       <= (others => '0');

				n_state_wait_cnt <= 0;
				n_crc_din_valid  <= '0';
				n_crc_rst        <= '1';
				n_new_frame      <= '0';

				--State transition
				if i_enable = '1' and i_byteisk = '1' and i_data = c_header then -- detect frame_header, go to FS_FRAME_COUNTER
					n_new_frame		<= '1';
					n_state			<= FS_FRAME_COUNTER;
					n_state_wait_cnt	<= 2;
				end if;


			when FS_FRAME_COUNTER =>
				n_crc_din_valid		<= '1';
				n_crc_rst		<= '0';

				n_frame_number( p_state_wait_cnt*8-1 downto (p_state_wait_cnt-1)*8) <= i_data;
				n_state_wait_cnt <= p_state_wait_cnt -1;
				if p_state_wait_cnt = 1 then
					n_state			<= FS_EVENT_COUNTER;
					n_state_wait_cnt	<= 2;
				end if;


			when FS_EVENT_COUNTER =>
				n_state_wait_cnt <= p_state_wait_cnt -1;

				if p_state_wait_cnt = 2 then
					n_frame_flags			<= i_data(7 downto 2);
					n_frame_len(9 downto 8)		<= i_data(1 downto 0);
				elsif p_state_wait_cnt = 1 then
					n_frame_len(7 downto 0)		<= i_data;
					n_frame_info_ready         <= '1';
					--Indicate the start of the new frame
					if p_frame_len(9 downto 8) = "00" and i_data = std_logic_vector(to_unsigned(0,8)) then
						n_state			<= FS_CRC_CALC;
						n_state_wait_cnt	<= 2;
					else
						n_state			<= FS_UNPACK;

						if p_frame_flags(4) = '0' then
							n_state_wait_cnt	<= N_BYTES_PER_WORD;
						else
							n_state_wait_cnt	<= N_BYTES_PER_WORD_SHORT;
							n_word			<= (others => '0');
						end if;

					end if;
				end if;


			when FS_UNPACK =>
				if p_frame_flags(4) = '0' then -- normal mode
					n_word(p_state_wait_cnt*8-1 downto (p_state_wait_cnt-1)*8) <= i_data;

					n_state_wait_cnt <= p_state_wait_cnt -1;

					if p_state_wait_cnt = 1 then
						n_new_word <= '1';

						n_word_cnt <= p_word_cnt+1;
						n_state_wait_cnt <= N_BYTES_PER_WORD;

						if p_word_cnt = unsigned(p_frame_len(9 downto 0))-1 then
							n_state			<= FS_CRC_CALC;
							n_state_wait_cnt	<= 2;
						end if;
					end if;
				else -- fast mode
					n_state_wait_cnt <= p_state_wait_cnt -1;

					if p_word_cnt(0) = '0' then
						n_word(n_word'high-(N_BYTES_PER_WORD_SHORT-p_state_wait_cnt)*8 downto n_word'high-(N_BYTES_PER_WORD_SHORT-p_state_wait_cnt+1)*8+1) <= i_data;
						if p_state_wait_cnt = 1 then
							n_state		<= FS_UNPACK_EXTRA;
						end if;
					else
						if p_state_wait_cnt = N_BYTES_PER_WORD_SHORT then
							n_word(n_word'high downto n_word'high-3) <= p_word_extra;
						end if;
						n_word(n_word'high-(N_BYTES_PER_WORD_SHORT-p_state_wait_cnt)*8-4 downto n_word'high-(N_BYTES_PER_WORD_SHORT-p_state_wait_cnt+1)*8-3) <= i_data;
						if p_state_wait_cnt = 1 then
							n_new_word	<= '1';
							if p_word_cnt = unsigned(p_frame_len) -1 then
								n_state			<= FS_CRC_CALC;
								n_state_wait_cnt	<= 2;
							else
								n_word_cnt		<= p_word_cnt +1;
								n_state_wait_cnt	<= N_BYTES_PER_WORD_SHORT;
							end if;
						end if;
					end if;

				end if;


			when FS_UNPACK_EXTRA =>
				if p_word_cnt(0) = '0' then
					n_word(n_word'high-N_BYTES_PER_WORD_SHORT*8 downto n_word'high-N_BYTES_PER_WORD_SHORT*8-3)	<= i_data(7 downto 4);
					n_word_extra	<= i_data(3 downto 0);
					n_new_word	<= '1';
					if p_word_cnt = unsigned(p_frame_len) -1 then
						n_state			<= FS_CRC_CALC;
						n_state_wait_cnt	<= 2;
					else
						n_state			<= FS_UNPACK;
						n_word_cnt		<= p_word_cnt +1;
						n_state_wait_cnt	<= N_BYTES_PER_WORD_SHORT;
					end if;
				end if;


			when FS_CRC_CALC =>
				n_state_wait_cnt <= p_state_wait_cnt -1;
				if p_state_wait_cnt = 1 then

		--			if i_byteisk = '1' and i_data = c_trailer(31 downto 24) then
						n_state <= FS_CRC_CHECK;
						--n_new_frame		<= '0';
					--	n_state_wait_cnt <= 2;
		--			else
		--				n_state <= FS_IDLE;
		--				n_crc_err_count <= std_logic_vector(unsigned(p_crc_err_count)+1);
		--			end if;
				end if;


			when FS_CRC_CHECK =>
				n_crc_din_valid <= '0';
				n_crc_rst       <= '1';
				n_end_of_frame  <= '1';
		--		n_state_wait_cnt	<= p_state_wait_cnt -1;
		--		if p_state_wait_cnt = 1 then
				if s_crc_result /= X"7FF2" then  -- CORRECT magic number
					--if s_crc_result /= X"FFFF" then  -- WRONG result, to test if crc works
				    n_crc_err_count <= std_logic_vector(unsigned(p_crc_err_count)+1);
				    n_crc_error <= '1';
				end if;
				n_state <= FS_IDLE;
		--		end if;


			when others => null;

		end case;
	end process; --}}}


end architecture;
