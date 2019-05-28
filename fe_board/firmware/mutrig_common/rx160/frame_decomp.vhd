--stic3.1 frame decomposition

library ieee;
use ieee.STD_LOGIC_1164.ALL;
use ieee.std_logic_arith.all;

entity frame_decomp is
	port (
			i_k_signal : in std_logic;
			i_byte_data : in std_logic_vector(7 downto 0);
			i_byte_clk : in std_logic;
			i_rst : in std_logic;
			o_event_data : out std_logic_vector(55 downto 0);
			o_event_ready : out std_logic;
			o_end_of_frame : out std_logic

		 );
	end entity frame_decomp;

architecture with_frame_cnt of frame_decomp is
	constant head_id    : std_logic_vector(7 downto 0) := "00011100"; --K28.0
	constant trail_id   : std_logic_vector(7 downto 0) := "10011100"; --K28.4
	constant komma_sync : std_logic_vector(7 downto 0) := "10111100"; --K28.5

	type decomp_states is (	S_WAIT_HEADER,
                           S_FRAME_NUM,
                           S_GET_EVENTS,
                           S_EVENT_COUNT
                         );

	signal s_state, n_state : decomp_states;


	signal data_pipe : std_logic_vector(55 downto 0);
--	signal event_count : std_logic_vector(7 downto 0); --THE NUMBER OF EVENTS IN THE CURRENT FRAME
	signal n_event_data_count, s_event_data_count : std_logic_vector(3 downto 0);
	signal n_event_ready, n_end_of_frame : std_logic;
	signal current_byte : std_logic_vector(7 downto 0); -- the currently sampled byte data
	signal s_frame_number, n_frame_number : std_logic_vector(7 downto 0); -- frame number of the current transmission 
	signal no_event : std_logic;

begin

	o_event_data <= s_frame_number & data_pipe(55 downto 8);
	current_byte <= data_pipe(55 downto 48);


	sample_bytes : process(i_byte_clk)  --SAMPLE THE BYTE FROM THE DECODER {{{
	--WE NEED TO SAMPLE THE INCOMING BYTES SINCE IT WILL CHANGE WITH THE FALLING EDGE OF THE BYTE CLOCK
	begin
		if rising_edge(i_byte_clk) then
			data_pipe <= i_byte_data & data_pipe(55 downto 8);	 --PIPELINE CONTAINING THE LAST 10 bits
		end if;
	end process sample_bytes; --}}}



	fsm_comb : process (s_state, current_byte, i_k_signal, s_frame_number, s_event_data_count) --THE TRANSITION PART OF THE FSM{{{
	--NOT REALLY A MOORE FSM SINCE THE DATA IS ALSO ASSIGNED TO THE EVENT DATA ETC
	begin
		n_state <= s_state;
		n_event_data_count <= (others => '-');
		n_frame_number <= s_frame_number;
		n_event_ready <= '0';
		n_end_of_frame <= '0';

		case s_state is

			when S_WAIT_HEADER =>				-- wait for header
				if current_byte = head_id and i_k_signal = '1' then
					n_state <= S_FRAME_NUM;
				end if;

			when S_FRAME_NUM =>					-- store frame number
				n_frame_number <= current_byte;
				n_event_data_count <= conv_std_logic_vector(5,4); -- 6 bytes
				n_state <= S_GET_EVENTS;

			when S_GET_EVENTS =>				-- get data or triler
				if current_byte = trail_id and i_k_signal = '1' then
					n_state <= S_EVENT_COUNT;
				else
					if unsigned(s_event_data_count) = 0 then		-- 6 bytes have been received
						n_event_ready <= '1';
						n_event_data_count <= conv_std_logic_vector(5,4); -- 6 bytes
					else
						n_event_data_count <= conv_std_logic_vector(unsigned(s_event_data_count) - 1, 4);
						n_event_ready <= '0';
					end if;
				end if;


		when S_EVENT_COUNT =>
--				event_count <= current_byte; --EVENT COUNT IS AT THE MOMENT NOT USED
            n_end_of_frame <= '1';
				n_state <= S_WAIT_HEADER;

			when others =>
				n_state <= S_WAIT_HEADER;

		end case;
	end process fsm_comb;
	

	fsm_sync : process (i_byte_clk) --THE SYNCHRONOUS CHANGE OF THE STATE {{{
	begin
		if falling_edge(i_byte_clk) then
			if i_rst = '1' then
				s_state  <= S_WAIT_HEADER;
				s_frame_number <= (others => '0');
			else
				s_state            <= n_state;
				s_frame_number     <= n_frame_number;
				s_event_data_count <= n_event_data_count;
				o_event_ready      <= n_event_ready;
				o_end_of_frame     <= n_end_of_frame;
			end if;
		end if;

	end process fsm_sync; --}}}


end architecture with_frame_cnt;
