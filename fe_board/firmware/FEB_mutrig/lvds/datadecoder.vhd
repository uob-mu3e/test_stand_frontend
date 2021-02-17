-----------------------------------
--
-- On detector FPGA for layer 0/1
-- Setup and data alignment for one FPGA link
-- Includes 8b/10b decoding
-- Niklaus Berger, Feb 2014
-- 
-- nberger@physi.uni-heidelberg.de
--
----------------------------------
-- April 2019, Konrad Briggl: Minor changes in signal and fsm state names for better readability, may require more than 1 k28.5 in the counting cycle


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.daq_constants.all;




entity data_decoder is 
	generic (
		EVAL_WINDOW_WORDCNT_BITS : natural := 8; -- number of bits of the counter used to check for the sync pattern
		EVAL_WINDOW_PATTERN_BITS : natural := 1; -- number of bits of the counter of the sync patterns found in the window (realign if not overflow)
		ALIGN_WORD	 : std_logic_vector(7 downto 0):=k28_5 -- pattern byte to search for
	);
	port (
		reset_n				: in std_logic;
--		checker_rst_n		: in std_logic;
		clk					: in std_logic;
		rx_in					: IN STD_LOGIC_VECTOR (9 DOWNTO 0);
		
		rx_reset				: OUT STD_LOGIC;
		rx_fifo_reset		: OUT STD_LOGIC;
		rx_dpa_locked		: IN STD_LOGIC;
		rx_locked			: IN STD_LOGIC;
		rx_bitslip				: OUT STD_LOGIC;
	
		ready					: OUT STD_LOGIC;
		data					: OUT STD_LOGIC_VECTOR(7 downto 0);
		k						: OUT STD_LOGIC;
		state_out			: out std_logic_vector(1 downto 0);		-- 4 possible states
		disp_err				: out std_logic
		);
end data_decoder;

architecture RTL of data_decoder is

type sync_state_type is (reset, waitforplllock, waitfordpalock, count_ALIGNWORD, eval_alignment); --, rxready);
signal sync_state		: sync_state_type;

--signal kcounter 		: std_logic_vector(3 downto 0);
signal alignment_phase_cnt 		: std_logic_vector(3 downto 0);

signal rx_decoded		: std_logic_vector(7 downto 0);
signal rx_k				: std_logic;

-- eval_alignment_ctr : Range of this counter defines window to look for k-words
signal eval_alignment_ctr	: std_logic_vector(EVAL_WINDOW_WORDCNT_BITS-1 downto 0);  -- Jens

-- k_seen: Range of this counter defines number of k-words expected in a sequence of the eval_alignment_ctr range words (255)
signal k_seen 			: std_logic_vector(EVAL_WINDOW_PATTERN_BITS-1 downto 0);

signal ready_buf		: std_logic;

begin

ready <= ready_buf;

process(reset_n, clk)
begin

if(reset_n = '0') then
	sync_state 		<= reset;
	state_out		<= "00";				  -- Jens
	rx_bitslip			<= '0';
	ready_buf				<= '0';
	rx_reset			<= '0';
	rx_fifo_reset	<= '0';
	eval_alignment_ctr		<= (others => '0'); -- Jens
	k_seen			<= (others => '0'); -- Jens
	alignment_phase_cnt			<= (others => '0');
elsif(clk'event and clk = '1') then

	-- to be adapted!
	rx_reset			<= '0';
	rx_fifo_reset	<= '0';
	
	case sync_state is
	
	when reset =>
		sync_state <= waitforplllock;
		state_out  <= "00";
		rx_reset			<= '1';
		rx_fifo_reset	<= '0';
		rx_bitslip		<= '0';
		ready_buf			<= '0';
		k_seen 		<= (others => '0');
		alignment_phase_cnt 	<= (others => '0');		
		eval_alignment_ctr	<= (others => '0');		
		
	when waitforplllock =>
		rx_reset			<= '1';
		rx_fifo_reset	<= '0';
		if(rx_locked = '1') then
			sync_state <= waitfordpalock;
			state_out  <= "00";
		end if;
		
	when waitfordpalock =>
		rx_reset			<= '0';
		rx_fifo_reset	<= '0';
		if(rx_locked = '0') then
			sync_state 	<= reset;
			ready_buf			<= '0';
		elsif (rx_dpa_locked = '1') then
			rx_fifo_reset	<= '1';
			sync_state <= count_ALIGNWORD;
			state_out  <= "01";
		end if;
		
	when count_ALIGNWORD =>
		rx_bitslip 	<= '0';
		eval_alignment_ctr	<= eval_alignment_ctr + 1;
		if(rx_locked = '0') then
			sync_state 	<= reset;
			ready_buf			<= '0';
		
		elsif(rx_decoded = ALIGN_WORD and rx_k = '1') then -- correct k-word coming in
			if(k_seen /= (0 to k_seen'high=>'1'))then
				k_seen <= k_seen + 1;
			end if;
		end if;
		if(eval_alignment_ctr = (0 to eval_alignment_ctr'high => '1'))then
			sync_state <= eval_alignment;
		end if;
			
	when eval_alignment =>
		eval_alignment_ctr	 <= (others => '0');
		if(k_seen /= (0 to k_seen'high=>'1'))then -- counter not at the limit : not enough k28.5 seen, initiate bitslip
			if(alignment_phase_cnt < x"A")then
				ready_buf 		<= '0';
				alignment_phase_cnt 	<= alignment_phase_cnt + 1;
				rx_bitslip 	<= '1';
				sync_state 	<= count_ALIGNWORD;
			else	-- we have tested all phases, reset the dpa circuitry!
				sync_state 	<= reset;
			end if;
		else
			sync_state 	<= count_ALIGNWORD;		-- we continously monitor that we see the komma words!!
			k_seen 		<= (others => '0');	-- so that we act directly during continous check
			state_out	<= "10";
			ready_buf			<= '1';
		end if;
	when others =>
		sync_state <= reset;
	end case;
end if;

end process;

d_checker : entity work.disparity_checker
	port map(
		reset_n				=> reset_n,
		clk					=> clk,
		rx_in					=> rx_in,	
		ready					=> ready_buf,
		disp_err				=> disp_err
		);


dec8b10b: entity work.decode8b10b 
	port map(
		reset_n				=> reset_n,
		clk					=> clk,
		input					=> rx_in,
		output				=> rx_decoded,
		k						=> rx_k
		);

		data 	<= rx_decoded;
		k		<= rx_k;
		
end RTL;
		
