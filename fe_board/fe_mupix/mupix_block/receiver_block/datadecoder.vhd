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

-- TODO: rename entity, mupix data aligner etc. 


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

use work.mudaq.all;

entity data_decoder is 
	port (
		reset_n				: in std_logic;
		checker_rst_n		: in std_logic;
		clk					: in std_logic;
		rx_in					: IN STD_LOGIC_VECTOR (9 DOWNTO 0);
		
		rx_reset				: OUT STD_LOGIC;
		rx_fifo_reset		: OUT STD_LOGIC;
		rx_dpa_locked		: IN STD_LOGIC;
		rx_locked			: IN STD_LOGIC;
		rx_align				: OUT STD_LOGIC;
	
		ready					: OUT STD_LOGIC;
		data					: OUT STD_LOGIC_VECTOR(7 downto 0);
		k						: OUT STD_LOGIC;
		state_out			: out std_logic_vector(1 downto 0);		-- 4 possible states
		disp_err				: out std_logic
		);
end data_decoder;

architecture RTL of data_decoder is

type sync_state_type is (reset, waitforplllock, waitfordpalock, check_k28_5, align); --, rxready);
signal sync_state		: sync_state_type := reset;

--signal kcounter 		: std_logic_vector(3 downto 0);
signal acounter 		: std_logic_vector(7 downto 0);

signal rx_decoded		: std_logic_vector(7 downto 0);
signal rx_k				: std_logic;

signal align_ctr		: std_logic_vector(8 downto 0);  -- Jens
signal k_seen 			: std_logic_vector(8 downto 0); -- Jens
signal rx_reversed      : std_logic_vector(9 downto 0);

signal ready_buf		: std_logic;

signal current_disparity, new_disparity : std_logic;
signal new_datak : std_logic;
signal new_data : std_logic_vector(7 downto 0);
signal disp_error, data_error : std_logic;

begin

ready <= ready_buf;

process(reset_n, clk)
begin
if(reset_n = '0') then
	sync_state 		<= reset;
	state_out		<= "00";				  -- Jens
	rx_align			<= '0';
	ready_buf				<= '0';
	rx_reset			<= '0';
	rx_fifo_reset	<= '0';
	align_ctr		<= (others => '0'); -- Jens
	k_seen			<= (others => '0'); -- Jens
	acounter			<= (others => '0');
elsif(clk'event and clk = '1') then
    for I in 0 to 9 loop
        rx_reversed(9-I) <= rx_in(I);
    end loop;

	-- to be adapted!
	rx_reset			<= '0';
	rx_fifo_reset	<= '0';
	
	case sync_state is
	
	when reset =>
		sync_state <= waitforplllock;
		state_out  <= "00";
		rx_reset			<= '1';
		rx_fifo_reset	<= '0';
		rx_align		<= '0';
		ready_buf			<= '0';
		k_seen 		<= (others => '0');
		acounter 	<= (others => '0');		
		align_ctr	<= (others => '0');		
		
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
			sync_state <= check_k28_5;
			state_out  <= "01";
		end if;
		
	when check_k28_5 =>
        rx_align 	<= '0';
-- -- -- -- -- -- THIS IS NIK's ALIGNMENT --> doesn't work for Jens
--		if(rx_k = '1' and rx_decoded = K28_5) then
--			kcounter <= kcounter + '1';
--			if(kcounter = "1111") then
--				sync_state		<= rxready;
--			end if;
--		else
--			kcounter <= (others => '0');
--			acounter	<= acounter + '1';
--			if(acounter = "1111") then
--				rx_align <= '1';
--			end if;
--		end if;
-- -- -- -- begin Jens align -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
		align_ctr	<= align_ctr + 1;
		if(rx_locked = '0') then
			sync_state 	<= reset;
			ready_buf			<= '0';
			
		-- we assume the Mupix data format here to fulfill the following criteria:
		-- reset mode: a lot of K28_5 in a row
		-- regular data format:
		-- maxcycend 6 bits: 64 hits in a row + counter +link identifier = 264 cycles without K28_5-word in data stream
		-- so within 512 cycles we should definitely see a few K28_5 words
		-- in worst case we check for 36 us (9*512*8ns) until we find the right pattern
			
--		elsif(rx_decoded = K28_5 and rx_k = '1') then -- correct k-word coming in
		--elsif(disp_err = '1') then -- correct k-word coming in
        elsif(rx_decoded /= K28_5 or rx_k /= '1') then -- correct k-word coming in
			if(k_seen < "111111111")then
				k_seen <= k_seen + 1;
			end if;
		end if;
		if(align_ctr = "111111111")then
			sync_state <= align;
		end if;
			
	when align =>  -- TODO: change stuff
		align_ctr	 <= (others => '0');
        k_seen 		<= (others => '0');
		if(k_seen > "111111100")then
			if(acounter < 40)then
				ready_buf 		<= '0';
				acounter 	<= acounter + 1;
				rx_align 	<= '1';
				sync_state 	<= check_k28_5;
			else	-- we have tested all phases, reset the dpa circuitry!
				sync_state 	<= reset;
			end if;
		else
			sync_state 	<= check_k28_5;		-- we continously monitor that we see the komma words!!
			state_out	<= "10";
			ready_buf			<= '1';
		end if;

--			rx_align <= '0';
--				if(k_seen = 0) then
--					align_ctr 	<= (others => '0');
--					k_seen 		<= k_seen + 1;
--				elsif(align_ctr = 256) then
--					align_ctr 	<= (others => '0');
--					k_seen 		<= k_seen + 1;
--					if(k_seen = 1000) then		-- find 1000 k-words "in a row"
--						sync_state 	<= rxready;
--						state_out  	<= x"5";
--						k_seen 		<= (others => '0');
--						align_ctr 	<= (others => '0');
--					end if;
--				end if;
--			elsif(rx_decoded = k28_0 and rx_k = '1') then -- correct k-word coming in
--	
--				
--			else -- not = k-word
--				if(align_ctr = 512) then
--					align_ctr 	<= (others => '0');
--					k_seen		<= (others => '0');
--					rx_align 	<= '1';
--				else
--					rx_align 	<= '0';
--					align_ctr 	<= align_ctr + 1;
--				end if;
--			end if; 
-- -- -- -- -- end Jens'align -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
--	when rxready =>
--
--		if(rx_locked = '0')then-- or checker_rst_n = '0') then
--			sync_state 	<= reset;
--			state_out	<= "00";
--			ready			<= '0';
--		else
--			sync_state 	<= rxready;
--			state_out	<= "11";
--			ready			<= '1';
--			rx_reset		<= '0';
--		end if;
		
	when others =>
		sync_state <= reset;
	end case;
end if;

end process;


-- TBD: change disparity checker to simple addition of all bits in a line
-- and use information to reject bad data, so add a pipeline for the data

--d_checker : work.disparity_checker -- disparity checker in same entity ? (Alex entity)  -- do not only check disparity
--	port map(
--		reset_n				=> reset_n,
--		clk					=> clk,
--		rx_in					=> rx_reversed,
--		ready					=> ready_buf,
--		disp_err				=> disp_err
--		);
--
--
--dec8b10b : work.decode8b10b -- also Alex ?
--	port map(
--		reset_n				=> reset_n,
--		clk					=> clk,
--		input					=> rx_reversed,
--		output				=> rx_decoded,
--		k						=> rx_k
--		);

		data 	<= rx_decoded;
		k		<= rx_k;
		
        

    e_8b10b_dec : entity work.dec_8b10b
    port map (
        i_data => rx_reversed,
        i_disp => current_disparity,
        o_data(7 downto 0) => new_data,
        o_data(8) => new_datak,
        o_disp => new_disparity,
        o_disperr => disp_error,
        o_err => data_error--,
    );

    process(clk)
    begin
    if rising_edge(clk) then
        rx_decoded <= new_data;
        rx_k <= new_datak;
        current_disparity <= new_disparity;
        disp_err <= disp_error or data_error;
    end if;
    end process;
    
end RTL;

