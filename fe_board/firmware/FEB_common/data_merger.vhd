-- data merger for mu3e FEB
-- Martin Mueller, May 2019

-- 2 states: 
--	merger state: state of this entity (idle, sending data, sending slowcontrol)
--	FEB state: "reset" state from FEB_state_controller (idle, run_prep, sync, running, terminating, link_test, sync_test, reset, outOfDaq)
-- do not confuse them !!!


-- @ future me: 
-- ToDo: 
-- prio slowcontrol or prio data input
--	error outputs (data does not start with start marker, data fifo not empty in sync, etc. )
-- end of event marker from fifo --> what to do with 31 downto 0 in this case ? (output has to be k28.4 & x"000000")
-- trailer content ? (k28.4, and ??????)
--	Fifo runs empty --> can i leave read req = 1 with protection circuit ? (alternative would need 2 clock) 



library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
--use work.protocol_new.all;

ENTITY data_merger_new is 
    PORT(
        clk:                    in  std_logic; -- 156.25 clk input
        reset:                  in  std_logic; 
	fpga_ID_in:		in  std_logic_vector(15 downto 0); -- will be set by 15 jumpers in the end, set this to something random for now 
	FEB_type_in:		in  std_logic_vector(5  downto 0); -- Type of the frontendboard (001010: mupix, 001000: mutrig, DO NOT USE 000111 or 000000 HERE !!!!)
        state_idle:             in  std_logic; -- "reset" states from state controller 
        state_run_prepare:      in  std_logic;
        state_sync:             in  std_logic;
        state_running:          in  std_logic;
        state_terminating:      in  std_logic;
        state_link_test:        in  std_logic;
        state_sync_test:        in  std_logic;
        state_reset:            in  std_logic;
        state_out_of_DAQ:       in  std_logic;
        data_out:               out std_logic_vector(31 downto 0); -- to optical transm.
        data_is_k:              out std_logic_vector(3 downto 0); -- to optical trasm.
        data_in:		in  std_logic_vector(35 downto 0); -- data input from FIFO (32 bit data, 4 bit ID (0010 Header, 0011 Trail, 0000 Data))
        data_in_slowcontrol:    in  std_logic_vector(35 downto 0); -- data input slowcontrol from SCFIFO (32 bit data, 4 bit ID (0010 Header, 0011 Trail, 0000 SCData))
        slowcontrol_fifo_empty: in  std_logic;
	data_fifo_empty:	in  std_logic;
        slowcontrol_read_req:   out std_logic;
        data_read_req:          out std_logic;
	terminated:             out std_logic; -- to state controller (when stop run acknowledge was transmitted the state controller can go from terminating into idle, this is the signal to tell him that)
	override_data_in:	in  std_logic_vector(31 downto 0); -- data input for states link_test and sync_test;
	override_data_is_k_in:  in  std_logic_vector(3 downto 0);
	leds:			out std_logic_vector(3 downto 0) -- debug
        
    );
END ENTITY data_merger_new;

architecture rtl of data_merger_new is

	    type data_merger_state is (idle, sending_data, sending_slowcontrol);
    --type feb_state is (idle, run_prep, sync, running, terminating, link_test, sync_test, reset_state, out_of_DAQ);

    constant HEADER_K:    std_logic_vector(31 downto 0) := x"000000bc";
	 constant HEADER_K_DATAK:    std_logic_vector(3 downto 0) := "0001";
	 constant WORD_ALIGN:    std_logic_vector(31 downto 0) := x"beefcafe";
    constant DATA_HEADER_ID:    std_logic_vector(5 downto 0) := "111010";
    constant DATA_SUB_HEADER_ID:    std_logic_vector(5 downto 0) := "111111";
    constant ACTIVE_SIGNAL_HEADER_ID:    std_logic_vector(5 downto 0) := "111101";
    constant RUN_TAIL_HEADER_ID:    std_logic_vector(5 downto 0) := "111110";
    constant TIMING_MEAS_HEADER_ID:    std_logic_vector(5 downto 0) := "111100";
    constant SC_HEADER_ID:    std_logic_vector(5 downto 0) := "111011"; 
	constant K285:		std_logic_vector(31 downto 0) :=x"000000c9";
	constant K285_datak:	std_logic_vector(3 downto 0):= "0001";
	constant run_prep_acknowledge:	std_logic_vector(31 downto 0):= x"000000fe";
	constant run_prep_acknowledge_datak:	std_logic_vector(3 downto 0):= "0001";
	constant RUN_END:	std_logic_vector(31 downto 0):= x"000000fe";
	constant RUN_END_DATAK:	std_logic_vector(3 downto 0):= "0001";


----------------components------------------


----------------signals---------------------
	signal merger_state 					: data_merger_state;
	signal run_prep_acknowledge_send  	: std_logic;


----------------begin data merger------------------------

BEGIN

-- debug led merger state
process (clk, reset)
    begin
		if(merger_state=idle) then 
			leds<=(0=>'1', others => '0');
		elsif (merger_state=sending_data) then
			leds<=(1=>'1', others => '0');
		elsif (merger_state=sending_slowcontrol)then
			leds<=(2=>'1', others => '0');
		end if;
end process;


process (clk, reset)
    begin
    if (reset = '1') then 
        merger_state <= idle;
        slowcontrol_read_req <= '0';
        data_read_req <= '0'; 
        terminated <= '0';
		  run_prep_acknowledge_send <= '0';
        data_is_k <= K285_datak;
        data_out <= K285;
    elsif (rising_edge(clk)) then
		  
		  ------------------------------- feb state link test or sync test ----------------------------
		  -- use override data input
		  -- wait for slowcontrol to finish before (to do)
		  
        if (state_link_test = '1' or state_sync_test = '1') then 
                    merger_state <= idle;
                    data_out  <= override_data_in;
						  data_is_k <= override_data_is_k_in;
		  
		  ------------------------------- feb state sync or reset or outOfDaq -------------------------
		  -- send only komma words
		  -- wait for slowcontrol to finish before (to do)
		  elsif(state_sync = '1' or state_reset = '1' or state_out_of_DAQ = '1') then
                    merger_state <= idle;
						  data_out <= K285;
                    data_is_k <=K285_datak;
						  
		  ------------------------------- feb state idle  ---------------------------------------------
        elsif(state_idle = '1')then
            run_prep_acknowledge_send <= '0';
				
            case merger_state is
            
                when idle =>
						  -- not sending something, in state idle, slowcontrol fifo not empty
                    if (slowcontrol_fifo_empty = '0') then
								slowcontrol_read_req <= '1'; 				-- need 2 cycles to get new data from fifo --> start reading now
								
								-- send SC header:
								data_is_k 					<= "0001"; 
								data_out (31 downto 26) <= "000111";
								data_out (25 downto 24)	<= data_in_slowcontrol(35 downto 34);
								data_out (23 downto 8)  <= fpga_ID_in;
								data_out (7  downto 0) 	<= x"bc";
                        merger_state <= sending_slowcontrol; 	-- go to sending slowcontrol state next
								
                    else 													-- no data --> do nothing
								slowcontrol_read_req <= '0';
								data_read_req			<= '0';
                        data_out <= K285;
                        data_is_k <= K285_datak;
                    end if;
                    
                when sending_slowcontrol => 							-- slowcontrol header is trasmitted, send slowcontrol data now
                    if slowcontrol_fifo_empty = '0' then 		-- send slowcontrol data
						      data_out <= data_in_slowcontrol(31 downto 0);
                        data_is_k <= "0000";
								slowcontrol_read_req <= '1';
                    else 													-- slowcontrol fifo empty --> send end marker and goto idle state
						      merger_state <= idle;
								slowcontrol_read_req <= '0';
                        data_out(7 downto 0) <= x"9C";
								data_out(31 downto 8)  <= (others => '0');
                        data_is_k <= "0001";  

                    end if;
                    
                when others =>											-- send data state in FEB state idle should not happen --> goto merger state idle
                    merger_state <= idle;
                    data_out <= K285;
                    data_is_k <= K285_datak;
            end case;		
				
		  ------------------------------- feb state run prep  ---------------------------------------------
		  
		  elsif(state_run_prepare = '1')then
				case merger_state is
					when idle =>
						if(run_prep_acknowledge_send = '0') then	-- send run_prep_acknowledge
							run_prep_acknowledge_send <='1';
							data_out 					<= run_prep_acknowledge;
							data_is_k					<= run_prep_acknowledge_datak;
						elsif (slowcontrol_fifo_empty = '1') then -- no Slowcontrol --> do nothing
							slowcontrol_read_req 	<= '0';
							data_out 					<= K285;
							data_is_k					<= K285_datak;
						else 
							slowcontrol_read_req <= '1'; 				-- need 2 cycles to get new data from fifo --> start reading now
							-- send SC header:
							data_is_k 					<= "0001"; 
							data_out (31 downto 26) <= "000111";
							data_out (25 downto 24)	<= data_in_slowcontrol(35 downto 34);
							data_out (23 downto 8)  <= fpga_ID_in;
							data_out (7  downto 0) 	<= x"bc";
							merger_state <= sending_slowcontrol; 	-- go to sending slowcontrol state next
						end if;
			  
					when sending_slowcontrol =>
						-- slowcontrol header is trasmitted, send slowcontrol data now
						if slowcontrol_fifo_empty = '0' then 		-- send slowcontrol data
							data_out <= data_in_slowcontrol(31 downto 0);
							data_is_k <= "0000";
							slowcontrol_read_req <= '1';
						else 													-- slowcontrol fifo empty --> send end marker and goto idle state
							merger_state <= idle;
							slowcontrol_read_req <= '0';
							data_out(7  downto 0) <= x"9C";
							data_out(31 downto 8)  <= (others => '0');
							data_is_k <= "0001";  

						end if;
					when others => 										-- it should not be possible to get here		      
						merger_state <= idle;
						data_out <= K285;
						data_is_k <= K285_datak;
				end case;
		  ------------------------------- feb state running   ---------------------------------------------			
			elsif(state_running = '1') then
				run_prep_acknowledge_send <= '0';
				case merger_state is
					when idle =>
						if (slowcontrol_fifo_empty = '1' and data_fifo_empty = '1') then -- no data, state is idle --> do nothing
							slowcontrol_read_req 	<= '0';
							data_out 					<= K285;
							data_is_k					<= K285_datak;
						
						elsif(slowcontrol_fifo_empty = '0') then
							slowcontrol_read_req <= '1'; 				-- need 2 cycles to get new data from fifo --> start reading now
							-- send SC header:
							data_is_k 					<= "0001"; 
							data_out (31 downto 26) <= "000111";
							data_out (25 downto 24)	<= data_in_slowcontrol(35 downto 34);
							data_out (23 downto 8)  <= fpga_ID_in;
							data_out (7  downto 0) 	<= x"bc";
							merger_state <= sending_slowcontrol; 	-- go to sending slowcontrol state next
							
						elsif(data_fifo_empty = '0') then
							data_read_req <= '1'; 						-- need 2 cycles to get new data from fifo --> start reading now
							-- send data header:
							data_is_k 					<= "0001"; 
							data_out (31 downto 26) <= FEB_type_in;
							data_out (25 downto 24)	<= "00";
							data_out (23 downto 8)  <= fpga_ID_in;
							data_out (7  downto 0) 	<= x"bc";
							merger_state <= sending_data; 			-- go to sending data state next
						end if;		
						
					when sending_data=>
						if(data_fifo_empty='1') then 					-- send k285 idle, leave read req = 1 ?
							data_out(31 downto 8) 	<= (others => '0');
							data_out(7 downto 0)  	<=  x"9C";
							data_is_k <= "1000";  
						elsif(data_in(33 downto 32)="11") then 	-- end of event marker
							merger_state 				<= idle;
							data_read_req				<= '0'; 
							data_out(31 downto 8) 	<= (others => '0');
							data_out(7 downto 0)  	<= x"9C";
							data_is_k 					<= "0001";
						else
							data_read_req				<= '1';
							data_out						<= data_in(31 downto 0);
							data_is_k					<= "0000";
						end if;	
						
					when sending_slowcontrol=>
						if(slowcontrol_fifo_empty='1') then			-- send k285 idle, leave read req = 1 ?
							data_out(31 downto 8) 	<= (others => '0');
							data_out(7 downto 0)  	<=  x"9C";
							data_is_k <= "1000";
						elsif(data_in_slowcontrol(33 downto 32)= "11") then -- end of event marker
							merger_state				<= idle;
							slowcontrol_read_req		<= '0';
							data_out(31 downto 8) 	<= (others => '0');
							data_out(7 downto 0)  	<= x"9C";
							data_is_k 					<= "0001";
						else
							slowcontrol_read_req		<= '1';
							data_out						<= data_in_slowcontrol(31 downto 0);
							data_is_k					<= "0000";
						end if;
					when others =>

				end case;
		  ------------------------------- feb state terminating  ---------------------------------------------		
		  elsif (state_terminating = '1') then
				case merger_state is
					when idle =>
						terminated <= '1';
						data_out 					<= RUN_END;
						data_is_k					<= RUN_END_DATAK;
						
					when sending_data =>
						if(data_fifo_empty='1') then 					-- send k285 idle, leave read req = 1 ?
							data_out(31 downto 8) 	<= (others => '0');
							data_out(7  downto 0)  	<=  x"9C";
							data_is_k <= "1000";  
						elsif(data_in(33 downto 32)="11") then 	-- end of event marker
							merger_state 				<= idle;
							data_read_req				<= '0'; 
							data_out(31 downto 8) 	<= (others => '0');
							data_out(7  downto 0)  	<= x"9C";
							data_is_k 					<= "0001";
						else
							data_read_req				<= '1';
							data_out						<= data_in(31 downto 0);
							data_is_k					<= "0000";
						end if;
						
					when sending_slowcontrol =>
						if(slowcontrol_fifo_empty='1') then			-- send k285 idle, leave read req = 1 ?
							data_out(31 downto 8) 	<= (others => '0');
							data_out(7 downto 0)  	<=  x"9C";
							data_is_k <= "1000";
						elsif(data_in_slowcontrol(33 downto 32)= "11") then-- end of event marker
							merger_state				<= idle;
							slowcontrol_read_req		<= '0';
							data_out(31 downto 8) 	<= (others => '0');
							data_out(7 downto 0)  	<= x"9C";
							data_is_k 					<= "0001";
						else
							slowcontrol_read_req		<= '1';
							data_out						<= data_in_slowcontrol(31 downto 0);
							data_is_k					<= "0000";
						end if;
				end case;
		  ------------------------------- feb state sync  ---------------------------------------------			
		  elsif(state_sync='1') then
				case merger_state is
					when idle =>
                    merger_state <= idle;
						  data_out <= K285;
                    data_is_k <=K285_datak;
					when sending_slowcontrol =>
						-- slowcontrol header is trasmitted, send slowcontrol data now
						if slowcontrol_fifo_empty = '0' then 		-- send slowcontrol data
							data_out <= data_in_slowcontrol(31 downto 0);
							data_is_k <= "0000";
							slowcontrol_read_req <= '1';
						else 													-- slowcontrol fifo empty --> send end marker and goto idle state
							merger_state <= idle;
							slowcontrol_read_req <= '0';
							data_out(7  downto 0) <= x"9C";
							data_out(31 downto 8)  <= (others => '0');
							data_is_k <= "0001";  

						end if;
					when others => 										-- it should not be possible to get here		      
						merger_state <= idle;
						data_out <= K285;
						data_is_k <= K285_datak;
				end case;
		  end if;
	  end if;
end process;    
END rtl;