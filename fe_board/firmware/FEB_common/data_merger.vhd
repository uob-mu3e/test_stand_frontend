-- data merger for mu3e FEB
-- Martin Mueller, May 2019

----------------------------------
-- PLEASE READ THE README !!!!!!!
----------------------------------

-- 2 states: 
--	merger state: state of this entity (idle, sending data, sending slowcontrol)
--	FEB state: "reset" state from FEB_state_controller (idle, run_prep, sync, running, terminating, link_test, sync_test, reset, outOfDaq)
-- do not confuse them !!!


-- @ future me: 
-- ToDo:
--	error outputs (data does not start with start marker, data fifo not empty in sync, etc. )

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;

ENTITY data_merger is 
	PORT(
		clk:                    in  std_logic; -- 156.25 clk input
		reset:                  in  std_logic; 
		fpga_ID_in:					in  std_logic_vector(15 downto 0); -- will be set by 15 jumpers in the end, set this to something random for now 
		FEB_type_in:				in  std_logic_vector(5  downto 0); -- Type of the frontendboard (111010: mupix, 111000: mutrig, DO NOT USE 000111 or 000000 HERE !!!!)
		run_state:					in  run_state_t;
		data_out:               out std_logic_vector(31 downto 0); -- to optical transm.
		data_is_k:              out std_logic_vector(3 downto 0);  -- to optical trasm.
		data_in:						in  std_logic_vector(35 downto 0); -- data input from FIFO (32 bit data, 4 bit ID (0010 Header, 0011 Trail, 0000 Data))
		data_in_slowcontrol:    in  std_logic_vector(35 downto 0); -- data input slowcontrol from SCFIFO (32 bit data, 4 bit ID (0010 Header, 0011 Trail, 0000 SCData))
		slowcontrol_fifo_empty: in  std_logic;
		data_fifo_empty:			in  std_logic;
		slowcontrol_read_req:   out std_logic;
		data_read_req:          out std_logic;
		terminated:             out std_logic; -- to state controller (when stop run acknowledge was transmitted the state controller can go from terminating into idle, this is the signal to tell him that)
		override_data_in:			in  std_logic_vector(31 downto 0); -- data input for states link_test and sync_test;
		override_data_is_k_in:  in  std_logic_vector(3 downto 0);
		override_req:				in	 std_logic;
		override_granted:			out std_logic;
		data_priority:				in  std_logic; -- 0: slowcontrol packets have priority, 1: data packets have priority
		leds:							out std_logic_vector(3 downto 0) -- debug
   );
END ENTITY data_merger;

architecture rtl of data_merger is

	type data_merger_state is (idle, sending_data, sending_slowcontrol);
	--type feb_state is (idle, run_prep, sync, running, terminating, link_test, sync_test, reset_state, out_of_DAQ);

	-- ToDo: Move this to some common location (for all boards !!)
	constant K285:									std_logic_vector(31 downto 0) :=x"000000bc";
	constant K285_datak:							std_logic_vector(3 downto 0)	:= "0001";
	constant K284:									std_logic_vector(7 downto 0) :=x"9c";
	constant K284_datak:							std_logic_vector(3 downto 0)	:= "0001";
	constant K307:									std_logic_vector(7 downto 0)	:= x"fe";
	
	constant run_prep_acknowledge:			std_logic_vector(31 downto 0)	:= x"000001fe";
	constant run_prep_acknowledge_datak:	std_logic_vector(3 downto 0) 	:= "0001";
	constant RUN_END:								std_logic_vector(31 downto 0)	:= x"000002fe";
	constant RUN_END_DATAK:						std_logic_vector(3 downto 0)	:= "0001";
	
	constant MERGER_FIFO_RUN_END_MARKER:	std_logic_vector(3 downto 0)	:= "0111";


----------------components------------------


----------------signals---------------------
	signal merger_state 							: data_merger_state;
	signal run_prep_acknowledge_send 		: std_logic;
	signal last_merger_fifo_control_bits	: std_logic_vector(3 downto 0); -- used for run termination


----------------begin data merger------------------------

BEGIN

-- debug led merger state
process (merger_state)
	begin
    leds <= (others => '0');
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
    if ( reset = '1' ) then
        merger_state                <= idle;
        slowcontrol_read_req        <= '0';
        data_read_req               <= '0';
        terminated                  <= '0';
        run_prep_acknowledge_send   <= '0';
        data_is_k                   <= K285_datak;
        data_out                    <= K285;
        override_granted            <= '0';
        --
    elsif rising_edge(clk) then
		  
		  ------------------------------- feb state link test or sync test ----------------------------
		  -- use override data input
		  -- wait for slowcontrol to finish before
		  
			if (run_state = RUN_STATE_LINK_TEST or run_state = RUN_STATE_SYNC_TEST) then
				case merger_state is
					when idle =>
						-- send override start (Problem: how to end this on switch side ??)
						if(override_req='1') then
							data_is_k 					<= "0001"; 
							data_out (31 downto 26) <= "010101";
							data_out (25 downto 24)	<= "00";
							data_out (23 downto 8)  <= fpga_ID_in;
							data_out (7  downto 0) 	<= x"bc";
							
							merger_state				<= sending_data;
							override_granted			<= '1';
						else
							data_out						<= K285;
							data_is_k 					<= K285_datak;
						end if;
					when sending_slowcontrol =>
						-- slowcontrol header is trasmitted, send slowcontrol data now
						if(slowcontrol_fifo_empty='1') then			-- send k285 idle, leave read req = 1 ?
							data_out(31 downto 0)  	<= K285;
							data_is_k 					<= K285_datak;  
						elsif(data_in_slowcontrol(33 downto 32)= "11") then -- end of packet marker
							merger_state				<= idle;
							slowcontrol_read_req		<= '0';
							data_out(31 downto 0)  	<= x"000000" & K284;
							data_is_k 					<= K284_datak;
						else
							slowcontrol_read_req		<= '1';
							data_out						<= data_in_slowcontrol(31 downto 0);
							data_is_k					<= "0000";
						end if;
					when others =>
					-- state controller does not allow to go from running or terminating into *_test state 
					-- --> merger_state sending_data is used for override data
						if(override_req='1')	then
							data_out 					<= override_data_in;
							data_is_k 					<= override_data_is_k_in;
						else
							merger_state 				<= idle;
							override_granted			<= '0';
							data_out(31  downto 0) 	<= x"000000" & K284;
							data_is_k 					<= K284_datak;
						end if;
				end case;

		  
		  ------------------------------- feb state sync or reset ------------------------------
		  -- send only komma words
		  -- wait for slowcontrol to finish before
		  elsif(run_state = RUN_STATE_SYNC or run_state = RUN_STATE_RESET) then
				case merger_state is
					when sending_slowcontrol =>
						-- slowcontrol header is trasmitted, send slowcontrol data now
						if(slowcontrol_fifo_empty='1') then			-- send k285 idle, leave read req = 1 ?
							data_out(31 downto 0)  	<= K285;
							data_is_k 					<= K285_datak;  
						elsif(data_in_slowcontrol(33 downto 32)= "11") then -- end of packet marker
							merger_state				<= idle;
							slowcontrol_read_req		<= '0';
							data_out(31 downto 0)  	<= x"000000" & K284;
							data_is_k 					<= K284_datak;
						else
							slowcontrol_read_req		<= '1';
							data_out						<= data_in_slowcontrol(31 downto 0);
							data_is_k					<= "0000";
						end if;
					when others =>
                    merger_state 				<= idle;
						  data_out 						<= K285;
                    data_is_k 					<=K285_datak;
				end case;
						  
		  ------------------------------- feb state idle or outOfDaq --------------------------
        elsif(run_state = RUN_STATE_IDLE or run_state = RUN_STATE_OUT_OF_DAQ)then
				terminated 							<= '0';
            run_prep_acknowledge_send 		<= '0';
				override_granted					<= '0';
				last_merger_fifo_control_bits <= "0000";
				
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
                        data_out 				<= K285;
                        data_is_k 				<= K285_datak;
                    end if;
                    
                when sending_slowcontrol => 							-- slowcontrol header is trasmitted, send slowcontrol data now
						if(slowcontrol_fifo_empty='1') then			-- send k285 idle, leave read req = 1 ?
							data_out(31 downto 0)  	<= K285;
							data_is_k 					<= K285_datak;  
						elsif(data_in_slowcontrol(33 downto 32)= "11") then -- end of packet marker
							merger_state				<= idle;
							slowcontrol_read_req		<= '0';
							data_out(31 downto 0)  	<= x"000000" & K284;
							data_is_k 					<= K284_datak;
						else
							slowcontrol_read_req		<= '1';
							data_out						<= data_in_slowcontrol(31 downto 0);
							data_is_k					<= "0000";
						end if;
                    
                when others =>											-- send data state in FEB state idle should not happen (except if this is the end of *_test state) --> goto merger state idle
                    merger_state <= idle;
                    data_out <= K285;
                    data_is_k <= K285_datak;
            end case;		
				
		  ------------------------------- feb state run prep  ---------------------------------------------
		  
		  elsif(run_state = RUN_STATE_PREP)then
				terminated <= '0';
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
						if(slowcontrol_fifo_empty='1') then			-- send k285 idle, leave read req = 1 ?
							data_out(31 downto 0)  	<= K285;
							data_is_k 					<= K285_datak;  
						elsif(data_in_slowcontrol(33 downto 32)= "11") then -- end of packet marker
							merger_state				<= idle;
							slowcontrol_read_req		<= '0';
							data_out(31 downto 0)  	<= x"000000" & K284;
							data_is_k 					<= K284_datak;
						else
							slowcontrol_read_req		<= '1';
							data_out						<= data_in_slowcontrol(31 downto 0);
							data_is_k					<= "0000";
						end if;
					when others => 										-- it should not be possible to get here		      
						merger_state <= idle;
						data_out <= K285;
						data_is_k <= K285_datak;
				end case;
				
		  ------------------------------- feb state running or terminating  ---------------------------------------------			
		
			elsif(run_state = RUN_STATE_RUNNING or run_state = RUN_STATE_TERMINATING) then
				run_prep_acknowledge_send <= '0';
				case merger_state is
					when idle =>
						if (slowcontrol_fifo_empty = '1' and data_fifo_empty = '1') then -- no data, state is idle --> do nothing
							slowcontrol_read_req 	<= '0';
							data_out 					<= K285;
							data_is_k					<= K285_datak;
						
						elsif (last_merger_fifo_control_bits = MERGER_FIFO_RUN_END_MARKER or data_in(35 downto 32)= MERGER_FIFO_RUN_END_MARKER) then 
							-- allows run end for idle and sending data, run end in state sending_data is always packet end 
							terminated 					<= '1';
							data_out 					<= RUN_END;
							data_is_k					<= RUN_END_DATAK;
							
						elsif((slowcontrol_fifo_empty = '0' and data_fifo_empty = '1') or (slowcontrol_fifo_empty = '0' and data_priority ='0')) then
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
							data_out(31 downto 0)  	<= K285;
							data_is_k 					<= K285_datak;  
						elsif(data_in(33 downto 32)="11") then 	-- run end(0111) in state sending_data is always packet end (XX11)
							merger_state 				<= idle;
							data_read_req				<= '0'; 
							data_out(31 downto 0)  	<= data_in(23 downto 0) & K284;
							data_is_k 					<= K284_datak;
							last_merger_fifo_control_bits <= data_in(35 downto 32); -- save them now --> if 35 downto 32 is actually 0111(run END) then terminate in merger_state idle
						else
							data_read_req				<= '1';
							data_out						<= data_in(31 downto 0);
							data_is_k					<= "0000";
						end if;	
						
					when sending_slowcontrol=>
						if(slowcontrol_fifo_empty='1') then			-- send k285 idle, leave read req = 1 ?
							data_out(31 downto 0)  	<= K285;
							data_is_k <= K285_datak;  
						elsif(data_in_slowcontrol(33 downto 32)= "11") then -- end of packet marker
							merger_state				<= idle;
							slowcontrol_read_req		<= '0';
							data_out(31 downto 0)  	<= x"000000" & K284;
							data_is_k 					<= K284_datak;
						else
							slowcontrol_read_req		<= '1';
							data_out						<= data_in_slowcontrol(31 downto 0);
							data_is_k					<= "0000";
						end if;
					when others =>
                        merger_state <= idle;

				end case;
	
		  end if;
	  end if;
end process;    
END rtl;