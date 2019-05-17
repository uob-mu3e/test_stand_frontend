-- data merger for mu3e FEB
-- Martin Mueller, May 2019

-- 2 states: 
--	merger state: state of this entity (idle, sending data, sending slowcontrol)
--	FEB state: "reset" state from FEB_state_controller (idle, run_prep, sync, running, terminating, link_test, sync_test, reset, outOfDaq)
-- do not confuse them !!!

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.protocol_new.all;

ENTITY data_merger_new is 
    PORT(
        clk:                    in  std_logic; -- 156.25 clk input
        reset:                  in  std_logic; 
		  fpga_ID_in:				  in  std_logic_vector(15 downto 0); -- will be set by 15 jumpers in the end, set this to something random for now 
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
        data_in:          		  in  std_logic_vector(35 downto 0); -- data input from FIFO (32 bit data, 4 bit ID (0010 Header, 0011 Trail, 0000 Data))
        data_in_slowcontrol:    in  std_logic_vector(35 downto 0); -- data input slowcontrol from SCFIFO (32 bit data, 4 bit ID (0010 Header, 0011 Trail, 0000 SCData))
        slowcontrol_fifo_empty: in  std_logic;
        data_fifo_empty:        in  std_logic;
        slowcontrol_read_req:   out std_logic;
        data_read_req:          out std_logic;
		  terminated:             out std_logic; -- to state controller (when stop run acknowledge was transmitted the state controller can go from terminating into idle, this is the signal to tell him that)
		  override_data_in:		  in  std_logic_vector(31 downto 0); -- data input for states link_test and sync_test;
		  override_data_is_k_in:  in  std_logic_vector(3 downto 0);
		  leds:						  out std_logic_vector(3 downto 0) -- debug
        
    );
END ENTITY data_merger_new;

architecture rtl of data_merger_new is

----------------components------------------


----------------signals---------------------
    signal merger_state : data_merger_state;
	 
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
        elsif(idle = '1')then
            active_send <= '0';
				
            case merger_state is
            
                when idle =>
						  -- not sending something, in state idle, slowcontrol fifo not empty
                    if (slowcontrol_fifo_empty = '0') then
								slowcontrol_read_req <= '1'; 				-- need 2 cycles to get new data from fifo --> start reading now
								
								-- send SC header:
								data_out (31 downto 24) <= x"bc";
								data_is_k 					<= "1000"; 
								data_out (23 downto 18) <= "000111"
								data_out (17 downto 16)	<= data_in_slowcontrol(35 downto 34);
								data_out (15 downto 0)  <= fpga_ID_in;
                        merger_state <= sending_slowcontrol; 	-- go to sending slowcontrol state next
								
                    else 													-- no data --> do nothing
								slowcontrol_read_req <= '0';
								data_read_req			<= '0';
                        data_out <= K285;
                        data_is_k <= K285_datak;
                    end if;
                    
                when sending_slowcontrol => 							-- slowcontrol header is trasmitted, send slowcontrol data now
                    if slowcontrol_fifo_empty = '0' then 		-- send slowcontrol data
						      data_out <= data_in_slowcontrol;
                        data_is_k <= "0000";
								slowcontrol_read_req <= '1';
                    else 													-- slowcontrol fifo empty --> send end marker and goto idle state
						      merger_state <= idle;
								slowcontrol_read_req <= '0';
                        data_out(31 downto 24) <= x"9C";
								data_out(23 downto 0)  <= (others => '0');
                        data_is_k <= "1000";  

                    end if;
                    
                when others =>											-- send data state in FEB state idle should not happen --> goto merger state idle
                    merger_state <= idle;
                    data_out <= K285;
                    data_is_k <= K285_datak;
            end case;		
				
				
				
				--- dummy for missing states---
		  elsif(state_running = '1' or state_terminating = '1' or state_run_prepare = '1' or state_sync='1') then
                    merger_state <= idle;
						  data_out <= K285;
                    data_is_k <=K285_datak;				
		  
		  end if;
	  end if;
end process;
    
END rtl;