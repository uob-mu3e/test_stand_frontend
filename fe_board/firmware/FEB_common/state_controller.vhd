-- run start and reset state machine for AriaV frontend board
-- Protocol defined in Mu3e-Note-46 "Run Start and Reset Protocol" (Version 12.11.2018) (+ Stop reset signal)

----------------------------------
-- PLEASE READ THE README !!!!!!!
----------------------------------

--states:
--idle
--run prepare
--Sync
--running
--terminating
--link test
--sync test
--Reset
--out of DAQ

-- Martin Mueller, 2018


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


ENTITY state_controller is 
    PORT(
        clk:                    in  std_logic; -- 125 Mhz clock from reset transceiver
        reset:                  in  std_logic; -- hard reset for testing
        reset_link_8bData :     in  std_logic_vector(7 downto 0); -- input reset line
        state_idle:             out std_logic; -- state outputs
        state_run_prepare:      out std_logic;
        state_sync:             out std_logic;
        state_running:          out std_logic;
        state_terminating:      out std_logic;
        state_link_test:        out std_logic;
        state_sync_test:        out std_logic;
        state_reset:            out std_logic;
        state_out_of_DAQ:       out std_logic;
        fpga_addr:              in  std_logic_vector(15 downto 0); -- FPGA address input for addressed reset commands(from jumpers on final FEB board)
        runnumber:              out std_logic_vector(31 downto 0); -- runnumber received from midas with the prep_run command
        reset_mask:             out std_logic_vector(15 downto 0); -- payload output of the reset command
        link_test_payload:      out std_logic_vector(15 downto 0); -- to be specified 
        sync_test_payload:      out std_logic_vector(15 downto 0); -- to be specified
        terminated:             in std_logic								 -- connect this to data merger "end of run was transmitted, run can be terminated"
    );
END ENTITY state_controller;

architecture rtl of state_controller is


----------------signals---------------------
    signal states:                          std_logic_vector(8 downto 0);
    signal recieving_payload:               std_logic;
    signal payload_byte_counter:            integer range 0 to 10;
    signal addressing_payload_byte_counter: integer range 0 to 10;
    --signal terminated:                      std_logic;
    
    -- address signals
    signal addressed:           std_logic;
    signal address_part1_match: std_logic;
    signal recieving_address:   std_logic;
    signal ignoring_signals:    std_logic;
    
    type state_type is (idle, run_prep, sync, running, terminating, link_test, sync_test, reset_state, out_of_DAQ);
    signal state : state_type;
    
----------------begin state controller------------------------



BEGIN


     --state vector assignments
    state_idle <= states(0);
    state_run_prepare <= states(1);
    state_sync <= states(2);
    state_running <= states(3);
    state_terminating <= states(4);
    state_link_test <= states(5);
    state_sync_test <= states(6);
    state_reset <= states(7);
    state_out_of_DAQ <= states(8);
    
        
--urun_termination_controller : component run_termination_controller
--     port map(
--        clk => clk,
--        reset => reset,
--        state_terminating => states(4),
--        terminated => terminated
--    );
    
    
    -- output process
    process (clk, recieving_payload)
    begin
        if (rising_edge (clk)) and (recieving_payload = '0') then
            case state is
                when idle =>
                    states <= (0=>'1', others => '0');
                when run_prep=>
                    states <= (1=>'1', others => '0');
                when sync =>
                    states <= (2=>'1', others => '0');
                when running =>
                    states <= (3=>'1', others => '0');
                when terminating =>
                    states <= (4=>'1', others => '0');
                when link_test =>
                    states <= (5=>'1', others => '0');
                when sync_test =>
                    states <= (6=>'1', others => '0');
                when reset_state =>
                    states <= (7=>'1', others => '0');
                when out_of_DAQ =>
                    states <= (8=>'1', others => '0');
            end case;
        end if;
    end process;
    
    -- address
    
    --- transition process
    process (clk, reset, addressed)
    begin
        if reset = '1' then
            state <= idle;
            payload_byte_counter <= 0;
            recieving_payload <= '0';
            reset_mask <= (others =>'0');
            link_test_payload <= (others =>'0');
            sync_test_payload <= (others =>'0');
            runnumber <= (others =>'0');
            
        elsif (rising_edge(clk) and addressed = '1') then
            
            if recieving_payload = '1' then -- recieve payload 
                payload_byte_counter <= payload_byte_counter - 1;
                if payload_byte_counter = 1 then 
                    recieving_payload <= '0';
                end if;
                
                -- write payload to correct output
                case state is
                    when run_prep =>
                        runnumber((payload_byte_counter ) * 8 - 1 downto (payload_byte_counter-1)*8) <= reset_link_8bData;
                    when link_test =>
                         link_test_payload((payload_byte_counter ) * 8 - 1 downto (payload_byte_counter-1)*8) <= reset_link_8bData;
                    when sync_test =>
                         sync_test_payload((payload_byte_counter ) * 8 - 1 downto (payload_byte_counter-1)*8) <= reset_link_8bData;
                    when reset_state => 
                         reset_mask((payload_byte_counter ) * 8 - 1 downto (payload_byte_counter-1)*8) <= reset_link_8bData;
                    when others => -- should not happen
                end case;
            
                    
            else  -- command incoming --> listen and do stuff 
                case state is
                        ------------ idle ------------
                        when idle =>
                            if reset_link_8bData = x"10" then  -- run prepare
                                recieving_payload <= '1';
                                payload_byte_counter <= 4;
                                state <= run_prep;
                            elsif reset_link_8bData = x"20" then -- start link test
                                recieving_payload <= '1';
                                payload_byte_counter <= 2;
                                state <= link_test;
                            elsif reset_link_8bData = x"24" then -- start sync test
                                recieving_payload <= '1';
                                payload_byte_counter <= 2;
                                state <= sync_test;
                            elsif reset_link_8bData = x"30" then -- reset
                                recieving_payload <= '1';
                                payload_byte_counter <= 2;
                                state <= reset_state;
                            elsif reset_link_8bData = x"33" then -- disable
                                state <= out_of_DAQ;
                            else 
                                state <= idle;
                            end if;


                        ------------ run prepare --------------
                        when run_prep =>
                            if reset_link_8bData = x"11" then  -- sync
                                state <= sync;
                            elsif reset_link_8bData = x"14" then -- abort run
                                state <= idle;
                            elsif reset_link_8bData = x"30" then -- reset
                                state <= reset_state;
                                recieving_payload <= '1';
                                payload_byte_counter <= 2;
                            elsif reset_link_8bData = x"33" then -- disable
                                state <= out_of_DAQ;
                            else 
                                state <= run_prep;
                            end if;
                                
                        ------------ sync --------------
                        when sync => 
                            if reset_link_8bData = x"12" then  -- start run
                                state <= running;
                            elsif reset_link_8bData = x"14" then -- abort run
                                state <= idle;
                            elsif reset_link_8bData = x"30" then -- reset 
                                state <= reset_state;
                                recieving_payload <= '1';
                                payload_byte_counter <= 2;
                            elsif reset_link_8bData = x"33" then -- disable
                                state <= out_of_DAQ;
                            else
                                state <= sync;
                            end if;
                        
                        ------------ running --------------
                        when running =>
                            if reset_link_8bData = x"13" then -- end run
                                state <= terminating;
                            elsif reset_link_8bData = x"14" then -- abort run
                                state <= idle;
                            elsif reset_link_8bData = x"30" then -- reset
                                state <= reset_state;
                                recieving_payload <= '1';
                                payload_byte_counter <= 2;
                            elsif reset_link_8bData = x"33" then -- disable
                                state <= out_of_DAQ;
                            else
                                state <= running;
                            end if;
                            
                        ------------ terminating --------------
                        when terminating =>
                            if terminated = '1' then            -- terminating finished
                                state <= idle;
                            elsif reset_link_8bData = x"14" then -- abort run
                                state <= idle;
                            elsif reset_link_8bData = x"30" then -- reset
                                state <= reset_state;
                                recieving_payload <= '1';
                                payload_byte_counter <= 2;
                            elsif reset_link_8bData = x"33" then -- disable
                                state <= out_of_DAQ;
                            else
                                state <= terminating;
                            end if;
                            
                       ------------ link test --------------
                       when link_test =>
                            if reset_link_8bData = x"21" then  -- stop link test
                                state <= idle;
                            elsif reset_link_8bData = x"30" then -- reset
                                state <= reset_state;
                                recieving_payload <= '1';
                                payload_byte_counter <= 2;
                            elsif reset_link_8bData = x"33" then -- disable
                                state <= out_of_DAQ;
                            else
                                state <= link_test;
                            end if;
                            
                       ------------ sync test --------------
                       when sync_test =>
                            if reset_link_8bData = x"25" then  -- stop sync test
                                state <= idle;
                            elsif reset_link_8bData = x"30" then -- reset
                                state <= reset_state;
                                recieving_payload <= '1';
                                payload_byte_counter <= 2;
                            elsif reset_link_8bData = x"33" then -- disable
                                state <= out_of_DAQ;
                            else
                                state <= sync_test;
                            end if;
                            
                        ------------ reset state --------------
                        when reset_state =>
                            if reset_link_8bData = x"31" then        -- reset finished
                                state <= idle;
                            elsif reset_link_8bData = x"33" then  -- disable
                                state <= out_of_DAQ;
                            else 
                                state <= reset_state;
                            end if;
                            
                        when out_of_DAQ =>
                            
                            if reset_link_8bData = x"32" then   -- enable
                                state <= idle;
                            end if;
								when others =>
									state <= idle;
                            
                    end case;
            end if;
        end if;
    end process;
    
    -- address process
    process (clk, reset)
    begin
        if reset = '1' then
            addressed <= '1';
            address_part1_match <= '0';
            recieving_address   <= '0';
            ignoring_signals    <= '0';
            addressing_payload_byte_counter <= 0;
            
        elsif rising_edge(clk) then
            
            
            -- address command, not part of a payload, --> compare the next 32 bits with own address
            if (recieving_payload = '0' and reset_link_8bData = x"40" and recieving_address = '0' and ignoring_signals = '0') then  
                addressed <= '0';
                recieving_address  <= '1';
                address_part1_match <= '0';
                addressing_payload_byte_counter <= 2;      -- 2 byte address
            end if;
            
            
            -- recieve address for 2 cycles
            if (recieving_address = '1' and ignoring_signals = '0') then
                addressing_payload_byte_counter <= addressing_payload_byte_counter -1;
                
                -- check first address byte
                if (addressing_payload_byte_counter = 2 and reset_link_8bData = fpga_addr(15 downto 8)) then
                    address_part1_match <= '1';
                end if;
                
                -- check second address byte
                if (addressing_payload_byte_counter = 1) then
                    recieving_address <= '0';
                    
                    if (reset_link_8bData = fpga_addr(7 downto 0) and address_part1_match = '1') then
                        addressed <= '1'; -- address match, go into big case statement next
                        address_part1_match <= '0';
                        
                    else 
                        addressed <= '0'; -- address mismatch
                    end if;
                end if;
            end if;
            
            
            -- full address recieved, address mismatch  --> this command is relevant to determine the cycle where fpga should start listening again (commands have different payload sizes)  
            if(addressed = '0' and addressing_payload_byte_counter = 0 and ignoring_signals = '0') then 
                   case reset_link_8bData is
                        when x"10" => addressing_payload_byte_counter <= 4; ignoring_signals <= '1';
                        when x"20" => addressing_payload_byte_counter <= 2; ignoring_signals <= '1';-- to be specified
                        when x"24" => addressing_payload_byte_counter <= 2; ignoring_signals <= '1';-- to be specified
                        when x"26" => addressing_payload_byte_counter <= 2; ignoring_signals <= '1';-- to be specified
                        when x"30" => addressing_payload_byte_counter <= 2; ignoring_signals <= '1';
                        when others => addressed <= '1'; address_part1_match <= '0'; ignoring_signals <= '0'; -- all other commands dont have payload --> back to normal
                    end case;
            end if;
            
            if(ignoring_signals = '1') then   -- ignore payload of commands to different fpgas
                addressing_payload_byte_counter <= addressing_payload_byte_counter - 1;
                if (addressing_payload_byte_counter = 1) then
                    ignoring_signals <= '0';
                    addressed <= '1';
                    address_part1_match <= '0';
                end if;
            end if;
            
        end if;
    end process;
END rtl;
