-- testbench for FEB common
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity feb_common_tb is
end entity feb_common_tb;

architecture rtl of feb_common_tb is

  component data_merger is
    port (
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
  end component;
  
  component state_controller is
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
	end component;

  
  

	signal	clk   : std_logic := '1';
	signal	reset : std_logic;
	signal	state_idle              :  std_logic; -- "reset" states from state controller 
	signal	state_run_prepare       :  std_logic;
	signal	state_sync              :  std_logic;
	signal	state_running           :  std_logic;
	signal	state_terminating       :  std_logic;
	signal	state_link_test         :  std_logic;
	signal	state_sync_test         :  std_logic;
	signal	state_reset             :  std_logic;
	signal	state_out_of_DAQ        :  std_logic;
	signal	data_out						:	std_logic_vector(31 downto 0);
	signal	data_is_k					:	std_logic_vector(3 downto 0);
	signal 	data_in						:	std_logic_vector(35 downto 0);
	signal 	data_in_slowcontrol		:	std_logic_vector(35 downto 0);
	signal 	slowcontrol_fifo_empty	:	std_logic;
	signal 	data_fifo_empty			: 	std_logic;
	signal 	slowcontrol_read_req		:	std_logic;
	signal 	data_read_req				:  std_logic;
	signal	terminated					:	std_logic;
	signal	reset_link					: 	std_logic_vector(7 downto 0);
	signal 	runnumber					: 	std_logic_vector(31 downto 0);


begin
  clk   <= not clk  after 4 ns; 
  reset <= '1', '0' after 24 ns;

  merger : data_merger
    port map (
      clk       => clk,
      reset     => reset,
		fpga_ID_in 					=> (others => '0'), -- will be set by 15 jumpers in the end, set this to something random for now 
		FEB_type_in 		      => "001010", -- Type of the frontendboard (001010: mupix, 001000: mutrig, DO NOT USE 000111 or 000000 HERE !!!!)
		state_idle              => state_idle, -- "reset" states from state controller 
		state_run_prepare       => state_run_prepare,
		state_sync              => state_sync,
		state_running           => state_running,
		state_terminating       => state_terminating,
		state_link_test         => state_link_test,
		state_sync_test         => state_sync_test,
		state_reset             => state_reset,
		state_out_of_DAQ        => state_out_of_DAQ,
		data_out                => data_out, -- to optical transm.
		data_is_k               => data_is_k, -- to optical trasm.
		data_in 						=> data_in, -- data input from FIFO (32 bit data, 4 bit ID (0010 Header, 0011 Trail, 0000 Data))
		data_in_slowcontrol     => data_in_slowcontrol, -- data input slowcontrol from SCFIFO (32 bit data, 4 bit ID (0010 Header, 0011 Trail, 0000 SCData))
		slowcontrol_fifo_empty  => slowcontrol_fifo_empty,
		data_fifo_empty 			=> data_fifo_empty,
		slowcontrol_read_req    => slowcontrol_read_req,
		data_read_req           => data_read_req,
		terminated              => terminated, -- to state controller (when stop run acknowledge was transmitted the state controller can go from terminating into idle, this is the signal to tell him that)
		override_data_in			=> (others => '0'), -- data input for states link_test and sync_test;
		override_data_is_k_in   => "0000",
		leds							=> open -- debug
		
      );
		
	ustate_controller : component state_controller
    port map(
        clk => clk,
        reset => reset,
        reset_link_8bData => reset_link,
        state_idle => state_idle,
        state_run_prepare => state_run_prepare,
        state_sync => state_sync,
        state_running => state_running,
        state_terminating => state_terminating,
        state_link_test => state_link_test,
        state_sync_test => state_sync_test,
        state_reset => state_reset,
        state_out_of_DAQ => state_out_of_DAQ,
        fpga_addr => (others => '0'), 
        runnumber => runnumber,
        reset_mask => open,
        link_test_payload => open,
        sync_test_payload => open,
        terminated => terminated
        
    );
    
	 

end architecture;