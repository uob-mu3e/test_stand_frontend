-- demerging data and slowcontrol from FEB on the switching board
-- Martin Mueller, May 2019

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.daq_constants.all;

ENTITY data_demerge is
port (
		clk:                    in  std_logic; -- receive clock (156.25 MHz)
 		reset:                  in  std_logic;
		aligned:						in  std_logic; -- word alignment achieved
		data_in:						in  std_logic_vector(31 downto 0); -- optical from frontend board
		datak_in:               in  std_logic_vector(3 downto 0);
		data_out:					out std_logic_vector(31 downto 0); -- to sorting fifos
		datak_out:					out std_logic_vector(3 downto 0); -- to sorting fifos
		data_ready:             out std_logic;							  -- write req for sorting fifos	
		sc_out:						out std_logic_vector(31 downto 0); -- slowcontrol from frontend board
		sck_out:						out std_logic_vector(3 downto 0); -- slowcontrol from frontend board
		sc_out_ready:				out std_logic;
		fpga_id:						out std_logic_vector(15 downto 0)  -- FPGA ID of the connected frontend board
);
end entity;

architecture rtl of data_demerge is	

----------------signals---------------------
    type   data_demerge_state is (idle,receiving_data, receiving_slowcontrol);
    signal demerge_state : 		data_demerge_state;
	signal slowcontrol_type:		std_logic_vector(1 downto 0);
    
----------------begin data_demerge------------------------
BEGIN

    process (clk, reset, aligned)
    begin
        if (reset = '1' or aligned = '0') then 
            demerge_state 		<= idle;
            data_ready 			<= '0';
            data_out 			<= (others => '0');
				datak_out			<= (others => '0');
				sc_out_ready		<= '0';
				sc_out				<= (others => '0');
            
        elsif (rising_edge(clk)) then
  
			 case demerge_state is
			 
				  when idle =>
						data_ready 			<= '0';
						data_out 			<= (others => '0');
						datak_out			<= (others => '0');
						sc_out_ready		<= '0';
						sc_out				<= (others => '0');
						sck_out				<= (others => '0');
						
						if (datak_in(3 downto 0) = "0001" and data_in(31 downto 29)="111") then -- Mupix or MuTrig preamble
							fpga_id					<=	data_in(23 downto 8);
							demerge_state 			<= receiving_data;
						elsif (datak_in(3 downto 0) = "0001" and data_in(31 downto 26)="000111") then -- SC preamble
							fpga_id					<=	data_in(23 downto 8);
							demerge_state 			<= receiving_slowcontrol;
							slowcontrol_type		<= data_in(25 downto 24);
							sc_out_ready			<= '1';
							sc_out					<= data_in;
							sck_out					<= datak_in;
						end if;
				  
				  when receiving_data =>
						data_ready 					<= '1';
						if(data_in (31 downto 0) = K28_4 and datak_in = "0001") then 
							 demerge_state 		<= idle;
							 data_ready				<= '0';					-- TODO: do something with the trailer bits (31 downto 8) here (They are used for Mutrig, DAQ Week 2019)
							 data_out				<= (others => '0');
							 datak_out				<= (others => '0');
							 
						elsif(data_in (31 downto 0)= K28_5 and datak_in = "0001") then
							 data_ready				<= '0';
							 data_out				<= (others => '0');
							 datak_out				<= (others => '0');
						else
							 data_out				<= data_in;
							 datak_out				<= datak_in;
						end if;
						
				  when receiving_slowcontrol =>
						sc_out_ready						<= '1';
						if(data_in (31 downto 0) = K28_4 and datak_in = "0001") then 
							 demerge_state 		<= idle;
							 sc_out_ready			<= '0';
							 sc_out					<= (others => '0');
							 sck_out					<= (others => '0');
						elsif(data_in (31 downto 0)= K28_5 and datak_in = "0001") then
							 sc_out_ready			<= '0';
							 sc_out					<= (others => '0');
							 sck_out					<= (others => '0');
						else
							 sc_out					<= data_in;
							 sck_out					<= datak_in;
						end if;
			 end case;
			 
		end if;
    end process;
END architecture;
