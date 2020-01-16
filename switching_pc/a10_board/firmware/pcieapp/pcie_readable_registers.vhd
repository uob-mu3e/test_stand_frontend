-----------------------------------------------------------------------------
-- PCIe to register application, pcie readable registers
--
-- Niklaus Berger, Heidelberg University
-- nberger@physi.uni-heidelberg.de
--
-----------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity pcie_readable_registers is
port (
		local_rstn:				in		std_logic;
		refclk:					in		std_logic;
	
		-- from IF
		rx_st_data0 :  		in 	STD_LOGIC_VECTOR (255 DOWNTO 0);
		rx_st_eop0 :  			in		STD_LOGIC;
		rx_st_sop0 :  			in 	STD_LOGIC;
		rx_st_ready0 :			out 	STD_LOGIC;
		rx_st_valid0 :			in 	STD_LOGIC;
		rx_bar :					in 	STD_LOGIC;
				
		-- to response engine
		readaddr :				out std_logic_vector(5 downto 0);
		readlength :			out std_logic_vector(9 downto 0);
		header2 :				out std_logic_vector(31 downto 0);
		readen :					out std_logic;
		inaddr32_r			: out STD_LOGIC_VECTOR (31 DOWNTO 0)

);
end entity;



architecture RTL of pcie_readable_registers is

	type receiver_state_type is (reset, waiting);
	signal state : receiver_state_type;
	
	signal inaddr32 : 			std_logic_vector(31 downto 0);
	signal regaddr  : 			std_logic_vector(5 downto 0);
	
	-- Decoding PCIe TLP headers
	signal fmt : 		std_logic_vector(1 downto 0);
	signal ptype :    std_logic_vector(4 downto 0);
	signal tc :			std_logic_vector(2 downto 0);
	signal td :			std_logic;
	signal ep :			std_logic;
	signal attr :		std_logic_vector(1 downto 0);
	signal plength :	std_logic_vector(9 downto 0);
	signal plength_reg : std_logic_vector(9 downto 0);
	signal fdw_be:		std_logic_vector(3 downto 0);
	signal ldw_be:		std_logic_vector(3 downto 0);
	signal ldw_be_reg: std_logic_vector(3 downto 0);
	
	

begin

	-- Endian chasing for addresses
	inaddr32 <= rx_st_data0(95 downto 66) & "00";
	--inaddr32 <= rx_st_data0(95 downto 74) & "0" & rx_st_data0(72 downto 66) & "00";
	
	regaddr	<= inaddr32(7 downto 2);
	
	
	-- decode TLP
	fmt		<=		rx_st_data0(30 downto 29);
	ptype		<=		rx_st_data0(28 downto 24);
	tc			<= 	rx_st_data0(22 downto 20);
	td			<= 	rx_st_data0(15);
	ep			<=		rx_st_data0(14);
	attr		<=		rx_st_data0(13 downto 12);
	plength	<= 	rx_st_data0(9 downto 8) & rx_st_data0(7 downto 0);
	fdw_be	<= 	rx_st_data0(35 downto 32);
	ldw_be	<= 	rx_st_data0(39 downto 36);
	
	
	
	process(local_rstn, refclk)
		
	begin	
	
	if(local_rstn = '0') then
		state 			<= reset;
		rx_st_ready0    <= '0';

	
	elsif (refclk'event and refclk = '1') then
		readen	<= '0';
		case state is
			when reset =>
				state 			<= waiting;
				rx_st_ready0   <= '0';
	-------------------------------------------------------------------------------------			
			when waiting =>
				rx_st_ready0   <= '1';
				if(rx_st_sop0 = '1' and rx_bar = '1') then -- and inaddr32 = x"fb480024"
					if(fmt = "00" and ptype = "00000") then -- 32 bit memory read request 	
						inaddr32_r <= rx_st_data0(95 downto 66) & "00";
						readaddr		<= regaddr;
						readlength 	<= plength;
						header2		<= rx_st_data0(63 downto 32);
						readen		<= '1';
						state			<= waiting;
					end if; -- 32 bit write/read request
				end if; -- if Start of Packet
		end case;	
	end if; -- if clk event
	end process;

end architecture;
