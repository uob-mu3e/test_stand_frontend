-----------------------------------------------------------------------------
-- PCIe to register application, pcie writeable registers
--
-- Niklaus Berger, Heidelberg University
-- nberger@physi.uni-heidelberg.de
--
-----------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.pcie_components.all;

entity pcie_writeable_registers is 
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
		
		-- registers
		writeregs :				out	reg32array;
		regwritten :			out   std_logic_vector(63 downto 0);
		
		-- to response engine
		readaddr :				out std_logic_vector(5 downto 0);
		readlength :			out std_logic_vector(9 downto 0);
		header2 :				out std_logic_vector(31 downto 0);
		readen :					out std_logic;
		inaddr32_w			: out STD_LOGIC_VECTOR (31 DOWNTO 0)
	
	);
	end pcie_writeable_registers;
	


architecture RTL of pcie_writeable_registers is

	type receiver_state_type is (reset, waiting);
	signal state : receiver_state_type;
	
	signal inaddr32 : 			std_logic_vector(31 downto 0);
	signal regaddr  : 			std_logic_vector(5 downto 0);
	signal regaddr_reg  : 		std_logic_vector(5 downto 0);
	
	-- Decoding PCIe TLP headers
	signal fmt : 		std_logic_vector(1 downto 0);
	signal ptype :    std_logic_vector(4 downto 0);
	signal tc :			std_logic_vector(2 downto 0);
	signal td :			std_logic;
	signal ep :			std_logic;
	signal attr :		std_logic_vector(1 downto 0);
	signal plength :	std_logic_vector(9 downto 0);
	signal plength_reg : std_logic_vector(9 downto 0);
	signal fdw_be:			std_logic_vector(3 downto 0);
	signal ldw_be:			std_logic_vector(3 downto 0);
	signal ldw_be_reg: 	std_logic_vector(3 downto 0);

	signal word3 : std_logic_vector(31 downto 0);
	signal word4 : std_logic_vector(31 downto 0);

	
	signal be3	 : std_logic;
	signal be4	 : std_logic;

	
	
	signal addr3: 	 std_logic_vector(5 downto 0);
	signal addr4:	 std_logic_vector(5 downto 0);

	
	-- registers
	signal writeregs_r	: reg32array;
	signal regwritten_r : std_logic_vector(63 downto 0);
	
	constant zero32	:  std_logic_vector(31 downto 0) := "00000000000000000000000000000000";

begin

	-- Endian chasing for addresses
	inaddr32 <= rx_st_data0(95 downto 66) & "00";
	--inaddr32 <= rx_st_data0(95 downto 74) & "1" & rx_st_data0(72 downto 66) & "00";
	
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
	
	writeregs <= writeregs_r;
	
	process(local_rstn, refclk)
	
	variable regaddr_var : std_logic_vector(5 downto 0);
	variable length_var  : std_logic_vector(9 downto 0);
	
	begin	
	
	if(local_rstn = '0') then
		state 			<= reset;
		rx_st_ready0    <= '0';
		writeregs_r		<= (others => zero32);
		regwritten		<= (others => '0');

	
	elsif (refclk'event and refclk = '1') then
		regwritten <= regwritten_r;
	
		readen	<= '0';
		word3		<= rx_st_data0(127 downto 96);	
		word4		<= rx_st_data0(159 downto 128);
		regwritten_r		<= (others => '0');
		

		
		
		-- do the actual writing
		if (be3 = '1') then
			writeregs_r(TO_INTEGER(UNSIGNED(addr3))) <= word3;
			regwritten_r(TO_INTEGER(UNSIGNED(addr3))) <= '1';
		end if;
		if (be4 = '1') then
			writeregs_r(TO_INTEGER(UNSIGNED(addr4))) <= word4;
			regwritten_r(TO_INTEGER(UNSIGNED(addr4))) <= '1';
		end if;
	
		
	
		
		case state is
			when reset =>
				state 			<= waiting;
				rx_st_ready0   <= '0';
				be3				<= '0';
				be4				<= '0';
	-------------------------------------------------------------------------------------			
			when waiting =>
				be3				<= '0';
				be4				<= '0';
				rx_st_ready0   <= '1';
				
				if(rx_st_sop0 = '1' and rx_bar = '1') then --  and inaddr32 = x"fb480040"
					if(fmt = "10" and ptype = "00000") then -- 32 bit memory write request
						if(inaddr32(2) = '1') then -- Unaligned write, first data word at word3
							addr3 	<= regaddr;
							if(fdw_be /= "0000") then
								be3 		<= '1';
							else
								be3 		<= '0';
							end if;
							state <= waiting;
						else -- aligned write, first data word at word4
							addr4 	<= regaddr;
							if(fdw_be /= "0000") then
								be4 		<= '1';
							else
								be4 		<= '0';
							end if;
							state <= waiting;
						end if; -- if aligned
					elsif(fmt = "00" and ptype = "00000") then -- 32 bit memory read request 	
						inaddr32_w <= rx_st_data0(95 downto 66) & "00";
						readaddr		<= regaddr;
						readlength 	<= plength;
						header2		<= rx_st_data0(63 downto 32);
						readen		<= '1';
						state			<= waiting;
					end if; -- 32 bit write/read request
				end if; -- if Start of Packet
				state <= waiting;
		end case;	
	end if; -- if clk event
	end process;
	
end RTL;