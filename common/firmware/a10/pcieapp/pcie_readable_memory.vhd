-----------------------------------------------------------------------------
-- PCIe to memory application, pcie readable memory
--
-- Niklaus Berger, Heidelberg University
-- nberger@physi.uni-heidelberg.de
--
-- Only handles reads which are 128 bit aligned
-----------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity pcie_readable_memory is
port (
		local_rstn:				in		std_logic;
		refclk:					in		std_logic;
	
		-- from IF
    i_rx_st_data0   : in    std_logic_vector(255 downto 0);
    i_rx_st_eop0    : in    std_logic;
    i_rx_st_sop0    : in    std_logic;
    o_rx_st_ready0  : out   std_logic;
    i_rx_st_valid0  : in    std_logic;
    i_rx_bar        : in    std_logic;

		-- to response engine
		readaddr :				out std_logic_vector(15 downto 0);
		readlength :			out std_logic_vector(9 downto 0);
		header2 :				out std_logic_vector(31 downto 0);
		readen :					out std_logic
);
end entity;



architecture RTL of pcie_readable_memory is

    signal rx_st_data0 : std_logic_vector(i_rx_st_data0'range);
    signal rx_st_sop0 : std_logic;
    signal rx_bar : std_logic;

	type receiver_state_type is (reset, waiting);
	signal state : receiver_state_type;
	
	signal inaddr32 : 			std_logic_vector(31 downto 0);
	signal memaddr  : 			std_logic_vector(13 downto 0);
	
	
	-- Decoding PCIe TLP headers
	signal fmt : 		std_logic_vector(1 downto 0);
	signal ptype :    std_logic_vector(4 downto 0);
	signal tc :			std_logic_vector(2 downto 0);
	signal td :			std_logic;
	signal ep :			std_logic;
	signal attr :		std_logic_vector(1 downto 0);
	signal plength :	std_logic_vector(9 downto 0);


begin

    rx_st_data0 <= i_rx_st_data0;
    rx_st_sop0 <= i_rx_st_sop0;
    rx_bar <= i_rx_bar;

    process(refclk, local_rstn)
    begin
    if ( local_rstn = '0' ) then
        o_rx_st_ready0 <= '0';
    elsif rising_edge(refclk) then
        if ( state = reset ) then
            o_rx_st_ready0 <= '0';
        else
            o_rx_st_ready0 <= '1';
        end if;
    end if;
    end process;

	-- Endian chasing for addresses
	inaddr32 <= rx_st_data0(95 downto 66) & "00";
	memaddr	<= inaddr32(17 downto 4);
	
	
	-- decode TLP
	fmt		<=		rx_st_data0(30 downto 29);
	ptype		<=		rx_st_data0(28 downto 24);
	tc			<= 	rx_st_data0(22 downto 20);
	td			<= 	rx_st_data0(15);
	ep			<=		rx_st_data0(14);
	attr		<=		rx_st_data0(13 downto 12);
	plength	<= 	rx_st_data0(9 downto 8) & rx_st_data0(7 downto 0);

	
	process(local_rstn, refclk)
	
	begin	
	
	if(local_rstn = '0') then
		state 			<= reset;

	
	elsif (refclk'event and refclk = '1') then
		readen	<= '0';
		case state is
			when reset =>
				state 			<= waiting;
	-------------------------------------------------------------------------------------			
			when waiting =>
				if(rx_st_sop0 = '1' and rx_bar = '1') then
					if(fmt = "00" and ptype = "00000") then -- 32 bit memory read request 	
						readaddr		<= memaddr & inaddr32(3 downto 2);
						readlength 	<= plength;
						header2		<= rx_st_data0(63 downto 32);
						readen		<= '1';
						state			<= waiting;
					end if; -- 32 bit write/read request
				end if; -- if Start of Packet
	----------------------------------------------------------------------------------
		end case;	
	end if; -- if clk event
	end process;

end architecture;
