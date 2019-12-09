-----------------------------------
-- Sebastian Dittmeier
----------------------------------



library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

entity fifo_Nb_sync is
	generic(
		NBITS: integer := 8
	);
port(
	clk_wr 	: in std_logic;
	clk_rd	: in std_logic;
	rst_n		: in std_logic;
	wrreq_in	: in std_logic;
	data_in	: in std_logic_vector(9 downto 0);
	data_out : out std_logic_vector(9 downto 0);
	data_valid: out std_logic
);
end fifo_Nb_sync;

architecture RTL of fifo_Nb_sync is

signal rdreq : std_logic;
signal wrreq : std_logic;
signal wrusedw : std_logic_vector(3 downto 0);
signal rdusedw: std_logic_vector(3 downto 0);
signal data	: std_logic_vector(9 downto 0);
signal q_out : std_logic_vector(9 downto 0);
signal fifo_aclr: std_logic;

begin
	
fifo_aclr	<= not rst_n;	
	
output_sync: work.fifo_Nb 
	generic map(
		NBITS	=> NBITS
	)
	port map(
		data		=> data,
		rdclk		=> clk_rd,
		rdreq		=> rdreq,
		wrclk		=> clk_wr,
		wrreq		=> wrreq,
		q			=> q_out,
		rdusedw	=> rdusedw,
		wrusedw	=> wrusedw,
		aclr		=> fifo_aclr
	);	
	
process(clk_wr, rst_n)	
begin
	if(rst_n = '0')then
		wrreq <= '0';
		data	<= (others => '0');
	elsif(rising_edge(clk_wr))then
		data	<=	data_in;		
		if(wrusedw < "1110")then	-- in order to not overflow this tiny buffer
			wrreq <= wrreq_in;
		else
			wrreq <= '0';
		end if;
	end if;
end process;

process(clk_rd, rst_n)	
begin
	if(rst_n = '0')then
		rdreq 		<= '0';
		data_out		<= (others => '0');	
		data_valid	<= '0';
	elsif(rising_edge(clk_rd))then
		if(rdusedw > "0011")then	-- in order to not underflow this tiny buffer
			rdreq <= '1';
		elsif(rdusedw < "0010")then
			rdreq <= '0';
		end if;
		
		data_out	<= q_out;
		
		if(rdreq = '1')then
			data_valid	<= '1';
		else
			data_valid	<= '0';			
		end if;
		
	end if;
end process;
	
end RTL;