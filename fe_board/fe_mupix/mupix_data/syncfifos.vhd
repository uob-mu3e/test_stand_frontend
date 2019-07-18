--- Synchronize data paths via a FIFO

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.datapath_components.all;
use work.mupix_types.all;


entity syncfifos is
generic(
	NCHIPS : integer := 4
);
port (
	ready:					in std_logic_vector(NCHIPS-1 downto 0); 
	clkout:					in std_logic; -- clock for outputs
	reset_n:					in std_logic;
	
	clkin:					in std_logic_vector(NCHIPS-1 downto 0);
	datain:					in std_logic_vector(8*NCHIPS-1 downto 0);
	kin:						in std_logic_vector(NCHIPS-1 downto 0);
	
	dataout:					out std_logic_vector(8*NCHIPS-1 downto 0);
	kout:						out std_logic_vector(NCHIPS-1 downto 0);
	
	data_valid:				out std_logic_vector(NCHIPS-1 downto 0)
--	fifo_underflow:		out chips_reg32;
--	fifo_overflow:			out chips_reg32;
--	fifo_rdusedw_out:		out reg32
);
end syncfifos;

architecture rtl of syncfifos is

signal syncfifoin 			: std_logic_vector(NCHIPS*9-1 downto 0);
signal syncfifoout 			: std_logic_vector(NCHIPS*9-1 downto 0);
signal syncfifo_wrena 		: std_logic_vector(NCHIPS-1 downto 0);

signal fifordusedw			: fifo_usedw_array;
signal fifowrusedw			: fifo_usedw_array;

signal fifo_rdreq 			: std_logic_vector(NCHIPS-1 downto 0);
--signal fifo_underflow_reg	: chips_reg32;
--signal fifo_overflow_reg	: chips_reg32;

--signal init_fifo 	: std_logic_vector(NCHIPS-1 downto 0);
signal fifo_aclr	: std_logic;

begin

fifo_aclr	<= not reset_n;

--fifo_underflow	<= fifo_underflow_reg;
--fifo_overflow	<= fifo_overflow_reg;

gensyncfifos:
FOR i in NCHIPS-1 downto 0 generate

		write_process : process(clkin(i), reset_n)
		begin
			if(reset_n = '0')then
				syncfifo_wrena(i) 					<= '0';
				syncfifoin(i*9+8 downto i*9)		<= (others => '0');
--				fifo_overflow_reg(i)					<= (others => '0');
			elsif(rising_edge(clkin(i)))then
				syncfifoin(i*9+8 downto i*9)		<= kin(i) & datain(i*8+7 downto i*8);			
				if(fifowrusedw(i) < "1110")then
					syncfifo_wrena(i) 					<= ready(i);
				else
					syncfifo_wrena(i) 					<= '0';	
--					fifo_overflow_reg(i)					<= fifo_overflow_reg(i) + 1;					
				end if;
			end if;
		end process write_process; 


sf:syncfifo
	PORT MAP
	(
		aclr		=> fifo_aclr,	
		data		=> syncfifoin(i*9+8 downto i*9),
		rdclk		=> clkout,
		rdreq		=> fifo_rdreq(i),
		wrclk		=> clkin(i),
		wrreq		=> syncfifo_wrena(i),
		q			=> syncfifoout(i*9+8 downto i*9),
		rdusedw	=> fifordusedw(i),
		wrusedw	=> fifowrusedw(i)
	);

		read_process : process(clkout, reset_n)
		begin
			if(reset_n = '0')then
				fifo_rdreq(i)				<= '0';
			--	fifo_underflow_reg(i)	<= (others => '0');
				data_valid(i)				<= '0';
--				fifo_rdusedw_out(8*(i+1)-1 downto 8*i)			<= (others => '0');
			elsif(rising_edge(clkout))then
			
				if(fifo_rdreq(i) = '1')then
					data_valid(i)	<= '1';
				else
					data_valid(i)	<= '0';			
				end if;

				if(fifordusedw(i) > "0011")then
					fifo_rdreq(i)	<= '1';
				elsif(fifordusedw(i) < "0010")then
					fifo_rdreq(i) 	<= '0';
			--		fifo_underflow_reg(i)	<= fifo_underflow_reg(i) + 1;
				end if;
						
--				fifo_rdusedw_out(8*(i+1)-5 downto 8*i) <= fifordusedw(i);
--				fifo_rdusedw_out(8*(i+1)-4)				<= fifo_rdreq(i);			
--				fifo_rdusedw_out(8*(i+1)-3)				<= '0';
--				fifo_rdusedw_out(8*(i+1)-2)				<= '0';	
--				fifo_rdusedw_out(8*(i+1)-1)				<= '0';				
			end if;
		end process read_process;
	
kout(i) 							<= syncfifoout(i*9+8);
dataout(i*8+7 downto i*8)	<= syncfifoout(i*9+7 downto i*9);	
	
	
END GENERATE;



end rtl;
