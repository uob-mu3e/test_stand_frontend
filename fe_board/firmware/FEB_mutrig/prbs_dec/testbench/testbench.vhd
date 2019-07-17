library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use ieee.math_real.all; -- for UNIFORM, TRUNC
use ieee.numeric_std.all; -- for TO_UNSIGNED
use ieee.std_logic_textio.all;	-- for write std_logic_vector to line
library std;
use std.textio.all;             --FOR LOGFILE WRITING

--library mutrig_sim;
--use mutrig_sim.txt_util.all;
--library modelsim_lib;
--use modelsim_lib.util.all;

--use work.daq_constants.all;
--use work.mutrig_constants.all;

--library mutrig_sim;
--use mutrig_sim.txt_util.all;
--use mutrig_sim.datapath_types.all;
--use mutrig_sim.datapath_helpers.all;

entity testbench is
end testbench; 

architecture RTL of testbench is
--dut definition
component prbs_decoder is
port (
--system
	i_coreclk	: in  std_logic;
	i_rst		: in  std_logic;
--data stream input
	i_data	: in std_logic_vector(63 downto 0);
	i_valid	: in std_logic;
--data stream output
	o_data	: out std_logic_vector(63 downto 0);
	o_valid	: out std_logic;
--disable block (make transparent)
	i_SC_disable_dec : in std_logic
);
end component; --prbs_decoder;


-- DUT signals --{{{
type t_stimulus_vec	is array (natural range <>) of std_logic_vector(63 downto 0);
signal s_stimulus	: t_stimulus_vec(1 to 100):=(X"4000F00D00010000",X"8002F8009095DB6B",X"81020355169C6979",X"80020AE165D136A3",X"800203E5EFD4B815",X"0000BEEF00010000",X"4000F00D00018000",X"8003FB7F36DFE513",X"81030375B6D3F867",X"80030982F91283DB",X"810303BD835D9B7B",X"80030025D15544C5",X"8003FA96A51CCF79",X"80030BD3F79157C9",X"800300D10954F549",X"0000BEEF00018000",X"4000F00D00020000",X"8004FAF7C5DE167D",X"8104032D9E5DC07B",X"80040B053952B391",X"8104035CD8159F15",X"800403A3E8C99191",X"8004FB5B5E8C0995",X"80040BB1211D4E3B",X"800403CD179CDFF9",X"0000BEEF00020000",X"4000F00D00028000",X"8005FA440C919B47",X"810502DDB59B68F7",X"800508AB209A22F5",X"810503DDF78D3CD3",X"800503A8EF4E1E05",X"8005FB21DE561AED",X"80050846721E82FD",X"800503B2889D0FD7",X"0000BEEF00028000",X"4000F00D00030000",X"8006FABFE558659F",X"810602CA62D1CAA3",X"80060A335C40F453",X"8106027CF4C2945F",X"8006038407002C4F",X"8006FBA39F47811B",X"80060A2B645B3E53",X"800603520E802B0F",X"0000BEEF00030000",X"4000F00D00038000",X"8007F89EF1D227D7",X"810702D5A586E909",X"80070B48AE838195",X"8107035721536517",X"80070320CA90A361",X"8007FB314658D4B1",X"80070B5A32D0EF41",X"800703E5FFC056DB",X"0000BEEF00038000",X"4000F00D00040000",X"8008FB84C2DC9A89",X"810802D3EBD1DF5F",X"80080AB63550D821",X"810803A0B896A92D",X"8008023884494B8D",X"8008FA72A4DC4979",X"80080BC9D20E234B",X"8008032A3E489A95",X"0000BEEF00040000",X"4000F00D00048000",X"8009FAF348D838F1",X"8109027E5AD20CA5",X"800908A7A0116963",X"810900B828D4A8E9",X"80090327765795AF",X"8009FA91F51DA643",X"80090BA1D75F15BF",X"800900FDA38B9FD7",X"0000BEEF00048000",X"4000F00D00050000",X"800AF995E3D7732F",X"810A02798356160B",X"800A0AE8FB9C9439",X"810A02C42398618B",X"800A021E041C145D",X"810A02F3ADD801B1",X"800AFA78B4DE1B7D",X"800A0B6E06DB7ECD",X"800A0339285F6FBF",X"800AFBEA5FCF870F",X"0000BEEF00050000",X"4000F00D00058000",X"800B0A0351903A21",X"810B001ED208D493",X"800B02BE1B9F75BF",X"810B02BAFD47EFC7",X"800BFB2A3641B79B",X"800B0BFED7D43169",X"800B02280C58AC71",X"800BFA7D08DF2A4D",X"800B0A28C4580707",X"800B03D1778F41CD",X"0000BEEF00058000");
signal i_data	  	: std_logic_vector(63 downto 0):=(others =>'0');
signal i_valid	: std_logic:='0';
signal o_data	  	: std_logic_vector(63 downto 0);
signal o_valid	: std_logic;

--system signals
signal i_rst		: std_logic:='0';
signal i_coreclk  	: std_logic:='0';

begin

-- basic stimulus for receiver
i_coreclk	<= not i_coreclk after  4 ns;	-- 125 MHz system core clock
i_rst		<= '1' after 20 ns, '0' after 200 ns;	-- Basic reset

--dut
dut: prbs_decoder
	port map (
		i_coreclk	=> i_coreclk,
		i_rst		=> i_rst,
    		i_data		=> i_data,
    		i_valid		=> i_valid,
    		o_data		=> o_data,
    		o_valid		=> o_valid,
		i_SC_disable_dec=> '0'
	);

stim: process
begin
	wait for 1 us;
	i_valid <='1';
	wait until rising_edge(o_valid);
	wait until rising_edge(i_coreclk);
	i_valid<='0';
	wait until rising_edge(i_coreclk);
	for i in 1 to 100 loop
		i_valid<='1';
		i_data<=s_stimulus(i);
		wait until rising_edge(i_coreclk);
	end loop;
	i_valid<='0';
	wait;	
end process;


-- fifo readout logger
stimulus_logging: process (i_coreclk)
file log_file : TEXT open write_mode is "stimulus_data.txt";
variable l : line;
begin
	if rising_edge(i_rst) then
		-- write header
		write(l, string'("#STIMULUS DATA"));
		writeline(log_file, l);
		write(l, string'("#--------------------------------------------------"));
		writeline(log_file, l);
		
	elsif (rising_edge(i_coreclk) and i_valid='1' and i_data/=X"0000000000000000") then
		if(i_data(63 downto 62)="01") then
			write(log_file,"Frame Header "&
				HT & "RAW "& to_hstring(i_data) & 
				HT & "FID="& "0x" & to_hstring(i_data(30 downto 15)) & LF);
		elsif(i_data(63 downto 62)="00") then
			write(log_file,"Frame Trailer "&
				HT & "RAW "& to_hstring(i_data) & 
					HT & "FID="& "0x" & to_hstring(i_data(30 downto 15)) & LF);
		elsif(i_data(63 downto 62)="10") then
			write(log_file,"Hit data (L) "& 
				HT & "RAW "& to_hstring(i_data) & 
				HT & "ASIC "& natural'image(to_integer(unsigned(i_data(59 downto 56)))) & 
				HT & "  CH "& natural'image(to_integer(unsigned(i_data(47 downto 43)))) & 
				HT & " TBH "& to_hstring(i_data(42 downto 42)) & 
				HT & " TCC "& to_hstring(i_data(41 downto 27)) & 
				HT & " TFC "& to_hstring(i_data(26 downto 22)) & 
				HT & " EBH "& to_hstring(i_data(21 downto 21)) & 
				HT & " ECC "& to_hstring(i_data(20 downto 6)) & 
				HT & " EFC "& to_hstring(i_data(5 downto 1)) & 
				HT & "   E "& to_hstring(i_data(0 downto 0)) & 
				LF);
		end if;
		flush(log_file);
	end if;
end process;

-- fifo readout logger
decoder_logging: process (i_coreclk)
file log_file : TEXT open write_mode is "decoder_data.txt";
variable l : line;
begin
	if rising_edge(i_rst) then
		-- write header
		write(l, string'("#DECODED DATA"));
		writeline(log_file, l);
		write(l, string'("#--------------------------------------------------"));
		writeline(log_file, l);
		
	elsif (rising_edge(i_coreclk) and o_valid='1') then
		if(o_data(63 downto 62)="01") then
			write(log_file,"Frame Header "&
				HT & "RAW "& to_hstring(o_data) & 
				HT & "FID="& "0x" & to_hstring(o_data(30 downto 15)) & LF);
		elsif(o_data(63 downto 62)="00") then
			write(log_file,"Frame Trailer "&
				HT & "RAW "& to_hstring(o_data) & 
					HT & "FID="& "0x" & to_hstring(o_data(30 downto 15)) & LF);
		elsif(o_data(63 downto 62)="10") then
			write(log_file,"Hit data (L) "& 
				HT & "RAW "& to_hstring(o_data) & 
				HT & "ASIC "& natural'image(to_integer(unsigned(o_data(59 downto 56)))) & 
				HT & "  CH "& natural'image(to_integer(unsigned(o_data(47 downto 43)))) & 
				HT & " TBH "& to_hstring(o_data(42 downto 42)) & 
				HT & " TCC "& to_hstring(o_data(41 downto 27)) & 
				HT & " TFC "& to_hstring(o_data(26 downto 22)) & 
				HT & " EBH "& to_hstring(o_data(21 downto 21)) & 
				HT & " ECC "& to_hstring(o_data(20 downto 6)) & 
				HT & " EFC "& to_hstring(o_data(5 downto 1)) & 
				HT & "   E "& to_hstring(o_data(0 downto 0)) & 
				LF);
		end if;
		flush(log_file);
	end if;
end process;

end architecture;
