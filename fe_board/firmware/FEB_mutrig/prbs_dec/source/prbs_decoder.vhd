---------------------------------------
--
-- On the fly PRBS decoder for mutrig/stic event data
-- Latency two CC.
-- data (or more precisely data_valid flag) is only passed after decoder memory has been initialized, 
-- Initialization takes ~32k CC.
-- Frame header and trailer tags are passed as is, word after header (supposed to be second part of header payload) is passed as is
-- Konrad Briggl May 2019
-- 
-- konrad.briggl@unige.ch
--
----------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
LIBRARY altera_mf;
USE altera_mf.altera_mf_components.all;

entity prbs_decoder is
port (
--system
	i_coreclk	: in  std_logic;
	i_rst		: in  std_logic;
	o_initializing  : out std_logic;
--data stream input
	i_A_data		: in std_logic_vector(33 downto 0);
	i_A_valid		: in std_logic;
	i_B_data		: in std_logic_vector(33 downto 0);
	i_B_valid		: in std_logic;

--data stream output
	o_A_data		: out std_logic_vector(33 downto 0);
	o_A_valid		: out std_logic;
	o_B_data		: out std_logic_vector(33 downto 0);
	o_B_valid		: out std_logic;

--disable block (make transparent)
	i_SC_disable_dec : in std_logic
);
end prbs_decoder;

architecture impl of prbs_decoder is
	component decoder_mem
		PORT
		(
			address		: IN STD_LOGIC_VECTOR (14 DOWNTO 0);
			clock		: IN STD_LOGIC  := '1';
			data		: IN STD_LOGIC_VECTOR (14 DOWNTO 0);
			wren		: IN STD_LOGIC  := '0';
			q		: OUT STD_LOGIC_VECTOR (14 DOWNTO 0)
		);
	end component;
	signal s_init : std_logic; --state vector, '1': Block is initializing
	signal s_init_prbs : std_logic_vector(14 downto 0); --initialization vector(original data)
	signal s_init_dec,s_init_dec_d : std_logic_vector(14 downto 0); --initialization vector(decoded data)

	signal s_A_addr,n_A_addr : std_logic_vector(14 downto 0);    --r/w address on port a (init and T decoding)
	signal s_A_valid_2 : std_logic;
	signal s_A_valid_1 : std_logic;
	signal s_A_data_bypass_2  : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_A_data_bypass_1  : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_A_data_bypass_0    : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_A_data_dec : std_logic_vector(14 downto 0); -- decoded data E timestamp
	signal s_A_select_bypass_2 : std_logic;
	signal s_A_select_bypass_1 : std_logic;
	signal s_A_select_bypass_0 : std_logic;
	signal s_A_is_header_3 : std_logic;
	signal s_A_is_header_2 : std_logic;
	signal s_A_is_header_1 : std_logic;
	signal s_A_is_header_0 : std_logic;

	signal s_B_addr,n_B_addr : std_logic_vector(14 downto 0);    --r/w address on port a (init and T decoding)
	signal s_B_valid_2 : std_logic;
	signal s_B_valid_1 : std_logic;
	signal s_B_data_bypass_2  : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_B_data_bypass_1  : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_B_data_bypass_0    : std_logic_vector(33 downto 0); -- undecoded data bypass
	signal s_B_data_dec : std_logic_vector(14 downto 0); -- decoded data E timestamp
	signal s_B_select_bypass_2 : std_logic;
	signal s_B_select_bypass_1 : std_logic;
	signal s_B_select_bypass_0 : std_logic;
	signal s_B_is_header_3 : std_logic;
	signal s_B_is_header_2 : std_logic;
	signal s_B_is_header_1 : std_logic;
	signal s_B_is_header_0 : std_logic;
begin


--decoder memory
--u_mem: decoder_mem -- replaced by direct implementation 
u_mem : altsyncram
	GENERIC MAP (
		address_reg_b => "CLOCK0",
		clock_enable_input_a => "BYPASS",
		clock_enable_input_b => "BYPASS",
		clock_enable_output_a => "BYPASS",
		clock_enable_output_b => "BYPASS",
		indata_reg_b => "CLOCK0",
		intended_device_family => "Stratix IV",
		lpm_type => "altsyncram",
		numwords_a => 32768,
		numwords_b => 32768,
		operation_mode => "BIDIR_DUAL_PORT",
		outdata_aclr_a => "NONE",
		outdata_aclr_b => "NONE",
		outdata_reg_a => "CLOCK0",
		outdata_reg_b => "CLOCK0",
		power_up_uninitialized => "FALSE",
		read_during_write_mode_mixed_ports => "DONT_CARE",
		read_during_write_mode_port_a => "NEW_DATA_NO_NBE_READ",
		read_during_write_mode_port_b => "NEW_DATA_NO_NBE_READ",
		widthad_a => 15,
		widthad_b => 15,
		width_a => 15,
		width_b => 15,
		width_byteena_a => 1,
		width_byteena_b => 1,
		wrcontrol_wraddress_reg_b => "CLOCK0"
	)
	PORT MAP (
		address_a => s_A_addr,
		address_b => s_B_addr,
		clock0 => i_coreclk,
		data_a => s_init_dec_d,
		wren_a => s_init,
		q_a => s_A_data_dec,
		q_b => s_B_data_dec,
		data_b => (others => '-'),
		wren_b => '-'
	);



--decoder init and bypass registers
p_sync: process(i_coreclk)
begin
	if rising_edge(i_coreclk) then
		--ram address and data input, data delay is only to compensate address pipelining.
		s_A_addr <= n_A_addr;
		s_init_dec_d <= s_init_dec;
		--memory initialisation state machine, data bypass control
		if(i_rst='1') then
		--init part
			s_init_prbs <= (0=>'1', others=>'0');
			s_init_dec  <= (0=>'1', others=>'0');
			s_init <='1';

		--A_part
			s_A_valid_2<='0';
			s_A_valid_1<='0';
			o_A_valid<='0';

		--B_part
			s_B_valid_2<='0';
			s_B_valid_1<='0';
			o_B_valid<='0';
		else
		--init part
			s_init_prbs<=s_init_prbs(13 downto 0) & not (s_init_prbs(14) xor s_init_prbs(13));
			--s_init_prbs<=std_logic_vector(unsigned(s_init_prbs)+1);
			s_init_dec<=std_logic_vector(unsigned(s_init_dec)+1);
			if(s_init_dec=(0 to s_init_dec'length-1 =>'0')) then
				s_init<='0';
			end if;

		--A_part
			s_A_valid_2<= i_A_valid;
			s_A_valid_1<= s_A_valid_2;
			o_A_valid<=s_A_valid_1 and ((not s_init) or i_SC_disable_dec); --A or init
			s_A_data_bypass_2 <= i_A_data;
			s_A_data_bypass_1 <= s_A_data_bypass_2;
			s_A_data_bypass_0 <= s_A_data_bypass_1;

			--scan input data for header or trailer, to be bypassed
			s_A_select_bypass_2 <=i_A_data(33);
			if(i_A_data(33 downto 32)="10") then
				s_A_is_header_3 <='1';
			else
				s_A_is_header_3 <='0';
			end if;
			if(i_SC_disable_dec='1') then s_A_select_bypass_2 <= '1'; end if;
			s_A_select_bypass_1<= s_A_select_bypass_2;
			s_A_select_bypass_0<= s_A_select_bypass_1;
			s_A_is_header_2 <=s_A_is_header_3;
			s_A_is_header_1 <=s_A_is_header_2;
			s_A_is_header_0 <=s_A_is_header_1;

		--B_part
			s_B_valid_2<= i_B_valid;
			s_B_valid_1<= s_B_valid_2;
			o_B_valid<=s_B_valid_1 and ((not s_init) or i_SC_disable_dec); --A or init
			s_B_data_bypass_2 <= i_B_data;
			s_B_data_bypass_1 <= s_B_data_bypass_2;
			s_B_data_bypass_0 <= s_B_data_bypass_1;

			--scan input data for header or trailer, to be bypassed
			s_B_select_bypass_2 <=i_B_data(33);
			if(i_B_data(33 downto 32)="10") then
				s_B_is_header_3 <='1';
			else
				s_B_is_header_3 <='0';
			end if;
			if(i_SC_disable_dec='1') then s_B_select_bypass_2 <= '1'; end if;
			s_B_select_bypass_1<= s_B_select_bypass_2;
			s_B_select_bypass_0<= s_B_select_bypass_1;
			s_B_is_header_2 <=s_B_is_header_3;
			s_B_is_header_1 <=s_B_is_header_2;
			s_B_is_header_0 <=s_B_is_header_1;
		end if;
	end if;
end process;

--address assignments: PRBS data for TCC and ECC or initialization vector
--pipelined to improve timing budget
n_A_addr <= s_init_prbs when s_init='1' else i_A_data(20 downto 6);
n_B_addr <= i_B_data(20 downto 6);

--output assignment: replace TCC and ECC by their decoded counterparts when not bypassing
o_A_data<=s_A_data_bypass_0	when s_A_select_bypass_0='1' or s_A_is_header_0='1' else
	s_A_data_bypass_0(33 downto 21) & s_A_data_dec(14 downto 0) & s_A_data_bypass_0(5 downto 0);

o_B_data<=s_B_data_bypass_0	when s_B_select_bypass_0='1' or s_B_is_header_0='1' else
	s_B_data_bypass_0(33 downto 21) & s_B_data_dec(14 downto 0) & s_B_data_bypass_0(5 downto 0);

o_initializing <= s_init;
end architecture;

