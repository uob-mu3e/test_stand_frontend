---------------------------------------
--
-- Multiplexer for data from different asics, to be connected after channel buffer fifo (mutrig_store/datachannel)
-- Fair round robin arbitration for hit data, frame headers and trailers are combined and replaced by a global header and trailer
-- Konrad Briggl May 2019
-- 
-- konrad.briggl@unige.ch
--
----------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;

use work.mutrig_constants.all;

entity framebuilder_mux is
generic(
	N_INPUTS : integer;
	N_INPUTID_BITS : integer
);
port (
	i_coreclk        : in  std_logic;                                     -- system clock
	i_rst            : in  std_logic;                                     -- reset, active low
--global timestamp
	i_timestamp_clk  : in  std_logic;	--125M timestamp clock
	i_timestamp_rst  : in  std_logic;	--timestamp reset, synced to i_timestamp_clk, high active
--event data inputs interface
	i_source_data	 : in mutrig_evtdata_array_t(N_INPUTS-1 downto 0);
	i_source_empty   : in std_logic_vector(N_INPUTS-1 downto 0);
	o_source_rd   	 : out std_logic_vector(N_INPUTS-1 downto 0);

--event data output interface to big buffer storage
	o_sink_data	 : out std_logic_vector(33 downto 0);		      -- event data output, asic number appended
	i_sink_full      :  in std_logic;
	o_sink_wr   	 : out std_logic;
--monitoring, write-when-fill is prevented internally
	o_sync_error     : out std_logic;
	i_SC_mask	 : in std_logic_vector(N_INPUTS-1 downto 0)		-- allow missing header or tailer from masked asic, block read requests from this 
);
end framebuilder_mux;

architecture impl of framebuilder_mux is
	--integer to one hot encoding
	function to_onehot(a:integer ; n:integer) return std_logic_vector is
	variable res: std_logic_vector(n-1 downto 0):=(others =>'0');
	begin
		res(a):='1';
		return res;
	end function;

	--one-hot encoded priority selection output based on request vector
	function Priority_Select(requests: std_logic_vector) return std_logic_vector is
	variable res : std_logic_vector(requests'range):=(others =>'0');
	begin
		--build priority selected result from input vector
		for i in requests'range loop
			if(requests(i)='1') then
				res:=to_onehot(i,requests'length);
				return res;
			end if;
		end loop;
		return res;
	end function;

	--very basic round robin encoded priority selection based on request vector and one-hot encoded previous selection 
	function RR_Select(requests: std_logic_vector ; token:std_logic_vector) return std_logic_vector is
	begin
		--build request vector for previous state, equivalent to Priority_res(token)
		return Priority_Select(requests and not token);
	end function;

--intput-data based combinatorics
	signal l_all_header	: std_logic;
	signal l_all_trailer	: std_logic;
	signal l_frameid_nonsync: std_logic;					--combining header, frame numbers do not match
	signal l_any_crc_err : std_logic;
	signal l_any_asic_overflow : std_logic;
	signal l_common_data	: std_logic_vector(55 downto 0); --select first non-masked data input for retreiving header and trailer information
--header data
	signal s_global_timestamp	: std_logic_vector(47 downto 0);

--input data reading
	signal s_is_valid, n_is_valid	: std_logic_vector(N_INPUTS-1 downto 0); --data at source input is valid
	signal l_request		: std_logic_vector(N_INPUTS-1 downto 0); --data at source input is valid and hit data
	signal s_read			: std_logic_vector(N_INPUTS-1 downto 0);
--selection and state machine
	--one-hot grant
	signal s_sel_gnt,n_sel_gnt	: std_logic_vector(N_INPUTS-1 downto 0);
	--selected data after mux (wide)
	signal s_sel_data		: std_logic_vector(55 downto 0);
	signal s_chnum			: std_logic_vector(N_INPUTID_BITS-1 downto 0);
	--selection of narrow data based on wide
	signal s_Tpart,n_Tpart		: std_logic;
	signal s_Hpart,n_Hpart		: std_logic;

	--fsm
	type fsm_state_t is (fs_idle,fs_headerH,fs_headerL,fs_hitH,fs_hitL,fs_trailer);
	signal s_state, n_state		: fsm_state_t;
--fifo writing
	signal s_sink_wr,n_sink_wr	: std_logic;
begin
--output assignments
o_sink_wr <= s_sink_wr;

--global timestamp generation
p_gen_timestamp: process(i_timestamp_clk)
begin
	if rising_edge(i_timestamp_clk) then
		if(i_timestamp_rst='1') then
			s_global_timestamp<=(others =>'0');	
		else
			s_global_timestamp<=std_logic_vector(unsigned(s_global_timestamp)+1);
		end if;
	end if;
end process;

--source data inspection: all trailer,all header,hit requests.
---> define signals l_all_header, l_all_trailer, l_request, common data (l_common_data, l_any_crc_err, l_any_asic_overflow) 
gen_combinatorics: process(i_source_data, s_is_valid,i_SC_mask)
begin
	l_all_header <='1';
	l_all_trailer<='1';
	l_request<=(others => '1');
	--incoming data is header or trailer?
	for i in N_INPUTS-1 downto 0 loop
		if (s_is_valid(i) = '0' and i_SC_mask(i)='0') then -- no valid data, no request, not all header or trailer
			l_all_header<='0';
			l_all_trailer<='0';
			l_request(i)<='0';
		else
			if (i_source_data(i)(51 downto 50)="10" or i_SC_mask(i)='1') then -- data is header
				l_request(i)<='0';	--do not request readout of header
			else 
				l_all_header<='0';
			end if;
			if (i_source_data(i)(51 downto 50)="11" or i_SC_mask(i)='1') then -- data is trailer
				l_request(i)<='0';	--do not request readout of trailer
			else
				l_all_trailer<='0';
			end if;
		end if;
	end loop;

	--common data: find a candidate for common frame delimiter data (frameID)
	--TODO: separate selection (may be slow based on flag, even false_path) and multiplexing (synchronous)
	l_common_data <=i_source_data(0);
	for i in 1 to N_INPUTS-1 loop
		if(i_SC_mask(i)='0') then l_common_data<= i_source_data(i); end if;
	end loop;
	--common data: "ERROR" flags : valid during trailer
	l_any_crc_err <= '0';
	l_any_asic_overflow <= '0';
	for i in N_INPUTS-1 downto 0 loop
		if(i_source_data(i)(16)='1') then l_any_crc_err <= '1'; end if;
		if(i_source_data(i)(17)='1') then l_any_asic_overflow <= '1'; end if;
	end loop;

end process;

--source data consistency_check (frame ID)
consistency_check : process (i_source_data)
variable frameid_nonsync : std_logic;
begin
	--check if all frameIDs match
	frameid_nonsync:='1';
	for i in N_INPUTS-2 downto 0 loop
		if(i_source_data(i+1)(15 downto 0) /= i_source_data(i)(15 downto 0)) then frameid_nonsync:='0'; end if;
	end loop;
	l_frameid_nonsync<=frameid_nonsync;
end process;
o_sync_error<=l_frameid_nonsync and s_Hpart; --show when valid, can be used for counting


--mux data from inputs and one-hot encoded grant vector: simple or of inputs and-ed with grant
def_mux_sel : process (i_source_data,s_sel_gnt)
begin
	s_sel_data<=(others =>'0');
	for i in N_INPUTS-1 downto 0 loop
		if(s_sel_gnt(i)='1') then
			s_sel_data<=i_source_data(i);
			s_chnum<= std_logic_vector(to_unsigned(i,s_chnum'length));
		end if;
	end loop;
end process;
------------------------------------------------------------------

p_read_fifos: process(i_source_empty, s_is_valid, s_read)
begin
	-- SOURCE FIFO READING
	n_is_valid <= s_is_valid;
	o_source_rd<= (others =>'0');
	for i in N_INPUTS-1 downto 0 loop
		--issue read from fifo when appropriate
		--read next when selected
		--read next when available and not valid
		if( (s_is_valid(i)='0' or s_read(i)='1') and i_source_empty(i)='0') then
			o_source_rd(i)<= '1';
			n_is_valid(i) <= '1';
		elsif(s_read(i)='1') then
			n_is_valid(i)<='0';
		end if;
	end loop;
end process;


------------------------------------------------------------------
p_fsm_async: process(s_state,l_all_header,l_all_trailer,l_request,s_sel_data, s_sel_gnt)
begin
	n_state <= s_state;
	n_sel_gnt <= s_sel_gnt;
	s_sink_wr <= '0';
	n_Hpart <= '0';
	s_read <= (others =>'0');
	n_Tpart <= '0';
	case s_state is
		when fs_idle =>
			--wait for request -- TODO: move next selection to common part to speed up process
			if(l_all_header) then
				n_state <= fs_headerH;
			elsif(l_all_trailer) then
				n_state <= fs_trailer;
			elsif(unsigned(l_request) /= 0) then
				n_sel_gnt <= Priority_Select(l_request);
				n_state <= fs_hitH;
			end if;
		when fs_headerH =>
			s_sink_wr <= '1';
			n_Hpart <= '1';
			n_state <= fs_headerL;
		when fs_headerL =>
			s_read <= (others =>'1');
			s_sink_wr <= '1';
			n_state <= fs_idle; -- TODO: select next already here
		when fs_trailer =>
			s_read <= (others =>'1');
			s_sink_wr <= '1';
			n_state <= fs_idle; -- TODO: select next already here
		when fs_hitH =>
			if(s_sel_data(48)='0') then --long event, continue with writing E-part
				s_sink_wr <= '1';
				n_state <= fs_hitL;
				n_Tpart <= '1';
			else
				s_sink_wr <= '1';
				s_read <= s_sel_gnt; 
				n_state <= fs_idle; -- TODO: select next already here
			end if;
		when fs_hitL =>
			s_sink_wr <= '1';
			s_read <= s_sel_gnt; 
			n_state <= fs_idle; -- TODO: select next already here
	end case;
end process;

p_sync: process(i_coreclk)
begin
	if rising_edge(i_coreclk) then
		if(i_rst='1') then
			s_is_valid<=(others => '0');
			s_state<=fs_idle;
			s_sel_gnt<=(others => '0');
			--s_sink_wr <='0';
			s_Tpart<='0';
			s_Hpart<='0';
		else
			s_is_valid<=n_is_valid;
			s_state<=n_state;
			s_sel_gnt<=n_sel_gnt;
--			s_sink_wr<=n_sink_wr;
			s_Hpart<=n_Hpart;
			s_Tpart<=n_Tpart;
--			s_read<=n_read;
		end if;
	end if;
end process;

--mux data assignment and definition, definition of data format (headerH,headerL, hitdataH, hitdataL, trailer)
--hitdataH (first part) is selected when  l_all_*='0' and s_Tpart='0'
--hitdataL (second part) is selected when l_all_*='0' and s_Tpart='1'
--headerdataH (first part) is selected when l_all_header='1' and  s_Hpart='0'
--headerdataL (second part) is selected when l_all_header='1' and  s_Hpart='1'
--trailerdata  is selected when l_all_trailer='1'
def_mux_out : process (s_sel_data, s_chnum, l_common_data,s_global_timestamp,l_frameid_nonsync,l_any_crc_err,l_any_asic_overflow, l_all_header,l_all_trailer, s_Tpart,s_Hpart)
begin

	if(l_all_trailer) then --select global trailer
		o_sink_data(33 downto 32) <= "11"; --identifier
		o_sink_data(31 downto 2) <= (others=>'0'); --filler
		o_sink_data(1) <= l_any_asic_overflow;  --asic fifo overflow flag
		o_sink_data(0) <= l_any_crc_err; --crc error flag
	elsif(l_all_header) then --select global header
		if (s_Hpart='0') then -- first header payload word
			o_sink_data(33 downto 32) <= "10"; --identifier (type header)
			o_sink_data(31 downto 0) <= s_global_timestamp(47 downto 16); --global timestamp 
		else 
			o_sink_data(33 downto 32) <= "00"; --identifier (is a payload : type data)
			o_sink_data(31 downto 16) <= s_global_timestamp(15 downto 0); --global timestamp 
			o_sink_data(15) <= l_frameid_nonsync;		--frameID nonsync
			o_sink_data(14 downto 0) <=l_common_data(14 downto 0);  --frameID
		end if;
	else --select data
		--data common part
		o_sink_data(33 downto 32) <= "00"; --identifier: data T part
		o_sink_data(31 downto 28) <= s_chnum; -- asic number
		o_sink_data(27)		  <= s_Tpart; -- type (0=TPART, 1=EPART)
		o_sink_data(26 downto 22)  <= s_sel_data(47 downto 43); --event data: chnum
		--data specific parts
		if(s_Tpart='0') then --select hit data, appending input ID after identifier
			o_sink_data(21 downto 0)  <= s_sel_data(42 downto 21); --T event data: ttime,eflag
		else
			o_sink_data(21 downto 0)  <= s_sel_data(20 downto 0) & s_sel_data(21); --E event data: etime,eflag(redun)
		end if;
	end if;
end process;

end architecture;
