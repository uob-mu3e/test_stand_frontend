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
component arb_selection_framecollect is
	generic(
		NPORTS : natural :=4 -- number of requesting ports
		);
	port(
			i_clk	: in std_logic;
			i_rst	: in std_logic;
			-- REQUEST SIGNALS
			i_req	: in std_logic_vector(NPORTS - 1 downto 0);
			-- GRANT VALID FLAG
			o_gnt_valid : out std_logic;
			i_ack 	: in std_logic;
			-- GRANT SELECTION OUTPUT
			o_gnt: out natural range NPORTS- 1 downto 0
	    );
end component; -- arb_selection_framecollect;


	signal l_all_header	: std_logic;
	signal l_all_trailer	: std_logic;
	signal l_frameid_nonsync: std_logic;					--combining header, frame numbers do not match

	signal s_is_valid, n_is_valid	: std_logic_vector(N_INPUTS-1 downto 0); --data at source input is valid
	signal l_request		: std_logic_vector(N_INPUTS-1 downto 0); --data at source input is valid and hit data
	signal src_read			: std_logic_vector(N_INPUTS-1 downto 0); --read signal to sources

	signal s_selection		: integer range 0 to N_INPUTS+1;	--selected data
	constant MUX_SELECT_HEAD	: integer := N_INPUTS;
	constant MUX_SELECT_TRAIL	: integer := N_INPUTS+1;
	signal s_selection_valid	: std_logic;
	signal s_selection_ack		: std_logic;
	signal s_sink_wr		: std_logic;
	signal l_common_data	: std_logic_vector(55 downto 0); --select first non-masked data input for retreiving header and trailer information
	signal s_Tpart,n_Tpart		: std_logic;
begin
--output assignments
o_source_rd <= src_read;
o_sink_wr <= s_sink_wr;

--source data inspection
gen_combinatorics: process(i_source_data, s_is_valid,i_SC_mask)
--variable v_is_header  : std_logic_vector(N_INPUTS-1 downto 0):=(others=>'0');
--variable v_is_trailer : std_logic_vector(N_INPUTS-1 downto 0):=(others=>'0');
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

	--find a candidate for common frame delimiter data
	l_common_data <=i_source_data(0);
	for i in 1 to N_INPUTS-1 loop
		if(i_SC_mask(i)='0') then l_common_data<= i_source_data(i); end if;
	end loop;
end process;


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

--store short event flag sent in header to correct data later

--definition of arbitration procedure
--roundrobin for hits, common-header wins over hits, common-trailer wins over header
selector: arb_selection_framecollect
	generic map ( NPORTS => N_INPUTS+2)
	port map(
		i_clk	=> i_coreclk,
		i_rst	=> i_rst,
		i_req	=> l_all_trailer & l_all_header & l_request, 
		o_gnt_valid => s_selection_valid,
		i_ack 	=> s_selection_ack,
		o_gnt => s_selection
	);




p_async: process(i_source_empty, s_is_valid, s_selection_valid, s_selection,i_sink_full,s_Tpart)
variable v_read : std_logic_vector(N_INPUTS+1 downto 0);
begin
	-- SINK FIFO WRITING
	-- ARBITRATION ACKNOWLEDGE
	v_read := (others => '0');
	s_sink_wr<='0';
	s_selection_ack <= '0';
	n_Tpart<='1';
	if(i_sink_full='0' and s_selection_valid='1') then
		s_selection_ack <= '1';
		v_read(s_selection) := '1';
		if(s_selection >= N_INPUTS) then --reading trailer or header, advance all inputs
			v_read:= (others=>'1');
		elsif(i_source_data(s_selection)(48)='0') then -- when long (=0), write e-part before advancing
			if(s_Tpart='1') then
				s_selection_ack<='0';
				n_Tpart<= '0';
			end if;
		end if;
		s_sink_wr<='1';
	end if;
	if(i_sink_full='0' and s_Tpart='0') then
		s_sink_wr<='1';
	end if;

	-- SOURCE FIFO READING
	n_is_valid <= s_is_valid;
	src_read <= (others =>'0');
	for i in N_INPUTS-1 downto 0 loop
		--issue read from fifo when appropriate
		--read next when selected
		--read next when available and not valid
		if( (s_is_valid(i)='0' or v_read(i)='1') and i_source_empty(i)='0') then
			src_read(i)<= '1';
			n_is_valid(i) <= '1';
		elsif(v_read(i)='1') then
			n_is_valid(i)<='0';
		end if;
	end loop;
end process;

p_sync: process(i_coreclk)
begin
	if rising_edge(i_coreclk) then
		if(i_rst='1') then
			s_is_valid<=(others => '0');
			o_sync_error<='0';
			s_Tpart<='0';
		else
			s_is_valid<=n_is_valid;
			s_Tpart<=n_Tpart;
			if(s_selection=MUX_SELECT_HEAD) then
				o_sync_error<=l_frameid_nonsync;
			end if;
		end if;
	end if;
end process;


--mux data assignment and definition, definition of data format (header, trailer, hit data)
def_mux : process (i_source_data, s_selection,l_common_data,l_frameid_nonsync,s_Tpart)
variable v_chnum : std_logic_vector (N_INPUTID_BITS-1 downto 0);
variable v_any_crc_err : std_logic;
variable v_any_asic_overflow : std_logic;
begin
	v_chnum := std_logic_vector(to_unsigned(s_selection,v_chnum'length));
	--"ERROR" flags : valid during trailer
	v_any_crc_err := '0';
	v_any_asic_overflow := '0';
	for i in N_INPUTS-1 downto 0 loop
		if(i_source_data(i)(16)='1') then v_any_crc_err := '1'; end if;
		if(i_source_data(i)(17)='1') then v_any_asic_overflow := '1'; end if;
	end loop;

	if(s_selection=MUX_SELECT_HEAD) then --select global header
		o_sink_data(33 downto 32) <= "10"; --identifier
		o_sink_data(31 downto 17) <= (others=>'0'); --filler
		o_sink_data(16) <= l_frameid_nonsync;		--frameID nonsync
		o_sink_data(15 downto 0) <=l_common_data(15 downto 0);  --frameID

	elsif(s_selection=MUX_SELECT_TRAIL) then --select global trailer
		o_sink_data(33 downto 32) <= "11"; --identifier
		o_sink_data(31 downto 2) <= (others=>'0'); --filler
		o_sink_data(1) <= v_any_asic_overflow;  --asic fifo overflow flag
		o_sink_data(0) <= v_any_crc_err; --crc error flag
	else
		--data common part
		o_sink_data(33 downto 32) <= "00"; --identifier: data T part
		o_sink_data(31 downto 28) <= v_chnum; -- asic number
		o_sink_data(27)		  <= s_Tpart; -- type (0=TPART, 1=EPART)
		o_sink_data(26 downto 22)  <= i_source_data(s_selection)(47 downto 43); --event data: chnum
		--data specific parts
		if(s_Tpart='1') then --select hit data, appending input ID after identifier
			o_sink_data(21 downto 0)  <= i_source_data(s_selection)(42 downto 21); --T event data: ttime,eflag
		else
			o_sink_data(21 downto 0)  <= i_source_data(s_selection)(20 downto 0) & i_source_data(s_selection)(21); --E event data: etime,eflag(redun)
		end if;
	end if;
end process;

end architecture;
