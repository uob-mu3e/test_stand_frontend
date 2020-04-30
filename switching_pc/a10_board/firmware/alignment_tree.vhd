library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

entity alginment_tree is 
generic (
	NLINKS : integer := 32;
	NFIRST : integer := 4;
	NSECOND : integer := 2;
    LINK_FIFO_ADDR_WIDTH : integer := 10 --;
);
port (
	i_clk_250 	: in  std_logic;
	i_reset_n	: in  std_logic;

	i_data 		: in std_logic_vector(NLINKS * 32 - 1 downto 0);
	i_datak		: in std_logic_vector(NLINKS * 4 - 1 downto 0);
	
	o_data	    : out std_logic_vector(255 downto 0);
	o_datak	    : out std_logic_vector(32 downto 0);
	o_error		: out std_logic_vector(3 downto 0)--;
);
end entity;

architecture rtl of alginment_tree is
----------------signals--------------------
signal reset : std_logic;

-- link fifos
signal link_fifo_wren       : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_data       : std_logic_vector(NLINKS * 36 - 1 downto 0);
signal link_fifo_ren        : std_logic_vector(NLINKS - 1 downto 0);
type data_array_type is array (NLINKS - 1 downto 0) of std_logic_vector(35 downto 0);
signal link_fifo_data_out   : data_array_type;
signal link_fifo_empty      : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_full       : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_almost_full: std_logic_vector(NLINKS - 1 downto 0);

-- first layer fifos
type first_layer_type is (wait_for_preamble, wait_one_cs, read_out_ts_1, read_out_ts_2, read_out_sub_header, ts_error, algin_first_layer, trailer);
signal first_layer_state : first_layer_type;
type array_16_type is array (NLINKS - 1 downto 0) of std_logic_vector(15 downto 0);
signal link_fpga_id : array_16_type;
signal link_sub_overflow : array_16_type;
type ts_array_type is array (NLINKS - 1 downto 0) of std_logic_vector(47 downto 0);
signal link_fpga_ts : ts_array_type;

type first_layer_array is array (NLINKS/2 - 1 downto 0) of std_logic_vector(37 downto 0);
signal first_layer_wren       : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_data       : first_layer_array;
signal first_layer_ren        : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_data_out   : first_layer_array;
signal first_layer_empty      : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_full       : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_almost_full: std_logic_vector(NFIRST - 1 downto 0);
signal saw_preamble			  : std_logic_vector(NLINKS - 1 downto 0);
signal saw_sheader			  : std_logic_vector(NLINKS - 1 downto 0);
signal saw_trailer			  : std_logic_vector(NLINKS - 1 downto 0);
constant all_ones 			  : std_logic_vector(NLINKS - 1 downto 0) := (others => '1');
constant all_zero 			  : std_logic_vector(NLINKS - 1 downto 0) := (others => '0');

begin

reset <= not i_reset_n;

-- write to link fifos
-- TODO: check https://www.intel.com/content/dam/www/programmable/us/en/pdfs/literature/ug/archives/ug-fifo-15.1.pdf
-- TODO: do not write if fifo is full --> error, throw package away
process(i_clk_250, i_reset_n)
begin
	if(i_reset_n = '0') then
		link_fifo_wren <= (others => '0');
		link_fifo_data <= (others => '0');
	elsif(rising_edge(i_clk_250)) then
		set_link_data : FOR i in 0 to NLINKS - 1 LOOP
			link_fifo_data(35 + i * 36 downto i * 36) <= i_data(31 + i * 32 downto i * 32) & i_datak(3 + i * 4 downto i * 4);
			if ( ( i_data(31 + i * 32 downto i * 32) = x"000000BC" and i_datak(3 + i * 4 downto i * 4) = "0001" ) or 
                 ( i_data(31 + i * 32 downto i * 32) = x"00000000" and i_datak(3 + i * 4 downto i * 4) = "1111" )                 
            ) then
	         	link_fifo_wren(i) <= '0';
        	else
				link_fifo_wren(i) <= '1';
      		end if;
		END LOOP set_link_data;
	end if;
end process;

-- generate fifos per link
link_fifos:
FOR i in 0 to NLINKS - 1 GENERATE
	
    e_link_fifo : entity work.ip_dcfifo_mixed_widths
    generic map(
        ADDR_WIDTH_w => LINK_FIFO_ADDR_WIDTH,
        DATA_WIDTH_w => 36,
        ADDR_WIDTH_r => LINK_FIFO_ADDR_WIDTH,
        DATA_WIDTH_r => 36,
        DEVICE 		 => "Arria 10"--,
	)
	port map (
		aclr 	=> reset,
		data 	=> link_fifo_data(35 + i * 36 downto i * 36),
		rdclk 	=> i_clk_250,
		rdreq 	=> link_fifo_ren(i),
		wrclk 	=> i_clk_250,
		wrreq 	=> link_fifo_wren(i),
		q 		=> link_fifo_data_out(i),
		rdempty => link_fifo_empty(i),
		rdusedw => open,
		wrfull 	=> link_fifo_full(i),
		wrusedw => open--,
	);

END GENERATE link_fifos;

-- write link data to first tree layer
process(i_clk_250, i_reset_n)
begin
	if( i_reset_n = '0' ) then
		-- state machine singals
		first_layer_state	<= wait_for_preamble;
		link_fifo_ren 		<= (others => '0');
		saw_preamble		<= (others => '0');
		saw_sheader			<= (others => '0');
		link_fpga_id		<= (others => (others => '0'));
		link_fpga_ts		<= (others => (others => '0'));
		o_error				<= (others => '0');
		saw_trailer			<= (others => '0');
		-- first layer signals
		first_layer_data 	<= (others => (others => '0'));
		first_layer_wren 	<= (others => '0');
	elsif( rising_edge(i_clk_250) ) then

		o_error			 <= (others => '0');
		first_layer_wren <= (others => '0');

		case first_layer_state is

			when wait_for_preamble =>
			-- TODO: timeout?
				if ( saw_preamble = all_ones ) then
					FOR I IN 0 to NLINKS - 1 LOOP
						if ( link_fifo_empty(I) = '0' ) then
							link_fifo_ren(I) <= '1';
						end if;
					END LOOP;
					first_layer_state <= wait_one_cs;
				else
					FOR I IN 0 to NLINKS - 1 LOOP
						if(	(link_fifo_data_out(I)(35 downto 30) = "111010" )
							and 
							(link_fifo_data_out(I)(11 downto 4) = x"bc")
							and
							(link_fifo_data_out(I)(3 downto 0) = "0001")
						) then
							saw_preamble(I) 	<= '1';
							link_fpga_id(I) 	<= link_fifo_data_out(I)(27 downto 12);
							link_fifo_ren(I) 	<= '0';
						else
							if ( link_fifo_empty(I) = '0' ) then
								link_fifo_ren(I) <= '1';
							end if;
						end if;
					END LOOP;
				end if;

			when wait_one_cs =>
				FOR I IN 0 to NLINKS - 1 LOOP
					if ( link_fifo_empty(I) = '0' ) then
						link_fifo_ren(I) <= '1';
					end if;
				END LOOP;
				if ( link_fifo_empty = all_zero ) then
					first_layer_state <= read_out_ts_1;
				end if;

			when read_out_ts_1 =>
				FOR I IN 0 to NLINKS - 1 LOOP
					link_fpga_ts(I)(47 downto 16) <= link_fifo_data_out(I)(35 downto 4);
				END LOOP;
				first_layer_state <= read_out_ts_2;

			when read_out_ts_2 =>
				FOR I IN 0 to NLINKS - 1 LOOP
					link_fpga_ts(I)(15 downto 0) <= link_fifo_data_out(I)(35 downto 20);
				END LOOP;
				first_layer_state <= read_out_sub_header;

			when read_out_sub_header =>
				saw_sheader	<= (others => '0');
				first_layer_state <= algin_first_layer;
				-- check if TS are all the same
				FOR I IN 1 to NLINKS - 1 LOOP
					if ( link_fpga_ts(0) /= link_fpga_ts(I) ) then
						first_layer_state <= ts_error;
					end if;
				END LOOP;

				FOR I IN 1 to NFIRST - 1 LOOP
					-- here we just send the same sheader 
					first_layer_data(I) <= (others => '1');
					first_layer_wren(I) <= '1';
				END LOOP;

			when algin_first_layer =>
				link_fifo_ren <= (others => '0');
				-- TODO: timeout?
				if ( saw_trailer = all_ones ) then
					first_layer_state <= trailer;
				elsif ( saw_sheader = all_ones ) then
					link_fifo_ren <= (others => '1');
					first_layer_state <= read_out_sub_header;
				end if;

				-- start with alignment here
				-- align 0-31 links
				FOR I IN 0 TO NLINKS/2 - 1 LOOP
					if(	(link_fifo_data_out(I*2)(31 downto 26) = "111111" ) ) then
						saw_sheader(I*2) <= '1';
						if(	link_fifo_data_out(I*2 + 1)(31 downto 26) = "111111" ) then
							saw_sheader(I*2 + 1) <= '1';
						else
							-- TODO add correct FPGA ID
							first_layer_data(I) <= link_fifo_data_out(I*2 + 1)(35 downto 4) & "000000";
							link_fifo_ren(I*2 + 1) <= '1';
						end if;

					elsif( link_fifo_data_out(I*2 + 1)(31 downto 26) = "111111" ) then
						saw_sheader(I*2 + 1) <= '1';
						-- TODO add correct FPGA ID
						first_layer_data(I) <= link_fifo_data_out(I*2)(35 downto 4) & "000000";
						link_fifo_ren(I*2) <= '1';

					elsif( (link_fifo_data_out(I*2)(11 downto 4) = x"9c" ) 
							and (link_fifo_data_out(I*2)(3 downto 0) = "0001") ) then
						saw_trailer(I*2) <= '1';
						if(	(link_fifo_data_out(I*2 + 1)(11 downto 4) = x"9c" )
							and (link_fifo_data_out(I*2 + 1)(3 downto 0) = "0001") ) then
							saw_trailer(I*2 + 1) <= '1';
						else
							-- TODO add correct FPGA ID
							first_layer_data(I) <= link_fifo_data_out(I*2 + 1)(35 downto 4) & "000000";
							link_fifo_ren(I*2 + 1) <= '1';
						end if;

					elsif( (link_fifo_data_out(I*2 + 1)(11 downto 4) = x"9c" )
							and (link_fifo_data_out(I*2 + 1)(3 downto 0) = "0001") ) then
						saw_trailer(I*2 + 1) <= '1';
						-- TODO add correct FPGA ID
						first_layer_data(I) <= link_fifo_data_out(I*2)(35 downto 4) & "000000";
						link_fifo_ren(I*2) <= '1';
					else
						if ( link_fifo_data_out(I*2)(35 downto 33) <= link_fifo_data_out(I*2 + 1)(35 downto 33) ) then
							-- TODO add correct FPGA ID
							first_layer_data(I) <= link_fifo_data_out(I*2)(35 downto 4) & "000000";
							link_fifo_ren(I*2) <= '1';
						else
							-- TODO add correct FPGA ID
							first_layer_data(I) <= link_fifo_data_out(I*2 + 1)(35 downto 4) & "000000";
							link_fifo_ren(I*2 + 1) <= '1';
						end if;
					end if;
            	END LOOP;

            when trailer =>
            	first_layer_state <= wait_for_preamble;
            	saw_sheader <= (others => '0');
            	saw_preamble <= (others => '0');
            	saw_trailer <= (others => '0');

			when ts_error =>
				o_error <= x"1";

			when others =>
            	first_layer_state <= wait_for_preamble;
		end case;
	end if;
end process;


end architecture;
