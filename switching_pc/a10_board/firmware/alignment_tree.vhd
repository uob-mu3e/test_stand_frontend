library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
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
signal link_fifo_data_out   : std_logic_vector(NLINKS * 36 - 1 downto 0);
signal link_fifo_empty      : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_full       : std_logic_vector(NLINKS - 1 downto 0);
signal link_fifo_almost_full: std_logic_vector(NLINKS - 1 downto 0);

-- first layer fifos
type first_layer_type is (wait_for_preamble, read_out_ts_1, read_out_ts_2, read_out_sub_header, ts_error, algin_first_layer);
signal first_layer_state : first_layer_type;
type array_16_type is array (NLINKS - 1 downto 0) of std_logic_vector(15 downto 0);
signal link_fpga_id : array_16_type;
signal link_sub_overflow : array_16_type;
type ts_array_type is array (NLINKS - 1 downto 0) of std_logic_vector(47 downto 0);
signal link_fpga_ts : ts_array_type;
type data_array_type is array (31 downto 0) of std_logic_vector(37 downto 0);

signal first_layer_wren       : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_data       : std_logic_vector(NFIRST * 38 - 1 downto 0);
signal first_layer_ren        : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_data_out   : std_logic_vector(NFIRST * 38 - 1 downto 0);
signal first_layer_empty      : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_full       : std_logic_vector(NFIRST - 1 downto 0);
signal first_layer_almost_full: std_logic_vector(NFIRST - 1 downto 0);
signal saw_preamble			  : std_logic_vector(NLINKS - 1 downto 0);
signal saw_sheader			  : std_logic_vector(NLINKS - 1 downto 0);
signal saw_trailer			  : std_logic_vector(NLINKS - 1 downto 0);
constant all_ones 			  : std_logic_vector(NLINKS - 1 downto 0) := (others => '1');

begin

reset <= not i_reset_n;

-- write to link fifos
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
	
    e_link_fifo : entity work.ip_scfifo
    generic map(
        ADDR_WIDTH 	=> LINK_FIFO_ADDR_WIDTH,
        DATA_WIDTH 	=> 36,
        DEVICE 		=> "Arria 10"--,
	)
	port map (
		data     		=> link_fifo_data(35 + i * 36 downto i * 36),
		wrreq    		=> link_fifo_wren(i),
		rdreq    		=> link_fifo_ren(i),
		clock    		=> i_clk_250,
		q    	 		=> link_fifo_data_out(35 + i * 36 downto i * 36),
		full     		=> link_fifo_full(i),
		empty    		=> link_fifo_empty(i),
		almost_empty 	=> open,
		almost_full 	=> link_fifo_almost_full(i),
		usedw 			=> open,
		sclr     		=> reset--,
	);

END GENERATE link_fifos;

-- write link data to first tree layer
process(i_clk_250, i_reset_n)
	variable temp_hit_ts : std_logic_vector (3 downto 0);  -- lower 4 bits of ts
	variable var_hits    : data_array_type; -- new hit sice
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
		first_layer_data 	<= (others => '0');
		first_layer_wren 	<= (others => '0');
	elsif( rising_edge(i_clk_250) ) then

		o_error			 <= (others => '0');
		first_layer_wren <= (others => '0');

		case first_layer_state is

			when wait_for_preamble =>
			-- TODO: timeout?
				if ( saw_preamble = all_ones ) then
					link_fifo_ren <= (others => '1');
					first_layer_state <= read_out_ts_1;
				else
					FOR I IN 0 to NLINKS - 1 LOOP
						if(	(link_fifo_data_out(35 + I * 36 downto I * 36 + 30) = "111010" )
							and 
							(link_fifo_data_out(11 + I * 36 downto I * 36 + 4) = x"bc")
							and
							(link_fifo_data_out(3 + I * 36 downto I * 36) = "0001")
						) then
							saw_preamble(I) 	<= '1';
							link_fpga_id(I) 	<= link_fifo_data_out(I * 36 + 23 downto I * 36 + 4);
							link_fifo_ren(I) 	<= '0';
						else
							link_fifo_ren(I) <= '1';
						end if;
					END LOOP;
				end if;

			when read_out_ts_1 =>
				FOR I IN 0 to NLINKS - 1 LOOP
					link_fpga_ts(I)(47 downto 16) <= link_fifo_data_out(35 + I * 36 downto I * 36 + 4);
				END LOOP;
				first_layer_state <= read_out_ts_2;

			when read_out_ts_2 =>
				FOR I IN 0 to NLINKS - 1 LOOP
					link_fpga_ts(I)(15 downto 0) <= link_fifo_data_out(35 + I * 36 downto I * 36 + 20);
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
					-- here we just send the same sheader 
					first_layer_data(I) <= (others => '1');
					first_layer_wren(I) <= '1';
				END LOOP;


			when algin_first_layer =>
				-- TODO: timeout?
				if ( saw_trailer = all_ones ) then
					first_layer_state <= trailer;
				elsif ( saw_sheader = all_ones ) then
					link_fifo_ren <= (others => '1');
					first_layer_state <= read_out_sub_header;
				else
					FOR I IN 0 to NLINKS - 1 LOOP
						if(	(link_fifo_data_out(31 + I * 36 downto I * 36 + 26) = "111111" )
							and 
							(link_fifo_data_out(11 + I * 36 downto I * 36 + 4) = x"bc")
							and
							(link_fifo_data_out(3 + I * 36 downto I * 36) = "0001")
						) then
							saw_sheader(I) 		<= '1';
							link_fifo_ren(I) 	<= '0';
						elsif( (link_fifo_data_out(11 + I * 36 downto I * 36 + 4) = x"9c")
							   and
							   (link_fifo_data_out(3 + I * 36 downto I * 36) = "0001")
						) then
							saw_trailer(I)   <= '1';
							link_fifo_ren(I) <= '0';
						end if;
					END LOOP;
				end if;
				-- start with alignment here
				-- align 0-31 links
				FOR K in 0 TO 31 LOOP
					-- TODO: add FPGA ID right
					var_hits(K) := link_fifo_data_out((K + 1) * 36 - 1 downto K * 36 + 4) & "000000";
				END LOOP;
				FOR J IN 0 TO 31 LOOP
	                FOR I IN 0 TO 31 - J LOOP
	                    if unsigned(var_array(I) > unsigned(var_array(I + 1) then
	                        temp_hit_ts 	:= var_array(I);
	                        var_hits(I) 	:= var_array(I + 1);
	                        var_hits(I + 1) := temp;
	                    end if;
	                END LOOP;
            	END LOOP;

            when trailer =>
            	first_layer_state <= wait_for_preamble;

			when ts_error =>
				o_error <= x"1";

			when others =>
            	first_layer_state <= wait_for_preamble;
		end case;
	end if;
end process;


end architecture;
