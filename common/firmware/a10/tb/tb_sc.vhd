library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


--  A testbench has no ports.
entity tb_sc is
end entity;

architecture rtl of tb_sc is
  --  Declaration of the component that will be instantiated.

  --  Specifies which entity is bound with the component.
  		
        constant NLINKS : integer := 2;
  		
  		signal clk : std_logic;
  		signal reset_n : std_logic := '1';
  		signal writememdata : std_logic_vector(31 downto 0);
  		signal writememdata_out : std_logic_vector(31 downto 0);
  		signal writememdata_out_reg : std_logic_vector(31 downto 0);
  		signal writememaddr : std_logic_vector(15 downto 0);
  		signal memaddr : std_logic_vector(15 downto 0);
  		signal memaddr_reg : std_logic_vector(15 downto 0);
  		signal mem_data_out : work.util.slv32_array_t(NLINKS-1 downto 0);
  		signal mem_datak_out : work.util.slv4_array_t(NLINKS-1 downto 0);
  		signal mem_data_out_slave : std_logic_vector(31 downto 0);
  		signal mem_datak_out_slave : std_logic_vector(3 downto 0);
  		signal mem_addr_out_slave, length : std_logic_vector(15 downto 0);
  		signal mem_wren_slave : std_logic;
  		signal link_enable : std_logic_vector(NLINKS - 1 downto 0) := (others => '1');

  		constant ckTime: 		time	:= 10 ns;
  		
  		signal writememwren, length_we, done, toggle_read, fifo_we : std_logic;

        signal fifo_wdata : std_logic_vector(35 downto 0);
        signal link_data : std_logic_vector(127 downto 0);
        signal link_datak : std_logic_vector(15 downto 0);

  		type state_type is (idle, write_sc, wait_state, read_sc);
        signal state : state_type;
		
begin
  --  Component instantiation.
    sc_main : entity work.swb_sc_main 
    generic map (
        NLINKS => NLINKS
    )
    port map(
        i_clk				=> clk,
        i_reset_n			=> reset_n,
        i_length_we		    => length_we,
        i_length		    => length,
        i_mem_data		    => writememdata_out,
        o_mem_addr			=> memaddr,
        o_mem_data		    => mem_data_out,
        o_mem_datak		    => mem_datak_out,
        o_done				=> done,
        o_state			    => open--,
    );
  
    sc_rx : entity work.sc_rx 
    port map(
        i_link_data     => mem_data_out(0),
        i_link_datak    => mem_datak_out(0),
        
        o_fifo_we       => fifo_we,
        o_fifo_wdata    => fifo_wdata,
        
        o_ram_addr      => open,
        o_ram_re        => open,
        
        i_ram_rvalid    => '1',
        i_ram_rdata     => (others => '1'),
        
        o_ram_we        => open,
        o_ram_wdata     => open,
        
        i_reset_n       => reset_n,
        i_clk           => clk--,
    );

    e_merger : entity work.data_merger
    generic map (
        feb_mapping => (3,2,1,0)--;
                )
    port map (
                 fpga_ID_in                 => x"000A",
                 FEB_type_in                => "111000",

                 run_state                  => (0 => '1', others =>'0'),
                 run_number                 => (others => '0'),

                 o_data_out    => link_data,
                 o_data_is_k    => link_datak,

                 slowcontrol_write_req      => fifo_we,
                 i_data_in_slowcontrol      => fifo_wdata,

                 data_write_req             => (others => '0'),
                 i_data_in                  => (others => '0'),
                 o_fifos_almost_full        => open,

                 override_data_in           => (others => '0'),
                 override_data_is_k_in      => (others => '0'),
                 override_req               => '0',
                 override_granted           => open,

                 can_terminate              => '0',
                 o_terminated               => open,
                 data_priority              => '0',
                 o_rate_count               => open,

                 reset                      => not reset_n,
                 clk                        => clk--,
             );

  
    sc_secondary : entity work.swb_sc_secondary
    generic map (
        NLINKS => NLINKS,
        skip_init => '1'
    )
    port map(
        clk					=> clk,
        reset_n				=> reset_n,
        i_link_enable		=> "01",
        link_data_in(0)     => link_data(31 downto 0),
        link_data_in_k(0)   => link_datak(3 downto 0),
        mem_data_out		=> mem_data_out_slave,
        mem_addr_out		=> mem_addr_out_slave,
        mem_wren 			=> mem_wren_slave,
        mem_addr_finished_out => open,
        stateout			=> open
    );

	wram : entity work.ip_ram
    generic map(
        ADDR_WIDTH_A => 8,
        ADDR_WIDTH_B => 8,
        DATA_WIDTH_A => 32,
        DATA_WIDTH_B => 32,
        OUTDATA_REG_A => "UNREGISTERED"--,
    )
  	port map (
        address_a => writememaddr(7 downto 0),
        address_b => memaddr(7 downto 0),
        clock_a => clk,
        clock_b => clk,
        data_a => writememdata,
        data_b => (others => '0'),
        wren_a => writememwren,
        wren_b => '0',
        q_a => open,
        q_b => writememdata_out--,
  	);

     rram : entity work.ip_ram
     generic map(
           ADDR_WIDTH_A => 8,
           ADDR_WIDTH_B => 8,
           DATA_WIDTH_A => 32,
           DATA_WIDTH_B => 32,
           OUTDATA_REG_A => "UNREGISTERED"--,
     )
     port map (
           address_a => mem_addr_out_slave(7 downto 0),
           address_b => (others => '0'),
           clock_a => clk,
           clock_b => '0',
           data_a => mem_data_out_slave,
           data_b => (others => '0'),
           wren_a => mem_wren_slave,
           wren_b => '0',
           q_a => open,
           q_b => open--, 
    );

  	-- generate the clock
	ckProc: process
	begin
		clk <= '0';
		wait for ckTime/2;
		clk <= '1';
		wait for ckTime/2;
	end process;

	inita : process
	begin
		reset_n	 <= '0';
		wait for 8 ns;
		reset_n	 <= '1';

		wait for 10 us;
		reset_n  <= '0';

		wait for 8 ns;
		reset_n  <= '1';
		
		wait;
	end process inita;

	memory : process(reset_n, clk)
	begin
	if(reset_n = '0')then
		writememdata <= (others => '0');
		writememaddr <= x"FFFF";
		writememwren <= '0';
		length_we    <= '0';
        toggle_read  <= '0';
		length       <= (others => '0');
		state        <= idle;
	elsif(rising_edge(clk))then
		writememwren    <= '0';
        length_we       <= '0';
       
        case state is
            
            when idle =>
                if ( done = '1' ) then
                    if ( toggle_read = '1' ) then
                        state <= read_sc;
                    else
                        state <= write_sc;
                    end if;
                end if;
                
            when write_sc =>
                if(writememaddr(3 downto 0) = x"F")then
                    writememdata <= x"1dffffbc";
                    writememaddr <= writememaddr + 1;
                    writememwren <= '1';
                elsif(writememaddr(3 downto 0)  = x"0")then
                    writememdata <= x"0000000a";
                    writememaddr <= writememaddr + 1;
                    writememwren <= '1';
                elsif(writememaddr(3 downto 0)  = x"1")then
                    writememdata <= x"00000001";
                    writememaddr <= writememaddr + 1;
                    writememwren <= '1';
                elsif(writememaddr(3 downto 0)  = x"2")then
                    writememdata <= x"0000000b";
                    writememaddr <= writememaddr + 1;
                    writememwren <= '1';
                elsif(writememaddr(3 downto 0)  = x"3")then
                    writememdata <= x"0000009c";
                    writememaddr <= writememaddr + 1;
                    writememwren <= '1';
                    length <= x"0003";
                elsif(writememaddr(3 downto 0)  = x"4")then
                    length_we <= '1';
                    writememaddr <= (others => '1');
                    state <= wait_state;
                end if;

            when read_sc =>
                if(writememaddr(3 downto 0) = x"F")then
                    writememdata <= x"1Effffbc";
                    writememaddr <= writememaddr + 1;
                    writememwren <= '1';
                elsif(writememaddr(3 downto 0)  = x"0")then
                    writememdata <= x"0000000a";
                    writememaddr <= writememaddr + 1;
                    writememwren <= '1';
                elsif(writememaddr(3 downto 0)  = x"1")then
                    writememdata <= x"00000001";
                    writememaddr <= writememaddr + 1;
                    writememwren <= '1';
                elsif(writememaddr(3 downto 0)  = x"2")then
                    writememdata <= x"0000009c";
                    writememaddr <= writememaddr + 1;
                    writememwren <= '1';
                    length <= x"0002";
                elsif(writememaddr(3 downto 0)  = x"3")then
                    length_we <= '1';
                    writememaddr <= (others => '1');
                    state <= wait_state;
                end if;
            
            when wait_state =>
                if ( done = '0' ) then
                    toggle_read <= not toggle_read;
                    state <= idle;
                end if;
            
            when others =>
                writememaddr <= (others => '0');
                state <= idle;
                
        end case;
	end if;
	end process memory;

end architecture;
