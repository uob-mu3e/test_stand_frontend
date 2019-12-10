library ieee;
use ieee.std_logic_1164.all;
use work.pcie_components.all;

package mudaq_components is
 
component reset_logic is
	port(
		clk:                 in                      std_logic;
		rst_n:               in                      std_logic;

		reset_register:      in                      std_logic_vector(31 downto 0); 
		reset_reg_written:   in                      std_logic;

		resets:              out                     std_logic_vector(31 downto 0);
		resets_n:            out                     std_logic_vector(31 downto 0)                                                                                                                                           
	);              
end component;

component data_demerge is 
	port(
		clk:                    in  std_logic; -- receive clock (156.25 MHz)
		reset:                  in  std_logic;
		aligned:						in  std_logic; -- word alignment achieved
		data_in:						in  std_logic_vector(31 downto 0); -- optical from frontend board
		datak_in:               in  std_logic_vector(3 downto 0);
		data_out:					out std_logic_vector(31 downto 0); -- to sorting fifos
		datak_out:					out std_logic_vector(3 downto 0); -- to sorting fifos
		data_ready:             out std_logic;							  -- write req for sorting fifos	
		sc_out:						out std_logic_vector(31 downto 0); -- slowcontrol from frontend board
		sc_out_ready:				out std_logic;
		sck_out:						out std_logic_vector(3 downto 0); 
		fpga_id:						out std_logic_vector(15 downto 0)  -- FPGA ID of the connected frontend board
	);
end component;

component event_counter is
  port (
	clk:               in std_logic;
	dma_clk:           in std_logic;
	reset_n:           in std_logic;
	rx_data:           in std_logic_vector (31 downto 0);
	rx_datak:          in std_logic_vector (3 downto 0);
	dma_wen_reg:       in std_logic;
	event_length:      out std_logic_vector (11 downto 0);
	dma_data_wren:     out std_logic;
	dmamem_endofevent: out std_logic; 
	dma_data:          out std_logic_vector (31 downto 0);
	state_out:         out std_logic_vector(3 downto 0)
	  );
end component event_counter;

component dma_evaluation is
    port(
		clk:               	in std_logic;
		reset_n:           	in std_logic;
		dmamemhalffull:    	in std_logic;
		dmamem_endofevent: 	in std_logic;
		halffull_counter: 	out std_logic_vector (31 downto 0);
		nothalffull_counter: 	out std_logic_vector (31 downto 0);
		endofevent_counter:  out std_logic_vector (31 downto 0);
		notendofevent_counter:  out std_logic_vector (31 downto 0)
);
end component dma_evaluation;

component ip_ram is
  port (
		data      : in  std_logic_vector(31 downto 0) := (others => 'X'); -- datain
		wraddress : in  std_logic_vector(11 downto 0) := (others => 'X'); -- wraddress
		rdaddress : in  std_logic_vector(11 downto 0) := (others => 'X'); -- rdaddress
		wren      : in  std_logic                     := 'X';             -- wren
		clock     : in  std_logic                     := 'X';             -- clock
		q         : out std_logic_vector(31 downto 0)                     -- dataout
  );
end component ip_ram;

component ip_tagging_fifo is
  port (
		data  : in  std_logic_vector(11 downto 0) := (others => 'X'); -- datain
		wrreq : in  std_logic                     := 'X';             -- wrreq
		rdreq : in  std_logic                     := 'X';             -- rdreq
		clock : in  std_logic                     := 'X';             -- clk
		q     : out std_logic_vector(11 downto 0);                    -- dataout
		full  : out std_logic;                                        -- full
		aclr  : in  std_logic;
		empty : out std_logic                                         -- empty
  );
end component ip_tagging_fifo;

component linear_shift is 
	generic (
		g_m             : integer           := 7;
		g_poly          : std_logic_vector  := "1100000" -- x^7+x^6+1 
	);
	port (
		i_clk           : in  std_logic;
		reset_n         : in  std_logic;
		i_sync_reset    : in  std_logic;
		i_seed          : in  std_logic_vector (g_m-1 downto 0);
		i_en            : in  std_logic;
		o_lsfr          : out std_logic_vector (g_m-1 downto 0)
	);
end component linear_shift;

component data_generator_a10 is
    port(
		clk:                 in  std_logic;
		reset:               in  std_logic;
		enable_pix:          in  std_logic;
		random_seed:			in  std_logic_vector (15 downto 0);
		start_global_time:	in  std_logic_vector(47 downto 0);
		data_pix_generated:  out std_logic_vector(31 downto 0);
		datak_pix_generated:  out std_logic_vector(3 downto 0);
		data_pix_ready:      out std_logic;
		slow_down:				in  std_logic_vector(31 downto 0);
		state_out:  out std_logic_vector(3 downto 0)
);
end component data_generator_a10;

component ip_clk_ctrl is
  port (
		inclk  : in  std_logic := 'X'; -- inclk
		outclk : out std_logic         -- outclk
  );
end component ip_clk_ctrl;

component sc_master is
	generic(
		NLINKS : integer :=4
	);
	port(
		clk:					in std_logic;
		reset_n:				in std_logic;
		enable:				in std_logic;
		mem_data_in:		in std_logic_vector(31 downto 0);
		mem_addr:			out std_logic_vector(15 downto 0);
		mem_data_out:		out std_logic_vector(NLINKS * 32 - 1 downto 0);
		mem_data_out_k:	out std_logic_vector(NLINKS * 4 - 1 downto 0);
		done:					out std_logic;
		stateout:			out std_logic_vector(27 downto 0)
		);		
end component sc_master;

component sc_slave is
	port(
		clk:				in std_logic;
		reset_n:			in std_logic;
		enable:				in std_logic;
		link_data_in:		in std_logic_vector(31 downto 0);
		link_data_in_k:		in std_logic_vector(3 downto 0);
		mem_data_out:		out std_logic_vector(31 downto 0);
		mem_addr_out:		out std_logic_vector(15 downto 0);
		mem_addr_finished_out:       out std_logic_vector(15 downto 0);
		mem_wren:			out std_logic;			
		stateout:			out std_logic_vector(3 downto 0)
		);		
end component sc_slave;

component transceiver_fifo is
  port (
		data    : in  std_logic_vector(35 downto 0) := (others => 'X'); -- datain
		wrreq   : in  std_logic                     := 'X';             -- wrreq
		rdreq   : in  std_logic                     := 'X';             -- rdreq
		wrclk   : in  std_logic                     := 'X';             -- wrclk
		rdclk   : in  std_logic                     := 'X';             -- rdclk
		aclr    : in  std_logic                     := 'X';             -- aclr
		q       : out std_logic_vector(35 downto 0);                    -- dataout
		rdempty : out std_logic;                                        -- rdempty
		wrfull  : out std_logic                                         -- wrfull
  );
end component transceiver_fifo;

component sw_algin_data is
	generic(
		NLINKS : integer := 4
	);
	port (
			clks_read 		: in  std_logic_vector(NLINKS - 1 downto 0); -- 312,50 MHZ
			clks_write     : in  std_logic_vector(NLINKS - 1 downto 0); -- 156,25 MHZ
			
			clk_node_write : in  std_logic; -- 312,50 MHZ
			clk_node_read  : in  std_logic;

			reset_n			: in  std_logic;
			
			data_in 			: in std_logic_vector(NLINKS * 32 - 1 downto 0);
			fpga_id_in		: in std_logic_vector(NLINKS * 16 - 1 downto 0);
					
			enables_in		: in std_logic_vector(NLINKS - 1 downto 0);
			
			node_rdreq		: in std_logic;
			
			data_out	      : out std_logic_vector((NLINKS / 4) * 64 - 1 downto 0);
			state_out		: out std_logic_vector(3 downto 0);
			node_full_out  : out std_logic_vector(NLINKS / 4 - 1 downto 0);
			node_empty_out	: out std_logic_vector(NLINKS / 4 - 1 downto 0)
);		
end component sw_algin_data;

component ip_sw_fifo_32 is
	  port (
			data  : in  std_logic_vector(31 downto 0) := (others => 'X'); -- datain
			wrreq : in  std_logic                     := 'X';             -- wrreq
			rdreq : in  std_logic                     := 'X';             -- rdreq
			rdclk : in  std_logic                     := 'X';             -- rdclk
			wrclk : in  std_logic                     := 'X';             -- wrclk
			aclr  : in  std_logic                     := 'X';             -- aclr
			q     : out std_logic_vector(31 downto 0);                    -- dataout
			wrfull  : out std_logic;                                        -- full
			rdempty : out std_logic                                         -- empty
	  );
end component ip_sw_fifo_32;

component ip_sw_fifo_64 is
	  port (
			data  : in  std_logic_vector(63 downto 0) := (others => 'X'); -- datain
			wrreq : in  std_logic                     := 'X';             -- wrreq
			rdreq : in  std_logic                     := 'X';             -- rdreq
			rdclk : in  std_logic                     := 'X';             -- rdclk
			wrclk : in  std_logic                     := 'X';             -- wrclk
			aclr  : in  std_logic                     := 'X';             -- aclr
			q     : out std_logic_vector(63 downto 0);                    -- dataout
			wrfull  : out std_logic;                                        -- full
			rdempty : out std_logic                                         -- empty
	  );
end component ip_sw_fifo_64;

component ip_pll_312 is
  port (
		rst      : in  std_logic := 'X'; -- reset
		refclk   : in  std_logic := 'X'; -- clk
		locked   : out std_logic;        -- export
		outclk_0 : out std_logic         -- clk
  );
end component ip_pll_312;

component ip_pll_125 is
  port (
		outclk_0 : out std_logic;        -- clk
		refclk   : in  std_logic := 'X'; -- clk
		rst      : in  std_logic := 'X' -- reset
);
end component ip_pll_125;

end package mudaq_components;
