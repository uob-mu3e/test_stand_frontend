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
	reset_n:           in std_logic;
	rx_data:           in std_logic_vector (31 downto 0);
	rx_datak:          in std_logic_vector (3 downto 0);
	data_ready:        in std_logic;
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

component ip_transiver is
	port (
		reconfig_write          : in  std_logic_vector(0 downto 0)   := (others => 'X'); -- write
      reconfig_read           : in  std_logic_vector(0 downto 0)   := (others => 'X'); -- read
      reconfig_address        : in  std_logic_vector(11 downto 0)  := (others => 'X'); -- address
      reconfig_writedata      : in  std_logic_vector(31 downto 0)  := (others => 'X'); -- writedata
      reconfig_readdata       : out std_logic_vector(31 downto 0);                     -- readdata
      reconfig_waitrequest    : out std_logic_vector(0 downto 0);                      -- waitrequest
      reconfig_clk            : in  std_logic_vector(0 downto 0)   := (others => 'X'); -- clk
      reconfig_reset          : in  std_logic_vector(0 downto 0)   := (others => 'X'); -- reset
      rx_analogreset          : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- rx_analogreset
      rx_cal_busy             : out std_logic_vector(3 downto 0);                      -- rx_cal_busy
      rx_cdr_refclk0          : in  std_logic                      := 'X';             -- clk
      rx_clkout               : out std_logic_vector(3 downto 0);                      -- clk
      rx_coreclkin            : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- clk
      rx_datak                : out std_logic_vector(15 downto 0);                     -- rx_datak
      rx_digitalreset         : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- rx_digitalreset
      rx_disperr              : out std_logic_vector(15 downto 0);                     -- rx_disperr
      rx_errdetect            : out std_logic_vector(15 downto 0);                     -- rx_errdetect
      rx_is_lockedtodata      : out std_logic_vector(3 downto 0);                      -- rx_is_lockedtodata
      rx_is_lockedtoref       : out std_logic_vector(3 downto 0);                      -- rx_is_lockedtoref
      rx_parallel_data        : out std_logic_vector(127 downto 0);                    -- rx_parallel_data
      rx_patterndetect        : out std_logic_vector(15 downto 0);                     -- rx_patterndetect
      rx_runningdisp          : out std_logic_vector(15 downto 0);                     -- rx_runningdisp
      rx_serial_data          : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- rx_serial_data
      rx_seriallpbken         : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- rx_seriallpbken
      rx_syncstatus           : out std_logic_vector(15 downto 0);                     -- rx_syncstatus
      tx_analogreset          : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- tx_analogreset
      tx_cal_busy             : out std_logic_vector(3 downto 0);                      -- tx_cal_busy
      tx_clkout               : out std_logic_vector(3 downto 0);                      -- clk
      tx_coreclkin            : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- clk
      tx_datak                : in  std_logic_vector(15 downto 0)  := (others => 'X'); -- tx_datak
      tx_digitalreset         : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- tx_digitalreset
      tx_parallel_data        : in  std_logic_vector(127 downto 0) := (others => 'X'); -- tx_parallel_data
      tx_serial_clk0          : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- clk
      tx_serial_data          : out std_logic_vector(3 downto 0);                      -- tx_serial_data
      unused_rx_parallel_data : out std_logic_vector(287 downto 0);                    -- unused_rx_parallel_data
      unused_tx_parallel_data : in  std_logic_vector(367 downto 0) := (others => 'X')  -- unused_tx_parallel_data
	);
end component ip_transiver;

component debouncer is
	generic (
        -- bus width
        W   : positive := 1;
        -- number of clock cycles (stable signal)
        N   : positive := 16#FFFF#--;
    );
    port (
        d       :   in  std_logic_vector(W-1 downto 0);
        q       :   out std_logic_vector(W-1 downto 0);
        arst_n  :   in  std_logic;
        clk     :   in  std_logic--;
    );
end component debouncer;

component sc_s4 is
	port(
		clk:                in std_logic;
		reset_n:            in std_logic;
		enable:             in std_logic;
		
		mem_data_in:        in std_logic_vector(31 downto 0);
		
		link_data_in:       in std_logic_vector(31 downto 0);
		link_data_in_k:     in std_logic_vector(3 downto 0);
		
		fifo_data_out:      out std_logic_vector(35 downto 0);
		fifo_we:            out std_logic;
		
		mem_data_out:       out std_logic_vector(31 downto 0);
		mem_addr_out:       out std_logic_vector(15 downto 0);
		mem_wren:           out std_logic;
		
		stateout:           out std_logic_vector(27 downto 0)
	);
end component sc_s4;

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

component rx_align is
    generic (
        Nb : positive := 4;
		  K : std_logic_vector(7 downto 0) := X"BC"--;
    );
    port (
        data    :   out std_logic_vector(8*Nb-1 downto 0);
        datak   :   out std_logic_vector(Nb-1 downto 0);

        lock    :   out std_logic;

        datain  :   in  std_logic_vector(8*Nb-1 downto 0);
        datakin :   in  std_logic_vector(Nb-1 downto 0);
		  
        syncstatus      :   in  std_logic_vector(Nb-1 downto 0);
        patterndetect   :   in  std_logic_vector(Nb-1 downto 0);
        enapatternalign :   out std_logic;

        errdetect   :   in  std_logic_vector(Nb-1 downto 0);
        disperr     :   in  std_logic_vector(Nb-1 downto 0);

        rst_n   :   in  std_logic;
        clk     :   in  std_logic
    );
end component;

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

component seg7_lut is
    port (
        hex : in  std_logic_vector(3 downto 0);
        seg : out std_logic_vector(6 downto 0)--;
    );
end component seg7_lut;

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

component watchdog is
    generic (
        W : positive := 1;
        N : positive := 16#FFFF#--;
    );
    port (
        d			: in std_logic_vector(W-1 downto 0);

        rstout_n 	: out std_logic;

        rst_n 		: in  std_logic;
        clk 		: in  std_logic--;
    );
end component;

component ip_xcvr_fpll is
  port (
		pll_cal_busy          : out std_logic;                                        -- pll_cal_busy
		pll_locked            : out std_logic;                                        -- pll_locked
		pll_powerdown         : in  std_logic                     := 'X';             -- pll_powerdown
		pll_refclk0           : in  std_logic                     := 'X';             -- clk
		reconfig_write0       : in  std_logic                     := 'X';             -- write
		reconfig_read0        : in  std_logic                     := 'X';             -- read
		reconfig_address0     : in  std_logic_vector(9 downto 0)  := (others => 'X'); -- address
		reconfig_writedata0   : in  std_logic_vector(31 downto 0) := (others => 'X'); -- writedata
		reconfig_readdata0    : out std_logic_vector(31 downto 0);                    -- readdata
		reconfig_waitrequest0 : out std_logic;                                        -- waitrequest
		reconfig_clk0         : in  std_logic                     := 'X';             -- clk
		reconfig_reset0       : in  std_logic                     := 'X';             -- reset
		tx_serial_clk         : out std_logic                                         -- clk
  );
end component ip_xcvr_fpll;

component ip_xcvr_phy is
  port (
		reconfig_write          : in  std_logic_vector(0 downto 0)   := (others => 'X'); -- write
		reconfig_read           : in  std_logic_vector(0 downto 0)   := (others => 'X'); -- read
		reconfig_address        : in  std_logic_vector(11 downto 0)  := (others => 'X'); -- address
		reconfig_writedata      : in  std_logic_vector(31 downto 0)  := (others => 'X'); -- writedata
		reconfig_readdata       : out std_logic_vector(31 downto 0);                     -- readdata
		reconfig_waitrequest    : out std_logic_vector(0 downto 0);                      -- waitrequest
		reconfig_clk            : in  std_logic_vector(0 downto 0)   := (others => 'X'); -- clk
		reconfig_reset          : in  std_logic_vector(0 downto 0)   := (others => 'X'); -- reset
		rx_analogreset          : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- rx_analogreset
		rx_cal_busy             : out std_logic_vector(3 downto 0);                      -- rx_cal_busy
		rx_cdr_refclk0          : in  std_logic                      := 'X';             -- clk
		rx_clkout               : out std_logic_vector(3 downto 0);                      -- clk
		rx_coreclkin            : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- clk
		rx_datak                : out std_logic_vector(15 downto 0);                     -- rx_datak
		rx_digitalreset         : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- rx_digitalreset
		rx_disperr              : out std_logic_vector(15 downto 0);                     -- rx_disperr
		rx_errdetect            : out std_logic_vector(15 downto 0);                     -- rx_errdetect
		rx_is_lockedtodata      : out std_logic_vector(3 downto 0);                      -- rx_is_lockedtodata
		rx_is_lockedtoref       : out std_logic_vector(3 downto 0);                      -- rx_is_lockedtoref
		rx_parallel_data        : out std_logic_vector(127 downto 0);                    -- rx_parallel_data
		rx_patterndetect        : out std_logic_vector(15 downto 0);                     -- rx_patterndetect
		rx_runningdisp          : out std_logic_vector(15 downto 0);                     -- rx_runningdisp
		rx_serial_data          : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- rx_serial_data
		rx_seriallpbken         : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- rx_seriallpbken
		rx_syncstatus           : out std_logic_vector(15 downto 0);                     -- rx_syncstatus
		tx_analogreset          : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- tx_analogreset
		tx_cal_busy             : out std_logic_vector(3 downto 0);                      -- tx_cal_busy
		tx_clkout               : out std_logic_vector(3 downto 0);                      -- clk
		tx_coreclkin            : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- clk
		tx_datak                : in  std_logic_vector(15 downto 0)  := (others => 'X'); -- tx_datak
		tx_digitalreset         : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- tx_digitalreset
		tx_parallel_data        : in  std_logic_vector(127 downto 0) := (others => 'X'); -- tx_parallel_data
		tx_serial_clk0          : in  std_logic_vector(3 downto 0)   := (others => 'X'); -- clk
		tx_serial_data          : out std_logic_vector(3 downto 0);                      -- tx_serial_data
		unused_rx_parallel_data : out std_logic_vector(287 downto 0);                    -- unused_rx_parallel_data
		unused_tx_parallel_data : in  std_logic_vector(367 downto 0) := (others => 'X')  -- unused_tx_parallel_data
  );
end component ip_xcvr_phy;

component ip_xcvr_reset is
  port (
		clock              : in  std_logic                    := 'X';             -- clk
		pll_cal_busy       : in  std_logic_vector(0 downto 0) := (others => 'X'); -- pll_cal_busy
		pll_locked         : in  std_logic_vector(0 downto 0) := (others => 'X'); -- pll_locked
		pll_powerdown      : out std_logic_vector(0 downto 0);                    -- pll_powerdown
		pll_select         : in  std_logic_vector(0 downto 0) := (others => 'X'); -- pll_select
		reset              : in  std_logic                    := 'X';             -- reset
		rx_analogreset     : out std_logic_vector(3 downto 0);                    -- rx_analogreset
		rx_cal_busy        : in  std_logic_vector(3 downto 0) := (others => 'X'); -- rx_cal_busy
		rx_digitalreset    : out std_logic_vector(3 downto 0);                    -- rx_digitalreset
		rx_is_lockedtodata : in  std_logic_vector(3 downto 0) := (others => 'X'); -- rx_is_lockedtodata
		rx_ready           : out std_logic_vector(3 downto 0);                    -- rx_ready
		tx_analogreset     : out std_logic_vector(3 downto 0);                    -- tx_analogreset
		tx_cal_busy        : in  std_logic_vector(3 downto 0) := (others => 'X'); -- tx_cal_busy
		tx_digitalreset    : out std_logic_vector(3 downto 0);                    -- tx_digitalreset
		tx_ready           : out std_logic_vector(3 downto 0)                     -- tx_ready
  );
end component ip_xcvr_reset;

component nios is
	port (
		avm_qsfp_address           : out   std_logic_vector(15 downto 0);                    -- address
		avm_qsfp_read              : out   std_logic;                                        -- read
		avm_qsfp_readdata          : in    std_logic_vector(31 downto 0) := (others => 'X'); -- readdata
		avm_qsfp_write             : out   std_logic;                                        -- write
		avm_qsfp_writedata         : out   std_logic_vector(31 downto 0);                    -- writedata
		avm_qsfp_waitrequest       : in    std_logic                     := 'X';             -- waitrequest
		clk_clk                    : in    std_logic                     := 'X';             -- clk
		flash_tcm_address_out      : out   std_logic_vector(27 downto 0);                    -- tcm_address_out
		flash_tcm_read_n_out       : out   std_logic_vector(0 downto 0);                     -- tcm_read_n_out
		flash_tcm_write_n_out      : out   std_logic_vector(0 downto 0);                     -- tcm_write_n_out
		flash_tcm_data_out         : inout std_logic_vector(31 downto 0) := (others => 'X'); -- tcm_data_out
		flash_tcm_chipselect_n_out : out   std_logic_vector(0 downto 0);                     -- tcm_chipselect_n_out
		i2c_sda_in                 : in    std_logic                     := 'X';             -- sda_in
		i2c_scl_in                 : in    std_logic                     := 'X';             -- scl_in
		i2c_sda_oe                 : out   std_logic;                                        -- sda_oe
		i2c_scl_oe                 : out   std_logic;                                        -- scl_oe
		pio_export                 : out   std_logic_vector(31 downto 0);                    -- export
		rst_reset_n                : in    std_logic                     := 'X';             -- reset_n
		spi_MISO                   : in    std_logic                     := 'X';             -- MISO
		spi_MOSI                   : out   std_logic;                                        -- MOSI
		spi_SCLK                   : out   std_logic;                                        -- SCLK
		spi_SS_n                   : out   std_logic    
);
end component nios;

component ip_pll_125 is
  port (
		outclk_0 : out std_logic;        -- clk
		refclk   : in  std_logic := 'X'; -- clk
		rst      : in  std_logic := 'X' -- reset
);
end component ip_pll_125;


end package mudaq_components;
