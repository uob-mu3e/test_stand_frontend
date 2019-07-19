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

component nios is
  port (
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
		reset_reset_n              : in    std_logic                     := 'X'              -- reset_n
  );
end component nios;

component datagenerator is
	Port (
		clk : 			in std_logic;
		reset_n:			in std_logic;
		writeregs:		in reg32array;
		slowdown:		in std_logic;
		data_out:		out std_logic_vector(31 downto 0);
		data_en:			out std_logic;
		end_of_event:	out std_logic;
		time_counter:	out std_logic_vector(63 downto 0);
		event_counter: out std_logic_vector(31 downto 0)
		);
end component;

component datagenerator64 is
	Generic (
		ENABLE_BIT: integer := 0);
	Port (
		clk : 			in std_logic;
		reset_n:			in std_logic;
		writeregs:		in reg32array;
		slowdown:		in std_logic;
		data_out:		out std_logic_vector(63 downto 0);
		data_en:			out std_logic;
		end_of_event:	out std_logic;
		time_counter:	out std_logic_vector(63 downto 0);
		event_counter: out std_logic_vector(31 downto 0)
		);
end component;

component counter_test is
	Port (
		clk            : 	in std_logic;
		reset_n			:	in std_logic;
		enable			:	in std_logic;
		fraccount      :	in std_logic_vector(7 downto 0);
		data_en			:	out std_logic;
		time_counter	:	out std_logic_vector(255 downto 0)
		);
end component;

component led_counter is
	Port (
		i_clock      : in  std_logic;
		i_enable     : in  std_logic;
		i_switch_1   : in  std_logic;
		i_switch_2   : in  std_logic;
		o_led_drive  : out std_logic
		);
end component;

-- component for dma speed test clk
component datagen_pll is
   Port (
      rst      : in  std_logic := 'X'; -- reset
      refclk   : in  std_logic := 'X'; -- clk
      locked   : out std_logic;        -- export
      outclk_0 : out std_logic         -- clk
      );
end component datagen_pll;

-- component for ip_fifo32
component ip_fifo32 is
   Port (
      data    : in  std_logic_vector(31 downto 0) := (others => 'X'); -- datain
      wrreq   : in  std_logic                     := 'X';             -- wrreq
      rdreq   : in  std_logic                     := 'X';             -- rdreq
      wrclk   : in  std_logic                     := 'X';             -- wrclk
      rdclk   : in  std_logic                     := 'X';             -- rdclk
      q       : out std_logic_vector(31 downto 0);                    -- dataout
      rdempty : out std_logic;                                        -- rdempty
      wrfull  : out std_logic;                                        -- wrfull
		aclr	  : in std_logic														 -- for resetting 
      );
end component ip_fifo32;

-- component for owne fifo32
component fifo32 is
	Generic (
		constant DATA_WIDTH  : positive := 32;
		constant FIFO_DEPTH	: positive := 32
	);
	Port ( 
		data			: in  std_logic_vector (DATA_WIDTH - 1 downto 0);
		wrreq			: in  std_logic;
		rdreq			: in  std_logic;
		clk			: in  std_logic;
		q				: out std_logic_vector (DATA_WIDTH - 1 downto 0);
		rdempty		: out std_logic;
		wrfull		: out std_logic;
		reset_n		: in  std_logic
	);
end component fifo32;

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
	port(
		clk:			in std_logic;
		din:			in std_logic;
		dout:			out std_logic
);		
end component debouncer;

component ip_clk_ctrl is
  port (
		inclk  : in  std_logic := 'X'; -- inclk
		outclk : out std_logic         -- outclk
  );
end component ip_clk_ctrl;

component counter is
  port (
		clk 				: 	in std_logic;
		reset_n			:	in std_logic;
		enable			:	in std_logic;
		data_en			:	out std_logic;
		time_counter 	: 	out std_logic_vector (255 downto 0)
  );
end component counter;

component data_demerge is 
    port(
        clk:                    in  std_logic;
        reset:                  in  std_logic;
        pixel_data:             out std_logic_vector(31 downto 0);
        pixel_data_ready:             out std_logic;
        feb_link_in:            in  std_logic_vector(31 downto 0);
        datak_in:               in  std_logic
    );
end component data_demerge;

component running is
	generic (
		g_m             	: integer           := 8;
		g_poly          	: std_logic_vector  := "10111000"; -- x^7+x^6+1
		chip_id_header_i	: std_logic_vector  := "001111";
		fpga_id_i		: std_logic_vector  	  := "0000110000110000"
	);
	port (
		clk 		: in std_logic;
		reset_n 	: in std_logic;
		random_seed	: in std_logic_vector (g_m - 1 downto 0);
		enable		: in std_logic;
		sync_reset	: in std_logic;
		data_en		: out std_logic;
		data_out	: out std_logic_vector (31 downto 0)
	);
end component running;

component data_fifo is
        port (
            data  : in  std_logic_vector(31 downto 0) := (others => 'X'); -- datain
            wrreq : in  std_logic                     := 'X';             -- wrreq
            rdreq : in  std_logic                     := 'X';             -- rdreq
            clock : in  std_logic                     := 'X';             -- clk
            q     : out std_logic_vector(31 downto 0);                    -- dataout
            usedw : out std_logic_vector(4 downto 0);                     -- usedw
            full  : out std_logic;                                        -- full
            empty : out std_logic                                         -- empty
        );
end component data_fifo;

component receiver_switching is
  port (
				clk_qsfp_clk                                    : in  std_logic                      := 'X'; -- clk
            reset_1_reset                                   : in  std_logic                      := 'X'; -- reset
            rx_bitslip_ch0_rx_bitslip                       : in  std_logic                      := 'X'; -- rx_bitslip
            rx_bitslip_ch1_rx_bitslip                       : in  std_logic                      := 'X'; -- rx_bitslip
            rx_bitslip_ch2_rx_bitslip                       : in  std_logic                      := 'X'; -- rx_bitslip
            rx_bitslip_ch3_rx_bitslip                       : in  std_logic                      := 'X'; -- rx_bitslip
            rx_cdr_refclk0_clk                              : in  std_logic                      := 'X'; -- clk
            rx_clkout_ch0_clk                               : out std_logic;                             -- clk
            rx_clkout_ch1_clk                               : out std_logic;                             -- clk
            rx_clkout_ch2_clk                               : out std_logic;                             -- clk
            rx_clkout_ch3_clk                               : out std_logic;                             -- clk
            rx_coreclkin_ch0_clk                            : in  std_logic                      := 'X'; -- clk
            rx_coreclkin_ch1_clk                            : in  std_logic                      := 'X'; -- clk
            rx_coreclkin_ch2_clk                            : in  std_logic                      := 'X'; -- clk
            rx_coreclkin_ch3_clk                            : in  std_logic                      := 'X'; -- clk
            rx_datak_ch0_rx_datak                           : out std_logic_vector(3 downto 0);          -- rx_datak
            rx_datak_ch1_rx_datak                           : out std_logic_vector(3 downto 0);          -- rx_datak
            rx_datak_ch2_rx_datak                           : out std_logic_vector(3 downto 0);          -- rx_datak
            rx_datak_ch3_rx_datak                           : out std_logic_vector(3 downto 0);          -- rx_datak
            rx_disperr_ch0_rx_disperr                       : out std_logic_vector(3 downto 0);          -- rx_disperr
            rx_errdetect_ch0_rx_errdetect                   : out std_logic_vector(3 downto 0);          -- rx_errdetect
            rx_is_lockedtoref_ch0_rx_is_lockedtoref         : out std_logic;                             -- rx_is_lockedtoref
            rx_is_lockedtoref_ch1_rx_is_lockedtoref         : out std_logic;                             -- rx_is_lockedtoref
            rx_is_lockedtoref_ch2_rx_is_lockedtoref         : out std_logic;                             -- rx_is_lockedtoref
            rx_is_lockedtoref_ch3_rx_is_lockedtoref         : out std_logic;                             -- rx_is_lockedtoref
            rx_parallel_data_ch0_rx_parallel_data           : out std_logic_vector(31 downto 0);         -- rx_parallel_data
            rx_parallel_data_ch1_rx_parallel_data           : out std_logic_vector(31 downto 0);         -- rx_parallel_data
            rx_parallel_data_ch2_rx_parallel_data           : out std_logic_vector(31 downto 0);         -- rx_parallel_data
            rx_parallel_data_ch3_rx_parallel_data           : out std_logic_vector(31 downto 0);         -- rx_parallel_data
            rx_patterndetect_ch0_rx_patterndetect           : out std_logic_vector(3 downto 0);          -- rx_patterndetect
            rx_patterndetect_ch1_rx_patterndetect           : out std_logic_vector(3 downto 0);          -- rx_patterndetect
            rx_ready0_rx_ready                              : out std_logic;                             -- rx_ready
            rx_ready1_rx_ready                              : out std_logic;                             -- rx_ready
            rx_ready2_rx_ready                              : out std_logic;                             -- rx_ready
            rx_ready3_rx_ready                              : out std_logic;                             -- rx_ready
            rx_serial_data_ch0_rx_serial_data               : in  std_logic                      := 'X'; -- rx_serial_data
            rx_serial_data_ch1_rx_serial_data               : in  std_logic                      := 'X'; -- rx_serial_data
            rx_serial_data_ch2_rx_serial_data               : in  std_logic                      := 'X'; -- rx_serial_data
            rx_serial_data_ch3_rx_serial_data               : in  std_logic                      := 'X'; -- rx_serial_data
            rx_seriallpbken_ch0_rx_seriallpbken             : in  std_logic                      := 'X'; -- rx_seriallpbken
            rx_seriallpbken_ch1_rx_seriallpbken             : in  std_logic                      := 'X'; -- rx_seriallpbken
            rx_seriallpbken_ch2_rx_seriallpbken             : in  std_logic                      := 'X'; -- rx_seriallpbken
            rx_seriallpbken_ch3_rx_seriallpbken             : in  std_logic                      := 'X'; -- rx_seriallpbken
            rx_syncstatus_ch0_rx_syncstatus                 : out std_logic_vector(3 downto 0);          -- rx_syncstatus
            rx_syncstatus_ch1_rx_syncstatus                 : out std_logic_vector(3 downto 0);          -- rx_syncstatus
            rx_errdetect_ch1_rx_errdetect                   : out std_logic_vector(3 downto 0);          -- rx_errdetect
            rx_disperr_ch1_rx_disperr                       : out std_logic_vector(3 downto 0);          -- rx_disperr
            rx_runningdisp_ch1_rx_runningdisp               : out std_logic_vector(3 downto 0);          -- rx_runningdisp
            rx_errdetect_ch2_rx_errdetect                   : out std_logic_vector(3 downto 0);          -- rx_errdetect
            rx_disperr_ch2_rx_disperr                       : out std_logic_vector(3 downto 0);          -- rx_disperr
            rx_runningdisp_ch2_rx_runningdisp               : out std_logic_vector(3 downto 0);          -- rx_runningdisp
            rx_patterndetect_ch2_rx_patterndetect           : out std_logic_vector(3 downto 0);          -- rx_patterndetect
            rx_syncstatus_ch2_rx_syncstatus                 : out std_logic_vector(3 downto 0);          -- rx_syncstatus
            rx_errdetect_ch3_rx_errdetect                   : out std_logic_vector(3 downto 0);          -- rx_errdetect
            rx_disperr_ch3_rx_disperr                       : out std_logic_vector(3 downto 0);          -- rx_disperr
            rx_runningdisp_ch3_rx_runningdisp               : out std_logic_vector(3 downto 0);          -- rx_runningdisp
            rx_patterndetect_ch3_rx_patterndetect           : out std_logic_vector(3 downto 0);          -- rx_patterndetect
            rx_syncstatus_ch3_rx_syncstatus                 : out std_logic_vector(3 downto 0);          -- rx_syncstatus
            unused_rx_parallel_data_unused_rx_parallel_data : out std_logic_vector(287 downto 0)         -- unused_rx_parallel_data
  );
end component receiver_switching;

component transceiver_switching is
	port (
		clk_qsfp_clk                          : in  std_logic                     := 'X';             -- clk
		pll_refclk0_clk                       : in  std_logic                     := 'X';             -- clk
		reset_1_reset                         : in  std_logic                     := 'X';             -- reset
		tx_datak_ch0_tx_datak                 : in  std_logic_vector(3 downto 0)  := (others => 'X'); -- tx_datak
		tx_datak_ch1_tx_datak                 : in  std_logic_vector(3 downto 0)  := (others => 'X'); -- tx_datak
		tx_datak_ch2_tx_datak                 : in  std_logic_vector(3 downto 0)  := (others => 'X'); -- tx_datak
		tx_datak_ch3_tx_datak                 : in  std_logic_vector(3 downto 0)  := (others => 'X'); -- tx_datak
		tx_parallel_data_ch0_tx_parallel_data : in  std_logic_vector(31 downto 0) := (others => 'X'); -- tx_parallel_data
		tx_parallel_data_ch1_tx_parallel_data : in  std_logic_vector(31 downto 0) := (others => 'X'); -- tx_parallel_data
		tx_parallel_data_ch2_tx_parallel_data : in  std_logic_vector(31 downto 0) := (others => 'X'); -- tx_parallel_data
		tx_parallel_data_ch3_tx_parallel_data : in  std_logic_vector(31 downto 0) := (others => 'X'); -- tx_parallel_data
		tx_ready_tx_ready                     : out std_logic_vector(3 downto 0);                     -- tx_ready
		tx_serial_data_ch0_tx_serial_data     : out std_logic;                                        -- tx_serial_data
		tx_serial_data_ch1_tx_serial_data     : out std_logic;                                        -- tx_serial_data
		tx_serial_data_ch2_tx_serial_data     : out std_logic;                                        -- tx_serial_data
		tx_serial_data_ch3_tx_serial_data     : out std_logic;                                         -- tx_serial_data
		tx_coreclkin_ch0_clk                  : in  std_logic                     := 'X';             -- clk
		tx_coreclkin_ch1_clk                  : in  std_logic                     := 'X';             -- clk
		tx_coreclkin_ch2_clk                  : in  std_logic                     := 'X';             -- clk
		tx_coreclkin_ch3_clk                  : in  std_logic                     := 'X';             -- clk
		tx_clkout_ch0_clk                     : out std_logic;                                        -- clk
		tx_clkout_ch1_clk                     : out std_logic;                                        -- clk
		tx_clkout_ch2_clk                     : out std_logic;                                        -- clk
		tx_clkout_ch3_clk                     : out std_logic  
	);
end component transceiver_switching;

component ip_receiving_fifo is
  port (
		data    : in  std_logic_vector(31 downto 0) := (others => 'X'); -- datain
		wrreq   : in  std_logic                     := 'X';             -- wrreq
		rdreq   : in  std_logic                     := 'X';             -- rdreq
		wrclk   : in  std_logic                     := 'X';             -- wrclk
		rdclk   : in  std_logic                     := 'X';             -- rdclk
		q       : out std_logic_vector(31 downto 0);                    -- dataout
		rdempty : out std_logic;                                        -- rdempty
		aclr	  : in  std_logic;
		wrfull  : out std_logic                                         -- wrfull
  );
end component ip_receiving_fifo;

component ip_fifodataoutpll is
  port (
		rst      : in  std_logic := 'X'; -- reset
		refclk   : in  std_logic := 'X'; -- clk
		locked   : out std_logic;        -- export
		outclk_0 : out std_logic        -- 156.25 clk
  );
end component ip_fifodataoutpll;

component seven_segment is
  port (
    clk     	 : 	in 	std_logic;
    reset     	 :   	in 	std_logic;
    HEX0_D      :   	out 	std_logic_vector(6 downto 0);
    HEX0_DP 	 : 	out 	std_logic;
	 HEX1_D      :   	out 	std_logic_vector(6 downto 0);
    HEX1_DP 	 : 	out 	std_logic
    );
end component seven_segment;

component data_scan is	
	generic (
		constant state_length  : positive := 50
	);
	port(
		clk 					: 	in std_logic;
		reset_n				:	in std_logic;
		rx_parallel_data 	: 	in std_logic_vector(31 downto 0);
		count_up				:  in std_logic;
		SMA_CLKOUT 			: 	out std_logic
		);
end component data_scan;

component linear_shift is 
	generic (
		G_M             : integer           := 7;
		G_POLY          : std_logic_vector  := "1100000" -- x^7+x^6+1 
	);
	port (
		i_clk           : in  std_logic;
		reset_n         : in  std_logic;
		i_sync_reset    : in  std_logic;
		i_seed          : in  std_logic_vector (G_M-1 downto 0);
		i_en            : in  std_logic;
		o_lsfr          : out std_logic_vector (G_M-1 downto 0)
	);
end component linear_shift;

component seg7_lut is
    port (
        clk : in  std_logic;
        hex : in  std_logic_vector(3 downto 0);
        seg : out std_logic_vector(6 downto 0)--;
    );
end component seg7_lut;


component i2c_nios is
  port (
		clk_clk                          : in    std_logic                    := 'X'; -- clk
		reset_reset_n                    : in    std_logic                    := 'X'; -- reset_n
		pio_0_external_connection_export : out   std_logic_vector(9 downto 0);        -- export
		i2c_scl_conduit_export           : out   std_logic;                           -- export
		i2c_sda_conduit_export           : inout std_logic                    := 'X'  -- export
  );
end component i2c_nios;
 
end package mudaq_components;
