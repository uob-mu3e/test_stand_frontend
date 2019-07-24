library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;


ENTITY mscb is 
    PORT(
			nios_clk:							in  std_logic;
			reset:								in  std_logic;
			mscb_to_nios_parallel_in:		out std_logic_vector(11 downto 0);
			mscb_from_nios_parallel_out:  in  std_logic_vector(11 downto 0);
			mscb_data_in:						in  std_logic;
			mscb_data_out:						out std_logic;
			mscb_oe:								out std_logic;
			mscb_counter_in: 					out unsigned(15 downto 0)
    );
END ENTITY mscb;

architecture rtl of mscb is

  COMPONENT uart_reciever is
  generic ( Clk_Ratio : integer := 347 );
    PORT(
     Clk:				in  std_logic;
     Reset:				in  std_logic; -- here asynchronous reset
     DataIn:			in  std_logic; -- enable
     ReadEnable:		out  std_logic; -- down when 1, else up
	  DataOut:			out std_logic_vector(8 downto 0)
     );
   END COMPONENT;
	
  component uart_transmitter is
  generic ( Clk_Ratio : integer := 347);
  port(
     Clk:			in  std_logic;
     Reset:			in  std_logic; 
     DataIn:		in  std_logic_vector(8 downto 0);
     DataReady:	in std_logic; -- data is ready to be read
     ReadRequest:	out  std_logic; 
     DataOut:		out std_logic;
	  Transmitting:out std_logic
  );
  end component;
	
  component MSCBDataFIFO
	PORT
	(
	  sclr		: IN STD_LOGIC ;
	  clock		: IN STD_LOGIC ;
	  data		: IN STD_LOGIC_VECTOR (8 DOWNTO 0);
	  rdreq		: IN STD_LOGIC ;
	  wrreq		: IN STD_LOGIC ;
	  empty		: OUT STD_LOGIC ;
	  full		: OUT STD_LOGIC ;
	  q		: OUT STD_LOGIC_VECTOR (8 DOWNTO 0);
	  usedw		: OUT STD_LOGIC_VECTOR (7 DOWNTO 0)
  );
  end component;

  
  component edge_detector
  Port(
    clk : in  STD_LOGIC;
    signal_in : in  STD_LOGIC;
    output    : out  STD_LOGIC
  );
  end component;
  
  component slow_counter is
  generic ( Clk_Ratio : integer := 100 );
  port(
     Clk	: in  std_logic;
     Reset		: in  std_logic; -- here asynchronous reset
     Enable		: in  std_logic; -- enable
     CountDown	: in  std_logic; -- down when 1, else up
     CounterOut	: out unsigned(15 downto 0);
     Init : in unsigned(15 downto 0)
  );
  end component;
	
	
------------------ Signal declaration ------------------------
  
  -- external connections
  signal clk : std_logic;

  -- nios io's
  --signal mscb_parallel_in: std_logic_vector(11 downto 0);
  --signal mscb_parallel_out: std_logic_vector(11 downto 0);
  --signal mscb_counter_in : unsigned(15 downto 0);
  --signal serial_mc_in : std_logic ;
  --signal serial_mc_out : std_logic ;
  
  
  -- mscb data flow
  signal uart_generated_data : std_logic;
  signal signal_in : std_logic;
  signal signal_out : std_logic;
  signal uart_serial_in : std_logic;
  signal uart_read_enable : std_logic;
  signal uart_parallel_out : std_logic_vector(8 downto 0);
  
  signal in_fifo_read_request : std_logic;
  signal in_fifo_write_request : std_logic;
  signal in_fifo_empty : std_logic;
  signal in_fifo_full : std_logic;
  signal in_fifo_data_out : std_logic_vector(8 downto 0);
  signal in_fifo_size_queue : std_logic_vector(7 downto 0);
  
  signal out_fifo_read_request : std_logic;
  signal out_fifo_write_request : std_logic;
  signal out_fifo_empty : std_logic;
  signal out_fifo_full : std_logic;
  signal out_fifo_data_out : std_logic_vector(8 downto 0);
  signal out_fifo_size_queue : std_logic_vector(7 downto 0);
  
  signal DataReady : std_logic;
  
  signal mscb_nios_out : std_logic_vector(8 downto 0);
  signal mscb_data_ready : std_logic;
  
  signal DataGeneratorEnable : std_logic;
  signal uart_serial_out : std_logic; --uart data for the output pin
  
  signal Transmitting  : std_logic; -- uart data is being send
  signal dummy : std_logic;
  
  --fpga status bit;
  signal fpga_status : std_logic_vector(31 downto 0);
  signal fpga_status2 : std_logic_vector(15 downto 0);
  signal fpga_setting : std_logic_vector(7 downto 0);
  
  
  
----------------------------------------
------------ Begin top level------------
----------------------------------------
  
begin

--------- external input connections -----------

  clk <= nios_clk;
------------- external input/output connections -----------------

  --i/o switches
  process(Transmitting,reset,Clk)
  begin
    if (Transmitting = '1') then
		  mscb_data_out <= uart_serial_out;
		  mscb_oe <= '1';
		  
    else
		  mscb_data_out <= 'Z';
		  --mscb_oe <= '0';  	-- single FPGA connected to converter chip
		  mscb_oe <= 'Z';  		-- multiple FPGAs 
    end if;
  end process;
  
  --hsma_d(0) <= 'Z' when Transmitting = '1' else 'Z';
  signal_in <= '1' when Transmitting = '1' else (mscb_data_in);
  
  ---------------internal connections-----------------------
  
  
  
  ---- parallel in for the nios ----
  mscb_to_nios_parallel_in(8 downto 0) <= in_fifo_data_out; -- 8+1 bit mscb words
  mscb_to_nios_parallel_in(9) <= in_fifo_empty;
  mscb_to_nios_parallel_in(10) <= in_fifo_full;
  mscb_to_nios_parallel_in(11) <= '1';

  mscb_nios_out <= mscb_from_nios_parallel_out(8 downto 0);
 

  DataReady <= not out_fifo_empty;

------------- Wire up components --------------------
  
  c1 : slow_counter generic map ( Clk_Ratio => 100 ) 
  port map(
    Clk	=> clk,
    Reset => reset,
    Enable => '1',
    CountDown => '0',
    CounterOut	=> mscb_counter_in,
    Init => to_unsigned(0,16)
  );
  
  -- wire up uart reciever for mscb
  u1 : uart_reciever generic map (Clk_Ratio => 1085 ) port map(
     Clk	=> clk,
     Reset => reset,
     DataIn => uart_serial_in,
     ReadEnable => uart_read_enable,
	  DataOut => uart_parallel_out
  );
  
  uDataIn: MSCBDataFIFO
  PORT MAP
  (
    sclr => reset,
	 clock => clk,
	 data => uart_parallel_out,
	 rdreq => in_fifo_read_request,
	 wrreq => in_fifo_write_request,
	 empty => in_fifo_empty,
	 full => in_fifo_full,
	 q => in_fifo_data_out,
	 usedw	=> in_fifo_size_queue
  );
  
  
  -- wire up uart transmitter for mscb
  u2 : uart_transmitter generic map (Clk_Ratio => 1085 ) port map(
     Clk	=> clk,
     Reset => reset,
     DataIn => out_fifo_data_out,
	  DataReady => DataReady,
     ReadRequest => out_fifo_read_request,
	  DataOut => uart_serial_out,
	  Transmitting => Transmitting
  );
  
  uDataOut: MSCBDataFIFO
  PORT MAP
  (
    sclr => reset,
	 clock => clk,
	 data => mscb_nios_out, 
	 rdreq => out_fifo_read_request,
	 wrreq => out_fifo_write_request,
	 empty => out_fifo_empty,
	 full => out_fifo_full,
	 q => out_fifo_data_out,
	 usedw	=> out_fifo_size_queue
  );


	uEdgeFIFORead : edge_detector -- make the fifo read request from the MC one clocktick long
	port map
	(
	  clk => clk,
     signal_in => mscb_from_nios_parallel_out(10),
     output => in_fifo_read_request
	);
	
	uEdgeFIFOWrite : edge_detector -- make the fifo read request from the MC one clocktick long
	port map
	(
	  clk => clk,
     signal_in => mscb_from_nios_parallel_out(9),
     output => out_fifo_write_request
	);
  

  process(Clk, reset)
    begin
    if reset = '1' then
		DataGeneratorEnable <= '0';
		--uart_serial_in<='1';
		--hsma_d(0)<='Z';
    elsif rising_edge(clk) then
	   DataGeneratorEnable <= '1';
		in_fifo_write_request <= uart_read_enable;
		uart_serial_in <= signal_in;
	 		
	end if;
	 
  end process;

END rtl;

  --	component MC is
--		port (
--			clk_clk                   : in    std_logic                     := 'X';             -- clk
--			counter_in_export         : in    std_logic_vector(15 downto 0) := (others => 'X'); -- export
--			fpga_setting_export       : out    std_logic_vector(7 downto 0)  := (others => 'X'); -- export
--			fpga_status_export        : in    std_logic_vector(31 downto 0)  := (others => 'X'); -- export
--			lcd_16207_0_external_RS   : out   std_logic;                                        -- RS
--			lcd_16207_0_external_RW   : out   std_logic;                                        -- RW
--			lcd_16207_0_external_data : inout std_logic_vector(7 downto 0)  := (others => 'X'); -- data
--			lcd_16207_0_external_E    : out   std_logic;                                        -- E
--			parallel1_export          : out   std_logic_vector(11 downto 0);                    -- export
--			parallel2_export          : in    std_logic_vector(11 downto 0) := (others => 'X'); -- export
--			reset_reset_n             : in    std_logic                     := 'X'              -- reset_n
--		);
--	end component MC;

--  u0 : MC  port map (
--    clk_clk => clk,
--	 counter_in_export => std_logic_vector(mscb_counter_in),
--	 fpga_setting_export => fpga_setting,
--	 fpga_status_export =>fpga_status,
--	 lcd_16207_0_external_RS  =>  open,        
--	 lcd_16207_0_external_RW  =>  open,                                 
--	 lcd_16207_0_external_data => open,              
--	 lcd_16207_0_external_E => open,
--	 parallel1_export => mscb_parallel_out,
--	 parallel2_export => mscb_parallel_in,
--	 reset_reset_n => not reset
--  );