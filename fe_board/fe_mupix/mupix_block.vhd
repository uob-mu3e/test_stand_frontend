library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mupix_block is
generic(
    NCHIPS : integer := 4--,
);
port (
   
    -- chip dacs
    i_CTRL_SDO_A    : in std_logic;
    o_CTRL_SDI_A    : out std_logic;
    o_CTRL_SCK1_A   : out std_logic;
    o_CTRL_SCK2_A   : out std_logic;
    o_CTRL_Load_A   : out std_logic;
    o_CTRL_RB_A     : out std_logic;
    i_data_chip_dacs : in std_logic_vector(31 downto 0);
    o_add_chip_dacs : out std_logic_vector(15 downto 0);
    
    
    
    -- board dacs
    i_SPI_DOUT_ADC_0_A      : in std_logic;
    o_SPI_DIN0_A            : out std_logic;
    o_SPI_CLK_A             : out std_logic;
    o_SPI_LD_ADC_A          : out std_logic;
    o_SPI_LD_TEMP_DAC_A     : out std_logic;
    o_SPI_LD_DAC_A          : out std_logic;
    
	 
	 
	 -- mupix dac regs
	 i_reg_add               : in std_logic_vector(7 downto 0);
	 i_reg_re                : in std_logic;
	 o_reg_rdata       		 : out std_logic_vector(31 downto 0);
	 i_reg_we   				 : in std_logic;
	 i_reg_wdata 				 : in std_logic_vector(31 downto 0);
    
	 
	 
    i_ckdiv         : in std_logic_vector(15 downto 0);

    i_reset         : in std_logic;
    -- 156.25 MHz
    i_clk           : in std_logic;
    i_clk125        : in std_logic--;
);
end entity;

architecture arch of mupix_block is

     signal reset_n : std_logic;
    
    -- chip dacs
	 signal mp8_busy_n : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_mem_data_out : std_logic_vector(31 downto 0);
	 signal mp8_wren : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_ld : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_rb : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_ctrl_dout : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_ctrl_din : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_ctrl_clk1 : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_ctrl_clk2 : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_ctrl_ld : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_ctrl_rb : std_logic_vector(NCHIPS - 1 downto 0);
	 signal mp8_dataout : std_logic_vector(NCHIPS*32 - 1 downto 0);
	 
	 -- board dacs
	 type state_spi is (waiting, starting, read_out_pix, write_pix, read_out_th, ending);
    signal spi_state : state_spi;
    signal A_spi_wren_front : std_logic_vector(2 downto 0);
    signal A_spi_busy_n_front : std_logic;
    signal A_spi_sdo_front : std_logic_vector(2 downto 0);
    signal A_spi_ldn_front : std_logic_vector(2 downto 0);
    signal threshold_low_out_A_front : std_logic_vector(15 downto 0);
    signal threshold_high_out_A_front : std_logic_vector(15 downto 0);
    signal injection1_out_A_front : std_logic_vector(15 downto 0);
    signal threshold_pix_out_A_front : std_logic_vector(15 downto 0);
    signal board_th_low : std_logic_vector(15 downto 0);
    signal board_th_high : std_logic_vector(15 downto 0);
    signal board_injection : std_logic_vector(15 downto 0);
    signal board_th_pix : std_logic_vector(15 downto 0);
    signal board_temp_dac : std_logic_vector(15 downto 0);
    signal board_temp_adc : std_logic_vector(15 downto 0);
    signal board_temp_dac_out : std_logic_vector(15 downto 0);
    signal board_temp_adc_out : std_logic_vector(31 downto 0);
    
    signal board_dac_data_we : std_logic_vector(31 downto 0);
    signal chip_dac_data_we : std_logic_vector(31 downto 0);
    signal board_dac_we : std_logic;
    signal chip_dac_we : std_logic;

    signal board_dac_data : std_logic_vector(31 downto 0);
	signal board_dac_ren : std_logic;
	signal board_dac_fifo_empty : std_logic;
	signal board_dac_ready : std_logic;

	signal chip_dac_data : std_logic_vector(31 downto 0);
	signal chip_dac_ren : std_logic;
	signal chip_dac_fifo_empty : std_logic;
	signal chip_dac_ready : std_logic;
    

begin

    reset_n <= not i_reset;
    
    -- chip dacs slow_controll
    e_mp8_sc_master : work.mp8_sc_master
	 generic map(NCHIPS => NCHIPS)
	 port map (
		  clk			   => i_clk,
		  reset_n		=> reset_n,
	 	  mem_data_in	=> i_data_chip_dacs,
		  busy_n		   => mp8_busy_n,
		
		  mem_addr		=> o_add_chip_dacs,
		  mem_data_out	=> mp8_mem_data_out,
		  wren			=> mp8_wren,
		  ctrl_ld		=> mp8_ld,
		  ctrl_rb		=> mp8_rb,
		  done			=> open, 
		  stateout		=> open--,
	 );
	 
    gen_slowc:
	 for i in 0 to NCHIPS-1 generate
	 e_mp8_slowcontrol : work.mp8_slowcontrol
	 port map(
		  clk			=> i_clk,
		  reset_n	=> reset_n,
		  ckdiv		=> i_ckdiv, -- this need to be set to a register at the moment 0
		  mem_data	=> mp8_mem_data_out,
		  wren		=> mp8_wren(i),
		  ld_in		=> mp8_ld(i),
		  rb_in		=> mp8_rb(i),
		  ctrl_dout	=> mp8_ctrl_dout(i),
		  ctrl_din	=> mp8_ctrl_din(i),
		  ctrl_clk1	=> mp8_ctrl_clk1(i),
		  ctrl_clk2	=> mp8_ctrl_clk2(i),
		  ctrl_ld	=> mp8_ctrl_ld(i),
		  ctrl_rb	=> mp8_ctrl_rb(i),
		  busy_n		=> mp8_busy_n(i),
		  dataout	=> mp8_dataout--,
	 );	
	 end generate gen_slowc;
	 
    process(i_clk)
	 begin
		if(rising_edge(i_clk)) then	
			mp8_ctrl_dout(0)	<= i_CTRL_SDO_A;
		end if;
	 end process;
	 
	 process(i_clk)
	 begin
		if(rising_edge(i_clk)) then	
			o_CTRL_SDI_A	<= mp8_ctrl_din(0);
			o_CTRL_SCK1_A	<= mp8_ctrl_clk1(0);
			o_CTRL_SCK2_A	<= mp8_ctrl_clk2(0);
			o_CTRL_Load_A	<= mp8_ctrl_ld(0);
			o_CTRL_RB_A		<= mp8_ctrl_rb(0);
		end if;
	 end process;
	 
	 
	 -- board dacs slow_controll
	 A_spi_sdo_front 		<= i_SPI_DOUT_ADC_0_A & "00";-- A_spi_dout_dac_front & A_dac4_dout_front;
	 o_SPI_LD_ADC_A 		<= A_spi_ldn_front(2);
    o_SPI_LD_TEMP_DAC_A <= A_spi_ldn_front(1);
	 o_SPI_LD_DAC_A 		<= A_spi_ldn_front(0);
     
     e_dac_fifo : work.dac_fifo
		port map (

		   	-- mupix dac regs
			i_reg_add         => i_reg_add,
			i_reg_re          => i_reg_re,
			o_reg_rdata       => o_reg_rdata,
			i_reg_we   		  => i_reg_we,
			i_reg_wdata 	  => i_reg_wdata,

		    -- mupix board dac data
		    o_board_dac_data  		=> board_dac_data,
		    i_board_dac_ren   		=> board_dac_ren,
		    o_board_dac_fifo_empty 	=> board_dac_fifo_empty,
		    o_board_dac_ready 		=> board_dac_ready,

		    i_board_dac_data  		=> board_dac_data_we,
		    i_board_dac_we  		=> board_dac_we,

		    -- mupix chip dac data
		    o_chip_dac_data  		=> chip_dac_data,
		    i_chip_dac_ren   		=> chip_dac_ren,
		    o_chip_dac_fifo_empty 	=> chip_dac_fifo_empty,
		    o_chip_dac_ready 		=> chip_dac_ready,

		    i_chip_dac_data 		=> chip_dac_data_we,
    		i_chip_dac_we  			=> chip_dac_we,
		    
		    i_reset_n         		=> reset_n,
		    -- 156.25 MHz
		    i_clk           		=> i_clk--,
	);

    process(i_clk, reset_n) -- handle dac fifo 
	begin
		if(reset_n = '0') then
	        board_dac_data_we       <= (others => '0');
	        board_dac_we   			<= '0';
	        board_dac_ren 			<= '0';
	        A_spi_wren_front        <= (others => '0');
	        spi_state <= waiting;
		elsif(rising_edge(i_clk)) then
	        board_dac_we   		<= '0';
	        board_dac_ren  		<= '0';
	        board_dac_data_we 	<= (others => '0');
	        A_spi_wren_front    <= (others => '0');
			
		case spi_state is
			when waiting =>
				if(board_dac_ready = '1') then
					board_dac_ren 	<= '1';
					spi_state       <= starting;
				end if;
                                      
			when starting =>
				board_dac_ren 		<= '1';
            	board_th_low        <= board_dac_data(15 downto 0);
            	board_th_high       <= board_dac_data(31 downto 16);
            	spi_state           <= write_pix;	
		
			when write_pix =>
            	A_spi_wren_front   	<= "001";
            	board_injection    	<= board_dac_data(15 downto 0);
            	board_th_pix       	<= board_dac_data(31 downto 16); 
            	spi_state          	<= read_out_th;		
	
			when read_out_th =>
	            board_dac_we              			<= '1';
	            board_dac_data_we(15 downto 0)  	<= threshold_low_out_A_front;
	            board_dac_data_we(31 downto 16)    	<= threshold_high_out_A_front;
	            spi_state <= read_out_pix;

			when read_out_pix =>
	            board_dac_we 						<= '1';
	            board_dac_data_we(15 downto 0)     	<= injection1_out_A_front;
	            board_dac_data_we(31 downto 16)    	<= threshold_pix_out_A_front;
	            spi_state <= waiting;
				
			when others =>
	            spi_state               <= waiting;
			
        end case;
			
    end if;
    end process;
	 
	 e_spi_master : work.spi_master 
	 port map(
		clk                   => i_clk,
		reset_n               => reset_n,
		injection1_reg        => board_injection,
		threshold_pix_reg     => board_th_pix,
		threshold_low_reg	    => board_th_low,
		threshold_high_reg    => board_th_high,
		temp_dac_reg		    => board_temp_dac,
		temp_adc_reg		    => board_temp_adc,	
		wren                  => A_spi_wren_front,
		busy_n                => A_spi_busy_n_front,
		spi_sdi               => o_SPI_DIN0_A,
		spi_sclk              => o_SPI_CLK_A,
		spi_load_n            => A_spi_ldn_front,
		
		spi_sdo               => A_spi_sdo_front,
		injection1_out        => injection1_out_A_front,
		threshold_pix_out     => threshold_pix_out_A_front,
		threshold_low_out     => threshold_low_out_A_front,
		threshold_high_out    => threshold_high_out_A_front,
		temp_dac_out          => board_temp_dac_out,
		temp_adc_out          => board_temp_adc_out
	 );	 

end architecture;
