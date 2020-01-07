library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;


--  A testbench has no ports.
entity dma_test_tb is
end entity;

architecture behav of dma_test_tb is
  --  Declaration of the component that will be instantiated.

	component dma_counter is
		port(
			i_clk 			: 	in STD_LOGIC;
			i_reset_n		:	in std_logic;
			i_enable		:	in std_logic;
			i_dma_wen_reg	:	in std_logic;
			i_fraccount     :	in std_logic_vector(7 downto 0);
			i_halffull_mode	:	in std_logic;
			i_dma_halffull	:	in std_logic;
			o_dma_end_event		:	out std_logic;
			o_dma_wen		:	out std_logic;
			o_cnt 			: 	out std_logic_vector (95 downto 0)--;
			);
	end component dma_counter;

    component dma_engine is
    generic (
        MEMWRITEADDRSIZE : integer := 13;
        MEMREADADDRSIZE  : integer := 11;
        MEMWRITEWIDTH     : integer := 64;
        IRQNUM          : std_logic_vector(4 downto 0) := "00000";
        ENABLE_BIT      : std_logic;
        NOW_BIT             : std_logic;
        ENABLE_INTERRUPT_BIT : std_logic
    );
    port (
        local_rstn:             in      std_logic;
        refclk:                 in      std_logic;

        -- Stuff for DMA writing
        dataclk:                    in      std_logic;
        datain:                 in  std_logic_vector(MEMWRITEWIDTH-1 downto 0);
        datawren:               in      std_logic;
        endofevent:             in  std_logic;
        memhalffull:            out std_logic;

        -- Bus and device number
        cfg_busdev:             in      std_logic_vector(12 downto 0);

        -- Comunication with completer
        dma_request:            out std_logic;
        dma_granted:            in      std_logic;
        dma_done:               out std_logic;
        tx_ready:               in      std_logic;
        tx_data:                    out std_logic_vector(255 downto 0);
        tx_valid:               out std_logic;
        tx_sop:                 out std_logic;
        tx_eop:                 out std_logic;
        tx_empty:               out std_logic_vector(1 downto 0);

        -- Interrupt stuff
        app_msi_req:            out std_logic;
        app_msi_tc:             out     std_logic_vector(2 downto 0);
        app_msi_num:            out std_logic_vector(4 downto 0);
        app_msi_ack:            in      std_logic;

        -- Configuration register
        dma_control_address:            in  std_logic_vector(63 downto 0);
        dma_data_address:           in  std_logic_vector(63 downto 0);
        dma_data_address_out:       out     std_logic_vector(63 downto 0);
        dma_data_mem_addr:          in  std_logic_vector(11 downto 0);
        dma_addrmem_data_written:   in  std_logic;
        dma_register:                   in    std_logic_vector(31 downto 0);
        dma_register_written:       in      std_logic;
        dma_data_pages:             in  std_logic_vector(19 downto 0);
        dma_data_pages_out:         out std_logic_vector(19 downto 0);
        dma_data_n_addrs:               in  std_logic_vector(11 downto 0);

        dma_status_register:        out std_logic_vector(31 downto 0);

        test_out:                       out std_logic_vector(71 downto 0)
    );
    end component dma_engine;

	signal clk 			: std_logic; -- 250 MHz
  	signal reset_n 		: std_logic := '1';
  	signal enable 		: std_logic := '0';
  	signal fraccount 	: std_logic_vector(7 downto 0);
  	signal dma_wen 		: std_logic;
  	signal dma_wen_reg  : std_logic;
  	signal halffull_mode  : std_logic := '0';
	signal dma_halffull  : std_logic := '0'; 
  	signal cnt 			: std_logic_vector(95 downto 0);
  	signal dma_end_event : std_logic;

    signal ENABLE_BIT : std_logic;
    signal DMA_BIT_NOW : std_logic;
    signal DMA_BIT_ENABLE_INTERRUPTS : std_logic;

    signal PCIE_REFCLK_p : std_logic; --100MHz
    signal datain : std_logic_vector(255 downto 0);

    signal dma_request : std_logic;
    signal dma_granted : std_logic;

    signal dma_done : std_logic;
    signal x_st_ready0 : std_logic;
    signal dma_tx_data : std_logic_vector(255 downto 0);
    signal dma_tx_valid : std_logic;
    signal dma_tx_sop : std_logic;
    signal dma_tx_eop : std_logic;
    signal dma_tx_empty : std_logic_vector(1 downto 0);

    -- Interrupt stuff
    signal app1_msi_req : std_logic;
    signal app1_msi_tc : std_logic_vector(2 downto 0);
    signal app1_msi_num : std_logic_vector(4 downto 0);
    signal app1_msi_ack : std_logic;

    -- Configuration register
    signal dma_control_address : std_logic_vector(63 downto 0);
    signal dma_data_address : std_logic_vector(63 downto 0);
    signal dma_data_address_out : std_logic_vector(63 downto 0);
    signal dma_data_mem_addr : std_logic_vector(11 downto 0);
    signal dma_addrmem_data_written : std_logic;
    signal dma_data_pages : std_logic_vector(19 downto 0);      
    signal dma_data_pages_out : std_logic_vector(19 downto 0);
    signal dma_data_n_addrs : std_logic_vector(11 downto 0);

    signal dma_register : std_logic_vector(31 downto 0);
    signal dma_register_written :  std_logic;
    signal dma_status_register : std_logic_vector(31 downto 0);
    signal test_out : std_logic_vector(71 downto 0);

  	constant ckTime		: time := 10 ns;

begin

	-- generate the clock
	ckProc: process
	begin
	   clk <= '0';
	   wait for ckTime/2;
	   clk <= '1';
	   wait for ckTime/2;
	end process;

    ckProc2: process
    begin
       PCIE_REFCLK_p <= '0';
       wait for ckTime;
       PCIE_REFCLK_p <= '1';
       wait for ckTime;
    end process;

	inita : process
	begin
	   reset_n	 <= '0';
	   wait for 8 ns;
	   reset_n	 <= '1';
	   wait for 20 ns;
       ENABLE_BIT <= '1';
	   enable    <= '1';
	   wait for 100 ns;
	   enable    <= '0';
       ENABLE_BIT <= '0';
	   wait for 50 ns;
	   enable    <= '1';
       ENABLE_BIT <= '1';

	   wait for 100 ns;
	   halffull_mode <= '1';
	   

	   wait;
	end process inita;

	fraccount <= x"FF";
	dma_wen_reg <= '1';
    DMA_BIT_ENABLE_INTERRUPTS <= '0';

	e_counter : component dma_counter
    port map (
		i_clk			=> clk,
		i_reset_n   	=> reset_n,
		i_enable    	=> enable,
		i_dma_wen_reg 	=> dma_wen_reg,
		i_fraccount 	=> fraccount,
		i_halffull_mode => halffull_mode,
		i_dma_halffull 	=> dma_halffull,
		o_dma_end_event => dma_end_event,
		o_dma_wen   	=> dma_wen,
		o_cnt     		=> cnt--,
	);

    datain(95 downto 0) <= cnt;
    datain(255 downto 96) <= (others => '0');

    -- dma comunication
    process(PCIE_REFCLK_p, reset_n)
    begin
        if(reset_n = '0') then
            dma_granted <= '0';
        elsif(PCIE_REFCLK_p'event and PCIE_REFCLK_p = '1') then
            if (dma_request = '1') then
                dma_granted <= '1';
            else
                dma_granted <= '0';
            end if;        
        end if;
    end process;

    
    dmaengine:dma_engine 
    generic map(
            MEMWRITEADDRSIZE => 11,
            MEMREADADDRSIZE  => 11,
            MEMWRITEWIDTH     => 256,
            IRQNUM            => "00000",
            ENABLE_BIT        => ENABLE_BIT,
            NOW_BIT           => DMA_BIT_NOW,
            ENABLE_INTERRUPT_BIT => DMA_BIT_ENABLE_INTERRUPTS
    )
    
    port map(
        local_rstn              => '1',
        refclk                  => PCIE_REFCLK_p,
        
        -- Stuff for DMA writing
        dataclk                 => clk,
        datain                  => datain,
        datawren                => dma_wen,
        endofevent              => dma_end_event,
        memhalffull             => dma_halffull,
        
                -- Bus and device number
        cfg_busdev              => (others => '0'),
        
        -- Comunication with completer
        dma_request             => dma_request,
        dma_granted             => dma_granted,
        dma_done                => dma_done,
        tx_ready                => x_st_ready0,
        tx_data                 => dma_tx_data,
        tx_valid                => dma_tx_valid,
        tx_sop                  => dma_tx_sop,
        tx_eop                  => dma_tx_eop,
        tx_empty                => dma_tx_empty,
        
        -- Interrupt stuff
        app_msi_req             => app1_msi_req,
        app_msi_tc              => app1_msi_tc,
        app_msi_num             => app1_msi_num,
        app_msi_ack             => app1_msi_ack,
        
        -- Configuration register
        dma_control_address             => dma_control_address,
        dma_data_address                => dma_data_address,
        dma_data_address_out            => dma_data_address_out,
        dma_data_mem_addr               => dma_data_mem_addr,
        dma_addrmem_data_written        => dma_addrmem_data_written,
        dma_data_pages                  => dma_data_pages,      
        dma_data_pages_out              => dma_data_pages_out,
        dma_data_n_addrs                => dma_data_n_addrs,
        
        dma_register                    => dma_register,
        dma_register_written            => dma_register_written,
        dma_status_register             => dma_status_register,
        test_out                        => test_out--,
        
        );

end architecture;
