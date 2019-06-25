library ieee;
use ieee.std_logic_1164.all;
use work.pcie_components.all;
--use work.gbt_components.all;

package mutrig_components is
component mutrig_slow is
	generic(
		C_NUM_SPI_CLIENTS	: integer := 1; 
		C_HANDLEID_WIDTH	: integer := 1;
		C_SPI_NUM_WORDS		: integer; 			-- THE TOTAL NUMBER OF WORDS TO READ FROM THE MEMORY
		C_SPI_FIRST_MSB		: integer			-- THE MSB OF THE FIRST WORD TO WRITE
	);
	Port (
		i_clk		: in  std_logic; -- (100 MHz)
		i_rst		: in  std_logic;
		i_start_spi	: in  std_logic;
		i_start_i2c	: in  std_logic; -- config via i2c of si3544
		o_done		: out std_logic;
		o_timeout	: out std_logic;
		i_handleid	: in  std_logic_vector(C_HANDLEID_WIDTH-1 downto 0);
		o_data		: out std_logic_vector(31 downto 0);
		i_data		: in  std_logic_vector(31 downto 0);
		o_wraddr		: out std_logic_vector(15 downto 0);
		o_rdaddr		: out std_logic_vector(15 downto 0);
		o_wren		: out std_logic;
		-- spi signals
		o_mdo		: out std_logic;                                    -- common output to all connected spi clients
		o_sclk		: out std_logic;                                    -- common clock  to all connected spi clients
		i_mdi		: in  std_logic_vector(C_NUM_SPI_CLIENTS-1 downto 0);	-- individual spi slave input
		o_cs		: out std_logic_vector(C_NUM_SPI_CLIENTS-1 downto 0);   -- individual chip select signals

		-- i2c signals
		io_scl		: inout  std_logic;
		io_sda		: inout  std_logic
	);
end component;

component mutrig_reset is
port(
	clk_pci            : in  std_logic;                       -- fast pcie clock 250MHz
	clk_base           : in  std_logic;                       -- 125MHz system clock, use 625 later?
	clk_slow           : in  std_logic;                       -- slow controle clock
	reset_n            : in  std_logic;                       -- 
	reset_reg          : in  std_logic_vector(31 downto 0);   -- reset register
	reset_reg_written  : in  std_logic;                       -- 
	o_reset_slow       : out std_logic_vector(31 downto 0);   -- resets sync to clk_slow (50MHz)
	o_reset_base       : out std_logic_vector(31 downto 0);   -- resets sync to clk_base (125MHz)
	o_reset_ch         : out std_logic;                       -- active high, (forces cc to 0)
	o_reset_chip       : out std_logic                        -- active high (1 cyle -> reset_ch, 
);
end component;

component spi_config_dummy is
generic (
	N_BITS : integer:=2357; -- for MuTRiG1
	N_CLIENTS : integer:=1
);
port (
	i_MOSI :       in std_logic;
	i_CSn :         in std_logic_vector(N_CLIENTS-1 downto 0);
	i_SCLK :        in std_logic;
	o_MISO :       out std_logic;
	o_data_first : out std_logic_vector(31 downto 0);
	o_data_last :  out std_logic_vector(31 downto 0)
);
end component; --spi_config_dummy;

component spi_hitcounter is
generic(
	C_NUM_CNT_BITS		: integer := 32*13 + 8;	-- length of CEC data register		
	C_TRANS_INTER		: integer := 2000;	-- the factor between sclk(2MHz) and the reading frenquency, e.g. 2MHz/2000 = 1KHz
	C_NUM_CLIENTS		: integer :=  1;
	C_HANDLEID_WIDTH 	: integer :=  1
);
port(
	i_clk			: in  std_logic;                                     -- system clock signal (125 MHz?)
	i_rst			: in  std_logic;                                     -- reset
	i_handleid 	   	: in  std_logic_vector(C_HANDLEID_WIDTH-1 downto 0); -- handle id of chip, will only talk to this one
	o_eot			: out std_logic;                                     -- end of transmission for daq controle
	o_hitcounter		: out std_logic_vector(C_NUM_CNT_BITS-1 downto 0);

	-- spi signals
	o_sclk			: out std_logic;                                     -- common clock  to all connected spi clients
	i_mdi			: in  std_logic_vector(C_NUM_CLIENTS-1 downto 0);	 -- individual spi slave input
	o_cs			: out std_logic_vector(C_NUM_CLIENTS-1 downto 0)	 -- individual chip select signals
);
end component;--spi_hitcounter;


component mutrig_datapath is
generic(
	N_ASICS : integer :=1;
	GEN_DUMMIES : boolean :=TRUE
);
port (
	i_rst			: in  std_logic;
	i_stic_txd		: in  std_logic_vector( N_ASICS-1 downto 0);-- serial data
	i_refclk_125		: in  std_logic;                 		-- ref clk for pll
	--global timestamp
	i_ts_clk		: in  std_logic;                 		-- ref clk for global timestamp
	i_ts_rst		: in  std_logic;				-- global timestamp reset, high active

	--interface to asic fifos
	i_clk_core		: in  std_logic; --fifo reading side clock
	o_fifo_empty		: out std_logic;
	o_fifo_data		: out std_logic_vector(35 downto 0);
	i_fifo_rd		: in  std_logic;
	--slow control
	i_SC_disable_dec	: in std_logic;
	i_SC_mask		: in std_logic_vector( N_ASICS-1 downto 0);
	i_SC_datagen_enable	: in std_logic;
	i_SC_datagen_shortmode	: in std_logic;
	i_SC_datagen_count	: in std_logic_vector(9 downto 0);

	--monitors
	o_receivers_usrclk	: out std_logic;              		-- pll output clock
	o_receivers_pll_lock	: out std_logic;			-- pll lock flag
	o_receivers_dpa_lock	: out std_logic_vector( N_ASICS-1 downto 0);			-- dpa lock flag per channel
	o_receivers_ready	: out std_logic_vector( N_ASICS-1 downto 0);-- receiver output ready flag
	o_frame_desync		: out std_logic;
	o_buffer_full		: out std_logic
);
end component;

component eventdata_pcie_bridge is
generic(
	C_ADDR_WIDTH     : integer := 64                                  -- PCIe address width
);
port (
	--global
	clk_pcie         : in  std_logic;                                     -- fast PCIe memory clk 
	reset_n          : in  std_logic;                                     -- reset, active low
	--fifo read side signals
	i_fifo_data	 : std_logic_vector(63 downto 0);
	i_fifo_empty	 : in  std_logic;
	o_fifo_rd	 : out std_logic;
	--pcie interface
	o_writeaddr      : out std_logic_vector(C_ADDR_WIDTH-1 downto 0);     -- PCIe address 
	o_write_data     : out std_logic_vector(31 downto 0);                 -- PCIe data word
	o_wren           : out std_logic;                                     -- PCIe enable write
	o_end_of_frame   : out std_logic
);
end component;

end package mutrig_components;
