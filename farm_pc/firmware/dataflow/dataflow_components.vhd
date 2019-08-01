library ieee;
use ieee.std_logic_1164.all;
--use work.pcie_components.all;

package dataflow_components is


	subtype tsrange_type is std_logic_vector(15 downto 0);
	subtype tsupper is natural range 31 downto 16;
	subtype tslower is natural range 15 downto 0;
	
	constant tsone : tsrange_type := (others => '1');
	constant tszero : tsrange_type := (others => '1');

	subtype dataplusts_type is std_logic_vector(271 downto 0);
	
	
	component data_flow is 
	port (
			reset_n		: 		in std_logic;
			
			-- Input from merging (first board) or links (subsequent boards)
			dataclk		: 		in std_logic;
			data_en		:		in std_logic;
			data_in		:		in std_logic_vector(255 downto 0);
			ts_in			:		in	std_logic_vector(31 downto 0);
			
			-- Input from PCIe demanding events
			pcieclk		:		in std_logic;
			ts_req_A		:		in std_logic_vector(31 downto 0);
			req_en_A		:		in std_logic;
			ts_req_B		:		in std_logic_vector(31 downto 0);
			req_en_B		:		in std_logic;
			tsblock_done:		in std_logic_vector(15 downto 0);
			
			-- Output to DMA
			dma_data_out	:	out std_logic_vector(255 downto 0);
			dma_data_en		:	out std_logic;
			dma_eoe			:  out std_logic;
			
			-- Output to links -- with dataclk
			link_data_out	:	out std_logic_vector(255 downto 0);
			link_ts_out		:	out std_logic_vector(31 downto 0);
			link_data_en	:	out std_logic;
			
			-- Interface to memory bank A
			A_mem_clk		: in std_logic;
			A_mem_ready		: in std_logic;
			A_mem_calibrated	: in std_logic;
			A_mem_addr		: out std_logic_vector(25 downto 0);
			A_mem_data		: out	std_logic_vector(255 downto 0);
			A_mem_write		: out std_logic;
			A_mem_read		: out std_logic;
			A_mem_q			: in std_logic_vector(255 downto 0);
			A_mem_q_valid	: in std_logic;

			-- Interface to memory bank B
			B_mem_clk		: in std_logic;
			B_mem_ready		: in std_logic;
			B_mem_calibrated	: in std_logic;
			B_mem_addr		: out std_logic_vector(25 downto 0);
			B_mem_data		: out	std_logic_vector(255 downto 0);
			B_mem_write		: out std_logic;
			B_mem_read		: out std_logic;
			B_mem_q			: in std_logic_vector(255 downto 0);
			B_mem_q_valid	: in std_logic
	
	);
	end component;
	
	
	component dflowfifo is
      port (
                        data    : in  std_logic_vector(271 downto 0) := (others => 'X'); -- datain
                        wrreq   : in  std_logic                      := 'X';             -- wrreq
                        rdreq   : in  std_logic                      := 'X';             -- rdreq
                        wrclk   : in  std_logic                      := 'X';             -- wrclk
                        rdclk   : in  std_logic                      := 'X';             -- rdclk
                        q       : out std_logic_vector(271 downto 0);                    -- dataout
                        wrusedw : out std_logic_vector(7 downto 0);                      -- wrusedw
                        rdempty : out std_logic;                                         -- rdempty
                        wrfull  : out std_logic                                          -- wrfull
                );
   end component dflowfifo;

	
	component tagram is
                port (
                        data    : in  std_logic_vector(31 downto 0) := (others => 'X'); -- datain
                        address : in  std_logic_vector(15 downto 0) := (others => 'X'); -- address
                        wren    : in  std_logic                     := 'X';             -- wren
                        clock   : in  std_logic                     := 'X';             -- clk
                        q       : out std_logic_vector(31 downto 0)                     -- dataout
                );
        end component tagram;
		  
		  component reqfifo is
                port (
                        data    : in  std_logic_vector(31 downto 0) := (others => 'X'); -- datain
                        wrreq   : in  std_logic                     := 'X';             -- wrreq
                        rdreq   : in  std_logic                     := 'X';             -- rdreq
                        wrclk   : in  std_logic                     := 'X';             -- wrclk
                        rdclk   : in  std_logic                     := 'X';             -- rdclk
                        q       : out std_logic_vector(15 downto 0);                    -- dataout
                        rdempty : out std_logic;                                        -- rdempty
                        wrfull  : out std_logic                                         -- wrfull
                );
        end component reqfifo;
		  
        component memreadfifo is
                port (
                        data    : in  std_logic_vector(37 downto 0) := (others => 'X'); -- datain
                        wrreq   : in  std_logic                     := 'X';             -- wrreq
                        rdreq   : in  std_logic                     := 'X';             -- rdreq
                        wrclk   : in  std_logic                     := 'X';             -- wrclk
                        rdclk   : in  std_logic                     := 'X';             -- rdclk
                        q       : out std_logic_vector(37 downto 0);                    -- dataout
                        rdempty : out std_logic;                                        -- rdempty
                        wrfull  : out std_logic                                         -- wrfull
                );
        end component memreadfifo;

         component memdatafifo is
                port (
                        data    : in  std_logic_vector(255 downto 0) := (others => 'X'); -- datain
                        wrreq   : in  std_logic                      := 'X';             -- wrreq
                        rdreq   : in  std_logic                      := 'X';             -- rdreq
                        wrclk   : in  std_logic                      := 'X';             -- wrclk
                        rdclk   : in  std_logic                      := 'X';             -- rdclk
                        q       : out std_logic_vector(255 downto 0);                    -- dataout
                        rdempty : out std_logic;                                         -- rdempty
                        wrfull  : out std_logic                                          -- wrfull
                );
        end component memdatafifo;
		  
		  
end package dataflow_components;