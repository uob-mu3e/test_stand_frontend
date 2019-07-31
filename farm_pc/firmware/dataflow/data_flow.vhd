-----------------------------------------------------------------------------
-- Handling of the data flow for the farm PCs
--
-- Niklaus Berger, JGU Mainz
-- niberger@uni-mainz.de
--
-----------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
--use work.ddr3_components.all;
--use work.pcie_components.all;
use work.dataflow_components.all;



entity data_flow is 
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
			A_mem_calibrated	: in std_logic;
			A_mem_ready		: in std_logic;
			A_mem_addr		: out std_logic_vector(25 downto 0);
			A_mem_data		: out	std_logic_vector(255 downto 0);
			A_mem_write		: out std_logic;
			A_mem_read		: out std_logic;
			A_mem_q			: in std_logic_vector(255 downto 0);
			A_mem_q_valid	: in std_logic;

			-- Interface to memory bank B
			B_mem_clk		: in std_logic;
			B_mem_calibrated	: in std_logic;
			B_mem_ready		: in std_logic;
			B_mem_addr		: out std_logic_vector(25 downto 0);
			B_mem_data		: out	std_logic_vector(255 downto 0);
			B_mem_write		: out std_logic;
			B_mem_read		: out std_logic;
			B_mem_q			: in std_logic_vector(255 downto 0);
			B_mem_q_valid	: in std_logic
	
	);
	end entity data_flow;
	
	architecture RTL of data_flow is
	
		type mem_mode_type is (disabled, ready, writing, reading);
		signal mem_mode_A : 	mem_mode_type;
		signal mem_mode_B :	mem_mode_type;
				
		type ddr3if_type is (disabled, ready, writing, reading, overwriting);
		signal ddr3if_state_A : ddr3if_type;
		signal ddr3if_state_B :	ddr3if_type;

		signal A_tsrange	:	tsrange_type;
		signal B_tsrange	:	tsrange_type;
		signal tsupper_last	: tsrange_type;
	
		signal A_memready	: std_logic;
		signal B_memready	: std_logic;
		
		signal A_writestate:	std_logic;
		signal B_writestate: std_logic;
		
		signal A_readstate:	std_logic;
		signal B_readstate:  std_logic;
		
		
		signal tofifo_A	: dataplusts_type;
		signal tofifo_B	: dataplusts_type;
		
		signal writefifo_A :	std_logic;
		signal writefifo_B : std_logic;
				
		signal A_fifo_empty	: std_logic;
		signal B_fifo_empty	: std_logic;
		
		signal A_reqfifo_empty	: std_logic;
		signal B_reqfifo_empty	: std_logic; 
		
		signal A_tagram_write	: std_logic;
		signal A_tagram_data		: std_logic_vector(31 downto 0);
		signal A_tagram_q			: std_logic_vector(31 downto 0);
		signal A_tagram_address	: tsrange_type;
		
		signal A_tagram_addrnext : tsrange_type;
		signal A_tagram_datanext : std_logic_vector(31 downto 0);
		
		signal A_mem_addr_reg		: std_logic_vector(25 downto 0);
		signal readfifo_A				: std_logic;
		signal qfifo_A					: dataplusts_type;
		signal A_tagts_last			: tsrange_type;
		signal A_wstarted				: std_logic; 
		signal A_numwords				: std_logic_vector(5 downto 0);
		
		signal A_readreqfifo			: std_logic;
		signal A_reqfifoq				: tsrange_type;
		signal A_req_last				: tsrange_type;
		
		type readsubstate_type is (fifowait, tagmemwait, reading);
		signal A_readsubstate 	:	readsubstate_type;
		signal A_readwords				: std_logic_vector(5 downto 0);
		
	begin
	
		process(reset_n, dataclk)
		variable tsupperchange : boolean;
		begin
		if(reset_n = '0') then
			mem_mode_A	<= disabled;
			mem_mode_B	<= disabled;
			tsupper_last	<= (others => '1');
			
			writefifo_A	<= '0';
			writefifo_B	<= '0';
			
			A_readstate	<= '0';
			B_readstate <= '0';
			A_writestate <= '0';
			B_writestate <= '0';

			link_data_en	<= '0';
			
		elsif(dataclk'event and dataclk = '1') then


			link_data_out	<= data_in;
			link_ts_out		<= ts_in;
		
			link_data_en  <= '0';
			if(mem_mode_A /= writing and mem_mode_B /= writing) then
				link_data_en <= data_en;
			end if;

		
			tofifo_A <= ts_in(tslower) & data_in;
			tofifo_B	<= ts_in(tslower) & data_in;
			
			writefifo_A	<= '0';
			writefifo_B	<= '0';
			
			A_readstate	<= '0';
			B_readstate <= '0';
			A_writestate <= '0';
			B_writestate <= '0';
			
			tsupperchange := false;
			if(data_en = '1') then
				tsupper_last <= ts_in(tsupper);
				if(ts_in(tsupper) /=  tsupper_last) then
					tsupperchange := true;
				end if;
			end if;
			
			case mem_mode_A is
				when disabled =>
					if(A_mem_calibrated = '1')then
						mem_mode_A <= ready;
					end if;
				when ready 	=>
					if(tsupperchange) then
						mem_mode_A	<= writing;
						A_tsrange	<= ts_in(tsupper);
						writefifo_A	<= '1';
						link_data_en 	<= '0';
					end if;
				when writing =>
					A_writestate <= '1';
				
					writefifo_A		<= data_en;
					if(tsupperchange) then
						mem_mode_A	<= reading;
						writefifo_A	<= '0';
					end if;
				when reading =>
					A_readstate	<= '1';
				
					if(tsblock_done = A_tsrange) then
						mem_mode_A <= ready;
					end if;
			end case;
			
			case mem_mode_B is
				when disabled =>
					if(B_mem_calibrated = '1')then
						mem_mode_B <= ready;
					end if;
				when ready 	=>
					if(tsupperchange and (mem_mode_A /= ready)) then
						mem_mode_B	<= writing;
						B_tsrange	<= ts_in(tsupper);
						writefifo_B	<= '1';
						link_data_en 	<= '0';
					end if;
				when writing =>
					A_writestate <= '1';
				
					writefifo_B		<= data_en;
					if(tsupperchange) then
						mem_mode_B	<= reading;
						writefifo_B	<= '0';
					end if;
				when reading =>
					A_readstate	<= '1';
				
					if(tsblock_done = B_tsrange) then
						mem_mode_B <= ready;
					end if;
			end case;
		

		end if;
		end process;
		
		tomemfifo_A:dflowfifo
			port map(
				data    => tofifo_A,
				wrreq   => writefifo_A,
            rdreq   => readfifo_A,
            wrclk   => dataclk,
            rdclk   => A_mem_clk,
            q       => qfifo_A,
            wrusedw => open,
            rdempty => A_fifo_empty,
            wrfull  => open
        );
		 
		A_mem_data		<= qfifo_A(255 downto 0);
		
		process(reset_n, A_mem_clk)
		begin
		if(reset_n = '0') then
			ddr3if_state_A	<= disabled;
			A_tagram_write	<= '0';
			readfifo_A		<= '0';
			A_readreqfifo  <= '0';
		elsif(A_mem_clk'event and A_mem_clk = '1') then
			A_tagram_write	<= '0';
			readfifo_A		<= '0';
			A_mem_write		<= '0';
			A_readreqfifo		<= '0';
			
			case ddr3if_state_A is
				when disabled =>
					if(A_mem_calibrated = '1')then
						A_tagram_address	<= (others => '1');
						ddr3if_state_A	<= overwriting;
-- Skip memory overwriting for simulation
-- synthesis translate_off
						ddr3if_state_A	<= ready;
-- synthesis translate on
					end if;
				when ready =>
					if(A_writestate = '1')then
						ddr3if_state_A	<= writing;
						A_mem_addr_reg		<= (others => '0');
						A_tagram_address	<= (others => '0');
						A_wstarted			<= '1';
						A_numwords			<= "000001";
					end if;
				when writing =>
					
					if(A_readstate = '1' and A_fifo_empty = '1') then
						ddr3if_state_A	<= reading;
						A_readsubstate <= fifowait;
					end if;
					
					if(A_fifo_empty = '0' and A_mem_ready = '1') then
						readfifo_A		<= '1';
						
						A_mem_write		<= '1';
						A_mem_addr  		<= A_mem_addr_reg;
						A_mem_addr_reg		<= A_mem_addr_reg + '1';
						A_tagts_last	<= qfifo_A(271 downto 256);
												
						if(A_tagts_last /= qfifo_A(271 downto 256) or A_wstarted = '1') then
							A_wstarted <= '0';
							A_tagram_write	<= '1';	
							A_tagram_addrnext	<= qfifo_A(271 downto 256);
							A_tagram_datanext(25 downto 0)		<= A_mem_addr_reg;
							
							A_tagram_write	<= '1';
							A_tagram_address	<= A_tagram_addrnext;
							A_tagram_data(25 downto 0)		<= A_tagram_datanext(25 downto 0);
							A_tagram_data(31 downto 26)	<= A_numwords;
							A_numwords							<= "000001";
						else
							if(A_numwords /= "111111") then
								A_numwords <= A_numwords + '1';
							end if;
						end if;
					end if;	
					
				when reading =>
					if(A_readstate = '0' and A_reqfifo_empty = '1' and A_readsubstate = fifowait)then
						ddr3if_state_A	<= overwriting;
						A_tagram_address	<= (others => '1');
					end if;
					
					case A_readsubstate is
						when fifowait =>
							if(A_reqfifo_empty = '0') then
								A_tagram_address <= A_reqfifoq;
								A_req_last		  <= A_reqfifoq;
								A_readreqfifo	  <= '1';
								if(A_reqfifoq /= A_req_last) then
									A_readsubstate <= tagmemwait;
								end if;
							end if;
						when tagmemwait =>
							A_mem_addr_reg	<= A_tagram_q(25 downto 0);
							A_readwords		<= A_tagram_q(31 downto 26);
							if(A_mem_ready = '1') then
								A_readsubstate	<= reading;
								A_mem_read		<= '1';
								if(A_tagram_q(31 downto 26) > "00001") then
									A_readsubstate	<= reading;
								else
									A_readsubstate <= fifowait;
								end if;
							end if;	
						when reading =>
							if(A_mem_ready = '1')then
								A_mem_addr_reg	<=	A_mem_addr_reg + '1';
								A_readwords    <= A_readwords - '1';
								A_mem_read		<= '1';
							end if;
							if(A_readwords > "00001") then
									A_readsubstate	<= reading;
							else
									A_readsubstate <= fifowait;
							end if;
					end case;
					
									
				when overwriting =>
					A_tagram_address        <= A_tagram_address + '1';
					A_tagram_write		<= '1';
					A_tagram_data		<= (others => '1');
					if(A_tagram_address = tsone and A_tagram_write = '1') then
						ddr3if_state_A	<= ready;
					end if;				
			end case;
		end if;
	end process;
		
			tagram_A:tagram
                port map(
                        data    => A_tagram_data,
                        address => A_tagram_address,
                        wren    => A_tagram_write,
                        clock   => A_mem_clk,
                        q       => A_tagram_q
                );

			 A_reqfifo:reqfifo
                port map(
                        data    => ts_req_A,
                        wrreq   => req_en_A,
                        rdreq   => A_readreqfifo,
                        wrclk   => pcieclk,
                        rdclk   => A_mem_clk,
                        q       => A_reqfifoq,
                        rdempty => A_reqfifo_empty,
                        wrfull  => open
                );


		
		
	end architecture RTL;
	