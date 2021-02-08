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
use work.dataflow_components.all;



entity data_flow is
	port (
			reset_n			: 		in std_logic;
			reset_n_ddr3	: 		in std_logic;
			
			-- Input from merging (first board) or links (subsequent boards)
			dataclk			: 		in std_logic;
			data_en			:		in std_logic;
			data_in			:		in std_logic_vector(255 downto 0);
			ts_in			:		in	std_logic_vector(31 downto 0);
			
			-- Input from PCIe demanding events
			pcieclk			:		in std_logic;
			ts_req_A		:		in std_logic_vector(31 downto 0);
			req_en_A		:		in std_logic;
			ts_req_B		:		in std_logic_vector(31 downto 0);
			req_en_B		:		in std_logic;
			tsblock_done	:		in std_logic_vector(15 downto 0);
			tsblocks		:		out std_logic_vector(31 downto 0);
			
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
			A_mem_calibrated: in std_logic;
			A_mem_ready		: in std_logic;
			A_mem_addr		: out std_logic_vector(25 downto 0);
			A_mem_data		: out	std_logic_vector(255 downto 0);
			A_mem_write		: out std_logic;
			A_mem_read		: out std_logic;
			A_mem_q			: in std_logic_vector(255 downto 0);
			A_mem_q_valid	: in std_logic;

			-- Interface to memory bank B
			B_mem_clk		: in std_logic;
			B_mem_calibrated: in std_logic;
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

		signal reset: std_logic;
		signal reset_ddr3: std_logic;
	
		type mem_mode_type is (disabled, ready, writing, reading);
		signal mem_mode_A : 	mem_mode_type;
		signal mem_mode_B :	mem_mode_type;
				
		type ddr3if_type is (disabled, ready, writing, reading, overwriting);
		signal ddr3if_state_A : ddr3if_type;
		signal ddr3if_state_B :	ddr3if_type;

		signal A_tsrange	:	tsrange_type;
		signal B_tsrange	:	tsrange_type;
		signal tsupper_last	: tsrange_type;
	
		--signal A_memready	: std_logic;
		--signal B_memready	: std_logic;
		
		signal A_writestate:	std_logic;
		signal B_writestate: std_logic;
		
		signal A_readstate:	std_logic;
		signal B_readstate:  std_logic;

		signal A_done:	     std_logic;
		signal B_done:	     std_logic;
		
		signal tofifo_A	: dataplusts_type;
		signal tofifo_B	: dataplusts_type;
		
		signal writefifo_A :	std_logic;
		signal writefifo_B : std_logic;
				
		signal A_fifo_empty	: std_logic;
		signal A_fifo_empty_last	: std_logic;
		signal B_fifo_empty	: std_logic;
		signal B_fifo_empty_last	: std_logic;
		
		signal A_reqfifo_empty	: std_logic;
		signal B_reqfifo_empty	: std_logic; 
		
		signal A_tagram_write	: std_logic;
		signal A_tagram_data		: std_logic_vector(31 downto 0);
		signal A_tagram_q			: std_logic_vector(31 downto 0);
		signal A_tagram_address	: tsrange_type;
				
        signal B_tagram_write	: std_logic;
		signal B_tagram_data		: std_logic_vector(31 downto 0);
		signal B_tagram_q			: std_logic_vector(31 downto 0);
		signal B_tagram_address	: tsrange_type;
		
		signal A_mem_addr_reg		: std_logic_vector(25 downto 0);
		signal A_mem_addr_tag		: std_logic_vector(25 downto 0);
		signal readfifo_A				: std_logic;
		signal qfifo_A					: dataplusts_type;
		signal A_tagts_last			: tsrange_type;
		signal A_wstarted				: std_logic;
 		signal A_wstarted_last				: std_logic;
		signal A_numwords				: std_logic_vector(5 downto 0);
		
        signal B_mem_addr_reg		: std_logic_vector(25 downto 0);
		signal B_mem_addr_tag		: std_logic_vector(25 downto 0);
		signal readfifo_B				: std_logic;
		signal qfifo_B					: dataplusts_type;
		signal B_tagts_last			: tsrange_type;
		signal B_wstarted				: std_logic;
 		signal B_wstarted_last				: std_logic;
		signal B_numwords				: std_logic_vector(5 downto 0);
				
		signal A_readreqfifo			: std_logic;
		signal A_reqfifoq				: tsrange_type;
		signal A_req_last				: tsrange_type;
		
        signal B_readreqfifo			: std_logic;
		signal B_reqfifoq				: tsrange_type;
		signal B_req_last				: tsrange_type;
		
		type readsubstate_type is (fifowait, tagmemwait_1, tagmemwait_2, tagmemwait_3, reading);
		signal A_readsubstate 	:	readsubstate_type;
		signal A_readwords				: std_logic_vector(5 downto 0);
		
        signal B_readsubstate 	:	readsubstate_type;
		signal B_readwords				: std_logic_vector(5 downto 0);

		signal A_memreadfifo_data: std_logic_vector(37 downto 0);
		signal A_memreadfifo_write: std_logic;
		signal A_memreadfifo_read: std_logic;
		signal A_memreadfifo_q: std_logic_vector(37 downto 0);
		signal A_memreadfifo_empty: std_logic;		 
		
		signal A_memdatafifo_empty: std_logic;
		signal A_memdatafifo_read:  std_logic;
		signal A_memdatafifo_q	 :  std_logic_vector(255 downto 0);
		
        signal B_memreadfifo_data: std_logic_vector(37 downto 0);
		signal B_memreadfifo_write: std_logic;
		signal B_memreadfifo_read: std_logic;
		signal B_memreadfifo_q: std_logic_vector(37 downto 0);
		signal B_memreadfifo_empty: std_logic;		 
		
		signal B_memdatafifo_empty: std_logic;
		signal B_memdatafifo_read:  std_logic;
        signal B_memdatafifo_q	 :  std_logic_vector(255 downto 0);
	
		
		type output_write_type is (waiting, eventA, eventB);
		signal output_write_state : output_write_type;
		signal nummemwords : std_logic_vector(5 downto 0);
		signal tagmemwait_3_state : std_logic_vector(3 downto 0);
		
		signal ts_in_upper : std_logic_vector(15 downto 0);
		signal ts_in_lower : std_logic_vector(15 downto 0);

		
	begin
	
		tsblocks <= B_tsrange & A_tsrange;
		
		-- TODO: make this dynamic
		ts_in_upper <= x"00" & ts_in(tsupper);
		ts_in_lower <= x"00" & ts_in(tslower);
	
		reset <= not reset_n;
		reset_ddr3 <= not reset_n_ddr3;

		process(reset_n_ddr3, dataclk)
		variable tsupperchange : boolean;
		begin
		if(reset_n_ddr3 = '0') then

            mem_mode_A   <= disabled;
			mem_mode_B   <= disabled;
			
			writefifo_A  <= '0';
			writefifo_B  <= '0';
			
			A_readstate  <= '0';
			B_readstate  <= '0';
			
			A_writestate <= '0';
			B_writestate <= '0';

			link_data_en <= '0';
			tsupper_last <= (others => '1');
			
		elsif(dataclk'event and dataclk = '1') then

			link_data_out	<= data_in;
			link_ts_out		<= ts_in;
		
            -- send data to next FPGA if mem is not writing
			link_data_en  <= '0';
			if(mem_mode_A /= writing and mem_mode_B /= writing) then
				link_data_en <= data_en;
			end if;
		
            -- keep lower time and data
			tofifo_A <= ts_in_lower & data_in;
			tofifo_B <= ts_in_lower & data_in;
			
			writefifo_A	<= '0';
			writefifo_B	<= '0';
			
			A_readstate	<= '0';
			B_readstate <= '0';
			
			A_writestate <= '0';
			B_writestate <= '0';
			
			tsupperchange := false;
			if(data_en = '1') then
				tsupper_last <= ts_in_upper;
				if(ts_in_upper /=  tsupper_last) then
					tsupperchange := true;
				end if;
			end if;
			
			case mem_mode_A is
				when disabled =>
					if(A_mem_calibrated = '1')then
						mem_mode_A <= ready;
					end if;
					
				when ready 	=>
					if(tsupperchange and A_done = '1') then
						mem_mode_A    <= writing;
						A_tsrange     <= ts_in_upper;
						writefifo_A   <= '1';
						link_data_en  <= '0';
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
					if(tsupperchange and (mem_mode_A /= ready or (mem_mode_A = ready and A_done = '0')) and B_done ='1') then
						mem_mode_B	<= writing;
						B_tsrange	<= ts_in_upper;
						writefifo_B	<= '1';
						link_data_en 	<= '0';
					end if;
					
				when writing =>
					B_writestate <= '1';
				
					writefifo_B		<= data_en;
					if(tsupperchange) then
						mem_mode_B	<= reading;
						writefifo_B	<= '0';
					end if;
					
				when reading =>
					B_readstate	<= '1';
				
					if(tsblock_done = B_tsrange) then
						mem_mode_B <= ready;
					end if;
			end case;
		end if;
		end process;
		
        tomemfifo_A : entity work.ip_dcfifo
        generic map(
            ADDR_WIDTH  => 8,
            DATA_WIDTH  => 272,
            DEVICE      => "Arria 10"--,
        )
        port map (
            data        => tofifo_A,
            wrreq       => writefifo_A,
            rdreq       => readfifo_A,
            wrclk       => dataclk,
            rdclk       => A_mem_clk,
            q           => qfifo_A,
            rdempty     => A_fifo_empty,
            rdusedw     => open,
            wrfull      => open,
            wrusedw     => open,
            aclr        => reset--,
        );
		 
		A_mem_data		<= qfifo_A(255 downto 0);
		
        tomemfifo_B : entity work.ip_dcfifo
        generic map(
            ADDR_WIDTH  => 8,
            DATA_WIDTH  => 272,
            DEVICE      => "Arria 10"--,
        )
        port map (
            data        => tofifo_B,
            wrreq       => writefifo_B,
            rdreq       => readfifo_B,
            wrclk       => dataclk,
            rdclk       => B_mem_clk,
            q           => qfifo_B,
            rdempty     => B_fifo_empty,
            rdusedw     => open,
            wrfull      => open,
            wrusedw     => open,
            aclr        => reset--,
        );
		 
		B_mem_data		<= qfifo_B(255 downto 0);

-- Process for writing the A memory
		process(reset_n_ddr3, A_mem_clk)
		begin
		if(reset_n_ddr3 = '0') then
			ddr3if_state_A	<= disabled;
			A_tagram_write	<= '0';
			readfifo_A	<= '0';
			A_readreqfifo   <= '0';
			A_mem_write	<= '0';
			A_mem_read	<= '0';
			A_memreadfifo_write <= '0';
			A_done 		   <= '0';
		elsif(A_mem_clk'event and A_mem_clk = '1') then
			A_tagram_write	<= '0';
			readfifo_A		<= '0';
			A_mem_write		<= '0';
			A_mem_read		<= '0';
			A_readreqfifo		<= '0';
			A_memreadfifo_write 	<= '0';
			A_wstarted_last		<= A_wstarted;
			A_fifo_empty_last	<= A_fifo_empty;
			case ddr3if_state_A is
				when disabled =>
					if(A_mem_calibrated = '1')then
						A_tagram_address	<= (others => '1');
						ddr3if_state_A	<= overwriting;
-- Skip memory overwriting for simulation
-- synthesis translate_off
						ddr3if_state_A	<= ready;
						A_done		<= '1';
-- synthesis translate_on
					end if;
					
				when ready =>
					if(A_writestate = '1')then
						ddr3if_state_A	<= writing;
						A_mem_addr_reg		<= (others => '0');
						A_tagram_address	<= (others => '0');
						A_wstarted			<= '1';
						A_numwords			<= "000001";
						A_done 				<= '0';
					end if;
					
				when writing =>
					if(A_readstate = '1' and A_fifo_empty = '1') then
						ddr3if_state_A	<= reading;
						A_readsubstate <= fifowait;
					end if;
					
					if(A_fifo_empty = '0' and A_fifo_empty_last = '0' and A_mem_ready = '1') then
						A_wstarted <= '0';

						readfifo_A		<= '1';
						
						A_mem_write		<= '1';
						A_mem_addr  		<= A_mem_addr_reg;
						A_mem_addr_tag		<= A_mem_addr_reg;	
						A_mem_addr_reg		<= A_mem_addr_reg + '1';
						A_tagts_last		<= qfifo_A(271 downto 256);
												
						if(A_tagts_last /= qfifo_A(271 downto 256) or A_wstarted_last = '1') then
							if(A_wstarted = '1') then
								A_tagram_write		<= '0';
							else	
								A_tagram_write		<= '1';
							end if;							
							A_tagram_address	<= qfifo_A(271 downto 256);
							A_tagram_data(25 downto 0)		<= A_mem_addr_tag;

							A_tagram_data(31 downto 26)	<= "000001";
							A_numwords			<= "000010";
						else

							A_tagram_write	<= '1';

							A_tagram_data(31 downto 26)	<= A_numwords;
							if(A_numwords /= "111111") then
								A_numwords <= A_numwords + '1';
							end if;
						end if;
					elsif(A_fifo_empty = '0' and A_mem_ready = '1' and A_wstarted = '0') then
						readfifo_A		<= '1';
					end if;	
					
				when reading =>
					if(A_readstate = '0' and A_reqfifo_empty = '1' and A_readsubstate = fifowait)then
						ddr3if_state_A		<= overwriting;
						A_tagram_address	<= (others => '1');
					end if;
					
					case A_readsubstate is
						when fifowait =>
							if(A_reqfifo_empty = '0') then
								A_tagram_address <= A_reqfifoq;
								A_req_last		  <= A_reqfifoq;
								A_readreqfifo	  <= '1';
								if(A_reqfifoq /= A_req_last) then
									A_readsubstate <= tagmemwait_1;
								end if;
							end if;
						when tagmemwait_1 =>
							A_readsubstate <= tagmemwait_2;	
						when tagmemwait_2 =>
							A_readsubstate <= tagmemwait_3;	
						when tagmemwait_3 =>
							if( A_tagram_q(31 downto 26) = "000000") then
                                tagmemwait_3_state <= x"A";
								A_readsubstate <= fifowait;
								A_memreadfifo_data <= "000000" & A_tsrange & A_tagram_address;
								A_memreadfifo_write <= '1';
-- synthesis translate_off
							elsif(Is_X(A_tagram_q(31 downto 26))) then
                                tagmemwait_3_state <= x"B";
								A_readsubstate <= fifowait;	
								A_memreadfifo_data <= "000000" & A_tsrange & A_tagram_address;
								A_memreadfifo_write <= '1';
-- synthesis translate_on					
							else
                                tagmemwait_3_state <= x"C";
								A_mem_addr	<= A_tagram_q(25 downto 0);
								A_readwords	<= A_tagram_q(31 downto 26)-'1';
								if(A_mem_ready = '1') then
									A_mem_read		<= '1';
									if(A_tagram_q(31 downto 26) > "00001") then--"00001") then
										A_readsubstate	<= reading;
										A_mem_addr_reg  <= A_tagram_q(25 downto 0) + '1';
									else
										A_readsubstate <= fifowait;
									end if;
									A_memreadfifo_data <= A_tagram_q(31 downto 26) & A_tsrange & A_tagram_address;
									A_memreadfifo_write <= '1';
								end if;
							end if;
						when reading =>
							if(A_mem_ready = '1')then
								A_mem_addr	<= A_mem_addr_reg ;
								A_mem_addr_reg  <= A_mem_addr_reg + '1';
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
					A_tagram_data		<= (others => '0');
					if(A_tagram_address = tsone and A_tagram_write = '1') then
						ddr3if_state_A	<= ready;
						A_done 		<= '1';
					end if;				
			end case;
		end if;
	end process;

	A_memreadfifo_read <= '1' when output_write_state = waiting and A_memreadfifo_empty = '0' 
				else '0';

	A_memdatafifo_read <= '1' when output_write_state = eventA and A_memdatafifo_empty = '0'
				else '0';
				
				
-- Process for writing the B memory
		process(reset_n_ddr3, B_mem_clk)
		begin
		if(reset_n_ddr3 = '0') then
			ddr3if_state_B	<= disabled;
			B_tagram_write	<= '0';
			readfifo_B	<= '0';
			B_readreqfifo   <= '0';
			B_mem_write	<= '0';
			B_mem_read	<= '0';
			B_memreadfifo_write <= '0';
			B_done 		   <= '0';
		elsif(B_mem_clk'event and B_mem_clk = '1') then
			B_tagram_write	<= '0';
			readfifo_B		<= '0';
			B_mem_write		<= '0';
			B_mem_read		<= '0';
			B_readreqfifo		<= '0';
			B_memreadfifo_write 	<= '0';
			B_wstarted_last		<= B_wstarted;
			B_fifo_empty_last	<= B_fifo_empty;
			case ddr3if_state_B is
				when disabled =>
					if(B_mem_calibrated = '1')then
						B_tagram_address	<= (others => '1');
						ddr3if_state_B	<= overwriting;
-- Skip memory overwriting for simulation
-- synthesis translate_off
						ddr3if_state_B	<= ready;
						B_done		<= '1';
-- synthesis translate on
					end if;
				when ready =>
					if(B_writestate = '1')then
						ddr3if_state_B	<= writing;
						B_mem_addr_reg		<= (others => '0');
						B_tagram_address	<= (others => '0');
						B_wstarted			<= '1';
						B_numwords			<= "000001";
						B_done 				<= '0';
					end if;
				when writing =>
					
					if(B_readstate = '1' and B_fifo_empty = '1') then
						ddr3if_state_B	<= reading;
						B_readsubstate <= fifowait;
					end if;
					
					if(B_fifo_empty = '0' and B_fifo_empty_last = '0' and B_mem_ready = '1') then
						B_wstarted <= '0';

						readfifo_B		<= '1';
						
						B_mem_write		<= '1';
						B_mem_addr  		<= B_mem_addr_reg;
						B_mem_addr_tag		<= B_mem_addr_reg;	
						B_mem_addr_reg		<= B_mem_addr_reg + '1';
						B_tagts_last		<= qfifo_B(271 downto 256);
												
						if(B_tagts_last /= qfifo_B(271 downto 256)  or B_wstarted_last ='1') then
							if(B_wstarted = '1') then
								B_tagram_write		<= '0';
							else	
								B_tagram_write		<= '1';
							end if;							
							B_tagram_address	<= qfifo_B(271 downto 256);
							B_tagram_data(25 downto 0)		<= B_mem_addr_tag;

							B_tagram_data(31 downto 26)	<= "000001";
							B_numwords			<= "000010";
						else

							B_tagram_write	<= '1';

							B_tagram_data(31 downto 26)	<= B_numwords;
							if(B_numwords /= "111111") then
								B_numwords <= B_numwords + '1';
							end if;
						end if;
					elsif(B_fifo_empty = '0' and B_mem_ready = '1' and B_wstarted = '0') then
						readfifo_B		<= '1';
					end if;	
					
				when reading =>
					if(B_readstate = '0' and B_reqfifo_empty = '1' and B_readsubstate = fifowait)then
						ddr3if_state_B		<= overwriting;
						B_tagram_address	<= (others => '1');
					end if;
					
					case B_readsubstate is
						when fifowait =>
							if(B_reqfifo_empty = '0') then
								B_tagram_address <= B_reqfifoq;
								B_req_last		  <= B_reqfifoq;
								B_readreqfifo	  <= '1';
								if(B_reqfifoq /= B_req_last) then
									B_readsubstate <= tagmemwait_1;
								end if;
							end if;
						when tagmemwait_1 =>
							B_readsubstate <= tagmemwait_2;	
						when tagmemwait_2 =>
							B_readsubstate <= tagmemwait_3;	
						when tagmemwait_3 =>
							if( B_tagram_q(31 downto 26) = "000000") then
								B_readsubstate <= fifowait;
								B_memreadfifo_data <= "000000" & B_tsrange & B_tagram_address;
								B_memreadfifo_write <= '1';
-- synthesis translate_off
							elsif(Is_X(B_tagram_q(31 downto 26))) then
								B_readsubstate <= fifowait;	
								B_memreadfifo_data <= "000000" & B_tsrange & B_tagram_address;
								B_memreadfifo_write <= '1';
-- synthesis translate on					
							else
								B_mem_addr	<= B_tagram_q(25 downto 0);
								B_readwords	<= B_tagram_q(31 downto 26)-'1';
								if(B_mem_ready = '1') then
									B_mem_read		<= '1';
									if(B_tagram_q(31 downto 26) > "00001") then
										B_readsubstate	<= reading;
										B_mem_addr_reg  <= B_tagram_q(25 downto 0) + '1';
									else
										B_readsubstate <= fifowait;
									end if;
									B_memreadfifo_data <= B_tagram_q(31 downto 26) & B_tsrange & B_tagram_address;
									B_memreadfifo_write <= '1';
								end if;
							end if;	
						when reading =>
							if(B_mem_ready = '1')then
								B_mem_addr	<= B_mem_addr_reg ;
								B_mem_addr_reg  <= B_mem_addr_reg + '1';
								B_readwords    <= B_readwords - '1';
								B_mem_read		<= '1';
							end if;
							if(B_readwords > "00001") then
									B_readsubstate	<= reading;
							else
									B_readsubstate <= fifowait;
							end if;
					end case;
					
									
				when overwriting =>
					B_tagram_address        <= B_tagram_address + '1';
					B_tagram_write		<= '1';
					B_tagram_data		<= (others => '0');
					if(B_tagram_address = tsone and B_tagram_write = '1') then
						ddr3if_state_B	<= ready;
						B_done 		<= '1';
					end if;				
			end case;
		end if;
	end process;

	B_memreadfifo_read <= '1' when output_write_state = waiting and B_memreadfifo_empty = '0' 
				else '0';

	B_memdatafifo_read <= '1' when output_write_state = eventA and B_memdatafifo_empty = '0'
				else '0';



	process(reset_n, pcieclk)
		begin
		if(reset_n = '0') then
			output_write_state <= waiting;
			dma_data_en	  <= '0';
			dma_eoe		   <= '0';
		elsif(pcieclk'event and pcieclk = '1') then
			dma_data_en	   <= '0';
			dma_eoe		   <= '0';
			case output_write_state is
				when waiting =>
				
					if(A_memreadfifo_empty = '0') then
						nummemwords <= A_memreadfifo_q(37 downto 32);
						if(A_memreadfifo_q(37 downto 32) = "000000")then
							output_write_state <= waiting;
							dma_eoe <= '1';	
						else
							output_write_state <= eventA;
						end if;

						dma_data_out <= x"0001" & x"0000" & -- ev ID and trigger mask
								A_memreadfifo_q(31 downto 0) & -- EventID (timestamp)
								A_memreadfifo_q(31 downto 0) & -- should be unix TS...
								x"00000" & "0" & A_memreadfifo_q(37 downto 32) & "01100" &-- Event data size
								x"00000" & "0" & A_memreadfifo_q(37 downto 32) & "01000" & -- All bank size (incl 8 bit header)
								x"00000000" & -- Flags
								x"4D485230" & --MHR0
								x"00030" & "0" & A_memreadfifo_q(37 downto 32) & "00000"; --TID_WORD and -- Length in bytes
						dma_data_en <= '1';
					
					elsif(B_memreadfifo_empty = '0') then
					
                        nummemwords <= B_memreadfifo_q(37 downto 32);
						if(B_memreadfifo_q(37 downto 32) = "000000")then
							output_write_state <= waiting;
							dma_eoe <= '1';	
						else
							output_write_state <= eventB;
						end if;

						dma_data_out <= x"0001" & x"0000" & -- ev ID and trigger mask
								B_memreadfifo_q(31 downto 0) & -- EventID (timestamp)
								B_memreadfifo_q(31 downto 0) & -- should be unix TS...
								x"00000" & "0" & B_memreadfifo_q(37 downto 32) & "01100" &-- Event data size
								x"00000" & "0" & B_memreadfifo_q(37 downto 32) & "01000" & -- All bank size (incl 8 bit header)
								x"00000000" & -- Flags
								x"4D485230" & --MHR0
								x"00030" & "0" & B_memreadfifo_q(37 downto 32) & "00000"; --TID_WORD and -- Length in bytes
						dma_data_en <= '1';
					end if;

		 		when eventA =>
					if(A_memdatafifo_empty = '0') then
						dma_data_en <= '1';
						dma_data_out		<= A_memdatafifo_q;
						nummemwords		<= nummemwords - '1';
						if(nummemwords = "000001")then
							output_write_state <= waiting;	
							dma_eoe <= '1';	
						end if;
					end if;
					
				when eventB =>
                    if(B_memdatafifo_empty = '0') then
						dma_data_en     <= '1';
						dma_data_out	<= B_memdatafifo_q;
						nummemwords		<= nummemwords - '1';
						if(nummemwords = "000001")then
							output_write_state <= waiting;	
							dma_eoe <= '1';	
						end if;
					end if;
				
			end case;	
		end if;
	end process;
	
            tagram_A : entity work.ip_ram_1_port
            generic map (
                ADDR_WIDTH    => 16,
                DATA_WIDTH    => 32,
                DEVICE        => "Arria 10"--,
            )
            port map (
                data    => A_tagram_data,
                address => A_tagram_address,
                wren    => A_tagram_write,
                clock   => A_mem_clk,
                q       => A_tagram_q--,
            );
            
            tagram_B : entity work.ip_ram_1_port
            generic map (
                ADDR_WIDTH    => 16,
                DATA_WIDTH    => 32,
                DEVICE        => "Arria 10"--,
            )
            port map (
                data    => B_tagram_data,
                address => B_tagram_address,
                wren    => B_tagram_write,
                clock   => B_mem_clk,
                q       => B_tagram_q--,
            );
            
            A_reqfifo : entity work.ip_dcfifo_mixed_widths
            generic map(
                ADDR_WIDTH_w    => 8,
                DATA_WIDTH_w    => 32,
                ADDR_WIDTH_r    => 9,
                DATA_WIDTH_r    => 16,
                DEVICE          => "Arria 10"--,
            )
            port map (
                data        => ts_req_A,
                wrreq       => req_en_A,
                rdreq       => A_readreqfifo,
                wrclk       => A_mem_clk,
                rdclk       => A_mem_clk,
                q           => A_reqfifoq,
                rdempty     => A_reqfifo_empty,
                rdusedw     => open,
                wrfull      => open,
                wrusedw     => open,
                aclr        => reset--,
            );
            
            B_reqfifo : entity work.ip_dcfifo_mixed_widths
            generic map(
                ADDR_WIDTH_w    => 8,
                DATA_WIDTH_w    => 32,
                ADDR_WIDTH_r    => 9,
                DATA_WIDTH_r    => 16,
                DEVICE          => "Arria 10"--,
            )
            port map (
                data        => ts_req_B,
                wrreq       => req_en_B,
                rdreq       => B_readreqfifo,
                wrclk       => B_mem_clk,
                rdclk       => B_mem_clk,
                q           => B_reqfifoq,
                rdempty     => B_reqfifo_empty,
                rdusedw     => open,
                wrfull      => open,
                wrusedw     => open,
                aclr        => reset--,
            );
            
            A_mreadfifo : entity work.ip_dcfifo
            generic map(
                ADDR_WIDTH  => 3,
                DATA_WIDTH  => 38,
                DEVICE      => "Arria 10"--,
            )
            port map (
                data        => A_memreadfifo_data,
                wrreq       => A_memreadfifo_write,
                rdreq       => A_memreadfifo_read,
                wrclk       => A_mem_clk,
                rdclk       => pcieclk,
                q           => A_memreadfifo_q,
                rdempty     => A_memreadfifo_empty,
                rdusedw     => open,
                wrfull      => open,
                wrusedw     => open,
                aclr        => reset_ddr3--,
            );
            
            B_mreadfifo : entity work.ip_dcfifo
            generic map(
                ADDR_WIDTH  => 3,
                DATA_WIDTH  => 38,
                DEVICE      => "Arria 10"--,
            )
            port map (
                data        => B_memreadfifo_data,
                wrreq       => B_memreadfifo_write,
                rdreq       => B_memreadfifo_read,
                wrclk       => B_mem_clk,
                rdclk       => pcieclk,
                q           => B_memreadfifo_q,
                rdempty     => B_memreadfifo_empty,
                rdusedw     => open,
                wrfull      => open,
                wrusedw     => open,
                aclr        => reset_ddr3--,
            );
            
            A_mdatafdfifo : entity work.ip_dcfifo
            generic map(
                ADDR_WIDTH  => 4,
                DATA_WIDTH  => 256,
                DEVICE      => "Arria 10"--,
            )
            port map (
                data        => A_mem_q,
                wrreq       => A_mem_q_valid,
                rdreq       => A_memdatafifo_read,
                wrclk       => A_mem_clk,
                rdclk       => pcieclk,
                q           => A_memdatafifo_q,
                rdempty     => A_memdatafifo_empty,
                rdusedw     => open,
                wrfull      => open,
                wrusedw     => open,
                aclr        => reset_ddr3--,
            );
            
            B_mdatafdfifo : entity work.ip_dcfifo
            generic map(
                ADDR_WIDTH  => 4,
                DATA_WIDTH  => 256,
                DEVICE      => "Arria 10"--,
            )
            port map (
                data        => B_mem_q,
                wrreq       => B_mem_q_valid,
                rdreq       => B_memdatafifo_read,
                wrclk       => B_mem_clk,
                rdclk       => pcieclk,
                q           => B_memdatafifo_q,
                rdempty     => B_memdatafifo_empty,
                rdusedw     => open,
                wrfull      => open,
                wrusedw     => open,
                aclr        => reset_ddr3--,
            );

    end architecture RTL;
