-- last change: M.Mueller, Oktober 2020 (muellem@uni-mainz.de)

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_misc.all;
use work.daq_constants.all;
use work.mupix_constants.all;
use work.mupix_registers.all;

entity mupix_block is
generic(
    NCHIPS                  : integer := 8;
    NCHIPS_SPI              : integer := 8;
    NPORTS                  : integer := 1;
    NLVDS                   : integer := 32;
    NINPUTS_BANK_A          : integer := 16;
    NINPUTS_BANK_B          : integer := 16--;
);
port (
    i_fpga_id               : in  std_logic_vector(7 downto 0);

    -- chip dacs
    i_CTRL_SDO_A            : in  std_logic; --TODO !!
    o_CTRL_SDI              : out std_logic_vector(NPORTS-1 downto 0);
    o_CTRL_SCK1             : out std_logic_vector(NPORTS-1 downto 0);
    o_CTRL_SCK2             : out std_logic_vector(NPORTS-1 downto 0);
    o_CTRL_Load             : out std_logic_vector(NPORTS-1 downto 0);
    o_CTRL_RB               : out std_logic_vector(NPORTS-1 downto 0);

    -- board dacs
    i_SPI_DOUT_ADC_0_A      : in  std_logic;
    o_SPI_DIN0_A            : out std_logic;
    o_SPI_CLK_A             : out std_logic;
    o_SPI_LD_ADC_A          : out std_logic;
    o_SPI_LD_TEMP_DAC_A     : out std_logic;
    o_SPI_LD_DAC_A          : out std_logic;  

    i_run_state_125           : in  run_state_t;
    i_run_state_156           : in  run_state_t;
    o_ack_run_prep_permission : out std_logic;

    -- mupix dac regs
    i_reg_add               : in  std_logic_vector(7 downto 0);
    i_reg_re                : in  std_logic;
    o_reg_rdata             : out std_logic_vector(31 downto 0);
    i_reg_we                : in  std_logic;
    i_reg_wdata             : in  std_logic_vector(31 downto 0);

    -- data 
    o_fifo_wdata            : out std_logic_vector(35 downto 0);
    o_fifo_write            : out std_logic;

    i_lvds_data_in          : in  std_logic_vector(NLVDS-1 downto 0);

    i_reset                 : in  std_logic;
    -- 156.25 MHz
    i_clk156                   : in  std_logic;
    i_clk125                : in  std_logic;
    i_lvds_rx_inclock_A     : in  std_logic;
    i_lvds_rx_inclock_B     : in  std_logic;
    i_sync_reset_cnt        : in  std_logic--;
);
end entity;

architecture arch of mupix_block is

signal reset_n : std_logic;

    -- chip dacs
    signal mp8_busy_n : std_logic_vector(NCHIPS_SPI - 1 downto 0);
    signal mp8_mem_data_out : std_logic_vector(31 downto 0);
    signal mp8_wren : std_logic_vector(NCHIPS_SPI - 1 downto 0);
    signal mp8_ld : std_logic_vector(NCHIPS_SPI - 1 downto 0);
    signal mp8_rb : std_logic_vector(NCHIPS_SPI - 1 downto 0);
    signal mp8_ctrl_dout : std_logic_vector(NCHIPS_SPI - 1 downto 0);
    signal mp8_ctrl_din : std_logic_vector(NCHIPS_SPI - 1 downto 0);
    signal mp8_ctrl_clk1 : std_logic_vector(NCHIPS_SPI - 1 downto 0);
    signal mp8_ctrl_clk2 : std_logic_vector(NCHIPS_SPI - 1 downto 0);
    signal mp8_ctrl_ld : std_logic_vector(NCHIPS_SPI - 1 downto 0);
    signal mp8_ctrl_rb : std_logic_vector(NCHIPS_SPI - 1 downto 0);

     -- board dacs
    type   state_spi is (waiting, starting, read_out_pix, write_pix, read_out_th, ending); -- to be used somewhere?
    signal spi_state : state_spi; -- to be used somewhere?

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

    signal chip_dac_data_we : std_logic_vector(31 downto 0);
    signal chip_dac_we : std_logic;

    signal chip_dac_data : std_logic_vector(31 downto 0);
    signal chip_dac_ren : std_logic;
    signal chip_dac_fifo_empty : std_logic;
    signal chip_dac_ready : std_logic;
    signal chip_dac_usedw : std_logic_vector(6 downto 0);
    signal reset_chip_dac_fifo : std_logic;
    signal ckdiv         : std_logic_vector(15 downto 0);
     
    signal write_regs_mupix : reg32array_t(NREGISTERS_MUPIX_WR-1+64 downto 64);
    signal read_regs_mupix : reg32array_t(NREGISTERS_MUPIX_RD-1+64 downto 64);

    signal reset_n_lvds : std_logic;

    -- regs nios
    signal debug_chip_select : std_logic_vector(31 downto 0);
    signal timestamp_gray_invert : std_logic_vector(31 downto 0);
    signal link_mask : std_logic_vector(31 downto 0);
    signal mux_read_regs_nios : std_logic_vector(6 downto 0);
    signal ro_prescaler : std_logic_vector(31 downto 0);
    signal read_regs_mupix_mux : std_logic_vector(31 downto 0);

    signal lvds_data_valid : std_logic_vector(NLVDS-1 downto 0);
    signal disable_conditions_for_run_ack : std_logic;

    signal reg_hits_ena_count : std_logic_vector(31 downto 0);

begin
    
    reset_n <= '0' when (i_reset='1' or i_run_state_156=RUN_STATE_SYNC) else '1';
    --reset_n <= '0' when (i_reset='1' or i_run_state_125=RUN_STATE_SYNC) else '1';

    e_mupix_run_start_ack : work.mupix_run_start_ack
    generic map (
        NLVDS                       => NLVDS--,
    )
    port map (
        i_clk156                    => i_clk156,
        i_reset                     => i_reset,
        i_disable                   => disable_conditions_for_run_ack, -- TODO: connect to sc
        i_stable_required           => x"F000", -- TODO: connect to sc
        i_lvds_err_counter          => read_regs_mupix(LVDS_ERRCOUNTER_REGISTER_R + NLVDS - 1 downto LVDS_ERRCOUNTER_REGISTER_R),
        i_lvds_data_valid           => lvds_data_valid,
        i_lvds_mask                 => write_regs_mupix(LINK_MASK_REGISTER_W + 1 downto LINK_MASK_REGISTER_W),
        i_sc_busy                   => or_reduce(mp8_wren & (not chip_dac_fifo_empty)),
        i_run_state_156             => i_run_state_156,
        o_ack_run_prep_permission   => o_ack_run_prep_permission--,
    );

    chip_dac_fifo_write : work.ip_scfifo
    generic map (
        ADDR_WIDTH => 7,
        DATA_WIDTH => 32--,
    )
    port map (
        empty           => chip_dac_fifo_empty,
        rdreq           => chip_dac_ren,
        q               => chip_dac_data,

        almost_empty    => open,
        almost_full     => open,
        usedw           => chip_dac_usedw,

        full            => open,
        wrreq           => chip_dac_we,
        data            => chip_dac_data_we,

        sclr            => reset_chip_dac_fifo,
        clock           => i_clk156--,
    );

    -- chip dacs slow_controll
    -- MK: since we have only one chip to configure at the moment
    -- we hard code this 
    e_mp8_sc_master : work.mp8_sc_master
    generic map(NCHIPS => NCHIPS_SPI)
    port map (
        clk             => i_clk156,
        reset_n         => reset_n,
        mem_data_in     => chip_dac_data,
        busy_n          => mp8_busy_n,--busy_n => mp8_busy_n,
        start           => chip_dac_ready,

        fifo_re         => chip_dac_ren,
        fifo_empty      => chip_dac_fifo_empty,
        mem_data_out    => mp8_mem_data_out,--mem_data_out => mp8_mem_data_out,
        wren            => mp8_wren,--mp8_wren,
        ctrl_ld         => mp8_ld,--mp8_ld,
        ctrl_rb         => mp8_rb,--mp8_rb,
        done            => open, 
        stateout        => open--,
    );

    gen_slowc:
    for i in 0 to NCHIPS_SPI-1 generate
    e_mp10_slowcontrol : work.mp10_slowcontrol
    port map(
        clk         => i_clk156,
        reset_n     => reset_n,
        ckdiv       => ckdiv, -- this need to be set to a register at the moment 0
        mem_data    => mp8_mem_data_out,
        wren        => mp8_wren(i),
        ld_in       => mp8_ld(i),
        rb_in       => mp8_rb(i),
        ctrl_dout   => mp8_ctrl_dout(i),
        ctrl_din    => mp8_ctrl_din(i),
        ctrl_clk1   => mp8_ctrl_clk1(i),
        ctrl_clk2   => mp8_ctrl_clk2(i),
        ctrl_ld     => mp8_ctrl_ld(i),
        ctrl_rb     => mp8_ctrl_rb(i),
        busy_n      => mp8_busy_n(i),
        dataout     => open--,
    );
    end generate gen_slowc;

    process(i_clk156)
    begin
        if(rising_edge(i_clk156)) then	
            mp8_ctrl_dout(0)    <= i_CTRL_SDO_A;
        end if;
    end process;

    process(i_clk156)
    begin
        if(rising_edge(i_clk156)) then	
            o_CTRL_SDI      <= mp8_ctrl_din;
            o_CTRL_SCK1     <= mp8_ctrl_clk1;
            o_CTRL_SCK2     <= mp8_ctrl_clk2;
            o_CTRL_Load     <= mp8_ctrl_ld;
            o_CTRL_RB       <= mp8_ctrl_rb;
        end if;
    end process;

    -- board dacs slow_controll
    A_spi_sdo_front         <= i_SPI_DOUT_ADC_0_A & "00";-- A_spi_dout_dac_front & A_dac4_dout_front;
    o_SPI_LD_ADC_A          <= A_spi_ldn_front(2);
    o_SPI_LD_TEMP_DAC_A     <= A_spi_ldn_front(1);
    o_SPI_LD_DAC_A          <= A_spi_ldn_front(0);

   -- regs reading
   board_dac_regs : process (i_clk156, reset_n)
   begin 
       if (reset_n = '0') then 
            board_th_low        <= (others => '0');
            board_th_high       <= (others => '0');
            board_injection     <= (others => '0');
            board_th_pix        <= (others => '0');
            A_spi_wren_front    <= (others => '0');
            o_reg_rdata         <= (others => '0');
            chip_dac_data_we    <= (others => '0');
            ckdiv               <= (others => '0');
            ro_prescaler        <= (others => '0');
            debug_chip_select   <= (others => '0');
            timestamp_gray_invert <= (others => '0');
            link_mask           <= (others => '0');
            chip_dac_we         <= '0';
            reset_chip_dac_fifo <= '0';
            chip_dac_ready      <= '0';
            reset_n_lvds        <= '0';
            mux_read_regs_nios  <= (others => '0');
            disable_conditions_for_run_ack <= '0';
        elsif rising_edge(i_clk156) then 
            
            chip_dac_we         <= '0';
            chip_dac_ready      <= '0';
            reset_chip_dac_fifo <= '0';
            o_reg_rdata         <= (others => '0');
            ckdiv               <= ckdiv;
            
            -- here we have to apply a register map with constants!
            
            -- DO NOT PUT ELSIF FOR R&W REGS HERE !! 
            -- Quartus does not know that i_req_we and i_req_re cannot be 1 at the same time
            -- Makes timing closure more difficult if elsif is used
            
            if ( i_reg_add = x"83" and i_reg_we = '1' ) then
                board_th_low    <= i_reg_wdata(15 downto 0);
                board_th_high   <= i_reg_wdata(31 downto 16);
            end if;
            
            if ( i_reg_add = x"84" and i_reg_we = '1' ) then
                board_injection <= i_reg_wdata(15 downto 0);
                board_th_pix    <= i_reg_wdata(31 downto 16);
            end if;
            
            if ( i_reg_add = x"85" and i_reg_we = '1' ) then
                board_temp_dac <= i_reg_wdata(15 downto 0);
                board_temp_adc <= i_reg_wdata(31 downto 16);
            end if;
            
            if ( i_reg_add = x"86" and i_reg_re = '1' ) then
                o_reg_rdata(15 downto 0) <= injection1_out_A_front;
            end if;
            
            if ( i_reg_add = x"87" and i_reg_re = '1' ) then
                o_reg_rdata(15 downto 0) <= threshold_pix_out_A_front;
            end if;
            
            if ( i_reg_add = x"88" and i_reg_re = '1' ) then
                o_reg_rdata(15 downto 0) <= threshold_low_out_A_front;
            end if;
            
            if ( i_reg_add = x"89" and i_reg_re = '1' ) then
                o_reg_rdata(15 downto 0) <= threshold_high_out_A_front;
            end if;
            
            if ( i_reg_add = x"8A" and i_reg_re = '1' ) then
                o_reg_rdata(15 downto 0) <= board_temp_dac_out;
            end if;
            
            if ( i_reg_add = x"8B" and i_reg_re = '1' ) then
                o_reg_rdata <= board_temp_adc_out;
            end if;
            
            if ( i_reg_add = x"8C" and i_reg_we = '1' ) then
                A_spi_wren_front <= i_reg_wdata(2 downto 0);
            end if;
            
            if ( i_reg_add = x"8D" and i_reg_we = '1' ) then
                chip_dac_data_we <= i_reg_wdata(31 downto 0);
                chip_dac_we      <= '1';
            end if;
            
            if ( i_reg_add = x"8E" and i_reg_we = '1' ) then
                chip_dac_ready      <= i_reg_wdata(0);
                ckdiv               <= i_reg_wdata(31 downto 16);
            end if;
            
            if ( i_reg_add = x"8F" and i_reg_we = '1' ) then
                reset_n_lvds    <= i_reg_wdata(0);
            else
                reset_n_lvds    <= '1';
            end if;

            if ( i_reg_add = x"90" and i_reg_we = '1' ) then
                ro_prescaler           <= i_reg_wdata;
            end if;
            
            if ( i_reg_add = x"91" and i_reg_we = '1' ) then
                debug_chip_select      <= i_reg_wdata;
            end if;
            
            if ( i_reg_add = x"92" and i_reg_we = '1' ) then
                timestamp_gray_invert  <= i_reg_wdata;
            end if;
            
            if ( i_reg_add = x"93" and i_reg_we = '1' ) then
                mux_read_regs_nios     <= i_reg_wdata(6 downto 0);
            end if;
            
            if ( i_reg_add = x"94" and i_reg_re = '1' ) then
                o_reg_rdata            <= read_regs_mupix_mux;
            end if;
            
            if ( i_reg_add = x"95" and i_reg_we = '1' ) then
                reset_chip_dac_fifo    <= i_reg_wdata(0);
            end if;
            
            if ( i_reg_add = x"96" and i_reg_we = '1') then
                link_mask          <= i_reg_wdata;
            end if;-- NO ELSIF HERE!!
            
            if ( i_reg_add = x"96" and i_reg_re = '1') then
                o_reg_rdata        <= link_mask;
            end if;
            
            if ( i_reg_add = x"97" and i_reg_re = '1' ) then
                if(NLVDS > 31) then
                    o_reg_rdata            <= lvds_data_valid(31 downto 0);
                else
                    o_reg_rdata            <= lvds_data_valid(NLVDS-1 downto 0);
                end if;
            end if;
            
            if ( i_reg_add = x"98" and i_reg_we = '1' ) then
                disable_conditions_for_run_ack  <= i_reg_wdata(0);
            end if;-- NO ELSIF HERE!!
            
            if( i_reg_add = x"98" and i_reg_re = '1') then
                o_reg_rdata(0)                  <= disable_conditions_for_run_ack;
                o_reg_rdata(31 downto 1)        <= (others => '0');
            end if;
            
            if ( i_reg_add = x"99" and i_reg_re = '1' ) then
                o_reg_rdata <= std_logic_vector(to_unsigned(2**chip_dac_usedw'length - to_integer(unsigned(chip_dac_usedw)), 32));
            end if;
            
            if ( i_reg_add = x"9A" and i_reg_re = '1' ) then
                o_reg_rdata <= reg_hits_ena_count;
            end if;
            
            if ( i_reg_add = x"9B" and i_reg_re = '1' ) then
                if(NLVDS > 32) then
                    for I in 0 to NLVDS-33 loop
                        o_reg_rdata(I)     <= lvds_data_valid(32 + I);
                    end loop;
                else
                    o_reg_rdata            <= (others => '0');
                end if;
            end if;
            
        end if;
    end process board_dac_regs;

    e_spi_master : work.spi_master 
    port map(
        clk                 => i_clk156,
        reset_n             => reset_n,
        injection1_reg      => board_injection,
        threshold_pix_reg   => board_th_pix,
        threshold_low_reg   => board_th_low,
        threshold_high_reg  => board_th_high,
        temp_dac_reg        => board_temp_dac,
        temp_adc_reg        => board_temp_adc,
        wren                => A_spi_wren_front,
        busy_n              => A_spi_busy_n_front,

        spi_sdi	            => o_SPI_DIN0_A,
        spi_sclk            => o_SPI_CLK_A,
        spi_load_n          => A_spi_ldn_front,
        spi_sdo             => A_spi_sdo_front,
        
        injection1_out      => injection1_out_A_front,
        threshold_pix_out   => threshold_pix_out_A_front,
        threshold_low_out   => threshold_low_out_A_front,
        threshold_high_out  => threshold_high_out_A_front,
        temp_dac_out        => board_temp_dac_out,
        temp_adc_out        => board_temp_adc_out
    );

    e_mupix_datapath : work.mupix_datapath
    generic map (
        NCHIPS              => NCHIPS,
        NLVDS               => NLVDS,
        NSORTERINPUTS       => NSORTERINPUTS,   --up to 4 LVDS links merge to one sorter
        NINPUTS_BANK_A      => NINPUTS_BANK_A,
        NINPUTS_BANK_B      => NINPUTS_BANK_B
    )
    port map (
        i_reset_n           => reset_n,
        i_reset_n_lvds      => reset_n_lvds,

        i_clk156            => i_clk156,
        i_clk125            => i_clk125,

        i_lvds_rx_inclock_A => i_lvds_rx_inclock_A,
        i_lvds_rx_inclock_B => i_lvds_rx_inclock_B,

        lvds_data_in        => i_lvds_data_in,

        write_sc_regs       => write_regs_mupix,
        read_sc_regs        => read_regs_mupix,

        o_fifo_wdata        => o_fifo_wdata,
        o_fifo_write        => o_fifo_write,
        o_lvds_data_valid   => lvds_data_valid,
        o_hits_ena_count    => reg_hits_ena_count,

        i_sync_reset_cnt    => i_sync_reset_cnt,
        i_fpga_id           => i_fpga_id,
        i_run_state_125     => i_run_state_125,
        i_run_state_156     => i_run_state_156--,
    );

    write_regs_mupix(RO_PRESCALER_REGISTER_W)               <= ro_prescaler;
    write_regs_mupix(DEBUG_CHIP_SELECT_REGISTER_W)          <= debug_chip_select;
    write_regs_mupix(TIMESTAMP_GRAY_INVERT_REGISTER_W)      <= timestamp_gray_invert;
    write_regs_mupix(LINK_MASK_REGISTER_W)                  <= link_mask;

    mux_read_regs : process (i_clk156, reset_n)
    begin 
        if (reset_n = '0') then 
            read_regs_mupix_mux <= (others => '0');
        elsif rising_edge(i_clk156) then 
        -- make sure we cannot access signals that are not there
            if(mux_read_regs_nios < NREGISTERS_MUPIX_RD)then
                read_regs_mupix_mux <= read_regs_mupix(conv_integer(mux_read_regs_nios));
            else
                read_regs_mupix_mux <= read_regs_mupix(NREGISTERS_MUPIX_RD-1);
            end if;
        end if;
    end process mux_read_regs;

end architecture;
