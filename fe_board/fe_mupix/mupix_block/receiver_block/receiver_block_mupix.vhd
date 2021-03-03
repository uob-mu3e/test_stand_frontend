-----------------------------------
--
-- On detector FPGA for layer 0/1
-- Receiver block for all the LVDS links
-- Niklaus Berger, May 2013
-- 
-- nberger@physi.uni-heidelberg.de
--
----------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.mupix_types.all;
use work.mupix_registers.all;
use work.daq_constants.all;

entity receiver_block_mupix is 
    generic(
        NINPUT : integer := 36;
        NCHIPS : integer := 15
    );
    port (
        i_reset_n       : in  std_logic;
        i_nios_clk      : in  std_logic;
        checker_rst_n   : in  std_logic_vector(NINPUT-1 downto 0);
        rx_in           : in  std_logic_vector(NINPUT-1 DOWNTO 0);
        rx_inclock_A    : in  std_logic;
        rx_inclock_B    : in  std_logic;
        o_rx_status     : out reg32array_t(NINPUT-1 downto 0);
        o_rx_ready      : out std_logic_vector(NINPUT-1 downto 0);
        rx_data         : out bytearray_t(NINPUT-1 downto 0);
        rx_k            : out std_logic_vector(NINPUT-1 downto 0);
        rx_clkout       : out std_logic_vector(2 downto 0);
        rx_doubleclk    : out std_logic_vector(1 downto 0)--;
    );
end receiver_block_mupix;


architecture rtl of receiver_block_mupix is

    signal reset_n              : std_logic_vector(NINPUT-1 DOWNTO 0);

    signal rx_out               : std_logic_vector(NINPUT*10-1 downto 0);
    signal rx_out_temp          : std_logic_vector(NINPUT*10-1 downto 0);
    signal rx_clk               : std_logic_vector(2 downto 0);

    signal rx_dpa_locked        : std_logic_vector (NINPUT-1 DOWNTO 0);
    signal rx_align             : std_logic_vector (NINPUT-1 DOWNTO 0);
    signal rx_fifo_reset        : std_logic_vector (NINPUT-1 DOWNTO 0);
    signal rx_reset             : std_logic_vector (NINPUT-1 DOWNTO 0);

    signal rx_locked            : std_logic_vector(1 downto 0);

    signal rx_inclock_A_pll     : std_logic;
    signal rx_enable_A          : std_logic;
    signal rx_synclock_A        : std_logic;
    signal rx_dpaclock_A        : std_logic;

    signal rx_inclock_B_pll     : std_logic;
    signal rx_enable_B          : std_logic;
    signal rx_synclock_B        : std_logic;
    signal rx_dpaclock_B        : std_logic;

    signal rx_inclock_FF_pll    : std_logic;
    signal rx_enable_FF         : std_logic;
    signal rx_synclock_FF       : std_logic;
    signal rx_dpaclock_FF       : std_logic;
    signal rx_dpa_locked_FF     : std_logic_vector (1 DOWNTO 0);
    signal rx_align_FF          : std_logic_vector (1 DOWNTO 0);
    signal rx_fifo_reset_FF     : std_logic_vector (1 DOWNTO 0);
    signal rx_reset_FF          : std_logic_vector (1 DOWNTO 0);

    signal rx_inclock_A_ctrl    : std_logic;

    signal rx_ready_FF          : std_logic_vector(1 downto 0);
    signal rx_k_FF              : std_logic_vector(1 downto 0);
    signal rx_state_FF          : std_logic_vector(7 downto 0);
    signal rx_out_FF            : std_logic_vector(19 downto 0);
    signal rx_ready             : std_logic_vector(NINPUT-1 downto 0);
    signal disp_err             : std_logic_vector(NINPUT-1 downto 0);
    type   disp_err_counter_t   is array (natural range <>) of std_logic_vector(27 downto 0);
    signal disp_err_counter     : disp_err_counter_t(NINPUT-1 downto 0);
    signal rx_state             : std_logic_vector(NINPUT*2-1 downto 0);
    signal lvds_status_empty    : std_logic_vector(NINPUT-1 downto 0);
    signal lvds_status_buffer   : reg32array_t    (NINPUT-1 downto 0);
    signal lvds_status_rdreq    : std_logic_vector(NINPUT-1 downto 0);

    signal decoder_ena          : std_logic_vector(NINPUT-1 downto 0);

begin

    rx_clk(0) <= rx_synclock_A;
    rx_clk(1) <= rx_synclock_B;	
    rx_clk(2) <= rx_synclock_FF;
    rx_clkout <= rx_clk;

    --rx_fifo_reset		<= rx_reset;

    -- a little clk stunt to make the lvds transceiver work:
    lvds_clk_ctrl : component work.cmp.clk_ctrl_single
    port map (
        inclk  => rx_inclock_A,
        outclk => rx_inclock_A_ctrl
    );

    lpll_A: entity work.lvdspll
    PORT MAP
    (
        refclk   => rx_inclock_A_ctrl,
        rst      => '0',
        outclk_0 => rx_inclock_A_pll,
        outclk_1 => rx_enable_A,
        outclk_2 => rx_synclock_A,
        outclk_3 => rx_dpaclock_A,
        outclk_4 => rx_doubleclk(0),
        locked   => rx_locked(0)
    );

    lvds_rec_small: entity work.lvds_receiver
    PORT MAP
    (
        pll_areset              => not rx_locked(0),
        rx_channel_data_align   => rx_align(26 downto 0),
        rx_dpaclock             => rx_dpaclock_A,
        rx_enable               => rx_enable_A,
        rx_fifo_reset           => rx_fifo_reset(26 downto 0),
        rx_in                   => rx_in(26 downto 0),
        rx_inclock              => rx_inclock_A_pll,
        rx_reset                => rx_reset(26 downto 0),
        rx_syncclock            => rx_synclock_A,
        rx_dpa_locked           => rx_dpa_locked(26 downto 0),
        rx_out                  => rx_out(269 downto 0)
    );

    lpll_B: entity work.lvdspll
    PORT MAP
    (
        refclk   => rx_inclock_B,
        rst      => '0',
        outclk_0 => rx_inclock_B_pll,
        outclk_1 => rx_enable_B,
        outclk_2 => rx_synclock_B,
        outclk_3 => rx_dpaclock_B,
        outclk_4 => rx_doubleclk(1),
        locked   => rx_locked(1)
    );

    lvds_rec: entity work.lvds_receiver_small
    PORT MAP
    (
        pll_areset              => not rx_locked(1),
        rx_channel_data_align   => rx_align(35 downto 27),
        rx_dpaclock             => rx_dpaclock_B,
        rx_enable               => rx_enable_B,
        rx_fifo_reset           => rx_fifo_reset(35 downto 27),
        rx_in                   => rx_in(35 downto 27),
        rx_inclock              => rx_inclock_B_pll,
        rx_reset                => rx_reset(35 downto 27),
        rx_syncclock            => rx_synclock_B,
        rx_dpa_locked           => rx_dpa_locked(35 downto 27),
        rx_out                  => rx_out_temp(359 downto 270)
    );

    -- Input D9 is inverted...
    rx_out(359 downto 350) <= not rx_out_temp(359 downto 350);
    rx_out(349 downto 270) <= rx_out_temp(349 downto 270);

    gendec:
    FOR i in NINPUT-1 downto 0 generate	
        datadec: entity work.data_decoder 
            port map(
                reset_n         => reset_n(i),
                checker_rst_n   => checker_rst_n(i),
                clk             => rx_clk(i/27),
                rx_in           => rx_out(i*10+9 downto i*10),

                rx_reset        => rx_reset(i),
                rx_fifo_reset   => rx_fifo_reset(i),
                rx_dpa_locked   => rx_dpa_locked(i),
                rx_locked       => rx_locked(i/26),
                rx_align        => rx_align(i),

                ready           => rx_ready(i),
                data            => rx_data(i),
                k               => rx_k(i),
                state_out       => rx_state(i*2+1 downto i*2),
                disp_err        => disp_err(i)
            );

        o_rx_ready <= rx_ready;

        process(rx_clk(i/27))
        begin
            if(rising_edge(rx_clk(i/27))) then
                if(disp_err(i)='1') then
                    disp_err_counter(i) <= disp_err_counter(i) + '1';
                end if;
            end if;
        end process;

        sync_fifo_cnt : entity work.ip_dcfifo
        generic map(
            ADDR_WIDTH  => 3,
            DATA_WIDTH  => 32,
            SHOWAHEAD   => "ON",
            OVERFLOW    => "ON",
            DEVICE      => "Arria V"--,
        )
        port map(
            aclr                                => '0',
            data(MP_LVDS_STATUS_DISP_ERR_RANGE) => disp_err_counter(i),
            data(MP_LVDS_STATUS_PLL_LOCKED_BIT) => rx_locked(i/27),
            data(MP_LVDS_STATUS_STATE_RANGE)    => rx_state(i*2+1 downto i*2),
            data(MP_LVDS_STATUS_READY_BIT)      => rx_ready(i),
            rdclk                               => i_nios_clk,
            rdreq                               => lvds_status_rdreq(i),
            rdempty                             => lvds_status_empty(i),
            wrclk                               => rx_clk(i/27),
            wrreq                               => '1',
            q                                   => lvds_status_buffer(i)
        );

        process(i_nios_clk)
        begin
            if(rising_edge(i_nios_clk)) then
                if(lvds_status_empty(i)='0') then
                    lvds_status_rdreq(i) <= '1';
                    o_rx_status(i)       <= lvds_status_buffer(i);
                else
                    lvds_status_rdreq(i) <= '0';
                end if;
            end if;
        end process;

        sync_fifo_rst : entity work.ip_dcfifo
        generic map(
            ADDR_WIDTH  => 2,
            DATA_WIDTH  => 1,
            SHOWAHEAD   => "OFF",
            OVERFLOW    => "ON",
            DEVICE      => "Arria V"--,
        )
        port map(
            aclr            => '0',
            data(0)         => i_reset_n,
            rdclk           => rx_clk(i/27), -- TODO: go with only 2 instead of NLVDS syncs here 
            rdreq           => '1',
            wrclk           => i_nios_clk,
            wrreq           => '1',
            q               => reset_n(i downto i)--,
        );

    end generate;

end rtl;