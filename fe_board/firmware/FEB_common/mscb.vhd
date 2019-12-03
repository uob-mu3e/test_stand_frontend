library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.daq_constants.all;

entity mscb is
generic (
    BAUD_RATE_g : positive := 115200;
    CLK_MHZ_g : real--;
);
port (
    -- avalon slave interface
    -- - read latency 1
    i_avs_address       : in    std_logic_vector(3 downto 0);
    i_avs_read          : in    std_logic;
    o_avs_readdata      : out   std_logic_vector(31 downto 0);
    i_avs_write         : in    std_logic;
    i_avs_writedata     : in    std_logic_vector(31 downto 0);
    o_avs_waitrequest   : out   std_logic;

    mscb_data_in                : in    std_logic;
    mscb_data_out               : out   std_logic;
    mscb_oe                     : out   std_logic;

    o_mscb_irq                  : out   std_logic;
    i_mscb_address              : in    std_logic_vector(15 downto 0);

    i_reset_n           : in    std_logic;
    i_clk               : in    std_logic--;
);
end entity;

architecture rtl of mscb is

    -- mscb data flow
    signal signal_in :                  std_logic;
    signal signal_out :                 std_logic;
    signal uart_serial_in :             std_logic;
    signal uart_read_enable :           std_logic;
    signal uart_parallel_out :          std_logic_vector(8 downto 0);

    signal in_fifo_read_request :       std_logic;
    signal in_fifo_write_request :      std_logic;
    signal in_fifo_empty :              std_logic;
    signal in_fifo_full :               std_logic;
    signal in_fifo_data_out :           std_logic_vector(8 downto 0);

    signal out_fifo_read_request :      std_logic;
    signal out_fifo_write_request :     std_logic;
    signal out_fifo_empty :             std_logic;
    signal out_fifo_full :              std_logic;
    signal out_fifo_data_out :          std_logic_vector(8 downto 0);

    signal mscb_nios_out :              std_logic_vector(8 downto 0);
    signal mscb_data_ready :            std_logic;

    signal uart_serial_out :            std_logic; --uart data for the output pin

    signal Transmitting  :              std_logic; -- uart data is being send

    signal addressing_data_in :         std_logic_vector(8 downto 0);
    signal addressing_data_out :        std_logic_vector(8 downto 0);
    signal addressing_wrreq :           std_logic;
    signal addressing_rdreq :           std_logic;
    signal rec_fifo_empty :             std_logic;

----------------------------------------
------------ Begin top level------------
----------------------------------------

begin

------------- external input/output connections -----------------

  --i/o switches
    process(Transmitting, i_reset_n, i_clk)
    begin
    if (Transmitting = '1') then
        mscb_data_out   <= uart_serial_out;
        mscb_oe         <= '1';
    else
        mscb_data_out   <= 'Z';
        --mscb_oe       <= '0';       -- single FPGA connected to converter chip
        mscb_oe         <= 'Z';         -- multiple FPGAs
    end if;
    end process;

    --hsma_d(0) <= 'Z' when Transmitting = '1' else 'Z';
    signal_in <= '1' when Transmitting = '1' else (mscb_data_in);

---------------internal connections-----------------------



    process(i_clk)
    begin
    if rising_edge(i_clk) then
        o_avs_readdata <= X"CCCCCCCC";

        if ( i_avs_address = X"0" and i_avs_read = '1' ) then
            o_avs_readdata <= (others => '0');
            o_avs_readdata(11 downto 0) <= '1' & in_fifo_full & in_fifo_empty & in_fifo_data_out;
        end if;

        in_fifo_read_request <= '0';
        out_fifo_write_request <= '0';
        if ( i_avs_address = X"0" and i_avs_write = '1' ) then
            in_fifo_read_request <= i_avs_writedata(10);
            out_fifo_write_request <= i_avs_writedata(9);
            mscb_nios_out <= i_avs_writedata(8 downto 0);
        end if;
        --
    end if;
    end process;

    ---- interrupt to nios ----
    o_mscb_irq                              <= not in_fifo_empty;

------------- Wire up components --------------------

  -- wire up uart reciever for mscb
    e_uart_rx : entity work.uart_reciever
    generic map (
        Clk_Ratio   => positive(1000000.0 * CLK_MHZ_g / real(BAUD_RATE_g))--,
    )
    port map (
        Clk         => i_clk,
        Reset       => not i_reset_n,
        DataIn      => uart_serial_in,
        ReadEnable  => uart_read_enable,
        DataOut     => uart_parallel_out
    );

    -- fifo addressing to nios (addressing is done, only commands intended for this mscb node)
    e_nios_fifo : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => 9,
        SHOWAHEAD   => "OFF"--,
    )
    port map (
        sclr            => not i_reset_n,
        clock           => i_clk,
        data            => addressing_data_out,
        rdreq           => in_fifo_read_request,
        wrreq           => addressing_wrreq,
        empty           => in_fifo_empty,
        full            => in_fifo_full,
        q               => in_fifo_data_out
    );

    -- fifo receiver to addressing
    e_fifo_data_in : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 3,
        DATA_WIDTH => 9,
        SHOWAHEAD   => "ON"--,
    )
    port map (
        sclr            => not i_reset_n,
        clock           => i_clk,
        data            => uart_parallel_out,
        rdreq           => addressing_rdreq,
        wrreq           => in_fifo_write_request,
        empty           => rec_fifo_empty,
        full            => open,
        q               => addressing_data_in
    );

    -- wire up uart transmitter for mscb
    e_uart_tx : entity work.uart_transmitter
    generic map (
        Clk_Ratio   => positive(1000000.0 * CLK_MHZ_g / real(BAUD_RATE_g))--,
    )
    port map (
        Clk             => i_clk,
        Reset           => not i_reset_n,
        DataIn          => out_fifo_data_out,
        DataReady       => not out_fifo_empty,
        ReadRequest     => out_fifo_read_request,
        DataOut         => uart_serial_out,
        Transmitting    => Transmitting
    );

    -- fifo nios to transmitter
    e_fifo_data_out : entity work.ip_scfifo
    generic map (
        ADDR_WIDTH => 8,
        DATA_WIDTH => 9,
        SHOWAHEAD   => "OFF"--,
    )
    port map (
        sclr            => not i_reset_n,
        clock           => i_clk,
        data            => mscb_nios_out,
        rdreq           => out_fifo_read_request,
        wrreq           => out_fifo_write_request,
        empty           => out_fifo_empty,
        full            => out_fifo_full,
        q               => out_fifo_data_out
    );

    e_mscb_addressing : entity work.mscb_addressing
    port map (
        i_clk           => i_clk,
        i_reset         => not i_reset_n,
        i_data          => addressing_data_in,
        i_empty         => rec_fifo_empty,
        i_address       => i_mscb_address,
        o_data          => addressing_data_out,
        o_wrreq         => addressing_wrreq,
        o_rdreq         => addressing_rdreq
    );


    process(i_clk, i_reset_n)
    begin
        if i_reset_n = '0' then
            --uart_serial_in            <='1';
            --hsma_d(0)                 <='Z';
        elsif rising_edge(i_clk) then
            in_fifo_write_request       <= uart_read_enable;
            uart_serial_in              <= signal_in;
        end if;
    end process;

end architecture;
