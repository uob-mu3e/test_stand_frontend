library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_misc.all;
use work.daq_constants.all;
use work.mupix_constants.all;

entity mupix_run_start_ack is
generic(
    NLVDS                   : integer := 32--;
);
port (
    i_clk125                    : in  std_logic;
    i_clk156                    : in  std_logic;
    i_reset                     : in  std_logic;
    i_disable                   : in  std_logic;
    i_stable_required           : in  unsigned(15 downto 0);
    i_lvds_err_counter          : in  reg32array_t(NLVDS-1 downto 0);
    i_lvds_data_valid           : in  std_logic_vector(NLVDS-1 downto 0);
    i_lvds_mask                 : in  reg32array_t(1 downto 0);
    i_sc_busy                   : in  std_logic;
    i_run_state_125             : in  run_state_t;
    o_ack_run_prep_permission   : out std_logic--;
);
end entity;

architecture arch of mupix_run_start_ack is

    --signal lvds_stable_count
    signal stable_counter   : unsigned(15 downto 0) := (others => '0');
    signal lvds_stable      : std_logic_vector(NLVDS-1 downto 0) := (others => '0');
    signal prev_err_counter : std_logic_vector(NLVDS*4-1 downto 0);
    signal lvds_mask        : std_logic_vector(NLVDS-1 downto 0);
    signal disable          : std_logic := '0';

    signal ack_run_prep_permission : std_logic;
    constant RDATA_RESET : std_logic_vector(0 downto 0) := (others => '0');
begin

    process(i_clk125)
        variable lvds_mask_slv  : std_logic_vector(63 downto 0) := (others => '0');
    begin
        if(rising_edge(i_clk125))then
            lvds_mask_slv       := i_lvds_mask(1) & i_lvds_mask(0);
            lvds_mask           <= lvds_mask_slv(NLVDS-1 downto 0);
        end if;
    end process;

    process (i_clk125, i_reset)
    begin
        if(i_reset = '1') then
            ack_run_prep_permission     <= '0';
            stable_counter              <= (others => '0');
            lvds_stable                 <= (others => '0');
            prev_err_counter            <= (others => '0');

        elsif (rising_edge(i_clk125)) then
            if(disable = '1') then
                ack_run_prep_permission   <= '1';
            elsif(i_run_state_125 = RUN_STATE_PREP and i_sc_busy='0' and stable_counter = i_stable_required and and_reduce(i_lvds_data_valid or lvds_mask)='1') then
                ack_run_prep_permission   <= '1';
            else
                ack_run_prep_permission   <= '0';
            end if;

            for I in NLVDS-1 downto 0 loop
                if(prev_err_counter((I+1)*4-1 downto I*4) = i_lvds_err_counter(I)(3 downto 0)) then
                    lvds_stable(I)          <= '1';
                else
                    lvds_stable(I)          <= '0';
                end if;
                prev_err_counter((I+1)*4-1 downto I*4)  <= i_lvds_err_counter(I)(3 downto 0);
            end loop;

            if (and_reduce(lvds_stable)='0') then
                stable_counter              <= (others => '0');
            elsif (stable_counter /= i_stable_required) then
                stable_counter              <= stable_counter+1;
            end if;
        end if;
    end process;

    e_fifo_sync0 : entity work.fifo_sync
    generic map (
        RDATA_RESET_g => RDATA_RESET--,
    )
    port map (
        o_rdata(0)  => o_ack_run_prep_permission,
        i_rreset_n  => '1',
        i_rclk      => i_clk156,
        i_wdata(0)  => ack_run_prep_permission,
        i_wreset_n  => '1',
        i_wclk      => i_clk125--,
    );

    e_fifo_sync1 : entity work.fifo_sync
    generic map (
        RDATA_RESET_g => RDATA_RESET--,
    )
    port map (
        o_rdata(0)  => disable,
        i_rreset_n  => '1',
        i_rclk      => i_clk156,
        i_wdata(0)  => i_disable,
        i_wreset_n  => '1',
        i_wclk      => i_clk125--,
    );

end architecture;
