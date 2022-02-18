----------------------------------------------------------------------------
-- storage for Mupix TDACs
-- M. Mueller, Feb 2022
-----------------------------------------------------------------------------

library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_misc.all;

use work.mupix.all;
use work.mudaq.all;

use IEEE.math_real."ceil";
use IEEE.math_real."log2";

entity tdac_memory is
    generic( 
        N_CHIPS_g                 : positive := 4;
        PAGE_ADDR_WIDTH_g         : positive := 2;
        ADDR_WIDTH_g              : positive := 8--;
    );
    port(
        i_clk               : in  std_logic;
        i_reset_n           : in  std_logic;

        o_tdac_dpf_we       : out std_logic_vector(N_CHIPS_g-1 downto 0);
        o_tdac_dpf_wdata    : out std_logic_vector(3 downto 0);
        i_tdac_dpf_empty    : in  std_logic_vector(N_CHIPS_g-1 downto 0);

        i_data              : in  std_logic_vector(31 downto 0);
        i_we                : in  std_logic;
        i_chip              : in  integer range 0 to N_CHIPS_g-1--;
    );
end entity tdac_memory;

architecture RTL of tdac_memory is

    constant N_PAGES_PER_CHIP : integer := 4;
    constant N_COLS_PER_PAGE  : integer := 128/N_PAGES_PER_CHIP;


    constant N_PAGES : integer := 8; -- TODO

    

    type TDAC_page_type is record
        addr            :   std_logic_vector(ADDR_WIDTH_g-PAGE_ADDR_WIDTH_g-1 downto 0);
        page_id         :   integer range 0 to N_PAGES_PER_CHIP-1;
        bit_in_tdac     :   integer range 0 to 6;
        in_use          :   boolean;
        full            :   boolean;
        chip            :   integer range 0 to N_CHIPS_g-1;
        --chip_page_ID    :   integer range 0 to TODO : max pages per chip;
    end record;

    type TDAC_page_array_type   is array( natural range <> ) of TDAC_page_type;
    signal TDAC_page_array      : TDAC_page_array_type(N_PAGES-1 downto 0);

    signal next_free_page       : std_logic_vector(PAGE_ADDR_WIDTH_g-1 downto 0);
    signal next_free_page_int   : integer range 0 to N_PAGES-1;
    signal current_write_page   : integer range 0 to N_PAGES-1;

    signal current_page_addr    : std_logic_vector(PAGE_ADDR_WIDTH_g-1 downto 0);
    signal addr_in_current_page : std_logic_vector(ADDR_WIDTH_g-PAGE_ADDR_WIDTH_g-1 downto 0);

    signal ram_we               : std_logic;
    signal ram_wdata            : std_logic_vector(31 downto 0);
    signal ram_waddr            : std_logic_vector(ADDR_WIDTH_g-1 downto 0);
    signal ram_raddr            : std_logic_vector(ADDR_WIDTH_g-1 downto 0);
    signal ram_rdata            : std_logic_vector(31 downto 0);

    subtype page_id_type        is integer range 0 to N_PAGES_PER_CHIP-1;
    type page_id_array_type     is array( natural range <>) of page_id_type;

    signal current_write_page_id : page_id_array_type(N_CHIPS_g-1 downto 0);
    signal current_read_page_id  : page_id_array_type(N_CHIPS_g-1 downto 0);

    type read_state_type         is (searching_match,reading);
    signal read_state            : read_state_type;

    signal page_cycler           : integer range 0 to N_PAGES-1;
    signal last_page_cycler      : integer range 0 to N_PAGES-1;
    signal cycler_last_full      : boolean;
    signal cycler_last_chip      : integer range 0 to N_CHIPS_g-1;

    signal read_chip             : integer range 0 to N_CHIPS_g-1;
    signal read_page             : integer range 0 to N_PAGES-1;



begin

    ram_waddr           <= current_page_addr & addr_in_current_page;

    process (i_clk, i_reset_n) is
    begin
        if(i_reset_n = '0') then
            addr_in_current_page    <= (others => '0');
            current_page_addr       <= (others => '0');
            current_write_page      <= 0;
            next_free_page          <= (others => '0');
            TDAC_page_array         <= (others => (full => false, in_use => false, bit_in_tdac => 0, page_id => 0));
            current_read_page_id    <= (others => 0);
            current_write_page_id   <= (others => 0);
            read_state              <= searching_match;
            page_cycler             <= 0;
            last_page_cycler        <= N_PAGES-1;
            cycler_last_full        <= false;
            cycler_last_chip        <= 0;
            
        elsif(rising_edge(i_clk)) then

            ---------------------------------------------
            -- write process
            ---------------------------------------------
            ram_we          <= i_we;
            ram_wdata       <= i_data;

            for I in 0 to N_PAGES-1 loop
                if(TDAC_page_array(I).in_use = false) then
                    next_free_page <= std_logic_vector(to_unsigned(I,PAGE_ADDR_WIDTH_g));
                    next_free_page_int <= I;
                end if;
            end loop;

            if(i_we = '1') then 
                if(addr_in_current_page= "000100") then  -- TODO insert proper end addr (complete cols)
                    TDAC_page_array(current_write_page).full     <= true;
                    addr_in_current_page                         <= (others => '0');
                    current_write_page                           <= next_free_page_int;
                    current_page_addr                            <= next_free_page;

                    if(TDAC_page_array(current_write_page).page_id = N_PAGES_PER_CHIP-1) then
                        TDAC_page_array(current_write_page).page_id <= 0;
                    else
                        TDAC_page_array(current_write_page).page_id <= TDAC_page_array(current_write_page).page_id + 1;
                    end if;
                else
                    addr_in_current_page <= addr_in_current_page + '1';
                    TDAC_page_array(current_write_page).chip    <= i_chip; 
                    TDAC_page_array(current_write_page).in_use  <= true; 
                end if;
            end if;


            -----------------------------------------------
            -- read process
            -----------------------------------------------
            o_tdac_dpf_we <= (others => '0');


            -- TODO select only correct read page ID for chip X
            if(page_cycler = N_PAGES-1) then 
                page_cycler <= 0;
            else
                page_cycler <= page_cycler + 1;
            end if;

            last_page_cycler <= page_cycler;
            cycler_last_chip <= TDAC_page_array(page_cycler).chip;
            cycler_last_full <= TDAC_page_array(page_cycler).full;


            case read_state is
              when searching_match =>
                for I in 0 to N_CHIPS_g loop
                    if(cycler_last_full= true and cycler_last_chip = I and i_tdac_dpf_empty(I) = '1') then
                        read_state <= reading;
                        read_chip <= I;
                        read_page <= last_page_cycler;
                    end if;
                end loop;
              when reading =>
                -- read the page for bit_in_tdac, incr. bit_in_tdac , if bit_in_tdac = 6 --> in_use=0
                if(TDAC_page_array(read_page).addr = "000011") then  -- end addr-1 to avoid cycler selecting the same (now empty) TDAC_page again
                    if(TDAC_page_array(read_page).bit_in_tdac = 6) then 
                        TDAC_page_array(read_page).full <= false;
                        TDAC_page_array(read_page).in_use <= false;
                    else
                        TDAC_page_array(read_page).bit_in_tdac <= TDAC_page_array(read_page).bit_in_tdac + 1;
                    end if;
                end if;

                if(TDAC_page_array(read_page).addr = "000100") then  -- TODO: put in correct end addr
                    read_state <= searching_match;
                end if;

                o_tdac_dpf_we(read_chip) <= '1';

                ram_raddr <= std_logic_vector(to_unsigned(read_page,PAGE_ADDR_WIDTH_g)) & TDAC_page_array(read_page).addr;

              when others =>
                read_state <= searching_match;
            end case;
            


        end if;
    end process;

    genwdata: for I in 0 to 3 generate
        o_tdac_dpf_wdata(I)<= ram_raddr(7+I+TDAC_page_array(read_page).bit_in_tdac);
    end generate;

    ram_1r1w_inst: entity work.ram_1r1w -- better split into multiple RAM IP's each with size of 1 page ?
      generic map (
        g_DATA_WIDTH       => 32,
        g_ADDR_WIDTH       => ADDR_WIDTH_g--,
      )
      port map (
        i_raddr => ram_raddr,
        o_rdata => ram_rdata,
        i_rclk  => i_clk,
        i_waddr => ram_waddr,
        i_wdata => ram_wdata,
        i_we    => ram_we,
        i_wclk  => i_clk
      );
end RTL;
