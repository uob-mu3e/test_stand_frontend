
library ieee;
use ieee.std_logic_1164.all;

package mupix is

    -----------------------------------------------------------------
    -- Things to clean up with the generics
    -----------------------------------------------------------------
    constant NINPUTS                :  integer := 36;
    constant NSORTERINPUTS          :  integer :=  1;
    constant NCHIPS                 :  integer := 12;

    -----------------------------------------------------------------
    -- conflicts between detectorfpga_constants and mupix_constants (to be checked & tested)
    -----------------------------------------------------------------

    constant HITSIZE                :  integer := 32;

    constant TIMESTAMPSIZE          :  integer := 11;

    subtype TSRANGE                 is integer range TIMESTAMPSIZE-1 downto 0;

    constant MHITSIZE               :  integer := HITSIZE+2;

    constant COARSECOUNTERSIZE      :  integer := 32;

    subtype  COLRANGE               is integer range 23 downto 16;
    subtype  ROWRANGE               is integer range 15 downto 8;

    constant CHIPRANGE              :  integer := 3;

    -----------------------------------------------------------
    -----------------------------------------------------------

    constant BINCOUNTERSIZE         :  integer := 24;
    constant CHARGESIZE_MP10        :  integer := 5;

    constant NOTSHITSIZE            :  integer := HITSIZE -TIMESTAMPSIZE;
    subtype NOTSRANGE               is integer range HITSIZE-1 downto TIMESTAMPSIZE;

    constant HITSORTERBINBITS       :  integer := 4;
    constant H                      :  integer := HITSORTERBINBITS;
    constant HITSORTERADDRSIZE      :  integer := TIMESTAMPSIZE + HITSORTERBINBITS;

    constant BITSPERTSBLOCK         :  integer := 4;
    subtype TSBLOCKRANGE            is integer range TIMESTAMPSIZE-1 downto BITSPERTSBLOCK;
    subtype TSNONBLOCKRANGE         is integer range BITSPERTSBLOCK-1 downto 0;

    constant COMMANDBITS            :  integer := 20;

    constant COUNTERMEMADDRSIZE     :  integer := 8;
    constant NMEMS                  :  integer := 2**(TIMESTAMPSIZE-COUNTERMEMADDRSIZE);
    constant COUNTERMEMDATASIZE     :  integer := 5;
    subtype COUNTERMEMSELRANGE      is integer range TIMESTAMPSIZE-1 downto COUNTERMEMADDRSIZE;
    subtype COUNTERMEMADDRRANGE     is integer range COUNTERMEMADDRSIZE-1 downto 0;

    -- Bit positions in the counter fifo of the sorter
    subtype MEMCOUNTERRANGE        is integer range 2*NCHIPS*HITSORTERBINBITS-1 downto 0;
    constant MEMOVERFLOWBIT        :  integer := 2*NCHIPS*HITSORTERBINBITS;
    constant HASMEMBIT             :  integer := 2*NCHIPS*HITSORTERBINBITS+1;
    subtype TSINFIFORANGE           is integer range HASMEMBIT+TIMESTAMPSIZE downto HASMEMBIT+1;
    subtype TSBLOCKINFIFORANGE      is integer range TSINFIFORANGE'left downto TSINFIFORANGE'right+BITSPERTSBLOCK;
    subtype TSINBLOCKINFIFORANGE    is integer range TSINFIFORANGE'right+BITSPERTSBLOCK-1  downto TSINFIFORANGE'right;
    subtype SORTERFIFORANGE         is integer range TSINFIFORANGE'left downto 0;
    subtype TSINBLOCKRANGE          is integer range BITSPERTSBLOCK-1 downto 0;
    
    constant NSORTERCOUNTERS        : integer := 40;

    -----------------------------------------------------------
    -- mupix ctrl constants
    -----------------------------------------------------------

    type mp_config_regs_length_t    is array (5 downto 0) of integer;
    constant MP_CONFIG_REGS_LENGTH  : mp_config_regs_length_t := (512, 896, 896, 80, 90, 210);

    type tdac_conversion_index_t    is array (511 downto 0) of integer;
    constant tdac_conversion_index  : tdac_conversion_index_t := (6,7,8,9,10,11,385, 383, 381, 379, 377, 375, 373, 371, 369, 367, 365, 363, 361, 359, 357, 355, 353, 351, 349, 347, 345, 343, 341, 339, 337, 335, 333, 331, 329, 327, 325, 323, 321, 319, 317, 315, 313, 311, 309, 307, 305, 303, 301, 299, 297, 295, 293, 291, 289, 287, 285, 283, 281, 279, 277, 275, 273, 271, 269, 267, 265, 263, 261, 259, 257, 386, 384, 382, 380, 378, 376, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356, 354, 352, 350, 348, 346, 344, 342, 340, 338, 336, 334, 332, 330, 328, 326, 324, 322, 320, 318, 316, 314, 312, 310, 308, 306, 304, 302, 300, 298, 296, 294, 292, 290, 288, 286, 284, 282, 280, 278, 276, 274, 272, 270, 268, 266, 264, 262, 260, 258, 256, 129, 127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 97, 95, 93, 91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 130, 128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 0,1,2,3,4,5, 511, 509, 507, 505, 503, 501, 499, 497, 495, 493, 491, 489, 487, 485, 483, 481, 479, 477, 475, 473, 471, 469, 467, 465, 463, 461, 459, 457, 455, 453, 451, 449, 447, 445, 443, 441, 439, 437, 435, 433, 431, 429, 427, 425, 423, 421, 419, 417, 415, 413, 411, 409, 407, 405, 403, 401, 399, 397, 395, 393, 391, 389, 387, 510, 508, 506, 504, 502, 500, 498, 496, 494, 492, 490, 488, 486, 484, 482, 480, 478, 476, 474, 472, 470, 468, 466, 464, 462, 460, 458, 456, 454, 452, 450, 448, 446, 444, 442, 440, 438, 436, 434, 432, 430, 428, 426, 424, 422, 420, 418, 416, 414, 412, 410, 408, 406, 404, 402, 400, 398, 396, 394, 392, 390, 388, 255, 253, 251, 249, 247, 245, 243, 241, 239, 237, 235, 233, 231, 229, 227, 225, 223, 221, 219, 217, 215, 213, 211, 209, 207, 205, 203, 201, 199, 197, 195, 193, 191, 189, 187, 185, 183, 181, 179, 177, 175, 173, 171, 169, 167, 165, 163, 161, 159, 157, 155, 153, 151, 149, 147, 145, 143, 141, 139, 137, 135, 133, 131, 254, 252, 250, 248, 246, 244, 242, 240, 238, 236, 234, 232, 230, 228, 226, 224, 222, 220, 218, 216, 214, 212, 210, 208, 206, 204, 202, 200, 198, 196, 194, 192, 190, 188, 186, 184, 182, 180, 178, 176, 174, 172, 170, 168, 166, 164, 162, 160, 158, 156, 154, 152, 150, 148, 146, 144, 142, 140, 138, 136, 134, 132);
    --(251,252,253,254,255,506,507,508,509,510,511, 315, 256, 316, 257, 317, 258, 318, 259, 319, 260, 320, 261, 321, 262, 322, 263, 323, 264, 324, 265, 325, 266, 326, 267, 327, 268, 328, 269, 329, 270, 330, 271, 331, 272, 332, 273, 333, 274, 334, 275, 335, 276, 336, 277, 337, 278, 338, 279, 339, 280, 340, 281, 341, 282, 342, 283, 343, 284, 344, 285, 345, 286, 346, 287, 347, 288, 348, 289, 349, 290, 350, 291, 351, 292, 352, 293, 353, 294, 354, 295, 355, 296, 356, 297, 357, 298, 358, 299, 359, 300, 360, 301, 361, 302, 362, 303, 363, 304, 364, 305, 365, 306, 366, 307, 367, 308, 368, 309, 369, 310, 370, 311, 371, 312, 372, 313, 373, 314, 374, 315, 62, 0, 63, 1, 64, 2, 65, 3, 66, 4, 67, 5, 68, 6, 69, 7, 70, 8, 71, 9, 72, 10, 73, 11, 74, 12, 75, 13, 76, 14, 77, 15, 78, 16, 79, 17, 80, 18, 81, 19, 82, 20, 83, 21, 84, 22, 85, 23, 86, 24, 87, 25, 88, 26, 89, 27, 90, 28, 91, 29, 92, 30, 93, 31, 94, 32, 95, 33, 96, 34, 97, 35, 98, 36, 99, 37, 100, 38, 101, 39, 102, 40, 103, 41, 104, 42, 105, 43, 106, 44, 107, 45, 108, 46, 109, 47, 110, 48, 111, 49, 112, 50, 113, 51, 114, 52, 115, 53, 116, 54, 117, 55, 118, 56, 119, 57, 120, 58, 121, 59, 122, 60, 123, 61, 124, 375, 441, 376, 442, 377, 443, 378, 444, 379, 445, 380, 446, 381, 447, 382, 448, 383, 449, 384, 450, 385, 451, 386, 452, 387, 453, 388, 454, 389, 455, 390, 456, 391, 457, 392, 458, 393, 459, 394, 460, 395, 461, 396, 462, 397, 463, 398, 464, 399, 465, 400, 466, 401, 467, 402, 468, 403, 469, 404, 470, 405, 471, 406, 472, 407, 473, 408, 474, 409, 475, 410, 476, 411, 477, 412, 478, 413, 479, 414, 480, 415, 481, 416, 482, 417, 483, 418, 484, 419, 485, 420, 486, 421, 487, 422, 488, 423, 489, 424, 490, 425, 491, 426, 492, 427, 493, 428, 494, 429, 495, 430, 496, 431, 497, 432, 498, 433, 499, 434, 500, 435, 501, 436, 502, 437, 503, 438, 504, 439, 505, 440, 187, 125, 188, 126, 189, 127, 190, 128, 191, 129, 192, 130, 193, 131, 194, 132, 195, 133, 196, 134, 197, 135, 198, 136, 199, 137, 200, 138, 201, 139, 202, 140, 203, 141, 204, 142, 205, 143, 206, 144, 207, 145, 208, 146, 209, 147, 210, 148, 211, 149, 212, 150, 213, 151, 214, 152, 215, 153, 216, 154, 217, 155, 218, 156, 219, 157, 220, 158, 221, 159, 222, 160, 223, 161, 224, 162, 225, 163, 226, 164, 227, 165, 228, 166, 229, 167, 230, 168, 231, 169, 232, 170, 233, 171, 234, 172, 235, 173, 236, 174, 237, 175, 238, 176, 239, 177, 240, 178, 241, 179, 242, 180, 243, 181, 244, 182, 245, 183, 246, 184, 247, 185, 248, 186, 249);

    type mp_conf_storage_interface_in is record
        spi_read        :   std_logic_vector(3 downto 0);
        mu3e_read       :   std_logic_vector(3 downto 0);
    end record;

    type mp_conf_storage_interface_out is record
        rdy             :   std_logic_vector(3 downto 0);
        spi_data        :   std_logic_vector(3 downto 0);
        conf            :   std_logic_vector(52 downto 0);
        vdac            :   std_logic_vector(52 downto 0);
        bias            :   std_logic_vector(52 downto 0);
        tdac            :   std_logic_vector(52 downto 0);
    end record;

    type mp_conf_array_in  is array( natural range <> ) of mp_conf_storage_interface_in;
    type mp_conf_array_out is array( natural range <> ) of mp_conf_storage_interface_out;


    type mp_link_order_t    is array (35 downto 0) of integer;
    constant MP_LINK_ORDER  : mp_link_order_t := (33,31,29,35,32,28,34,30,27,26,25,20,24,23,21,22,19,18,15,11,9,17,13,10,16,14,12,5,3,2,6,4,1,8,7,0);
    constant MP_LINK_ORDER_TELESCOPE : mp_link_order_t := (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,35,26,30,34,21,25,23,24,22,33,32,31,29,28,27,20,19,18);
    constant MP_LINK_ORDER_TELESCOPE_DOWNSTREAM : mp_link_order_t :=
    (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,35,20,19,18,26,30,34,21,25,33,32,31,24,23,22,29,28,27);

    -- following mp links are inverted on DAB integration run version:
        --0,19,30,35
    -- more links are inverted on the FEB:
        -- 12,14,16,27,34,35
    -- additional inverts: (we dont know from where, list is incomplete):
        -- 11,15,32,28,  2,3,5,24,23,21,20,25,26,28,32,29,31,10,13,17 
    -- additional inverts found in the integration run cage before craning:(with this the list is hopefully complete)
        -- 0,7,8,18,19,22,9,30,33,1,4,6
      constant MP_LINK_INVERT : std_logic_vector(35 downto 0) := "011110111111111101111111111111111110";
      constant MP_LINK_INVERT_TELESCOPE : std_logic_vector(35 downto 0) := "001110011111111111111111111111111110";


    type ts_array_t                 is array (natural range <>) of std_logic_vector(10 downto 0);
    type row_array_t                is array (natural range <>) of std_logic_vector(7 downto 0);
    type col_array_t                is array (natural range <>) of std_logic_vector(7 downto 0);
    type ch_ID_array_t              is array (natural range <>) of std_logic_vector(5 downto 0);
    type tot_array_t                is array (natural range <>) of std_logic_vector(5 downto 0);

    subtype hit_t is                std_logic_vector(HITSIZE-1 downto 0);
    subtype cnt_t is                std_logic_vector(COARSECOUNTERSIZE-1  downto 0);
    subtype ts_t is                 std_logic_vector(TSRANGE);
    subtype nots_t                  is std_logic_vector(NOTSHITSIZE-1 downto 0);
    subtype addr_t                  is std_logic_vector(HITSORTERADDRSIZE-1 downto 0);
    subtype counter_t               is std_logic_vector(HITSORTERBINBITS-1 downto 0);

    constant counter1               :  counter_t := (others => '1');

    type wide_hit_array             is array (NINPUTS-1 downto 0) of hit_t;
    type hit_array                  is array (NCHIPS-1 downto 0) of hit_t;

    type wide_cnt_array             is array (NINPUTS-1 downto 0) of cnt_t;
    type cnt_array                  is array (NCHIPS-1 downto 0) of cnt_t;

    type ts_array                   is array (NCHIPS-1 downto 0) of ts_t;

    type nots_hit_array             is array (NCHIPS-1 downto 0) of nots_t;
    type addr_array                 is array (NCHIPS-1 downto 0) of addr_t;

    type counter_chips              is array (NCHIPS-1 downto 0) of counter_t;
    subtype counter2_chips          is std_logic_vector(2*NCHIPS*HITSORTERBINBITS-1 downto 0);

    subtype sorterfifodata_t        is std_logic_vector(SORTERFIFORANGE);

    type hitcounter_sum3_type is array (NCHIPS/3-1 downto 0) of integer;

    subtype chip_bits_t             is std_logic_vector(NCHIPS-1 downto 0);

    subtype muxhit_t                is std_logic_vector(HITSIZE+1 downto 0);
    type muxhit_array               is array ((NINPUTS/4) downto 0) of muxhit_t;

    subtype byte_t                  is std_logic_vector(7 downto 0);
    type inbyte_array               is array (NINPUTS-1 downto 0) of byte_t;

    type state_type                 is (INIT, START, PRECOUNT, COUNT);

    subtype block_t                 is std_logic_vector(TSBLOCKRANGE);

    subtype command_t               is std_logic_vector(COMMANDBITS-1 downto 0);
    constant COMMAND_HEADER1        :  command_t := X"80000";
    constant COMMAND_HEADER2        :  command_t := X"90000";
    constant COMMAND_SUBHEADER      :  command_t := X"C0000";
    constant COMMAND_FOOTER         :  command_t := X"E0000";

    subtype doublecounter_t         is std_logic_vector(COUNTERMEMDATASIZE-1 downto 0);
    type doublecounter_array        is array (NMEMS-1 downto 0) of doublecounter_t;
    type doublecounter_chiparray    is array (NCHIPS-1 downto 0) of doublecounter_t;
    type alldoublecounter_array     is array (NCHIPS-1 downto 0) of doublecounter_array;

    subtype counteraddr_t           is std_logic_vector(COUNTERMEMADDRSIZE-1 downto 0);
    type counteraddr_array          is array (NMEMS-1 downto 0) of counteraddr_t;
    type counteraddr_chiparray      is array (NCHIPS-1 downto 0) of counteraddr_t;
    type allcounteraddr_array       is array (NCHIPS-1 downto 0) of counteraddr_array;

    type counterwren_array          is array (NMEMS-1 downto 0) of std_logic;
    type allcounterwren_array       is array (NCHIPS-1 downto 0) of counterwren_array;
    subtype countermemsel_t         is std_logic_vector(COUNTERMEMADDRRANGE);
    type reg_array                  is array (NCHIPS-1 downto 0) of work.mudaq.reg32;
    type sorter_reg_array           is array (NSORTERCOUNTERS-1 downto 0) of work.mudaq.reg32;

end package;
