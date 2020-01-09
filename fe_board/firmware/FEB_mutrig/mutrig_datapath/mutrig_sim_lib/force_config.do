#force standard values for clock divider,emulators 
#script parameter $1: path of DUT to force, e.g. "testbench/dut"

#configuration: all channels enabled, gen_idle, short event mode, no receive all
#force -freeze sim:/$1/u_digital_all/u_spi/o_config 100000000000000000000100000000000000000011100101101101110110000110111000010010110101011100011000000000000110100111011110000101001110000000001101111001101111010101000000000000011100000011011111100100101010111001110011011001100000010000000000000010110101101110011110111001111101001010011101111101111110000000000000011101111100010110001110101011101100010101110011001000110000000000000100111111001000100000000001000000110111101010110001001001000000000000111000011100011010000011111101110000101000101010100000110000000000000001100000111110111110010001001010011100101001010110111001000000000000011110001110010000001110110011011111000001010010011111110000000000000011011001101000000101110111100011010101111100010001010010000000000000011000111010100111010101010101001000110100000100000111100000000000000010110010100101100000100010110110111011000011011101001110000000000000011011011100011110000100011010101010011101111111100100111000000000000101110001111010011011001010100010000000011010010000001010000000000000110010110101111000010011001000100110011100110011101101111000000000000100110011110010011100111001110100010110000001100101100010000000000000110111110111001001010010010011111101110110101101010110001000000000000110101101110000101111110111000010010100111000010000100010000000000000111010100011111000000110100000011110000111001110101111110000000000000101000110101110101100110110100001000011011011011000101100000000000000110010001111100101101000010110100101000011001100011110001000000000000101011111011010001010000111011111001110010110001111000010000000000000011001010100011001001110011001110111011010010010100111111000000000000101001011110011000000111101000010010001110011100001101011000000000000010011101001011001000001111000111011110110011100100001000000000000000100111000011101111000001111011000000100000100111101011100000000000000011000010110111100101100101111110111100010100100001100001000000000000101111000010111111111010100101001011111011000111110100011000000000000011110000000011000000111111111101011011101000101111110101000000000000001011110010110010001001011001111000000000101110011011000000000000000011011100100011010010001011001010000000100010011011111010000000000000111011000010101111101000110010010001110111011011111001000000011010111000011000110011101000001000110111101111110010101011101110110101111110001000101011001011011101010111110000010000000 

#configuration: all channels enabled, gen_idle, long event mode, no receive all
#force -freeze sim:/$1/u_digital_all/u_spi/o_config 100000000000000000000000000000000000000011100101101101110110000110111000010010110101011100011000000000000110100111011110000101001110000000001101111001101111010101000000000000011100000011011111100100101010111001110011011001100000010000000000000010110101101110011110111001111101001010011101111101111110000000000000011101111100010110001110101011101100010101110011001000110000000000000100111111001000100000000001000000110111101010110001001001000000000000111000011100011010000011111101110000101000101010100000110000000000000001100000111110111110010001001010011100101001010110111001000000000000011110001110010000001110110011011111000001010010011111110000000000000011011001101000000101110111100011010101111100010001010010000000000000011000111010100111010101010101001000110100000100000111100000000000000010110010100101100000100010110110111011000011011101001110000000000000011011011100011110000100011010101010011101111111100100111000000000000101110001111010011011001010100010000000011010010000001010000000000000110010110101111000010011001000100110011100110011101101111000000000000100110011110010011100111001110100010110000001100101100010000000000000110111110111001001010010010011111101110110101101010110001000000000000110101101110000101111110111000010010100111000010000100010000000000000111010100011111000000110100000011110000111001110101111110000000000000101000110101110101100110110100001000011011011011000101100000000000000110010001111100101101000010110100101000011001100011110001000000000000101011111011010001010000111011111001110010110001111000010000000000000011001010100011001001110011001110111011010010010100111111000000000000101001011110011000000111101000010010001110011100001101011000000000000010011101001011001000001111000111011110110011100100001000000000000000100111000011101111000001111011000000100000100111101011100000000000000011000010110111100101100101111110111100010100100001100001000000000000101111000010111111111010100101001011111011000111110100011000000000000011110000000011000000111111111101011011101000101111110101000000000000001011110010110010001001011001111000000000101110011011000000000000000011011100100011010010001011001010000000100010011011111010000000000000111011000010101111101000110010010001110111011011111001000000011010111000011000110011101000001000110111101111110010101011101110110101111110001000101011001011011101010111110000010000000


#configuration: all channels enabled, gen_idle, long event mode, no receive all, prbs enabled, prbs-many
force -freeze sim:/$1/u_digital_all/u_spi/o_config 000000000000000000010000000000000000000011100101101101110110000110111000010010110101011100011000000000000110100111011110000101001110000000001101111001101111010101000000000000011100000011011111100100101010111001110011011001100000010000000000000010110101101110011110111001111101001010011101111101111110000000000000011101111100010110001110101011101100010101110011001000110000000000000100111111001000100000000001000000110111101010110001001001000000000000111000011100011010000011111101110000101000101010100000110000000000000001100000111110111110010001001010011100101001010110111001000000000000011110001110010000001110110011011111000001010010011111110000000000000011011001101000000101110111100011010101111100010001010010000000000000011000111010100111010101010101001000110100000100000111100000000000000010110010100101100000100010110110111011000011011101001110000000000000011011011100011110000100011010101010011101111111100100111000000000000101110001111010011011001010100010000000011010010000001010000000000000110010110101111000010011001000100110011100110011101101111000000000000100110011110010011100111001110100010110000001100101100010000000000000110111110111001001010010010011111101110110101101010110001000000000000110101101110000101111110111000010010100111000010000100010000000000000111010100011111000000110100000011110000111001110101111110000000000000101000110101110101100110110100001000011011011011000101100000000000000110010001111100101101000010110100101000011001100011110001000000000000101011111011010001010000111011111001110010110001111000010000000000000011001010100011001001110011001110111011010010010100111111000000000000101001011110011000000111101000010010001110011100001101011000000000000010011101001011001000001111000111011110110011100100001000000000000000100111000011101111000001111011000000100000100111101011100000000000000011000010110111100101100101111110111100010100100001100001000000000000101111000010111111111010100101001011111011000111110100011000000000000011110000000011000000111111111101011011101000101111110101000000000000001011110010110010001001011001111000000000101110011011000000000000000011011100100011010010001011001010000000100010011011111010000000000000111011000010101111101000110010010001110111011011111001000000011010111000011000110011101000001000110111101111110010101011101110110101111110001000101011001011011101010111110000010000000 
