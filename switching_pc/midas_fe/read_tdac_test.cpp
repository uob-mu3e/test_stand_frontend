#include <iostream>
#include <fstream>
#include <istream>
#include <vector>
using namespace std;

int main () {

  uint32_t counter = 0;
  std::vector<uint32_t> vec(256*64);

  ifstream file;
  file.open ("example.bin");
  file.read(reinterpret_cast<char*>(&vec[0]), 256*64*sizeof(uint32_t));
  file.close();

  for(int col = 0; col < 256; col++){
    for(int row4 = 0; row4< 64; row4++){
        std::cout<<"row "<<row4<<" "<<vec.at(counter)<<std::endl;
        counter++;
    }
  }
  return 0;

}