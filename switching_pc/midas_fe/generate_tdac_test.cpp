#include <iostream>
#include <fstream>
#include <vector>
using namespace std;

int main () {

  std::vector<uint32_t> vec;
  uint32_t counter = 0;

  for(int col = 0; col < 256; col++){
    for(int row4 = 0; row4< 64; row4++){
        vec.push_back(counter);
        counter++;
    }
  }

  ofstream file;
  file.open ("example.bin");
  file.write((char*) vec.data(), vec.size() * sizeof(uint32_t));
  file.close();
  return 0;
}