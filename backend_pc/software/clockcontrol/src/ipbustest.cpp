#include <iostream>

#include "clockboard.h"

using namespace std;

int main()
{
    cout << "Starting" << endl;
    clockboard cb("10.32.113.218", 50001);
    if(!cb.isConnected()){
        cout << "No connection!" << endl;
        return -1;
    }
    cout << "Connected" << endl;
    cb.init_12c();
    uint8_t data;
    cb.read_i2c(0x68, data);
    cout << "Data " << (int)data << endl;

    return 1;

}
