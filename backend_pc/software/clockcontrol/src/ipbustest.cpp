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
    cb.init_clockboard();
    uint16_t ic = cb.read_inverted_tx_channels();
    cout << "Inverted Channels " << (int)ic << endl;

    int current = cb.read_mother_board_current();
    cout << "MB Current " << current << endl;

    int voltage = cb.read_mother_board_current();
    cout << "MB Voltage " << voltage << endl;

    return 1;

}
