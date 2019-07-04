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

    float current = cb.read_mother_board_current();
    cout << "MB Current " << current << " mA" << endl;

    float voltage = cb.read_mother_board_voltage();
    cout << "MB Voltage " << voltage << " mV" << endl;

    for(int i=0; i < 8; i++){
        if(!cb.daughter_present(i))
            continue;

        current = cb.read_daughter_board_current(i);
        cout << "DB Current " << current << " mA" << endl;

        voltage = cb.read_daughter_board_voltage(i);
        cout << "DB Voltage " << voltage << " mV" << endl;
    }

    cout << cb.read_rx_firefly_temp() << endl;
    cout << cb.read_tx_firefly_temp() << endl;

    return 1;

}
