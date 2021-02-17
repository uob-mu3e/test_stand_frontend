#include "DummyFEBSlowcontrolInterface.h"

#include <iostream>
#include <cstdlib>

#include <thread>
#include <chrono>

#include "link_constants.h"
#include "feb_constants.h"

using std::endl;
using std::cout;


DummyFEBSlowcontrolInterface::DummyFEBSlowcontrolInterface(mudaq::MudaqDevice &mdev):
    FEBSlowcontrolInterface(mdev),
    scregs(MAX_LINKS_PER_SWITCHINGBOARD,vector<uint32_t>(FEB_SC_ADDR_RANGE_HI+1,0))
{
    for(uint i = 0; i < MAX_LINKS_PER_SWITCHINGBOARD; i++){
        for(uint j = 0; j < FEB_SC_ADDR_RANGE_HI+1; j++){
            scregs[i][j] = scregs[i][j] + std::rand()/((RAND_MAX +1u)/4096);
        }
    }


    t = thread(&DummyFEBSlowcontrolInterface::operator(),this);
}

DummyFEBSlowcontrolInterface::~DummyFEBSlowcontrolInterface()
{

}

void DummyFEBSlowcontrolInterface::operator()()
{
    while(1){
        for(uint i = 0; i < MAX_LINKS_PER_SWITCHINGBOARD; i++){
            for(uint j = 0; j < FEB_SC_ADDR_RANGE_HI+1; j++){
                scregs[i][j] = scregs[i][j] + std::rand()/((RAND_MAX +1u)/257)-128;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int DummyFEBSlowcontrolInterface::FEB_write(uint32_t FPGA_ID, uint32_t startaddr, vector<uint32_t> data, bool nonincrementing)
{
    if(startaddr > FEB_SC_ADDR_RANGE_HI){
        cout << "Address out of range: " << std::hex << startaddr << endl;
        return ERRCODES::ADDR_INVALID;
     }

    if(data.size() > FEB_SC_DATA_SIZE_RANGE_HI){
        cout << "Length too big: " << data.size() << endl;
        return ERRCODES::SIZE_INVALID;
     }

    if(!data.size()){
        cout << "Length zero" << endl;
        return ERRCODES::SIZE_ZERO;
     }

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(sc_mutex);

    for(size_t i =0; i < data.size(); i++)
        scregs[FPGA_ID][startaddr+i*(!nonincrementing)] = data[i];

     return ERRCODES::OK;
}

int DummyFEBSlowcontrolInterface::FEB_read(uint32_t FPGA_ID, uint32_t startaddr, vector<uint32_t> &data, bool nonincrementing)
{
    if(startaddr > FEB_SC_ADDR_RANGE_HI){
        cout << "Address out of range: " << std::hex << startaddr << endl;
        return ERRCODES::ADDR_INVALID;
     }

    if(data.size() > FEB_SC_DATA_SIZE_RANGE_HI){
        cout << "Length too big: " << data.size() << endl;
        return ERRCODES::SIZE_INVALID;
     }

    if(!data.size()){
        cout << "Length zero" << endl;
        return ERRCODES::SIZE_ZERO;
     }

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(sc_mutex);

    for(size_t i =0; i < data.size(); i++)
       data[i] = scregs[FPGA_ID][startaddr+i*(!nonincrementing)] ;

    return ERRCODES::OK;

}

int DummyFEBSlowcontrolInterface::FEBsc_NiosRPC(uint32_t FPGA_ID, uint16_t command, vector<vector<uint32_t> > payload_chunks)
{
     return ERRCODES::OK;
}



