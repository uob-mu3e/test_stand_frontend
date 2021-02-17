#ifndef DUMMYFEBSLOWCONTROLINTERFACE_H
#define DUMMYFEBSLOWCONTROLINTERFACE_H

#include "FEBSlowcontrolInterface.h"

#include <vector>
#include <thread>

using std::vector;
using std::thread;

class DummyFEBSlowcontrolInterface: public FEBSlowcontrolInterface
{
public:
    DummyFEBSlowcontrolInterface(mudaq::MudaqDevice & mdev /*,Add midas connection here */);
    virtual ~DummyFEBSlowcontrolInterface();
    // There should only be one SC interface, forbid copy and assignment
    DummyFEBSlowcontrolInterface() = delete;
    DummyFEBSlowcontrolInterface(const FEBSlowcontrolInterface &) = delete;
    DummyFEBSlowcontrolInterface& operator=(const FEBSlowcontrolInterface&) = delete;

    // We use the () operator to simulate changing values in the SC registers using a separate thread
    void operator()() [[noreturn]];

    virtual int FEB_write(uint32_t FPGA_ID, uint32_t startaddr, vector<uint32_t> data);
    // expects data vector with read-length size
    virtual int FEB_read(uint32_t FPGA_ID, uint32_t startaddr, vector<uint32_t> & data);

    virtual void FEBsc_resetMain(){}
    virtual void FEBsc_resetSecondary(){}
    virtual int FEBsc_NiosRPC(uint32_t FPGA_ID, uint16_t command, vector<vector<uint32_t> > payload_chunks);

protected:
    vector<vector<uint32_t> > scregs;
    thread t;
};

#endif // DUMMYFEBSLOWCONTROLINTERFACE_H
