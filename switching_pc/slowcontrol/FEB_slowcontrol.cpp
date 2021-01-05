#include "FEB_slowcontrol.h"

#include <iostream>
#include <thread>
#include <chrono>

#include "feb_constants.h"

#include "midas.h"

using std::cout;
using std::endl;

FEB_slowcontrol::FEB_slowcontrol(mudaq::MudaqDevice & _mdev):
    mdev(_mdev),
    last_fpga_rmem_addr(0),
    m_FEBsc_wmem_addr(0),
    m_FEBsc_rmem_addr(0)
{
    FEBsc_resetMain();
    FEBsc_resetSecondary();
}

FEB_slowcontrol::~FEB_slowcontrol()
{
    // We do not close the mudaq device here
}



/*
 *  PCIe packet and software interface
 *  20b: N: packet length for following payload(in 32b words)

 *  N*32b: packet payload:
 *      0xBC, 4b type=0xC, 2b SC type = 0b11, 16b FPGA ID
 *      start addr(32b, user parameter)
 *      (N-2)*data(32b, user parameter)
 *
 *      1 word as dummy: 0x00000000
 *      Write length from 0xBC -> 0x9c to SC_MAIN_LENGTH_REGISTER_W
 *      Write enable to SC_MAIN_ENABLE_REGISTER_W
 */

int FEB_slowcontrol::FEB_write(uint32_t FPGA_ID, uint32_t startaddr, vector<uint32_t> data)
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

    if(!(mdev.read_register_ro(SC_MAIN_STATUS_REGISTER_R)&0x1)){ // FPGA is busy, should not be here...
       cout << "FPGA busy" << endl;
       return ERRCODES::FPGA_BUSY;
    }

    uint32_t FEB_PACKET_TYPE_SC = 0x7;
    uint32_t FEB_PACKET_TYPE_SC_WRITE = 0x3; // this is 11 in binary

    // two most significant bits are 0
    mdev.write_memory_rw(0, FEB_PACKET_TYPE_SC << 26 | FEB_PACKET_TYPE_SC_WRITE << 24 | (uint16_t) FPGA_ID << 8 | 0xBC);
    mdev.write_memory_rw(1, startaddr);
    mdev.write_memory_rw(2, data.size());

    for (uint32_t i = 0; i < data.size(); i++) {
        mdev.write_memory_rw(3 + i, data[i]);
    }
    mdev.write_memory_rw(3 + data.size(), 0x0000009c);

    // SC_MAIN_LENGTH_REGISTER_W starts from 1
    // length for SC Main does not include preamble and trailer, thats why it is 2+length
    mdev.write_register(SC_MAIN_LENGTH_REGISTER_W, 2 + data.size());
    mdev.write_register(SC_MAIN_ENABLE_REGISTER_W, 0x0);
    mdev.toggle_register(SC_MAIN_ENABLE_REGISTER_W, 0x1,100);
    // firmware regs SC_MAIN_ENABLE_REGISTER_W so that it only starts on a 0->1 transition


    // check if SC Main is done
    uint32_t count = 0;
    while(count < 10){
        if ( mdev.read_register_ro(SC_MAIN_STATUS_REGISTER_R) & 0x1 ) break;
        count++;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if(count==10){
        cout << "MudaqDevice::FEB_write Timeout for done reg" << endl;
        return ERRCODES::FPGA_TIMEOUT;
    }

    // check for acknowledge packet


    return OK;
}

int FEB_slowcontrol::FEB_read(uint32_t FPGA_ID, uint32_t startaddr, vector<uint32_t> &data)
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

    if(!(mdev.read_register_ro(SC_MAIN_STATUS_REGISTER_R)&0x1)){ // FPGA is busy, should not be here...
       cout << "FPGA busy" << endl;
       return ERRCODES::FPGA_BUSY;
    }

    uint32_t FEB_PACKET_TYPE_SC = 0x7;
    uint32_t FEB_PACKET_TYPE_SC_READ = 0x2; // this is 10 in binary

    mdev.write_memory_rw(0, FEB_PACKET_TYPE_SC << 26 | FEB_PACKET_TYPE_SC_READ << 24 | (uint16_t) FPGA_ID << 8 | 0xBC);
    mdev.write_memory_rw(1, startaddr);
    mdev.write_memory_rw(2, data.size());
    mdev.write_memory_rw(3, 0x0000009c);

    // SC_MAIN_LENGTH_REGISTER_W starts from 1
    // length for SC Main does not include preamble and trailer, thats why it is 2
    mdev.write_register(SC_MAIN_LENGTH_REGISTER_W, 2);
    mdev.write_register(SC_MAIN_ENABLE_REGISTER_W, 0x0);
    // firmware regs SC_MAIN_ENABLE_REGISTER_W so that it only starts on a 0->1 transition
    mdev.toggle_register(SC_MAIN_ENABLE_REGISTER_W, 0x1, 100);



    return ERRCODES::OK;
}

void FEB_slowcontrol::FEBsc_resetMain()
{
    cm_msg(MINFO, "FEB_slowcontrol" , "Resetting slow control main");
    //clear memory to avoid sending old packets again -- TODO: should not be necessary
    for(int i = 0; i <= 64*1024; i++){
        mdev.write_memory_rw(i, 0);
    }
    //reset our pointer
    m_FEBsc_wmem_addr=0;
    //reset fpga entity
    mdev.toggle_register(RESET_REGISTER_W, SET_RESET_BIT_SC_MAIN(0), 1000);
}

void FEB_slowcontrol::FEBsc_resetSecondary()
{
    cm_msg(MINFO, "FEB_slowcontrol" , "Resetting slow control secondary");
    cout << "FEB_slowcontrol::FEBsc_resetSecondary(): " << endl;
    //reset our pointer
    m_FEBsc_rmem_addr=0;
    //reset fpga entity
    mdev.toggle_register(RESET_REGISTER_W, SET_RESET_BIT_SC_SECONDARY(0), 1000);
    //wait until SECONDARY is reset, clearing the ram takes time
    uint16_t timeout_cnt=0;
    //poll register until reset. Should be 0xff... during reset and zero after, but we might be bombarded with packets, so give some margin for data to enter.
    while((mdev.read_register_ro(MEM_WRITEADDR_LOW_REGISTER_R) > 0xff) && timeout_cnt++ < 50){
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        printf("."); fflush(stdout);
    };
    if(timeout_cnt>=50){
        cout << "\n ERROR: Slow control secondary reset FAILED with timeout\n";
    }else{
        cout << " DONE\n";
    };
}

int FEB_slowcontrol::FEBsc_read_packets()
{
    int packetcount = 0;
    uint32_t fpga_rmem_addr=(mdev.read_register_ro(MEM_WRITEADDR_LOW_REGISTER_R)+1) & 0xffff;
    while(fpga_rmem_addr!=m_FEBsc_rmem_addr){

        if ((mdev.read_memory_ro(m_FEBsc_rmem_addr) & 0x1c0000bc) != 0x1c0000bc)
            return 0; //TODO: correct when no event is to be written?

        if(((fpga_rmem_addr > m_FEBsc_rmem_addr) && (fpga_rmem_addr-m_FEBsc_rmem_addr) < 4)
            || (MUDAQ_MEM_RO_LEN - m_FEBsc_rmem_addr + fpga_rmem_addr) <4   ){ // This is the wraparound case
            cout << "Incomplete packet!" << endl;
            return -1;
        }

        SC_reply_packet packet;
        packet.push_back(mdev.read_memory_ro(m_FEBsc_rmem_addr)); //save preamble
        rmenaddrIncr();
        packet.push_back(mdev.read_memory_ro(m_FEBsc_rmem_addr)); //save startaddr
        rmenaddrIncr();
        packet.push_back(mdev.read_memory_ro(m_FEBsc_rmem_addr)); //save length word
        rmenaddrIncr();

        if(((fpga_rmem_addr > m_FEBsc_rmem_addr) && (fpga_rmem_addr-m_FEBsc_rmem_addr) < packet.GetLength() +1)
            || (MUDAQ_MEM_RO_LEN - m_FEBsc_rmem_addr + fpga_rmem_addr) < packet.GetLength() + 1  ){ // This is the wraparound case
            cout << "Incomplete packet!" << endl;
            return -1;
        }
        for (uint32_t i = 0; i < packet.GetLength(); i++) {
            packet.push_back(mdev.read_memory_ro(m_FEBsc_rmem_addr)); //save data
            rmenaddrIncr();
        }

        packet.push_back(mdev.read_memory_ro(m_FEBsc_rmem_addr));
        rmenaddrIncr();

        if(packet[packet.size()-1]!=0x9c){
            cout << "Did not see trailer: something is wrong.\n" << endl;
            packet.Print();
            return -1;
        }

        sc_packet_deque.push_back(packet);
        packetcount++;
    }
    return packetcount;
}




}






