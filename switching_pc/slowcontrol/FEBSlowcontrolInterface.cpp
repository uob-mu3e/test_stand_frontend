#include "FEBSlowcontrolInterface.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <math.h>

#include "feb_constants.h"

#include "midas.h"

using std::cout;
using std::endl;

FEBSlowcontrolInterface::FEBSlowcontrolInterface(mudaq::MudaqDevice & _mdev):
    mdev(_mdev),
    last_fpga_rmem_addr(0),
    m_FEBsc_wmem_addr(0),
    m_FEBsc_rmem_addr(0)
{
    FEBsc_resetMain();
    FEBsc_resetSecondary();
}

FEBSlowcontrolInterface::~FEBSlowcontrolInterface()
{
    // We do not close the mudaq device here on purpose
}



/*
 *  PCIe packet and software interface
 *  20b: N: packet length for following payload(in 32b words)

 *  N*32b: packet payload:
 *      0xBC, 4b type=0xC, 2b SC type = 0b11, 16b FPGA ID
 *      start addr(32b, user parameter)
 *      (N-2)*data(32b, user parameter)
 *
 *      1 word as dummy: 0x00000000 NOTE: MK: why this?
 *      Write length from 0xBC -> 0x9c to SC_MAIN_LENGTH_REGISTER_W
 *      Write enable to SC_MAIN_ENABLE_REGISTER_W
 */

int FEBSlowcontrolInterface::FEB_write(const mappedFEB & FEB, const uint32_t startaddr, const vector<uint32_t> & data, const bool nonincrementing, const bool broadcast)
{
    uint32_t FPGA_ID = FEB.SB_Port();
    if(broadcast)
        FPGA_ID = ADDRS::BROADCAST_ADDR;

     if(startaddr >= pow(2,16)){
        cout << "Address out of range: " << std::hex << startaddr << endl;
        return ERRCODES::ADDR_INVALID;
     }

    if(FPGA_ID > 15 and FPGA_ID != ADDRS::BROADCAST_ADDR){
        cout << "FPGA ID out of range: " << FPGA_ID << endl;
        return ERRCODES::ADDR_INVALID;
     }

    if(!data.size()){
        cout << "Length zero" << endl;
        return ERRCODES::SIZE_ZERO;
     }

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(sc_mutex);

    if(!(mdev.read_register_ro(SC_MAIN_STATUS_REGISTER_R)&0x1)){ // FPGA is busy, should not be here...
       cout << "FPGA busy" << endl;
       return ERRCODES::FPGA_BUSY;
    }

    uint32_t packet_type = PACKET_TYPE_SC_WRITE;
    if(nonincrementing)
        packet_type = PACKET_TYPE_SC_WRITE_NONINCREMENTING;

    // two most significant bits are 0
    mdev.write_memory_rw(0, PACKET_TYPE_SC << 26 | packet_type << 24 | ((uint16_t)(1UL << FPGA_ID)) << 8 | 0xBC);
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
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if(count==10){
        cout << "MudaqDevice::FEB_write Timeout for done reg" << endl;
        return ERRCODES::FPGA_TIMEOUT;
    }

    // TODO: Are there other cases where we do not get a return packet?
    if(FPGA_ID == ADDRS::BROADCAST_ADDR)
        return OK;

    // check for acknowledge packet
    count = 0;
    while(count<20){
        if(FEBsc_read_packets() > 0 && sc_packet_deque.front().IsWR()) break;
        // for some reason there is a read acknowledge at the front of the queue...
        if(FEBsc_read_packets() > 0 ){
            cout << "wrong packet type" << endl;
            sc_packet_deque.pop_front();
        };

        count++;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if(count==20){
        cm_msg(MERROR, "MudaqDevice::FEBsc_write" , "Timeout occured waiting for reply");
        cm_msg(MERROR, "MudaqDevice::FEBsc_write", "Wanted to write to FPGA %d, Addr %d, length %zu", FPGA_ID, startaddr, data.size());
        return ERRCODES::FPGA_TIMEOUT;
    }
    if(!sc_packet_deque.front().Good()){
        cm_msg(MERROR, "MudaqDevice::FEBsc_write" , "Received bad packet");
        sc_packet_deque.pop_front();
        return ERRCODES::BAD_PACKET;
    }
    if(!sc_packet_deque.front().IsResponse()){
        cm_msg(MERROR, "MudaqDevice::FEBsc_write" , "Received request packet, this should not happen...");
        sc_packet_deque.pop_front();
        return ERRCODES::BAD_PACKET;
    }

    // Message was consumed, drop it
    sc_packet_deque.pop_front();

    return OK;
}

int FEBSlowcontrolInterface::FEB_write(const mappedFEB & FEB, const uint32_t startaddr, const uint32_t data)
{
    return FEB_write(FEB, startaddr, vector<uint32_t>(1, data) );
}


int FEBSlowcontrolInterface::FEB_broadcast(const uint32_t startaddr, const uint32_t data)
{
    return FEB_write(mappedFEB(), startaddr, vector<uint32_t>(1, data),false, true );
}

int FEBSlowcontrolInterface::FEB_broadcast(const uint32_t startaddr, const vector<uint32_t> & data, const bool nonincrementing)
{
    return FEB_write(mappedFEB(), startaddr, data, nonincrementing, true );
}

int FEBSlowcontrolInterface::FEB_read(const mappedFEB & FEB, const uint32_t startaddr, vector<uint32_t> &data, const bool nonincrementing)
{
    uint32_t FPGA_ID = FEB.SB_Port();

     if(startaddr >= pow(2,16)){
        cout << "Address out of range: " << std::hex << startaddr << endl;
        return ERRCODES::ADDR_INVALID;
     }

    if(FPGA_ID > 15){
        cout << "FPGA ID out of range: " << FPGA_ID << endl;
        return ERRCODES::ADDR_INVALID;
     }

    if(!data.size()){
        cout << "Length zero" << endl;
        return ERRCODES::SIZE_ZERO;
     }

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(sc_mutex);

    if(!(mdev.read_register_ro(SC_MAIN_STATUS_REGISTER_R)&0x1)){ // FPGA is busy, should not be here...
       cout << "FPGA busy" << endl;
       return ERRCODES::FPGA_BUSY;
    }

    uint32_t packet_type = PACKET_TYPE_SC_READ;
    if(nonincrementing)
        packet_type = PACKET_TYPE_SC_READ_NONINCREMENTING;

    mdev.write_memory_rw(0, PACKET_TYPE_SC << 26 | packet_type << 24 | ((uint16_t)(1UL << FPGA_ID)) << 8 | 0xBC);
    mdev.write_memory_rw(1, startaddr);
    mdev.write_memory_rw(2, data.size());
    mdev.write_memory_rw(3, 0x0000009c);

    // SC_MAIN_LENGTH_REGISTER_W starts from 1
    // length for SC Main does not include preamble and trailer, thats why it is 2
    mdev.write_register(SC_MAIN_LENGTH_REGISTER_W, 2);
    mdev.write_register(SC_MAIN_ENABLE_REGISTER_W, 0x0);
    // firmware regs SC_MAIN_ENABLE_REGISTER_W so that it only starts on a 0->1 transition
    mdev.toggle_register(SC_MAIN_ENABLE_REGISTER_W, 0x1, 100);

    int count = 0;
    while(count<10){
        int retval = FEBsc_read_packets();
        if(retval > 0 && sc_packet_deque.front().IsRD()) break;
        // for some reason there is a write acknowledge at the front of the queue...
        if(retval > 0 ){
            cout << "wrong packet type" << endl;
            sc_packet_deque.pop_front();
            return ERRCODES::BAD_PACKET;
        };
        if(retval < 0){
            cout << "Receiving failed, resetting" << endl;
            FEBsc_resetSecondary();
            return ERRCODES::BAD_PACKET;
        }
        count++;
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    if(count==10){
        cm_msg(MERROR, "MudaqDevice::FEBsc_read" , "Timeout occured waiting for reply");
        cm_msg(MERROR, "MudaqDevice::FEBsc_read",  "Wanted to read from FPGA %d, Addr %d, length %zu, memaddr %d", FPGA_ID, startaddr, data.size(), m_FEBsc_rmem_addr);
        return ERRCODES::FPGA_TIMEOUT;
    }
    if(!sc_packet_deque.front().Good()){
        cm_msg(MERROR, "MudaqDevice::FEBsc_read" , "Received bad packet, resetting");
        sc_packet_deque.pop_front();
        FEBsc_resetSecondary();
        return ERRCODES::BAD_PACKET;
    }
    if(!sc_packet_deque.front().IsResponse()){
        cm_msg(MERROR, "MudaqDevice::FEBsc_read" , "Received request packet, this should not happen..., resetting");
        sc_packet_deque.pop_front();
        FEBsc_resetSecondary();
        return ERRCODES::BAD_PACKET;
    }
    if(sc_packet_deque.front().GetLength()!=data.size()){
        cm_msg(MERROR, "MudaqDevice::FEBsc_read", "Wanted to read from FPGA %d, Addr %d, length %zu", FPGA_ID, startaddr, data.size());
        cm_msg(MERROR, "MudaqDevice::FEBsc_read" , "Received packet fails size check, communication error, resetting");
        sc_packet_deque.pop_front();
        FEBsc_resetSecondary();
        return ERRCODES::WRONG_SIZE;
    }

    for(uint32_t index =0; index < data.size(); index++){
        data[index] = sc_packet_deque.front().data()[index+3];
    }

    // Message was consumed, drop it
    sc_packet_deque.pop_front();

    return ERRCODES::OK;
}

int FEBSlowcontrolInterface::FEB_read(const mappedFEB & FEB, const uint32_t startaddr, uint32_t &data)
{
    vector<uint32_t> d(1,0);
    int status = FEB_read(FEB, startaddr, d);
    data = d[0];
    return status;
}


void FEBSlowcontrolInterface::FEBsc_resetMain()
{
    //cm_msg(MINFO, "FEB_slowcontrol" , "Resetting slow control main");
    //clear memory to avoid sending old packets again -- TODO: should not be necessary
    for(int i = 0; i <= 64*1024; i++){
        mdev.write_memory_rw(i, 0);
    }
    //reset our pointer
    m_FEBsc_wmem_addr=0;
    //reset fpga entity
    mdev.toggle_register(RESET_REGISTER_W, SET_RESET_BIT_SC_MAIN(0), 1000);
}

void FEBSlowcontrolInterface::FEBsc_resetSecondary()
{
    //cm_msg(MINFO, "FEB_slowcontrol" , "Resetting slow control secondary");
    //reset our pointer
    m_FEBsc_rmem_addr=0;
    //reset fpga entity
    mdev.toggle_register(RESET_REGISTER_W, SET_RESET_BIT_SC_SECONDARY(0), 1000);
    //wait until SECONDARY is reset, clearing the ram takes time
    uint16_t timeout_cnt=0;
    // poll register until addr of sc secondary is 0xffff (and of init state)
    // NOTE: we have to wait 2**16 * 156.25MHz here, but we wait a bit longer
    while ( (mdev.read_register_ro(SC_STATE_REGISTER_R) & 0x20000000) != 0x20000000 ) {
    //while((mdev.read_register_ro(MEM_WRITEADDR_LOW_REGISTER_R) == 0x0) && timeout_cnt++ < 500){
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        printf("."); fflush(stdout);
    };
    if(timeout_cnt>=50){
        cm_msg(MERROR, "FEBsc_resetSecondary()", "Slow control secondary reset FAILED with timeout");
    }
}

int FEBSlowcontrolInterface::FEBsc_NiosRPC(const mappedFEB & FEB, uint16_t command, vector<vector<uint32_t> > payload_chunks)
{
    int status = 0;
    int index = 0;

    // Write the payload
    for(auto chunk: payload_chunks){
        status=FEB_write(FEB, (uint32_t) index+OFFSETS::FEBsc_RPC_DATAOFFSET, chunk);
         if(status < 0)
             return status;
        index += chunk.size();
    }
    if(index >= 1<<16)
        return ERRCODES::WRONG_SIZE;

    // Write the position of the payload in the offset register
    status=FEB_register_write(FEB, CMD_OFFSET_REGISTER_RW, vector<uint32_t>(1,OFFSETS::FEBsc_RPC_DATAOFFSET));
    if(status < 0)
        return status;

    // Write the command in the upper 16 bits of the length register and
    // the size of the payload in the lower 16 bits
    // This triggers the callback function on the frontend board
    status=FEB_register_write(FEB, CMD_LEN_REGISTER_RW,
                     vector<uint32_t>(1,(((uint32_t)command) << 16) | index));
    
    if(status < 0)
        return status;

    //Wait for remote command to finish, poll register
    uint timeout_cnt = 0;
    vector<uint32_t> readback(1,0);
    while(1){
        if(++timeout_cnt >= 500) return ERRCODES::NIOS_RPC_TIMEOUT;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        status=FEB_register_read(FEB, CMD_LEN_REGISTER_RW, readback);
        if(status < 0)
            return status;

        if(timeout_cnt > 5) printf("MudaqDevice::FEBsc_NiosRPC(): Polling for command %x @%d: %x, %x\n",command,timeout_cnt,readback[0],readback[0]&0xffff0000);
        if((readback[0]&0xffff0000) == 0) break;
    }
    return readback[0]&0xffff;
}

int FEBSlowcontrolInterface::FEBsc_read_packets()
{


    int packetcount = 0;

    uint32_t fpga_rmem_addr=(mdev.read_register_ro(MEM_WRITEADDR_LOW_REGISTER_R)+1) & 0xffff;
    while(fpga_rmem_addr!=m_FEBsc_rmem_addr){

        if ((mdev.read_memory_ro(m_FEBsc_rmem_addr) & 0x1c0000bc) != 0x1c0000bc){
            cout << "Start pattern not seen at addr " << std::hex << m_FEBsc_rmem_addr << " seeing " << mdev.read_memory_ro(m_FEBsc_rmem_addr) << std::dec << endl;
            return -1;
        }

        // the eqaulity case is taken care of by the while condition
        if(((fpga_rmem_addr > m_FEBsc_rmem_addr) && (fpga_rmem_addr-m_FEBsc_rmem_addr < MIN_SC_MESSAGE_SIZE))
            || ((fpga_rmem_addr < m_FEBsc_rmem_addr) && (MUDAQ_MEM_RO_LEN - m_FEBsc_rmem_addr + fpga_rmem_addr) < MIN_SC_MESSAGE_SIZE)   ){ // This is the wraparound case
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

        if(((fpga_rmem_addr >= m_FEBsc_rmem_addr) && (fpga_rmem_addr-m_FEBsc_rmem_addr) < packet.GetLength() +1) // Plus 1 for the trailer
            || ((fpga_rmem_addr < m_FEBsc_rmem_addr) && (MUDAQ_MEM_RO_LEN - m_FEBsc_rmem_addr + fpga_rmem_addr) < packet.GetLength() + 1 ) ){ // This is the wraparound case
            cout << "Incomplete packet!" << endl;
            return -1;
        }
        // Read data
        for (uint32_t i = 0; i < packet.GetLength(); i++) {
            packet.push_back(mdev.read_memory_ro(m_FEBsc_rmem_addr)); //save data
            rmenaddrIncr();
        }

        // Read trailer
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




void FEBSlowcontrolInterface::SC_reply_packet::Print(){
   printf("--- Packet dump ---\n");
   printf("Type %x\n", this->at(0)&0x1f0000bc);
   printf("FPGA ID %x\n", this->GetFPGA_ID());
   printf("startaddr %x\n", this->GetStartAddr());
   printf("length %ld\n", this->GetLength());
   printf("packet: size=%lu length=%lu IsRD=%c IsWR=%c, IsResponse=%c, IsGood=%c\n",
     this->size(),this->GetLength(),
     this->IsRD()?'y':'n',
     this->IsWR()?'y':'n',
     this->IsResponse()?'y':'n',
     this->Good()?'y':'n'
   );
   //report and check
   for(size_t i=0 ;i<10;i++){
      if(i>= this->size()) break;
      printf("data: +%lu: %16.16x\n",i,this->at(i));
   }
   printf("--- *********** ---\n");
}






