#include "ipbus.h"
#include "unistd.h"

#include <boost/array.hpp>
#include <iostream>


using namespace boost::asio;
using std::cout;
using std::endl;
using std::hex;

ipbus::ipbus(const char * _addr, unsigned short _port):
    addr(_addr),port(_port), connected(false), ios(), socket(ios), packetnumber(0)
{
    socket.open(ip::udp::v4());
    remote_endpoint = ip::udp::endpoint(ip::address::from_string(addr),port);

    if(socket.is_open())
        connected = true;

    socket.non_blocking(true);

    int status = Status();

    if(status < 0){
        socket.close();
        connected = false;
        cout << "Second connection attempt " << endl;
        socket.open(ip::udp::v4());
        if(socket.is_open())
            connected = true;
        socket.non_blocking(true);

        status = Status();
        if(status < 0){
            socket.close();
            connected = false;
            cout << "Connection failed" << endl;
            return;
        }
    }


    packetnumber = static_cast<uint16_t>(status);

    cout << "Starting at packet number " << packetnumber << endl;
}


ipbus::~ipbus(){
    if(connected)
        socket.close();
}

int ipbus::write(uint32_t addr, vector<uint32_t> data, bool nonicrementing)
{
    if(data.size() == 0)
        return -1;

    StartPacket();

    int nwords = data.size();
    int ntransactions = 0;
    while(nwords > 0){
        uint32_t header = 0;
        header |= 0x1f; //info code and type ID
        if(nonicrementing)
            header |= 0x20; // type ID is 3 for nonincrementing writes
        if(nwords < 256){
            header |= nwords<<8; // number
        } else {
            header |= 255 << 8;
        }

        header |= ((transactionnumber++) & 0xfff) << 16;
        transactionnumber = transactionnumber & 0xfff;
        header |= 2 << 28; // protocol version number

        sendbuffer.push_back(header);
        sendbuffer.push_back(addr);

        //cout << "sb :" << std::hex << sendbuffer[0] << endl;
        //cout << "wh :" << std::hex << header << endl;
        //cout << "wa :" << std::hex << addr << endl;

        if(nwords < 256){
            for(int i=0; i < data.size(); i++)
                sendbuffer.push_back(data[i]);
            nwords -= 256;
        } else {
            for(int i=0; i < 255; i++)
                sendbuffer.push_back(data[i]);
            nwords -= 255;
        }
        ntransactions++;
    }


    SendPacket();


    vector<uint32_t> receivebuffer(ntransactions+1,0);

    usleep(1000);

    if( ReadFromSocket(receivebuffer) != (ntransactions+1)*4){
        int s = Status();
     //   cout << "Status: " << s << endl;
        if(s < 0)
            return -3;
        if(s == packetnumber-1){
            SendPacket();
        } else if(s == packetnumber){
            CreateAndSendResendRequest(packetnumber);
        }
    }

    //cout << "w0 :" << std::hex << receivebuffer[0] << endl;
    //cout << "w1 :" << std::hex << receivebuffer[1] << endl << endl;

    return 0;
}

int ipbus::read(uint32_t addr, uint8_t size, vector<uint32_t> &data, bool nonicrementing)
{
    if(size==0)
        return -1;

    StartPacket();
    uint32_t header = 0;
    header |= 0x0f; //info code and type ID
    if(nonicrementing)
        header |= 0x20; // type ID is 2 for nonincrementing writes
    header |= size<<8; // number


    header |= ((transactionnumber++) & 0xfff) << 16;
    transactionnumber = transactionnumber & 0xfff;
    header |= 2 << 28; // protocol version number

    //cout << "h :" << std::hex << header << endl;
    //cout << "a :" << std::hex << addr << endl;

    sendbuffer.push_back(header);
    sendbuffer.push_back(addr);
    SendPacket();

    vector<uint32_t> receivebuffer(size+2,0);
    usleep(1000);

    if( ReadFromSocket(receivebuffer) != (size+2)*4){
        int s = Status();
        if(s < 0)
            return -3;
        if(s == packetnumber-1){
            SendPacket();
            usleep(1000);
        } else if(s == packetnumber){
            CreateAndSendResendRequest(packetnumber);
            usleep(1000);
        }
    }
    //cout << 0 << " :" << std::hex << receivebuffer[0] << endl;
    //cout << 1 << " :" << std::hex << receivebuffer[1] << endl;

    data.clear();
    for(unsigned int i =2; i < receivebuffer.size(); i++){
      //  cout << i << " :" << std::hex << receivebuffer[i] << endl;
        data.push_back(receivebuffer[i]);
    }

    //cout << endl;

    return 0;

}

int ipbus::write(uint32_t addr, uint32_t data)
{
    vector<uint32_t> v;
    v.push_back(data);
    return write(addr,v,false);
}

uint32_t ipbus::read(uint32_t addr)
{
    vector<uint32_t> v(1,0);
    read(addr,1,v,false);
    return v[0];
}

uint32_t ipbus::readModifyWriteBits(uint32_t addr, uint32_t andterm, uint32_t orterm)
{
    StartPacket();
    uint32_t header = 0;
    header |= 0x4f; //info code and type ID
    header |= 1<<8; // number of words is 1


    header |= ((transactionnumber++) & 0xfff) << 16;
    transactionnumber = transactionnumber & 0xfff;
    header |= 2 << 28; // protocol version number

    sendbuffer.push_back(header);
    sendbuffer.push_back(addr);
    sendbuffer.push_back(andterm);
    sendbuffer.push_back(orterm);
    SendPacket();

    vector<uint32_t> receivebuffer(3,0);
    usleep(1000);

    if( ReadFromSocket(receivebuffer) != 3*4){
        int s = Status();
        if(s < 0)
            return -3;
        if(s == packetnumber-1){
            SendPacket();
            usleep(1000);
        } else if(s == packetnumber){
            CreateAndSendResendRequest(packetnumber);
            usleep(1000);
        }
    }
    // This should contain the register content before the RMW
    return receivebuffer[2];
}


void ipbus::StartPacket(){
    uint32_t word = 0;
    word |= 0x0;            // control packet
    word |= 0xf << 4;       // endians
    word |=(packetnumber++)<<8; // running packet number
    word |=(0x20) << 24; // Protcol version and reserved bits

    sendbuffer.clear();
    sendbuffer.push_back(word);
}

void ipbus::CreateStatusPacket()
{
    uint32_t word = 0;
    word |= 0xf1 << 24; // different endianness...
    word |= 0x20;


    statusbuffer.clear();
    statusbuffer.push_back(word);
    for(int i = 1; i < 16; i++){
        statusbuffer.push_back(0);
    }


}

int ipbus::Status(unsigned int timeout)
{
    CreateStatusPacket();
    SendStatusPacket();

    vector<uint32_t>::size_type statuspacketsize = 16;

    vector<uint32_t> receivebuffer(statuspacketsize,0);

    usleep(timeout);

    if(ReadFromSocket(receivebuffer) != 16*4)
        return -1;

    // Middle 16 bits with swapped endinans
    return (receivebuffer[3]&0xFF00) | ((receivebuffer[3]&0xFF0000)>>16);
}

error_code ipbus::SendStatusPacket(){
    error_code err;
    socket.send_to(buffer(statusbuffer), remote_endpoint, 0, err);
    return err;
}




error_code ipbus::SendPacket(){
    error_code err;
    socket.send_to(buffer(sendbuffer), remote_endpoint, 0, err);
    return err;
}

error_code ipbus::CreateAndSendResendRequest(uint16_t packet)
{
        uint32_t word = 0;
        word |= 0x2;            // resend packet
        word |= 0xf << 4;       // endians
        word |= packet;         // running packet number
        word |=(0x20) << 24; // Protcol version and reserved bits

        std::vector<uint32_t> resendbuffer;
        resendbuffer.push_back(word);
        error_code err;
        socket.send_to(buffer(resendbuffer), remote_endpoint, 0, err);
        return err;
}

int ipbus::ReadFromSocket(vector<uint32_t> & rbuffer)
{
    int nbyte =0;
    boost::system::error_code err;
    try{
        nbyte = socket.receive_from(buffer(rbuffer), remote_endpoint, 0, err);
        if(err == boost::asio::error::try_again){
            cout << "Trying again, nbyte " << nbyte << endl;
            usleep(20000);
            nbyte = socket.receive_from(buffer(rbuffer), remote_endpoint, 0, err);
            if(err == boost::asio::error::try_again){
                cout << "UDP Timeout" << endl;
                return -2;
            }
         }
        if(err)
            throw boost::system::system_error(err);
    }
    catch(std::exception& e) {
        std::cerr << e.what() << endl;
    }

    /*
    cout << "Bytes received " << nbyte << endl;
    cout << hex;
    for(int i=0; i < nbyte/4; i++){
        cout << rbuffer[i] << " ";
    }
    cout << endl;*/

    return nbyte;
}



