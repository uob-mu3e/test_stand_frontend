#ifndef IPBUS_H
#define IPBUS_H

#include <boost/asio.hpp>
#include <vector>
#include <stdint.h>

using std::vector;
using boost::system::error_code;

class ipbus{
public:
    ipbus(std::string addr, unsigned short port);
    ~ipbus();
    bool isConnected(){return connected;}

    int write(uint32_t addr, vector<uint32_t> data, bool nonicrementing = false);
    int read(uint32_t addr, uint8_t size, vector<uint32_t> & data, bool nonicrementing = false);

    int write(uint32_t addr, uint32_t data);
    uint32_t read(uint32_t addr);
    // Register X <= (X & andterm)|orterm
    uint32_t readModifyWriteBits(uint32_t addr, uint32_t andterm, uint32_t orterm);

protected:
    int disconnect();
    int connect();

    error_code SendPacket();
    void StartPacket();

    error_code SendStatusPacket();
    void CreateStatusPacket();

    error_code CreateAndSendResendRequest(uint16_t packet);

    int ReadFromSocket(vector<uint32_t> &rbuffer); // length of buffer determines read size

    int Status(unsigned int timeout=20000);

    std::string addr;
    const unsigned short port;
    bool connected;
    boost::asio::io_service ios;
    boost::asio::ip::udp::socket socket;
    boost::asio::ip::udp::endpoint remote_endpoint;

    std::vector<uint32_t> sendbuffer;
    std::vector<uint32_t> statusbuffer;


    uint16_t packetnumber;
    uint16_t transactionnumber;

    uint32_t ntimeouts;
};


#endif // IPBUS_H
