#ifndef TCPCLIENT_H
#define TCPCLIENT_H

#include <boost/asio.hpp>

using namespace boost::asio; 
using namespace boost::asio::ip; 


class TCPClient{

	public:
	
		TCPClient(std::string IP,int port);
		~TCPClient();
		bool Connect();
		bool Write(std::string str);
		bool ReadReply(std::string *str,int = 3,int = 100);
		bool FlushQueu();
		int GetWaitTime() { return default_wait; }
		
	private:
	  boost::asio::io_service io_service;
    ip::tcp::socket* socket;
    std::string ip;
    int port;
    int default_wait;
    std::string read_stop;
};

#endif
