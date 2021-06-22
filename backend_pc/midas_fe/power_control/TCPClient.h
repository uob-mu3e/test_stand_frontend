#ifndef TCPCLIENT_H
#define TCPCLIENT_H

#include <boost/asio.hpp>

using namespace boost::asio; 
using namespace boost::asio::ip; 


class TCPClient{

	public:
	
		TCPClient(std::string IP,int port,int=2000);
		TCPClient(std::string IP, int port,int=2000, std::string hostname="");
		~TCPClient();
		bool Connect();
		bool Write(std::string str);
		bool ReadReply(std::string *str,int = 3);
		bool FlushQueu();
		int GetWaitTime() { return default_wait; }
		void SetDefaultWaitTime(int value){ default_wait = value; }
		
	private:
	  boost::asio::io_service io_service;
    ip::tcp::socket* socket;
    std::string ip;
    std::string hostname;
    int port;
    int default_wait;
    std::string read_stop;
    
    int read_time_out;
};

#endif
