#include "TCPClient.h"
#include <iostream>
#include <chrono>
#include <thread>


using boost::asio::ip::tcp;

TCPClient::TCPClient(std::string IP,int P, int to)
{
	socket = new ip::tcp::socket(io_service);
	ip = IP;
	port = P;
	default_wait = 7;
	read_time_out = to;
	read_stop = "\n";
}

bool TCPClient::Connect()
{
	boost::system::error_code ec;
	socket->connect( tcp::endpoint( boost::asio::ip::address::from_string(ip), port ) , ec);
	if (ec)
	{
	  std::cout << " socket->connect failed with err:" << ec << std::endl;
	  return false;
	}
	socket->non_blocking(true);
	FlushQueu();
	return true;
}

TCPClient::~TCPClient()
{
	socket->close();
}

bool TCPClient::Write(std::string str)
{
  if(!socket->is_open()) return false;
  boost::system::error_code error;
  boost::asio::write( *socket, boost::asio::buffer(str), error );
  if( !error )
  {
		std::cout << "Client sent message: " << str << std::endl;
	}
	else
	{
		std::cout << "send failed: " << error.message() << std::endl;
		return false;
   }

  return true;
}

bool TCPClient::FlushQueu()
{
	boost::system::error_code error;
	std::size_t data_size;
	char data[1024];
	data_size = socket->available(error);
	if(error) { std::cout << " size request failed " << std::endl; return false; }
	while( int(data_size) > 0)
	{
		size_t length = socket->read_some(boost::asio::buffer(data), error);
		if(error) { std::cout << " size request failed " << std::endl; return false; }
		data_size = socket->available(error);
		if(error) { std::cout << " size request failed " << std::endl; return false; }
	}
	return true;
	
}

bool TCPClient::ReadReply(std::string *str,int min_size) 
{ 
	std::size_t data_size;
	auto start = std::chrono::system_clock::now();
	int time_elapsed = 0;
	boost::system::error_code error;
	
	// wait for at least 3 characters (minimum reply for  is "*\n")
	while( time_elapsed < read_time_out && data_size < min_size) 
	{	
		data_size = socket->available(error);
		if(error) std::cout << " size request failed " << std::endl;
  	std::this_thread::sleep_for(std::chrono::milliseconds(default_wait));
		auto end = std::chrono::system_clock::now();
		time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	}	
	
	//std::cout << " data of size " << data_size << " waiting after " << time_elapsed << " ms " << std::endl;
	//waiting for valid data failed
	if(data_size < min_size  || time_elapsed > read_time_out)
	{
		*str="";
		return false;
	}
	
  //read
  streambuf buf; 
  read_until(*socket, buf, read_stop); 
  std::string data = buffer_cast<const char*>(buf.data());
  //std::cout << "data : " << data << std::endl; 
  //std::size_t found = data.find('\r');
	//if (found!=std::string::npos)  std::cout << "first 'needle' found at: " << found << '\n';
  //remove newline
  data.erase(std::remove(data.begin(), data.end(), '\n'), data.end());
  data.erase(std::remove(data.begin(), data.end(), '\r'), data.end()); //don`t get why there is a /r in the reply string FW
  *str=data;
  //std::cout << " --- data read : " << *str << std::endl;
  return true;
 
} 


