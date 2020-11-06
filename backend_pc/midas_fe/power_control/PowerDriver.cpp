#include "PowerDriver.h"

PowerDriver::PowerDriver()
{
	std::cout << "Warning: empty base class instantiated" << std::endl;
}

PowerDriver::PowerDriver(std::string n, EQUIPMENT_INFO* inf)
{
	name=n;
	info=inf;
	initialized=false;
}


INT PowerDriver::ConnectODB()
{
	//general settings
	settings.connect("/Equipment/"+name+"/Settings");
	settings["IP"]("10.10.10.10");
	settings["NChannels"](2);
	settings["Global Reset On FE Start"](true);
  
	//variables
	variables.connect("/Equipment/"+name+"/Variables");
  
	relevantchange=0.005; //only take action when values change more than this value
	return FE_SUCCESS;
}


INT PowerDriver::Connect()
{
	client = new TCPClient(settings["IP"],settings["port"],settings["reply timout"]);
	ss_sleep(100);
	std::string ip = settings["IP"];
	min_reply_length = settings["min reply"];
	
	if(!client->Connect())
	{
		cm_msg(MERROR, "Connect to power supply ... ", "could not connect to %s", ip.c_str()); 
		return FE_ERR_HW;
	}		
	else cm_msg(MINFO,"power_fe","Init Connection to %s alive",ip.c_str());
	
	return FE_SUCCESS;
}


bool PowerDriver::Enabled()
{
	midas::odb common("/Equipment/"+name+"/Common");
	bool value = common["Enabled"];
	return value;
}


bool PowerDriver::SelectChannel(int ch)
{
	std::string cmd;
	bool success;
	std::string reply;
  
	cmd = "INST:NSEL " + std::to_string(ch)+ "\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	success = OPC();
	if(!success)
	{
		cm_msg(MERROR,"power_fe","Not able to select channel %d ",ch);
		return false;
	}
	return true;
}


bool PowerDriver::OPC()
{
	client->Write("*OPC?\n");
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	std::string reply;
	bool status = client->ReadReply(&reply,min_reply_length);
	return status;
}



void PowerDriver::Print()
{
	std::cout << "ODB settings: " << std::endl << settings.print() << std::endl;
	std::cout << "ODB variables: " << std::endl << variables.print() << std::endl;
}



// *****************   Read functions *************** //



float PowerDriver::Read(std::string cmd, INT& error)
{
	error = FE_SUCCESS;
	bool success;
	std::string reply;
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	success = client->ReadReply(&reply,min_reply_length);
	if(!success)
	{
		cm_msg(MERROR, "Power supply read ... ", "could not read after command %s", cmd.c_str());
		error = FE_ERR_DRIVER;		
	}
	float value = std::stof(reply);
	return value;
}



std::string PowerDriver::ReadIDCode(int channel, INT& error)
{
	std::string cmd;
	bool success;
	std::string reply="";
	error=FE_SUCCESS;

	if(channel>=0) SelectChannel(channel);

	cmd = "*IDN?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));	
	success = client->ReadReply(&reply,min_reply_length);
	if(!success)
	{
		cm_msg(MERROR, "Power supply read ... ", "could not read id supply with address %d", channel);
		error = FE_ERR_DRIVER;
	}
	
	return reply;
	
}


bool PowerDriver::ReadState(int channel,INT& error)
{
	std::string cmd;
	bool success;
	std::string reply;
	error=FE_SUCCESS;
	bool value;
  
	if(channel>=0) SelectChannel(channel);
  
	cmd = "OUTP:STAT?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	success = client->ReadReply(&reply,min_reply_length);
	//std::cout << "here ********* , reply: " << reply << std::endl;
	
	if(!success)
	{
		cm_msg(MERROR, "power supply read ... ", "could not read %s state supply/channel: %d of", name.c_str(),channel);
		error = FE_ERR_DRIVER;
	}

	if(reply=="0") value=false;
	else if(reply=="1") value=true;
	else
	{ 
		cm_msg(MERROR, "power supply read ... ", "could not read %s valid state of supply/channel: %d", name.c_str(),channel);
		std::cout << "reply on state request = "<< reply << "." <<std::endl; 
		error = FE_ERR_DRIVER;
	}
	//std::cout << "here ********* , reply: " << reply << std::endl;
	return value; 
}


float PowerDriver::ReadVoltage(int channel,INT& error)
{
	error = FE_SUCCESS;
	float value = 0.0;
	if( SelectChannel(channel) )  {	  value = Read("MEAS:VOLT?\n",error);	}
		else error = FE_ERR_DRIVER;
	return value; 
}


float PowerDriver::ReadSetVoltage(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("VOLT?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


float PowerDriver::ReadCurrent(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("MEAS:CURR?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


float PowerDriver::ReadCurrentLimit(int channel,INT& error)
{
  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(channel))  {	  value = Read("CURR?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


// ****************** Set functions ********************* //

bool PowerDriver::Set(std::string cmd, INT& error)
{
	bool success;
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	success = OPC();
	if(!success) { error=FE_ERR_DRIVER; cm_msg(MERROR, "Power supply ... ", "command %s not succesful for %s supply", cmd.c_str(),name.c_str() ); }
	return success;
}


