#include "PowerDriver.h"
#include <thread>

PowerDriver::PowerDriver()
{
	std::cout << "Warning: empty base class instantiated" << std::endl;
}

PowerDriver::PowerDriver(std::string n, EQUIPMENT_INFO* inf) : info{inf}, name{n}, read{0}, stop{0}, readstatus(FE_ERR_DISABLED), n_read_faults(0)
{
}

PowerDriver::~PowerDriver()
{
    stop = 1;
    if(readthread.joinable())
        readthread.join();
}


INT PowerDriver::ConnectODB()
{
	//general settings
	settings.connect("/Equipment/"+name+"/Settings");
	settings["IP"]("10.10.10.10");
	settings["Use Hostname"](true);
	settings["Hostname"]("");
	settings["NChannels"](2);
	settings["Global Reset On FE Start"](false);
	settings["Read ESR"](false);
	
	//variables
	variables.connect("/Equipment/"+name+"/Variables");
  
	relevantchange=0.005; //only take action when values change more than this value
	return FE_SUCCESS;
}


INT PowerDriver::Connect()
{
	std::string hostname = "";
	if(settings["Use Hostname"] == true)
	{
		hostname = settings["Hostname"];
		//make a hostname from the eq name
		if (hostname.length() < 2)
		{
			hostname = name;
			std::transform( hostname.begin(), hostname.end(), hostname.begin(), [](unsigned char c){ return std::tolower(c); } );
			settings["Hostname"] = hostname;
			std::cout << " set hostname key as " << settings["Hostname"] << std::endl;
		}
	}
	client = new TCPClient(settings["IP"],settings["port"],settings["reply timout"],hostname);
	ss_sleep(100);
	std::string ip = settings["IP"];
	min_reply_length = settings["min reply"];
	
	if(!client->Connect())
	{
		cm_msg(MERROR, "Connect to power supply ... ", "could not connect to %s", ip.c_str()); 
		return FE_ERR_HW;
	}		
	else cm_msg(MINFO,"power_fe","Init Connection to %s alive",ip.c_str());

    //Also start the read thread here
    readthread = std::thread(&PowerDriver::ReadLoop, this);
	
    return FE_SUCCESS;
}

void PowerDriver::ReadLoop()
{
    while(!stop){
        if(!read){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        } else {
            readstatus = ReadAll();
            read = 0;
        }
    }
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
    float value = 0.;
	if(!success)
	{
		cm_msg(MERROR, "Power supply read ... ", "could not read after command %s", cmd.c_str());
		error = FE_ERR_DRIVER;		
	}
    try {
	    value = std::stof(reply);
    }
    catch (const std::exception& e) {
        cm_msg(MERROR, "Power supply read...", "could not convert to float %s (%s)", reply.c_str(), e.what());
        error = FE_ERR_DRIVER;
    }
	return value;
}



std::string PowerDriver::ReadIDCode(int index, INT& error)
{
	std::string cmd;
	bool success;
	std::string reply="";
	error=FE_SUCCESS;

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

	if(index>=0) SelectChannel(instrumentID[index]);

	cmd = "*IDN?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));	
	success = client->ReadReply(&reply,min_reply_length);
	if(!success)
	{
		cm_msg(MERROR, "Power supply read ... ", "could not read id supply with address %d", instrumentID[index]);
		error = FE_ERR_DRIVER;
	}
	
	return reply;
	
}


std::vector<std::string> PowerDriver::ReadErrorQueue(int index, INT& error)
{
	std::string cmd;
	bool success;
	std::string reply="";
	error=FE_SUCCESS;

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

	if(index>=0) SelectChannel(instrumentID[index]);

	std::vector<std::string> error_queue;
	
	while(1)
	{
		cmd = "SYST:ERR?\n";
		client->Write(cmd);
		std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));	
		success = client->ReadReply(&reply,min_reply_length);
		if(!success)
		{
			cm_msg(MERROR, "Power supply read ... ", "could not read error supply with address %d", instrumentID[index]);
			error = FE_ERR_DRIVER;
		}
		//std::cout << " error queue " << reply << std::endl;
		error_queue.push_back(reply);
		if(reply.substr(0,1)=="0") break;
	}
	return error_queue;
}


int PowerDriver::ReadESR(int index, INT& error)
{
	std::string cmd;
	bool success;
	std::string reply="";
	error=FE_SUCCESS;

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

	if(index>=0) SelectChannel(instrumentID[index]);

	cmd = "*ESR?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));	
	success = client->ReadReply(&reply,min_reply_length);
	if(!success)
	{
		cm_msg(MERROR, "Power supply read ... ", "could not read ESR supply with address %d", instrumentID[index]);
		error = FE_ERR_DRIVER;
	}
	int value = std::stoi(reply);
	return value;
}


WORD PowerDriver::ReadQCGE(int index, INT& error)
{
	std::string cmd;
	bool success;
	std::string reply="";
	error=FE_SUCCESS;

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

	if(index>=0) SelectChannel(instrumentID[index]);
	
	cmd = "STAT:QUES?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));	
	success = client->ReadReply(&reply,min_reply_length);
	if(!success)
	{
		cm_msg(MERROR, "Power supply read ... ", "could not read ESR supply with address %d", instrumentID[index]);
		error = FE_ERR_DRIVER;
	}
	int value = std::stoi(reply);
	
	return WORD(value);
}


bool PowerDriver::ReadState(int index,INT& error)
{
	std::string cmd;
	bool success;
	std::string reply;
	error=FE_SUCCESS;
    bool value =false;

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);
  
	if(index>=0) SelectChannel(instrumentID[index]);
  
	cmd = "OUTP:STAT?\n";
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	success = client->ReadReply(&reply,min_reply_length);
	//std::cout << "here ********* , reply: " << reply << std::endl;
	
	if(!success)
	{
		cm_msg(MERROR, "power supply read ... ", "could not read %s state supply/channel: %d of", name.c_str(),instrumentID[index]);
		error = FE_ERR_DRIVER;
	}

	if(reply=="0") value=false;
	else if(reply=="1") value=true;
	else
	{ 
		cm_msg(MERROR, "power supply read ... ", "could not read %s valid state of supply/channel: %d", name.c_str(),instrumentID[index]);
		std::cout << "reply on state request = "<< reply << "." <<std::endl; 
		error = FE_ERR_DRIVER;
	}
	//std::cout << "here ********* , reply: " << reply << std::endl;
	return value; 
}


float PowerDriver::ReadVoltage(int index,INT& error)
{
    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

	error = FE_SUCCESS;
	float value = 0.0;
	if( SelectChannel(instrumentID[index]) )  {	  value = Read("MEAS:VOLT?\n",error);	}
		else error = FE_ERR_DRIVER;
	return value; 
}


float PowerDriver::ReadSetVoltage(int index,INT& error)
{
    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(instrumentID[index]))  {	  value = Read("VOLT?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


float PowerDriver::ReadCurrent(int index,INT& error)
{
    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(instrumentID[index]))  {	  value = Read("MEAS:CURR?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}


float PowerDriver::ReadCurrentLimit(int index,INT& error)
{
    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

  error = FE_SUCCESS;
	float value = 0.0;
  if(SelectChannel(instrumentID[index]))  {	  value = Read("CURR?\n",error);	}
	else error = FE_ERR_DRIVER;
  return value; 
}

float PowerDriver::ReadOVPLevel(int index,INT& error)
{
    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

	error = FE_SUCCESS;
	float value = 0.0;
	if( SelectChannel(instrumentID[index]) )  {	  value = Read("VOLT:PROT:LEV?\n",error);	}
		else error = FE_ERR_DRIVER;
	return value; 
}




// ****************** Set functions ********************* //


bool PowerDriver::Set(std::string cmd, INT& error)
{
    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

	bool success;
	client->Write(cmd);
	std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
	success = OPC();
	if(!success) { error=FE_ERR_DRIVER; cm_msg(MERROR, "Power supply ... ", "command %s not succesful for %s supply", cmd.c_str(),name.c_str() ); }
	return success;
}



void PowerDriver::SetCurrentLimit(int index, float value,INT& error)
{
  error = FE_SUCCESS;
  
  if(value<-0.1 || value > 90.0) //check valid range 
  {
  	cm_msg(MERROR, "Power supply ... ", "current limit of %f not allowed",value );
  	variables["Current Limit"][index]=currentlimit[index]; //disable request
  	error=FE_ERR_DRIVER;
  	return;  	
  }
  

  if( SelectChannel(instrumentID[index]) ) //'channel' is already instrument channel
  {
	bool success = Set("CURR "+std::to_string(value)+"\n",error);
  	if(!success) error=FE_ERR_DRIVER;
  	else // read changes
  	{
	  	voltage[index]=ReadVoltage(index,error);
	  	variables["Voltage"][index]=voltage[index];
  	}		
  }
  else error=FE_ERR_DRIVER;
}



void PowerDriver::SetState(int index, bool value,INT& error)
{
	std::string cmd;
	bool success;
	error = FE_SUCCESS;
	std::cout << " **** Request to set channel " << instrumentID[index] << " to : " << value << std::endl;   

	if(value==true)
	{
		if(!AskPermissionToTurnOn(index))
		{
			cm_msg(MERROR, "Genesys supply ... ", "changing of channel %d not allowed",instrumentID[index] );
			variables["Set State"][index]=false; //disable request
			error=FE_ERR_DISABLED;
			return;
		}
	}

    // From here on we grab the mutex until the end of the function: One transaction at a time
    const std::lock_guard<std::mutex> lock(power_mutex);

	if( SelectChannel(instrumentID[index]) )
	{
		if(value==true) { cmd="OUTP:STAT 1\n"; }
		else { cmd = "OUTP:STAT 0\n"; }
		client->Write(cmd);
		std::this_thread::sleep_for(std::chrono::milliseconds(client->GetWaitTime()));
		success = OPC();
		if(!success) error=FE_ERR_DRIVER;  	  		
	}
	else error=FE_ERR_DRIVER;
}



void PowerDriver::SetVoltage(int index, float value,INT& error)
{
	error = FE_SUCCESS;
	if(value<-0.1 || value > 25.) //check valid range 
	{
		cm_msg(MERROR, "Power supply ... ", "voltage of %f not allowed",value );
		variables["Demand Voltage"][index]=demandvoltage[index]; //disable request
		error=FE_ERR_DRIVER;
		return;  	
	}
    
	if( SelectChannel(instrumentID[index]) ) // module address in the daisy chain to select channel, or 1/2/3/4 for the HAMEG
	{
		bool success = Set("VOLT "+std::to_string(value)+"\n",error);
		if(!success) error=FE_ERR_DRIVER;
		else // read changes
		{
			voltage[index]=ReadVoltage(index,error);
			variables["Voltage"][index]=voltage[index];
			current[index]=ReadCurrent(index,error);
			variables["Current"][index]=current[index];
		}		
	}
	else error=FE_ERR_DRIVER;
}


void PowerDriver::SetOVPLevel(int index, float value,INT& error)
{
	error = FE_SUCCESS;
	if(value<-0.1 || value > 25.) //check valid range 
	{
		cm_msg(MERROR, "Power supply ... ", "voltage protection level of %f not allowed",value );
		variables["Demand OVP Level"][index]=OVPlevel[index]; //disable request
		error=FE_ERR_DRIVER;
		return;  	
	}
  


	if( SelectChannel(instrumentID[index]) ) // module address in the daisy chain to select channel, or 1/2/3/4 for the HAMEG
	{
		bool success = Set("VOLT:PROT:LEV "+std::to_string(value)+"\n",error);
		if(!success) error=FE_ERR_DRIVER;
		else // read changes
		{
			voltage[index]=ReadVoltage(index,error);
			variables["Voltage"][index]=voltage[index];
			current[index]=ReadCurrent(index,error);
			variables["Current"][index]=current[index];
			OVPlevel[index]=ReadOVPLevel(index,error);
			variables["OVP Level"][index]=OVPlevel[index];
		}		
	}
	else error=FE_ERR_DRIVER;
}



// ******************* Watch functions ******************** //

void PowerDriver::CurrentLimitChanged()
{
	INT err;
	int nChannelsChanged = 0;
	for(unsigned int i=0; i<currentlimit.size(); i++)
	{
		float value = variables["Current Limit"][i];
		if( fabs(value-currentlimit[i]) > fabs(relevantchange*currentlimit[i]) ) //compare to local book keeping, look for significant change
		{
			SetCurrentLimit(i,value,err);   // 
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Power ... ", "changing %s current limit of channel %d to %f failed, error %d",name.c_str(), instrumentID[i],value,err);
			else
			{
				cm_msg(MINFO, "Power ... ", "changing %s current limit of channel %d to %f", name.c_str(),i,value);
				nChannelsChanged++;
				currentlimit[i]=value;
			}
		}	
	}	
	if(nChannelsChanged < 1) cm_msg(MINFO, "Power ... ", "changing current limit request of %s rejected",name.c_str());
}



void PowerDriver::SetStateChanged()
{
	INT err;
	int nChannelsChanged = 0;
	
	for(unsigned int i=0; i<state.size(); i++)
	{
		bool value = variables["Set State"][i];
		//cm_msg(MINFO, "Power ... ", "set state = %d, current state = %d of index = %d, channel = %d ", value,(int)state[i],i,instrumentID[i]);
		if(value!=state[i]) //compare to local book keeping
		{
			SetState(i,value,err);
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Power ... ", "changing %s state of channel %d to %d failed, error %d", name.c_str(), instrumentID[i],value,err);
			else{ cm_msg(MINFO, "Power ... ", "changing %s state of channel %d to %d", name.c_str(),instrumentID[i],value);	nChannelsChanged++;	}
		}			
	}
	
	//if(nChannelsChanged < 1) cm_msg(MINFO, "Power ... ", "changing %s state request failed",name.c_str()); //this is triggered, even when it works. Maybe a double call or so? FW
	//else // read changes back from device 
	{
		int nChannels = instrumentID.size();
		for(int i = 0; i<nChannels; i++ ) 
		{
			bool value=ReadState(i,err);
			if(err==FE_SUCCESS) state[i]=value;
		} 
		variables["State"]=state; //push to odb
	}
}



void PowerDriver::DemandVoltageChanged()
{
	INT err;
	int nChannelsChanged = 0;
	for(unsigned int i=0; i<voltage.size(); i++)
	{
		float value = variables["Demand Voltage"][i];
		if( fabs(value-voltage[i]) > fabs(relevantchange*voltage[i]) ) //compare to local book keeping, look for significant change
		{
			SetVoltage(i,value,err);
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Power ... ", "changing %s voltage of channel %d to %f failed, error %d", name.c_str(), instrumentID[i],value,err);
			else
			{
				cm_msg(MINFO, "Power ... ", "changing %s voltage of channel %d to %f", name.c_str(), instrumentID[i],value);
				nChannelsChanged++;
				demandvoltage[i]=value;
			}
		}			
	}	
	if(nChannelsChanged < 1) cm_msg(MINFO, "Genesys supply ... ", "changing voltage request rejected");
}


void PowerDriver::DemandOVPLevelChanged()
{
	INT err;
	int nChannelsChanged = 0;
	for(unsigned int i=0; i<OVPlevel.size(); i++)
	{
		float value = variables["Demand OVP Level"][i];
		if( fabs(value-OVPlevel[i]) > fabs(relevantchange*OVPlevel[i]) ) //compare to local book keeping, look for significant change
		{
			SetOVPLevel(i,value,err);
			if(err!=FE_SUCCESS ) cm_msg(MERROR, "Power ... ", "changing %s voltage protection level of channel %d to %f failed, error %d", name.c_str(), instrumentID[i],value,err);
			else
			{
				cm_msg(MINFO, "Power ... ", "changing %s voltage protection level of channel %d to %f", name.c_str(), instrumentID[i],value);
				nChannelsChanged++;
			}
		}			
	}	
	if(nChannelsChanged < 1) cm_msg(MINFO, "Genesys supply ... ", "changing voltage protection level request rejected");
}





