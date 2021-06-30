//****************************************************************************************
//
//	Base Driver for the LV power supplies. Use derived class fro TDK or HAMEG or .. supply
//
//	F.Wauters - Nov. 2020
//	

#ifndef POWERDRIVER_H
#define POWERDRIVER_H

#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>

#include "midas.h"
#include "odbxx.h"
#include "TCPClient.h"



class PowerDriver{	

	public:
	
		PowerDriver();
		PowerDriver(std::string, EQUIPMENT_INFO*);
        virtual ~PowerDriver();
		
		virtual INT ConnectODB();
		INT Connect();
		virtual INT Init(){return FE_ERR_DRIVER;};
		virtual INT ReadAll(){return FE_ERR_DRIVER;}
    INT GetReadStatus(){return readstatus;}
    void ReadLoop();
    void StartReading(){read = 1;}
		virtual void ReadESRChanged(){};
		std::vector<bool> GetState() const { return state; }
		std::vector<float> GetVoltage() const { return voltage; }
		std::vector<float> GetCurrent() const { return current; }
		void Print(); 
		
		bool Initialized() const { return initialized; }
		bool Enabled();
		void SetInitialized() { initialized = true; }
		void UnsetInitialized() { initialized = false; }
		std::string ReadIDCode(int,INT&);
		
		virtual bool AskPermissionToTurnOn(int) { std::cout << "Ask permissions in derived class!" << std::endl; return false;};
		
		bool ReadState(int,INT&);
		float ReadVoltage(int,INT&);
		float ReadCurrent(int,INT&);
		int ReadESR(int,INT&);
		
		WORD ReadQCGE(int,INT&);
		std::vector<std::string> ReadErrorQueue(int,INT&);
		std::string GetName() { return name; }
		
		
		EQUIPMENT_INFO GetInfo() { return *info; } //by value, so you cant modify the original

		void AddReadFault(){n_read_faults = n_read_faults + 1;}
		void ResetNReadFaults(){ n_read_faults = 0; }
		int GetNReadFaults() { return n_read_faults; }
		

	protected:

	
		//read
		float Read(std::string,INT&);
		float ReadSetVoltage(int,INT&);
		float ReadCurrentLimit(int,INT&);
		float ReadOVPLevel(int,INT&);

		
		//set
		bool SelectChannel(int);
		bool OPC();
		bool Set(std::string,INT&);
		void SetCurrentLimit(int,float,INT&);
		
		//watch
		void CurrentLimitChanged();
		void SetStateChanged();
		void SetState(int,bool,INT&);
		void SetVoltage(int,float,INT&);
		void DemandVoltageChanged();
		
		void DemandOVPLevelChanged();
		void SetOVPLevel(int,float,INT&);
		
		EQUIPMENT_INFO* info;
		std::string name;
		midas::odb settings;
		midas::odb variables;
		
		TCPClient* client;

    std::mutex power_mutex;
    std::thread readthread;

    std::atomic<int> read;
    std::atomic<int> stop;
    std::atomic<INT> readstatus;

		int n_read_faults;
		
		float relevantchange;

		
		//local copies of hardware state
		std::vector<bool> state;
		std::vector<float> voltage;
		std::vector<float> demandvoltage;
		std::vector<float> current;
		std::vector<float> currentlimit;
		
		std::vector<float> OVPlevel;
		
		std::vector<int> instrumentID;
		
		
		
		
	private:
		bool initialized = false;
		int min_reply_length;
		

};

#endif
