//****************************************************************************************
//
//	Base Driver for the LV power supplies. Use derived class for TDK or HAMEG or .. supply
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

enum COMMAND_TYPE {
    SetVoltage,
    SetCurrent,
    Reset,
    Beep,
    CLearStatus,
    SelectChannel,
    ReadErrorQueue,
    ReadVoltage,
    ReadSetVoltage,
    ReadCurrent,
    ReadCurrentLimit,
    SetState,

    ReadESR,
    OPC,
    ReadQCGE,
    ReadState,
    ReadOVPLevel,
    SetOVPLevel
    //TODO Complete with full list of action commands
};

class PowerDriver{	
	public:
		PowerDriver();
		PowerDriver(std::string, EQUIPMENT_INFO*);
        virtual ~PowerDriver();

        std::vector<bool> GetState() const { return state; }
        std::vector<float> GetVoltage() const { return voltage; }
        std::vector<float> GetCurrent() const { return current; }
        std::string ReadIDCode(int,INT&);
        std::vector<std::string> ReadErrorQueue(int,INT&);
        std::string GetName() { return name; }

        INT Connect();
        INT GetReadStatus(){return readstatus;}

        void ReadLoop();
        void StartReading(){read = 1;}
        void Print();
        void SetInitialized() { initialized = true; }
        void UnsetInitialized() { initialized = false; }
        void AddReadFault(){n_read_faults = n_read_faults + 1;}
        void ResetNReadFaults(){ n_read_faults = 0; }

        bool Initialized() const { return initialized; }
        bool Enabled();
        bool ReadState(int,INT&);

        float ReadVoltage(int,INT&);
        float ReadCurrent(int,INT&);

        int ReadESR(int,INT&);
        int GetNReadFaults() { return n_read_faults; }

        WORD ReadQCGE(int,INT&);
        EQUIPMENT_INFO GetInfo() { return *info; } //by value, so you cant modify the original

        virtual INT ConnectODB();
        virtual INT Init(){return FE_ERR_DRIVER;};
        virtual INT ReadAll(){return FE_ERR_DRIVER;}
        virtual void ReadESRChanged(){};
        virtual bool AskPermissionToTurnOn(int) { std::cout << "Ask permissions in derived class!" << std::endl; return false;};

	protected:
        EQUIPMENT_INFO* info;
        std::string name;
        midas::odb settings;
        midas::odb variables;

        TCPClient* client;

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

        std::mutex power_mutex;
        std::thread readthread;

        std::atomic<int> read;
        std::atomic<int> stop;
        std::atomic<INT> readstatus;

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

        virtual std::string GenerateCommand(COMMAND_TYPE, float);

	private:
		bool initialized = false;
		int min_reply_length;
		

};

#endif
