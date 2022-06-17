/** author: A.Loreti
  * date: Feb. 2021
  * abstract: retrives data from binary file in MIDAS format
  */ 

#include "iostream"
#include "string"
#include "fstream"
#include "sstream"
#include <cstdlib>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <cmath>
//#include </opt/midas_software/midas/include/midas.h>
//#include <TH1D.h>

using namespace std;

struct EVENT_H{
   	short int event_id;           /**< event ID starting from one   2 bytes   */
   	short int trigger_mask;       /**< hardware trigger mask        2 bytes   */
   	unsigned int serial_number;   /**< serial number starting from one */
   	unsigned int time_stamp;      /**< time of production of event     */
   	unsigned int data_size;       /**< size of event in bytes w/o header */
}event;

struct BANK_H{
	unsigned int data_size;              // 4 bytes: number of data-banks
    unsigned int flags;                  // 4 bytes: it tells if the bank is 16-bit or 32-bit
}bank_header;

struct BANK16{
   	char name[4];                // 4 bytes          
   	short int type;              // 2 bytes              
   	short int data_size;         // 2 bytes                
}bank16;

struct BANK32{
	char name[4];                // 4 bytes          
  	int type;              // 2 bytes              
	int data_size;         // 2 bytes                
}bank32;

int main(int argc, char** argv){

	//ROOT HIST
	/*TH1D *h_trt = new TH1D("h_t1", "RT temperature sensor", 256,-255,0);
	TH1D *h_tho = new TH1D("h_t2", "HoneyWell Temperature sensor", 101,0,100);
	TH1D *h_hum = new TH1D("h_hum", "HoneyWell Humidity sensor", 101,0,100);*/

	//////////////////////////////////////////////////////////////////////////////
	//////////////////////////FILES I/O///////////////////////////////////////////
	if (argc < 2) {
        cout<< "Usage: myAnalyser <input_file_name> "<< endl;
        return -1;
    }
	if (argc >= 2) {
        cout<< "Parameters after "<< argv[1] << " will be ignored." << endl;
    }

	//File streams 
	ifstream f(argv[1], ios::binary);

	/////////////////////////////////////////////////////////////////////////////
	//////////////////////////READING DATA FROM MIDAS FILE/////////////////////////

	if (f.is_open()) {
		
		//while (!f.eof()) {			
			// EVENT HEADER
			f.read((char*)&event.event_id, 2);
			f.read((char*)&event.trigger_mask, 2);
			f.read((char*)&event.serial_number, 4);
			f.read((char*)&event.time_stamp, 4);
			f.read((char*)&event.data_size, 4);
            cout << "EVENT: " << event.event_id << '\t' << event.trigger_mask << '\t' << event.serial_number\
			<< '\t' << event.time_stamp << '\t' << event.data_size << '\n';

			// BANK HEADER
			f.read((char*)&bank_header.data_size, 4);
			f.read((char*)&bank_header.flags, 4);
			cout << "BANK HEADER: data size =  " << bank_header.data_size << "flags =  " << bank_header.flags << endl;

			for(unsigned int k=0; k<bank_header.data_size; k++){
				if(sizeof(bank_header.flags == 4)){
					// 16-bit bank
					f.read((char*)&bank16.name,4);
					f.read((char*)&bank16.type,4);
					f.read((char*)&bank16.data_size, 4);
					char * dat = new char[bank32.data_size];
					f.read((char*)dat, bank32.data_size);
					cout << "16-bit BANK " << bank16.name << '\t' << bank16.type << '\t' << bank16.data_size<< endl; 
				}
				else{
					// 32-bit bank
					f.read((char*)&bank32.name,4);
					f.read((char*)&bank32.type,4);
					f.read((char*)&bank32.data_size, 4);	
					char * dat = new char[bank32.data_size];
					f.read((char*)dat, bank32.data_size);
					cout << "32-bit BANK " << bank32.name << '\t' << bank32.type << '\t' << bank32.data_size<< endl; 
				}
			}
		//}
	}
	else cout << "Unable to open file" << '\n';

	f.close();
	
	return 0;
}

