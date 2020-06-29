#include "MupixRawBank.h"

#include <iostream>

// NOT BASED ON SPECBOOOK
// CHECK WITH LUIGI's VERSION!


MupixRawBank::MupixRawBank(int bklen, int bktype, const char* name, void *pdata):
    TGenericData(bklen, bktype, name, pdata) {}

bool MupixRawBank::IsGood(){
    	if(!(GetSize()>7+1 && (GetSize()-7-1)%2==0)){
		printf("MupixRawBank::MupixRawBank(): Size check failed (Size=%d, modulo=%d)\n",GetSize(),(GetSize()-7-1)%2);
		DumpRaw(); 
		DumpInfo();
		return false;
	};
//    	if(GetFPGAID()&&0xfff0!=0xfeb0){			printf("MupixRawBank::MupixRawBank(): FEBID check failed\n"); DumpRaw(); DumpInfo(); throw;};
    	if(GetHeaderFPGATime()==0xfabeabbafabe){
		printf("MupixRawBank::MupixRawBank(): FPGA time check failed\n");
		DumpRaw();
		DumpInfo();
		return false;
	};
	return true;
}


void MupixRawBank::Print(int level) {
  std::cout << "Mutrig decoder for bank " << GetName().c_str() << std::endl;
}


//return raw header data
MupixBankInfo MupixRawBank::GetBankInfo() const{
	if(GetSize()<6){
		printf("MupixRawBank::GetBankInfo(): Bank is too short!\n");
		throw;
	}
MupixBankInfo info;
	info[0]=GetData32()[0]; //preamble 
	// Skip 0xFABEABBA word
	info[1]=GetData32()[2]; //timestamp msb
	info[2]=GetData32()[3]; //timestamp lsb
	info[3]=GetData32()[4]; //event counter
				// Skip overflow 
	info[4]=GetData32()[6]; //chip counter
	info[5]=GetData32()[GetSize()-1]; //trailer
	return info;
}

//return raw header data 
uint64_t MupixRawBank::GetHeaderRaw() const{
    uint64_t raw= ((uint64_t)GetData32()[0] << 32) | GetData32()[1];
    return raw;
};

//return 48b FPGA timestamp of this frame
uint64_t MupixRawBank::GetHeaderFPGATime() const{
	// uint64_t raw= (u_int64_t)GetData32()[7];
	// return raw;
    uint64_t raw= (uint64_t(GetData32()[2]) << 32) | uint64_t(GetData32()[3]);
    return (raw>>16) & 0xFFFFFFFFFFFF;
};

//return FEB ID (for FEB mode only)
uint16_t MupixRawBank::GetFPGAID() const{
	return (GetData32()[0] >> 8) & 0xffff;
};

// TODO: Add get functions for event counter and ChipID 

uint32_t MupixRawBank::GetEventCounter() const{
	return (GetData32()[4]) & 0x7FFFFFFF;
}

// uint16_t MupixRawBank::GetHeaderChipCounter() const{
// 	return (GetData32()[6] >> (2+13) & 0x7FFFFFFF;
// }

//return raw trailer data
uint32_t MupixRawBank::GetTrailerRaw() const{
	return GetData32()[GetSize()-1]; 
};

// // Check function implements bit shift correctly to get the sub-header timestamp
// uint32_t MupixRawBank::GetSubHeaderFPGATime() const{
// 	uint32_t raw = GetData32()[BANKHEADER_LEN+PACKETHEADER_LEN-1] >> 16 
// 	return raw & 0xffff;
// };

// Get raw number of Hits in bank 
int MupixRawBank::GetNTimestamps() const{
    // TODO: Check if this is correct number either minus 8 or minus 10?
    return (GetSize()-7-1)/2; 
};

int MupixRawBank::GetNHits() {
	if(m_hits.size()==0 && GetSize()>7+1)
		CollectHits();
	return m_hits.size();
};

void MupixRawBank::CorrectHitsColRow() {
    for (auto it : m_hits) {
        it.TransformColRow();
    }
}

std::vector<MupixHit>& MupixRawBank::GetHits(){
	if(m_hits.size()==0 && GetSize()>6+1)
		CollectHits();
	return m_hits;
};

// TODO: Handle sub-header which is sent after final hit
unsigned int MupixRawBank::GetPacketCounter(){
    return (unsigned int)(*(GetData32()+1));
}

unsigned int MupixRawBank::GetFPGATimeStampOld() {
    return (unsigned int)((0xffffffff00000000 & uint64_t(*(GetData32()+2)) << 32) |  (0x00000000ffffffff & uint64_t(*(GetData32()+3))));
}

unsigned int MupixRawBank::GetPacketCounterZeroSuppressed(){
    return (unsigned int)(*(GetData32()+4));
}

unsigned int MupixRawBank::GetMupixChipCounter(){
    return (unsigned int)(*(GetData32()+5));
}

int MupixRawBank::ForEachHit(std::function<int(MupixHit& hit)> func){
	const uint32_t* ptr=GetData32()+7;
	int nW=7;
	int status=0;
	while (nW<GetSize()-1){
		MupixHit hit(ptr);
	//	if(hit.)
		status=func(hit);
			if(status<0) break;
		ptr+=2;
		nW+=2;
        }
	return status;
};

void MupixRawBank::CollectHits(){
        ForEachHit([this](MupixHit& hit){
		m_hits.emplace_back(hit);
		return 0;
	});
}

void MupixRawBank::DumpRaw(){
	const uint32_t* ptr=GetData32();
	printf("Bank raw data dump (including headers): \n");
	for(int i=0;i<GetSize();i++,ptr++){
		printf("%d at %p: data %8.8x\n",i,ptr,*ptr);
	}
};

void MupixRawBank::DumpInfo(){
        std::cout<<"Bank name "<<this->GetName()<<" Length "<<this->GetSize()<<std::endl;
        printf("Header FPGA ID:    %8.8lx\n",this->GetFPGAID());
        printf("Header FPGATime:    %8.8lx\n",this->GetHeaderFPGATime());
        printf("Header Event count  %8.8lx\n",this->GetEventCounter());
        printf("Header Chip  count  %8.8lx\n",this->GetMupixChipCounter());
        printf("Timestamps: %d\n",this->GetNTimestamps());
//        printf("Hits: %d\n",this->GetNHits());

};
