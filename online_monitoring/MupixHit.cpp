//////////////////////////////////////////////////////////////////////////////////
// Implementation of Mupix hit type
// Based on leightweight stic3 data type from KIP (K. Briggl)
// Author(s) K. Briggl
#include "MupixHit.h"
#include <assert.h>
#include <stdio.h>
//#define NDEBUG

MupixHit::MupixHit(){m_raw=0;};

MupixHit::MupixHit(const uint32_t* raw){this->Fill(raw);};

void MupixHit::SetBankID(uint8_t bID){m_bankID=bID;};
uint8_t MupixHit::GetBankID(){return m_bankID;};

void MupixHit::SetEventID(uint16_t eID){m_eventID=eID;};
uint16_t MupixHit::GetEventID(){return m_eventID;};

void MupixHit::SetClusterID(uint16_t cID){m_clusterID=cID;};
uint16_t MupixHit::GetClusterID(){return m_clusterID;};

const uint32_t* MupixHit::Fill(const uint32_t* raw){
        this->m_raw= (0xffffffff00000000&(uint64_t(raw[1])<<32)) + ((0x00000000ffffffff)&uint64_t(raw[0])); 
        //printf("MupixHit::Fill(), raw[0]=%8.8x raw[1]=%8.8x, m_raw=%16.16lx\n",raw[0],raw[1],m_raw);
    return raw+2;
};

//Getters
unsigned char	MupixHit::GetChip() const{
        return (unsigned char)  	((m_raw>>26) & 0x3f);
}

unsigned int	MupixHit::GetChipID() const{
//        return (unsigned int)  	((m_raw>>22) & 0x3f);
//        return (unsigned int)  	(((m_raw&0xffffffff)>>26) & 0xff);
        return (unsigned int)  	(((m_raw)>>26) & 0x3f);
}

unsigned int	MupixHit::GetTimeStamp() const{
//        return (unsigned int)  	((m_raw>>28) & 0x0f);
    return (unsigned int)  	((m_raw >> 32) & 0x3ff);
}

unsigned int	MupixHit::GetTimeStamp2() const{
    return (unsigned int)  	((m_raw >> (32+10)) & 0x3f);
}

unsigned int	MupixHit::GetCol() const{
//        return (unsigned int)  	((m_raw>>6) & 0xff);
        return (unsigned int)  	((m_raw) & 0xff);
}

unsigned int	MupixHit::GetRow() const{
//        return (unsigned int)  	((m_raw>>14) & 0xff);
        return (unsigned int)  	((m_raw>>8) & 0xff);
}

unsigned int	MupixHit::GetToT() const{
//    return (unsigned int)  	((m_raw >> 42) & 0x3f);
    return (GetTimeStamp2() - GetTimeStamp()) &0x3f; //TODO: as a function of clkdividend(2)
}

MupixHit * MupixHit::ConvertToSubHeader() {
    if (isSubHeader())
        return static_cast<MupixHitSubHeader*>(this);
     else
        return this;
}

void MupixHit::TransformColRow(unsigned int &col, unsigned int &row) {
    uint8_t newcol = (col&0xff)-128; //(~col) & mask;
    //unsigned int newcol = col; //(~col) & mask;
    row &= 0xff;
    uint8_t newrow;
    if(row>=240)
        newrow=99-(255-row);
    else if(row<240 && row>=140)
        newrow = row - 40;
    else if(row<140 && row>=56)
        newrow = row-56;
    else
        {
      //  cout <<"non existing row "<<(uint16_t)row<<endl;
        newrow = row+200;
    }
    col=(unsigned int)newcol;
    row=(unsigned int)newrow;
}

void MupixHit::TransformColRow() {
    UInt_t col = GetCol();
    UInt_t row = GetRow();
    TransformColRow(col, row);
    m_raw = (m_raw & (~uint64_t(0xffff))) + (row << 8) + col;
}

int MupixHit::CheckMaskPix() {
    UInt_t col = GetCol();
    UInt_t row = GetRow();
    UInt_t layer = GetChipID();

    // TODO: make me better
    if (layer == 0 and col == 3 and row == 34) return -1;
    if (layer == 1 and col == 28 and row == 100) return -1;
    if (layer == 1 and col == 11 and row == 96) return -1;
    if (layer == 2 and col == 41 and row == 123) return -1;
    if (layer == 3 and col == 31 and row == 144) return -1;

    //printf("col %i row %i layer %i\n", col, row, layer);

    return 0;
}


void MupixHit::DumpLine(){
    if (!isSubHeader())
        printf("CHIP%u\tCol%u\tRow%u\tToT%u\tTS%u\n",GetChipID(),GetCol(),GetRow(),GetToT(), GetTimeStamp());
    else
        printf("CHIP%u\tTS%u\n",GetChipID(), GetTimeStamp());
}

void MupixHit::Dump(){
	printf("RAW data: %16.18lX \n",m_raw);
	printf("Chip-ID:    %u\n",GetChip());
        if (!isSubHeader()) {
            printf("Col:        %u\n",GetCol());
            printf("Row:        %u\n",GetRow());
            printf("ToT:        %u\n",GetToT());
        }
        printf("TS:         %u\n",GetTimeStamp());
        printf("-----------------\n");
}

#ifdef Mupix_HIT_ROOT_DERIVED
ClassImp(MupixHit);
#endif


MupixHit GenerateSimMupixHit(unsigned char _chip,unsigned char _channel, uint32_t _time){
	uint32_t raw=0;
	raw|=uint32_t(_chip&0x0f) << (28);
	raw|=uint32_t(_channel&0x1f) << (22);
	raw|=uint32_t(_time&0xfffff) << (1); //CC+fine time
        return MupixHit(&raw);
}; 

unsigned int	MupixHitSubHeader::GetTimeStamp() const{
        return (unsigned int)  	((m_raw>>16) & 0x3f);
}

unsigned int	MupixHitSubHeader::GetOverflow() const{
        return (unsigned int)  	((m_raw) & 0xffff);
}
