//////////////////////////////////////////////////////////////////////////////////
// Definition of Mutrig hit type
// Based on leightweight stic3 data type from KIP (K. Briggl)
// Author(s) K. Briggl

#ifndef MUPIX_HIT_H__
#define MUPIX_HIT_H__
#include "steering.h"
#include <stdint.h>
#include "TObject.h"


//forward declarations
class CalibrationBox;

class MupixHit
#ifdef MUPIX_HIT_ROOT_DERIVED
 : public TObject
#endif
{
public:
	//Construct emptied hit to be filled
    MupixHit();

    MupixHit(const uint32_t* raw);
    ~MupixHit(){}

    //Return value: pointer to the data to be checked against next
    const uint32_t* Fill(const uint32_t* raw);
	
	//printout
	void Dump();
	void DumpLine();

    //Getters
	//Sidenote: defining the methods const enables them to be shown in the ROOT browser, e.g. when saved to a Tree!
	//Sidenote2: the comments behind the methods actually make it into the ROOT dictionary, e.g. as an explanation!
    unsigned char	GetChip() const;	   //ChipID
    unsigned int 	GetChipID() const;	   //ChipID
    unsigned int	GetTimeStamp() const;  //time stamp of beginning of pulse
    unsigned int	GetTimeStamp2() const; //time stamp of end of pulse
    unsigned int    GetCol() const;        //Column
    unsigned int    GetRow() const;        //Row
    unsigned int    GetToT() const;        //Time-over-Threshold

    void TransformColRow(unsigned int & col, unsigned int & row);
    void TransformColRow();
    int CheckMaskPix();

    bool isSubHeader () const {return (bool)(GetChipID() == 0x3f);}

    MupixHit *ConvertToSubHeader();

    uint64_t	GetRawData() const {return m_raw;}
    uint32_t*	GetRawDataPointer() const {return (uint32_t*)&m_raw;}

	void SetBankID(uint8_t bID);
	uint8_t GetBankID();
	void SetEventID(uint16_t eID);
	uint16_t GetEventID();
	void SetClusterID(uint16_t cID);
	uint16_t GetClusterID();

	//access to change things...
    friend class CalibrationBox;
protected:
    uint64_t	m_raw;	//Raw data containing everything
	uint8_t		m_bankID; //reference to parent container;
	uint16_t	m_eventID; //combined cluster & event ID;
	uint16_t	m_clusterID; //combined cluster & event ID;
#ifdef MUPIX_HIT_ROOT_REFLECTED
ClassDef(MupixHit,1);
#endif
};

MupixHit GenerateSimMupixHit(unsigned char _chip,unsigned char _channel, uint32_t _time);

class MupixHitSubHeader : public MupixHit
{
public:
    MupixHitSubHeader();
    MupixHitSubHeader(const uint32_t* raw);

    unsigned int	GetTimeStamp() const;  //time including CC and FC, no DNL correction

    unsigned int GetOverflow () const;
    //~MupixHitSubHeader(); //WHY NO DESTRUCTOR?

};

#endif

