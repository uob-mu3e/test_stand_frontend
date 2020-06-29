//File: 	MupixRawBank.h
//Desc: 	Structure holding the raw event data from a "MUTR" midas bank
//		Provides access to the header and hits
//Author(s):	K. Briggl


#ifndef _MUPIXRAWBANK_H__
#define _MUPIXRAWBANK_H__


#include "TGenericData.hxx"
#include "MupixHit.h"
#include <vector>
#include <array>
#include <functional>

#ifdef MUPIX_BANK_ROOT_DERIVED
#include "TObject.h"
#endif

//FEB_MODE: if defined, use bank format given by A10 farm frontend.
//          if not, use bank format produced by stratix4 evalcard
#define FEB_MODE

class MupixBankInfo : public std::array<uint32_t,6> {
public:
  MupixBankInfo(){};
  MupixBankInfo(std::array<uint32_t,6> raw): std::array<uint32_t,6>(raw){};
  uint16_t GetHeaderFPGAID() const; //return ID of FPGA this packet came from
  uint64_t GetHeaderFPGATime() const; //return 48b FPGA timestamp of this frame
};

class MupixRawBank: public TGenericData
#ifdef MUPIX_BANK_ROOT_DERIVED
, public TObject
#endif
{
public:
  MupixRawBank(int bklen, int bktype, const char* name, void *pdata);
  ~MupixRawBank() {}

  //Check on header, size validity
  bool IsGood();

  void DumpRaw();
  void DumpInfo();

  //Header information
  MupixBankInfo GetBankInfo() const; //return raw header data
  uint64_t GetHeaderRaw() const; //return raw header data
  uint16_t GetFPGAID() const; //return ID of FPGA this packet came from
  uint64_t GetHeaderFPGATime() const; //return 48b FPGA timestamp of this frame
  uint32_t GetEventCounter() const;

  //Trailer information
  uint32_t GetTrailerRaw() const; //return raw trailer data

  bool	   GetTrailerASICsatFlag() const; //OR combination of L2 fifo full flags captured in this frame
  bool	   GetTrailerCRCErrorFlag() const; //OR combination of CRC errors captured in this frame

  //Mupix Hits headers
  unsigned int GetPacketCounter();
  unsigned int GetPacketCounterZeroSuppressed();
  unsigned int GetMupixChipCounter();
  unsigned int GetFPGATimeStampOld();

  //Hit data access
  // Get number of timestamps in bank 
  int GetNTimestamps() const;

  // Get actual number of hits in bank 
  // If the list of hits is not built, this is done.
  // The list building is an expensive operation, see below for other ways.
  int GetNHits();

  //Get vector of hits from bank.
  //If the internal vector of hits is not built already this is done.
  //Iterates over raw data and fills the vector -> costly and only recommended
  //if the full list is required in memory anyway.
  //Ownership is kept at this
  std::vector<MupixHit>& GetHits();

  //Iterate over raw data and generate hits as GetHits(), but does not store each (and do not generate an internal vector).
  //If not all hits need to be available at the same time in the following algorithm this should be a bit more performant as no time is spent allocating the vector.
  //Func must return a value >=0 for the loop to continue, otherwise it is stopped.
  //Return value: return value of func in it's last call
  int ForEachHit(std::function<int(MupixHit& hit)> func);

  void CorrectHitsColRow();

  //print something about this instance with switchable depth
  void Print(int level);

private:
  //Iterate over raw data and store hits in m_hits.
  void CollectHits();
  std::vector<MupixHit> m_hits;
};

#endif // TMupixData_h	
