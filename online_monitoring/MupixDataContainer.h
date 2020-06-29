//File: 	MupixDataContainer.h
//Desc: 	Class containing hits from different banks, bank header information is held in this class.
//		Provides access to the header and hits
//Author(s):	K. Briggl


#ifndef _MupixDATACONTAINER_H__
#define _MupixDATACONTAINER_H__


#include "MupixRawBank.h"
#include "MupixHit.h"
#include <array>
#include <vector>
#include <functional>

#ifdef Mupix_DATACONTAINER_ROOT_DERIVED
#include "TObject.h"
#endif

class MupixDataContainer
#ifdef Mupix_CONTAINER_ROOT_DERIVED
: public TObject
#endif
{
public:
  MupixDataContainer();
  ~MupixDataContainer() {}

  int Append(MupixRawBank* bnk);
  int AppendBank(MupixRawBank* bnk);
  int AppendHit(MupixHit& hit);
  void InsertBank();  
  void Clear();


  void DumpRaw();
  void DumpBankInfo(int bID);

  //Header information (Getters throw when out of range)
  unsigned GetNbanks() const; //return number of bank headers stored
  MupixBankInfo& GetBankInfo(unsigned bNo){return m_bankInfo[bNo];};
  std::vector<MupixBankInfo> GetBankInfos(){return m_bankInfo;};

  // Get number of hits in container (all banks)
  unsigned GetNHits() const;
  //Get number of Events in cluster after T-segmentation
  unsigned GetNEvents() const;
  //TODO: Add MupixDataContainer CollectHits(functor)

  //Get vector of hits from bank (ownership kept at this)
  std::vector<MupixHit>& GetHits();
  virtual int GetHitX(MupixHit& hit);
  virtual int GetHitY(MupixHit& hit);
  virtual int GetHitZ(MupixHit& hit);
  virtual int GetTimeStamp(MupixHit& hit);


  typedef std::pair<std::vector<MupixHit>::iterator, std::vector<MupixHit>::iterator> cluster_bnd_t;
//  cluster_bnd_t GetCluster(int cID){foreachhit - when cID==hit->cID start=it; when cID!=hit->cID stop=it; break;};
//
// int DoTSegmentation(int maxGapT);
//	partition by time, m_hits is supposed to be sorted already
// int DoZXYSegmentation(int maxGapY);
 /*
	sort by zxy
	partition by layer
	partition by x
	partition by y
  }*/
  //TODO: DumpObject(level) - for events/clusters

  cluster_bnd_t GetEvent(int eID);
  //cluster_bnd_t GetCluster(cluster_bnd_t event, int cID); TODO

  void GetClusterXYZ(cluster_bnd_t range, float& x, float& y, float& z);
  int GetSize(cluster_bnd_t range){return range.second - range.first;};
  float GetMeanTime(cluster_bnd_t range);
  float GetEsum(cluster_bnd_t range);

  void ForEachHit(cluster_bnd_t range, std::function<int(MupixHit& hit)> func);
  void ForEachHit(std::function<int(MupixHit& hit)> func){ForEachHit(cluster_bnd_t(m_hits.begin(),m_hits.end()),func);};
  // void ForEachCluster(cluster_bnd_t event,std::function<int(cluster_bnd_t range)> func);
  // void ForEachEvent(std::function<int(cluster_bnd_t range)> func);

private:
  std::vector<MupixHit> m_newbank_hits;
  std::vector<MupixHit> m_hits;
  std::vector<MupixBankInfo> m_bankInfo; //contains header and trailer information
  //after clustering
  unsigned m_nEvents;
protected:
static bool compare_ts(const MupixHit& lhs, const MupixHit& rhs);
};

#endif // MupixDataContainer_h	
