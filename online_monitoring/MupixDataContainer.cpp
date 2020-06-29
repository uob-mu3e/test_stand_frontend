#include "MupixDataContainer.h"

#include <iostream>
#include <algorithm>

#define BANKHEADER_LEN 2
#define PACKETHEADER_LEN 5
#define TRAILER_LEN 1


MupixDataContainer::MupixDataContainer():
	m_nEvents(1)
	{}


//Header information (Getters throw when out of range)

//return number of bank headers stored
unsigned MupixDataContainer::GetNbanks() const {return m_bankInfo.size();}
unsigned MupixDataContainer::GetNEvents() const {return m_nEvents;}
unsigned MupixDataContainer::GetNHits() const{return m_hits.size();}

//return ID of FPGA this packet came from
uint16_t MupixBankInfo::GetHeaderFPGAID() const{
	return (at(0) >> 8) & 0xffff;
}

//return 48b FPGA timestamp of this frame
uint64_t MupixBankInfo::GetHeaderFPGATime() const{
    uint64_t raw= (static_cast<uint64_t>(at(1)) << 32) | at(2);
    return raw>>16;
}

std::vector<MupixHit>& MupixDataContainer::GetHits(){
	return m_hits;
}

int MupixDataContainer::Append(MupixRawBank* bnk){
	m_bankInfo.push_back(bnk->GetBankInfo());
	size_t bnkID=m_bankInfo.size();
	int i=0;
	/*
	std::vector<MupixHit> newhits;
	newhits.reserve(bnk->GetNTimestamps());
	bnk->ForEachHit([this,bnkID,&i,&newhits](MupixHit& hit){
		hit.SetBankID(bnkID);
		newhits.push_back(hit);
		i++;
		return 0;
	});
	*/
	m_hits.reserve(m_hits.size()+bnk->GetNHits());
	bnk->ForEachHit([this,bnkID,&i](MupixHit& hit){
		hit.SetBankID(bnkID);
        hit.TransformColRow();
		m_hits.push_back(hit);
		i++;
		return 0;
	});
	return i;
}

int MupixDataContainer::AppendBank(MupixRawBank* bnk){
	m_bankInfo.push_back(bnk->GetBankInfo());
	size_t bnkID=m_bankInfo.size();
	int i=0;
	//std::vector<MutrigHit> newhits;
	//newhits.reserve(bnk->GetNTimestamps());
	m_newbank_hits.clear();
	m_newbank_hits.reserve(bnk->GetNTimestamps());
	return 0;
}

int MupixDataContainer::AppendHit(MupixHit& hit){
	hit.SetBankID(GetNbanks()-1);
	//newhits.push_back(hit);
	m_newbank_hits.push_back(hit);
	return 1;
}

void MupixDataContainer::InsertBank(){
	std::sort(m_newbank_hits.begin(),m_newbank_hits.end(), MupixDataContainer::compare_ts);
	std::vector<MupixHit> tmp;
	std::merge(m_hits.begin(),m_hits.end(),m_newbank_hits.begin(),m_newbank_hits.end(),std::back_inserter(tmp),MupixDataContainer::compare_ts);
	std::swap(m_hits,tmp);
	m_newbank_hits.clear();
	return;
}

void MupixDataContainer::Clear(){
	m_bankInfo.clear();
	m_hits.clear();
}

void MupixDataContainer::DumpRaw(){
	for(int i=0;i<m_bankInfo.size();i++)
		DumpBankInfo(i);
	printf("Hits: \n");
	for(int i=0;i<GetNHits();i++){
		printf("#%d BK%d E%u C%u: data %8.8x\n",i,m_hits[i].GetBankID(),m_hits[i].GetEventID(),m_hits[i].GetClusterID(),m_hits[i].GetRawData());
	}
}

void MupixDataContainer::DumpBankInfo(int bID){
        printf("Bank ID%d     -----------\n",bID);
        printf("FPGA:               %8.8x\n",this->GetBankInfo(bID).GetHeaderFPGAID());
        printf("Header FPGATime:    %8.8lx\n",this->GetBankInfo(bID).GetHeaderFPGATime());
        //printf("Hits: %d\n",this->GetNHits());
}

void MupixDataContainer::ForEachHit(cluster_bnd_t range, std::function<int(MupixHit& hit)> func){
        if(range.first > range.second){
      	  for(auto it=range.first;it!=m_hits.end();++it){
      		  if(func(*it)<0) return;
      	  }
      	  range.first=m_hits.begin();
        }
        for(auto it=range.first;it!=range.second;++it){
      	  if(func(*it)<0) return;
        }
};

//void MupixDataContainer::ForEachCluster(cluster_bnd_t range, std::function<int(cluster_bnd_t range)> func){}; 

//void MupixDataContainer::ForEachEvent(std::function<int(cluster_bnd_t range)> func){}; 

//MupixDataContainer::cluster_bnd_t MupixDataContainer::GetEvent(int eID){}

//MupixDataContainer::cluster_bnd_t MupixDataContainer::GetCluster(cluster_bnd_t event, int cID){} TODO

//int MupixDataContainer::DoTSegmentation(int maxGapT){}

//int MupixDataContainer::DoZXYSegmentation(int maxGapY){}

//Position mapping for hit. Implemented in MupixDataContainer because this includes the bank header information required to determine where we are globally (FEBID)
int MupixDataContainer::GetHitX(MupixHit& hit){
	return hit.GetCol();
}
int MupixDataContainer::GetHitY(MupixHit& hit){
	return hit.GetRow();
}
int MupixDataContainer::GetHitZ(MupixHit& hit){
	return hit.GetChipID();
}
int MupixDataContainer::GetTimeStamp(MupixHit& hit){
    return hit.GetTimeStamp();
}

void MupixDataContainer::GetClusterXYZ(cluster_bnd_t range, float& x, float& y, float& z){
        //TODO: implement hash map: if(m_map(find)){... return}; 
        x=0;y=0;z=0;
        for(auto it=range.first;it!=range.second;++it){
      	  x+=GetHitX(*it);
      	  y+=GetHitY(*it);
      	  z+=GetHitZ(*it);
        }
        x/=GetSize(range);
        y/=GetSize(range);
        z/=GetSize(range);
}
// float MupixDataContainer::GetMeanTime(cluster_bnd_t range){
        // //TODO: implement hash map: if(m_map(find)){... return}; 
        // float t=0;
        // for(auto it=range.first;it!=range.second;++it){
      	//   t+=it->GetTime();
        // }
        // t/=GetSize(range);
        // return t;
// }
// float MupixDataContainer::GetEsum(cluster_bnd_t range){
//         //TODO: implement hash map: if(m_map(find)){... return}; 
//         float e=0;
//         for(auto it=range.first;it!=range.second;++it){
//       	  e+=it->GetEnergy();
//         }
//         e/=GetSize(range);
//         return e;
// }

bool MupixDataContainer::compare_ts(const MupixHit& lhs, const MupixHit& rhs){
	return lhs.GetTimeStamp() < rhs.GetTimeStamp();
}