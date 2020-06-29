// Online analysis of SciFi data, starting with basic data quality, continuing with clustering in t and x, tracking
// July 2019, Konrad Briggl <konrad.briggl@unige.ch>
// Juni 2020, Marius Koeppel <mkoeppel@uni-mainz.de>

#include <stdio.h>
#include <iostream>
#include <time.h>

#include "TRootanaEventLoop.hxx"

#include "MupixRawBank.h"
#include "MupixDataContainer.h"

#include "TRawEventHistograms.h"
#include "TMupixDQHistogram.h"
#include "CorHistograms.h"
#include "CorHistogramsT.h"

#include "ftplib.h"
#include "midas.h"
#include "mdsupport.h"

class Analyzer: public TRootanaEventLoop {
uint32_t last_Serial;
MupixDataContainer m_bank_data_pixel;
std::vector<std::string> m_banks_pixel;

public:
  bool initialized;
  int m_nhits;
  int m_events_skip;
  int nDisplay;

  //DQM histogram managers. Filled only on request
  TMupixDQHistogramManager* m_hitHistograms_pixel;

  SpatCorHistos_PP* m_CorHisto_PP;
  
  TGraph g_graph_pixel;
  
  float m_crcerrors;
  
  Analyzer() {
    UseBatchMode();
  };

  virtual ~Analyzer() {};

  void Initialize() {
#ifdef HAVE_THTTP_SERVER
    std::cout << "Using THttpServer in read/write mode" << std::endl;
    SetTHttpServerReadWrite();
    if(GetTHttpServer())
        GetTHttpServer()->SetCors("*");
#endif
  }

  void InitManager(){
    printf("=======================Init manager============================\n");
    if(!m_banks_pixel.empty()){
       if(m_hitHistograms_pixel) m_hitHistograms_pixel->CreateHistograms();
    }

    if(m_CorHisto_PP) m_CorHisto_PP->CreateHistograms();

    initialized=true;
    m_nhits=0;
    printf("**************[ana] init manager finished\n");
  }
  
  
  void BeginRun(int transition,int run,int time){
    InitManager();
  }
  void EndRun(int transition,int run,int time){
    g_graph_pixel.SetName("nhits_pixel");
    g_graph_pixel.Write();
  }

 bool ProcessMidasEvent(TDataContainer& dataContainer){
   bool done=true;
   bool good=true;
   bool badblock=false;

   if(!initialized) InitManager();
   if(--m_events_skip >= 0 ) return false;
   printf("------- Event %d\n",dataContainer.GetMidasEvent().GetSerialNumber());
  
   // Collect mupix banks
   for(auto b: m_banks_pixel){
      MupixRawBank *data = dataContainer.GetEventData<MupixRawBank>(b.c_str());
      if(!data) continue;
      if(!data->IsGood()){
	      //Bad data, throw the full bank block later
	      badblock=true;
	      break;
      }
      //printf("-- TBank %s\n",b.c_str());
      //data->DumpInfo();
      //data->DumpRaw();
      //TODO: implement append hit by hit.
      //TOOD: implement noisy pixel masking (in add_pixel loop, using some json file)
      //time calibration, constant and PRBS shifting, store in DNL map
      
      m_bank_data_pixel.AppendBank(data);
      int tmp_m_hits = m_nhits;
      data->ForEachHit([this](MupixHit& hit)->int{
         hit.TransformColRow();
         if (hit.CheckMaskPix() != 0) return 0;
         if(m_hitHistograms_pixel) {
             m_hitHistograms_pixel->UpdateHistograms(hit);
         }
         m_bank_data_pixel.AppendHit(hit);
         m_nhits++;
         return 0;
      });
      // only masked data, throw full bank block
      // TODO: this is only removing the data if the full bank is noise. Change to mask all hits from noisy pixels, requires TODOs above to be implemented
      if (tmp_m_hits == m_nhits) continue;
      m_bank_data_pixel.InsertBank();
      // m_bank_data_pixel.Append(data);
      //printf("---\n");
   }
       //check if we have a missing subdetector, throw awa
       if(m_bank_data_pixel.GetNbanks() > 200) good=false;

       if(!good || badblock){//TODO can we implement a function to enable/disable the requirement that we MUST have data from all subdetector; because it will be convinent to disable this when we only want to tunning a single sub-detector. We cannot be so lucky that we plug everything and have data from all sub-detector
	    printf("-> Dropping data...\n");
    	m_bank_data_pixel.Clear();
	    return 0;
    }

    if(badblock) m_events_skip=5; //give some grace time to recover, probably something was wrong in the readout 

    //check if we have enough data (if not, return and wait for next event)
    //only look at the pixel bank size if we are running pixel-only.
    //TODO: extend this using FPGA timestamp for pixel to check if more pixel banks are required.
    //Not really relevant for DESY testbeam
    if(!m_banks_pixel.empty() && m_bank_data_pixel.GetNbanks() < 10) return 0;

    g_graph_pixel.SetPoint(g_graph_pixel.GetN(),dataContainer.GetMidasEvent().GetSerialNumber(),m_bank_data_pixel.GetNHits());

   try{
    //generate correlation plots between subdetectors
	
    // mupix mupix cor hit
    m_bank_data_pixel.ForEachHit([this](MupixHit& hitPix)->int{
        m_bank_data_pixel.ForEachHit([&](MupixHit& hitPix2)->int{
            m_CorHisto_PP->UpdateHistograms(&m_bank_data_pixel, &m_bank_data_pixel, hitPix, hitPix2);
              return 0;
        });
        return 0;
    });
 
    }catch(...){
        printf("Exception caught during histogram filling\n");
    } 
    
    m_bank_data_pixel.Clear(); 

    //TODO: pixel banks

 return true;
 }

 void Usage() {
    printf("\t--pbank=name		: Try to find this bank in the event for pixels. Multiple occurences possible. Possible enums for type: mutrig,mupix\n");
    printf("\t--skip=%d			: Skip a certain number of events before starting the analysis\n");
    printf("\t--genDQhist		: Produce data quality histograms (such as hitmaps, time distribution plots, etc.) for all subdetectors. Slows down analysis by ~ x2-x3\n");
 }

 virtual bool CheckOption(std::string option){

//--------------------------------
	char c[64];
	int i;

    if(sscanf(option.c_str(),"--pbank=%s",c)==1){
		if(strlen(c)!=4){
			printf("Size mismatch on bank name parameter %s\n",c);
			return false;
		}
		m_banks_pixel.emplace_back(c);
	}
	
    if(sscanf(option.c_str(),"--skip=%d",&i)==1){
		m_events_skip=i;
	}
	
    if(option == std::string("--genDQhist")){
		m_hitHistograms_pixel=new TMupixDQHistogramManager("Pixels",128);
	}
	return true;
      }

}; 

int main(int argc, char *argv[])
{

  Analyzer::CreateSingleton<Analyzer>();
  Analyzer::Get().SetOutputFilename("ana_output");
  
  printf("******************[ana:tile] SetDQManagerSciFi... \n");
//  static_cast<Analyzer&>(Analyzer::Get()).SetRawEventManagerSciFi(new TRawEventHistogramManager({{0,0}}));
//  static_cast<Analyzer&>(Analyzer::Get()).SetClusteredEventManagerSciFi(new TClusteredEventHistogramManager(1));
  printf("******************[ana:tile] SetDQManagerTile... \n");
//  static_cast<Analyzer&>(Analyzer::Get()).SetRawEventManagerTile(new TRawEventHistogramManager({{0,0}}));
//  static_cast<Analyzer&>(Analyzer::Get()).SetClusteredEventManagerTile(&ClusteredEventHistMan); --- TODO: implement algorithm for clustering
  printf("******************[ana:pixel] SetDQManagerPixel... \n");
//  static_cast<Analyzer&>(Analyzer::Get()).SetRawEventManagerTile(new TRawEventHistogramManager({{0,0}}));
//  static_cast<Analyzer&>(Analyzer::Get()).SetClusteredEventManagerTile(&ClusteredEventHistMan); --- TODO: implement algorithm for clustering

printf("=======================Execute Loop============================\n");
  return Analyzer::Get().ExecuteLoop(argc, argv);

}


