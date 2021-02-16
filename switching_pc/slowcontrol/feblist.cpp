#include "feblist.h"

#include "odbxx.h"

using midas::odb;

void FEBList::RebuildFEBList()
{
    //clear vectors, we will rebuild them now
    mFEBs.clear();
    mpFEBs.clear();
    mPixelFEBs.clear();
    mSciFiFEBs.clear();
    mTileFEBs.clear();

    // get odb instance for links settings
    odb links_settings("/Equipment/Links/Settings");

    //fields to assemble fiber-driven name
    auto febtype = links_settings["FrontEndBoardType"];
    auto linkmask = links_settings["LinkMask"];
    auto sbnames = links_settings["SwitchingBoardNames"];
    auto febnames = links_settings["FrontEndBoardNames"];

    // fill our list. Currently only mapping primaries;
    // secondary fibers for SciFi are implicitely mapped to the preceeding primary
    int lastPrimary=-1;
    int nSecondaries=0;
    char reportStr[255];
    for(uint16_t ID=0;ID<MAX_N_FRONTENDBOARDS;ID++){
        std::string name_link;
        std::string febnamesID;
        sbnames[ID/MAX_LINKS_PER_SWITCHINGBOARD].get(name_link);
        febnames[ID].get(febnamesID);
        name_link+=":";
        name_link+= febnamesID;
        if((uint32_t) febtype[ID] != FEBTYPE::Undefined  && (uint32_t) febtype[ID] != FEBTYPE::FibreSecondary){
            lastPrimary=mFEBs.size();
            mFEBs.push_back({ID,linkmask[ID],name_link.c_str()});
            mpFEBs.push_back(mFEBs.back());
            printf(reportStr,"TX Fiber %d is mapped to Link %u \"%s\"                            --> SB=%u.%u %s",
                                ID,mFEBs[lastPrimary].GetLinkID(),mFEBs[lastPrimary].GetLinkName().c_str(),
                                mFEBs[lastPrimary].SB_Number(),mFEBs[lastPrimary].SB_Port(),
                                !mFEBs[lastPrimary].IsScEnabled()?"\t[SC disabled]":"");
            cm_msg(MINFO,"FEBList::RebuildFEBList","%s",reportStr);
            if(((uint32_t) febtype[ID]) == FEBTYPE::Pixel){
                mPixelFEBs.push_back(mFEBs.back());
            } else if(((uint32_t) febtype[ID]) == FEBTYPE::Fibre)    {
                mSciFiFEBs.push_back(mFEBs.back());
            } else if(((uint32_t) febtype[ID]) == FEBTYPE::Tile)    {
                mTileFEBs.push_back(mFEBs.back());
            } else {
                cm_msg(MERROR,"FEBList::RebuildFEBList","Invalid FEB Type");
            }
        } else if((uint32_t) febtype[ID] == FEBTYPE::FibreSecondary) {
            if(lastPrimary==-1){
                cm_msg(MERROR,"FEBList::RebuildFEBList","Fiber #%d is set to type secondary but without primary",ID);
                return;
            }
            sprintf(reportStr,"TX Fiber %d is secondary, remap SC to primary ID %d, Link %u \"%s\" --> SB=%u.%u %s",
                    ID,lastPrimary,mFEBs[lastPrimary].GetLinkID(),mFEBs[lastPrimary].GetLinkName().c_str(),
                    mFEBs[lastPrimary].SB_Number(),mFEBs[lastPrimary].SB_Port(),
                    !mFEBs[lastPrimary].IsScEnabled()?"\t[SC disabled]":"");
            cm_msg(MINFO,"MuFEB::RebuildFEBsMap","%s", reportStr);
            lastPrimary=-1;
            nSecondaries++;
        }
    }
    sprintf(reportStr,"Found %lu FEBs, remapping %d secondaries.",mFEBs.size(),nSecondaries);
    cm_msg(MINFO,"FEBList::RebuildFEBList","%s", reportStr);

}
