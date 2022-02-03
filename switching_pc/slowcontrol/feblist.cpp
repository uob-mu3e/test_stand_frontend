#include "feblist.h"

#include "odbxx.h"

using midas::odb;

FEBList::FEBList(uint16_t SB_index_):SB_index(SB_index_){
    RebuildFEBList();
};

void FEBList::RebuildFEBList()
{
    //clear vectors, we will rebuild them now
    mFEBs.clear();
    mPrimaryFEBs.clear();
    mActiveFEBs.clear();
    mPixelFEBs.clear();
    mSciFiFEBs.clear();
    mTileFEBs.clear();

    mFEBMask =  0;
    mLinkMask = 0;

    mPixelFEBMask =0;
    mSciFiFEBMask =0;
    mTileFEBMask =0;

    // get odb instance for links settings
    odb links_settings("/Equipment/Links/Settings");

    //fields to assemble fiber-driven name
    auto febtype = links_settings["FrontEndBoardType"];
    auto linkmask = links_settings["LinkMask"];
    auto sbnames = links_settings["SwitchingBoardNames"];
    auto febnames = links_settings["FrontEndBoardNames"];

    odb febversion("/Equipment/Switching/Variables/FEBFirmware/FEB Version");

    odb febcrate("/Equipment/FEBCrates/Settings/FEBCrate");
    odb febslot("/Equipment/FEBCrates/Settings/FEBSlot");

    // fill our list. Currently only mapping primaries;
    // secondary fibers for SciFi are implicitely mapped to the preceeding primary
    int lastPrimary=-1;
    int nSecondaries=0;
    char reportStr[255];
    for(uint16_t ID=0;ID<MAX_LINKS_PER_SWITCHINGBOARD;ID++){
        uint16_t global_index = ID + SB_index*MAX_LINKS_PER_SWITCHINGBOARD;
        std::string name_link;
        std::string febnamesID;
        sbnames[SB_index].get(name_link);
        febnames[global_index].get(febnamesID);
        name_link+=":";
        name_link+= febnamesID;
        switch ((uint32_t)febtype[global_index]) {
            case FEBTYPE::Unused:
                mFEBs.push_back({linkstats[ID],
                                 global_index, 
                                 linkmask[global_index], 
                                 name_link.c_str(), 
                                 febcrate[global_index], 
                                 febslot[global_index], 
                                 febversion[global_index]});
                mPrimaryFEBs.push_back(mFEBs.back());
                break;
            case FEBTYPE::FibreSecondary:
                mLinkMask   |= 1ULL << ID;
                if(lastPrimary==-1){
                    cm_msg(MERROR,"FEBList::RebuildFEBList","Fiber #%d is set to type secondary but without primary",ID);
                    return;
                }
                sprintf(reportStr,"TX Fiber %d is secondary, remap SC to primary ID %d, Link %u \"%s\" --> SB=%u.%u %s",
                        ID, lastPrimary, mFEBs[lastPrimary].GetLinkID(), mFEBs[lastPrimary].GetLinkName().c_str(),
                        mFEBs[lastPrimary].SB_Number(), mFEBs[lastPrimary].SB_Port(),
                        !mFEBs[lastPrimary].IsScEnabled() ? "\t[SC disabled]" : "");
                cm_msg(MINFO,"MuFEB::RebuildFEBsMap","%s", reportStr);
                lastPrimary=-1;
                nSecondaries++;
                break;
            default:
                mFEBMask    |= 1ULL << ID;
                mLinkMask   |= 1ULL << ID;

                lastPrimary=mFEBs.size();
                mFEBs.push_back({linkstats[ID],
                                global_index,
                                linkmask[global_index],
                                name_link.c_str(),
                                febcrate[global_index], 
                                febslot[global_index], 
                                febversion[global_index]});
                mPrimaryFEBs.push_back(mFEBs.back());
                mActiveFEBs.push_back(mFEBs.back());

                sprintf(reportStr,"TX Fiber %d is mapped to Link %u \"%s\"  --> SB=%u.%u %s",
                        ID,mFEBs[lastPrimary].GetLinkID(),mFEBs[lastPrimary].GetLinkName().c_str(),
                        mFEBs[lastPrimary].SB_Number(),mFEBs[lastPrimary].SB_Port(),
                        !mFEBs[lastPrimary].IsScEnabled()?"\t[SC disabled]":"");

                cm_msg(MINFO,"FEBList::RebuildFEBList","%s",reportStr);

                if(((uint32_t) febtype[global_index]) == FEBTYPE::Pixel){
                    mPixelFEBs.push_back(mFEBs.back());
                    mPixelFEBMask |= 1ULL << ID;
                } else if(((uint32_t) febtype[global_index]) == FEBTYPE::Fibre)    {
                    mSciFiFEBs.push_back(mFEBs.back());
                    mSciFiFEBMask |= 1ULL << ID;
                } else if(((uint32_t) febtype[global_index]) == FEBTYPE::Tile)    {
                    mTileFEBs.push_back(mFEBs.back());
                    mTileFEBMask |= 1ULL << ID;
                } else {
                    cm_msg(MERROR,"FEBList::RebuildFEBList","Invalid FEB Type");
                }
                break;
        }
    }
    sprintf(reportStr,"Found %lu FEBs, remapping %d secondaries.",mFEBs.size(),nSecondaries);
    cm_msg(MINFO,"FEBList::RebuildFEBList","%s", reportStr);

}
