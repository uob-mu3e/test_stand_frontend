#ifndef FEBLIST_H
#define FEBLIST_H

#include <vector>
#include <string>
#include <stdint.h>

#include "mappedfeb.h"
#include "linkstatus.h"

namespace midas{
    class odb;
}

// TODO: Are we multithredaed enought that we need to block all read accesses
// with a mutex during the rebuild?? I guess not, as remapping will always be
// user initated and should not happen during a run


// NOTE: The mappedFEB is a lightweight, read-only object, so we produce copies
// whenever we repopulate the list. This is intentional and saves all the pointer
// indirects that would otherwise be necessary

class FEBList
{
public:
    FEBList(uint16_t SB_index_);
    size_t nFEBS() const {return mFEBs.size();}
    size_t nPixelFEBS() const {return mPixelFEBs.size();}
    size_t nSciFiFEBS() const {return mSciFiFEBs.size();}
    size_t nTileFEBS() const {return mTileFEBs.size();}

    const mappedFEB getFEB(size_t i) const {return mFEBs.at(i);}
    const mappedFEB getPixelFEB(size_t i) const {return mPixelFEBs.at(i);}
    const mappedFEB getSciFiFEB(size_t i) const {return mSciFiFEBs.at(i);}
    const mappedFEB getTileFEB(size_t i) const {return mTileFEBs.at(i);}

    const std::vector<mappedFEB> & getFEBs(){return mpFEBs;}
    const std::vector<mappedFEB> & getPixelFEBs(){return mPixelFEBs;}
    const std::vector<mappedFEB> & getSciFiFEBs(){return mSciFiFEBs;}
    const std::vector<mappedFEB> & getTileFEBs(){return mTileFEBs;}

    uint64_t & getFEBMask(){return mFEBMask;}
    uint64_t & getLinkMask(){return mLinkMask;}
    uint64_t & getPixelFEBMask(){return mPixelFEBMask;}
    uint64_t & getSciFiFEBMask(){return mSciFiFEBMask;}
    uint64_t & getTileFEBMask(){return mTileFEBMask;}

    void RebuildFEBList();

protected:
    std::vector<LinkStatus> linkstats;
    const uint16_t SB_index;
    uint64_t mFEBMask;
    uint64_t mLinkMask;
    uint64_t mPixelFEBMask;
    uint64_t mSciFiFEBMask;
    uint64_t mTileFEBMask;
    std::vector<mappedFEB> mFEBs;
    std::vector<mappedFEB> mpFEBs;
    std::vector<mappedFEB> mPixelFEBs;
    std::vector<mappedFEB> mSciFiFEBs;
    std::vector<mappedFEB> mTileFEBs;
};

#endif // FEBLIST_H
