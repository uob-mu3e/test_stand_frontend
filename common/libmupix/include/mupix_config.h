/********************************************************************\

  Name:         mupix_config.h
  Created by:   Konrad Briggl (Dummy)

  Contents:     Assembly of bitpatterns from ODB

  Created on:   Nov 05 2019

\********************************************************************/

#ifndef MUPIX_CONFIG_H
#define MUPIX_CONFIG_H
#include "midas.h"
#include "mupix_config.h"
#include "mupix_MIDAS_config.h"
namespace mupix{
class MupixConfig{
	public:
	MupixConfig(){}
	void reset(){};
        void Parse_ChipDACs_from_struct(MUPIX_CHIPDACS mt){};

};
class MupixBoardConfig{
	public:
	MupixBoardConfig(){}
	void reset(){};
        void Parse_BoardDACs_from_struct(MUPIX_BOARDDACS mt){};
};
}
#endif //MUPIX_CONFIG_H
