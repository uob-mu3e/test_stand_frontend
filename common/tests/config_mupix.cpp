#include <stdio.h>
#include <cassert>
#include <iostream>
#include "mupix_config.h"


int main(int,char**){
   mupix::MupixConfig config;
   config.setParameter("unused_10",12,true);
   std::cout<<config;
   printf("\n");
   return 0;
}
