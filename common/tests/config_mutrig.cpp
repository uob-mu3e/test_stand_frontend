#include <stdio.h>
#include <cassert>
#include <iostream>
#include "MutrigConfig.h"


int main(int,char**){
   mutrig::MutrigConfig config;
   config.setParameter("ethresh_5",12);
   std::cout<<config;
   printf("\n");
   return 0;
}
