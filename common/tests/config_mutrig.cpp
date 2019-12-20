#include <stdio.h>
#include <cassert>
#include <iostream>
#include "mutrig_config.h"


int main(int,char**){
   mutrig::Config config;
   config.setParameter("ethresh_5",12);
   std::cout<<config;
   printf("\n");
   return 0;
}
