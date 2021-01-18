//
// Created by makoeppe on 6/8/20.
//

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <unistd.h>

#include "midas.h"
#include "odbxx.h"

#include <sstream>
#include <fstream>

#include "mfe.h"

using namespace std;


int main ()
{
    cm_connect_experiment(NULL, NULL, "test", NULL);
    midas::odb::set_debug(true);

    midas::odb stream_settings = {
            {"Test_odb_api", {
                                     {"Divider", 1000},     // int
                                     {"Enable", false},     // bool
                             }},
    };
    stream_settings.connect("/Equipment/Test/Settings", true);

    midas::odb datagen("/Equipment/Test/Settings/Test_odb_api");
    std::cout << "Datagenerator Enable is " << datagen["Enable"] << std::endl;
    cm_disconnect_experiment();
}
