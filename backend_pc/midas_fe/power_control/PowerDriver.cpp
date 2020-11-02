#include "PowerDriver.h"

PowerDriver::PowerDriver()
{
	std::cout << "Warning: empty base class instantiated" << std::endl;
}

PowerDriver::PowerDriver(std::string n, EQUIPMENT_INFO* inf)
{
	name=n;
	info=inf;
}
