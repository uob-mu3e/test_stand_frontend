
/*
 *  author: Valentin Henkys (fhenkys@students.uni-mainz.de)
 */
#pragma once

#include <chrono>
#include <map>
#include <string>

class Time_Measurement {
  public:
    uint64_t                                                    total = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::chrono::time_point<std::chrono::high_resolution_clock> end;
    uint                                                        times_taken = 0;
};

/**
 * Time measurement helper class.
 * Provides common functionality to measure time using the c++ high precision clock.
 */
class Timer {
  public:
    static void     start(std::string name);
    static uint64_t stop(const std::string& name, bool print = false);

    static uint64_t total(const std::string& name);
    static double   average(const std::string& name);

    static void clear() { Timer::timer_map.clear(); };

    static std::map<std::string, Time_Measurement> timer_map;
};
