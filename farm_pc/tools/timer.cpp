
/*
 *  author: Valentin Henkys (fhenkys@students.uni-mainz.de)
 */
#include "./timer.hpp"

#include <iostream>
#include <tuple>

std::map<std::string, Time_Measurement> Timer::timer_map;

void Timer::start(std::string name) {
    auto& measurement = Timer::timer_map[name];
    measurement.start = std::chrono::high_resolution_clock::now();
}

uint64_t Timer::stop(const std::string& name, bool print) {
    auto& measurement = Timer::timer_map[name];
    measurement.end   = std::chrono::high_resolution_clock::now();
    const uint64_t elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(measurement.end - measurement.start)
            .count();
    measurement.total += elapsed;
    measurement.times_taken++;
    if (print)
        std::cout << name << ": " << elapsed << "micro s\n";
    return elapsed;
}

uint64_t Timer::total(const std::string& name) {
    auto& measurement = Timer::timer_map[name];
    return measurement.total;
}

double Timer::average(const std::string& name) {
    auto& measurement = Timer::timer_map[name];
    if (measurement.times_taken > 0)
        return (measurement.total / static_cast<double>(measurement.times_taken));
    return -1;
}
