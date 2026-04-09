#include "Timing/Timer.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <vector>

Timer& Timer::get() {
    static Timer instance;
    return instance;
}

void Timer::start(const char* region) {
    auto& e = entries_[region];
    e.t0      = std::chrono::high_resolution_clock::now();
    e.running = true;
}

void Timer::stop(const char* region) {
    auto now = std::chrono::high_resolution_clock::now();
    auto& e  = entries_[region];
    if (!e.running) return;
    std::chrono::duration<double> dur = now - e.t0;
    e.total_sec += dur.count();
    e.calls++;
    e.running = false;
}

void Timer::reset() {
    entries_.clear();
}

void Timer::report(int mpi_rank) const {
    if (mpi_rank != 0) return;
    if (entries_.empty()) { std::cout << "[Timer] No regions recorded.\n"; return; }

    // Collect and sort by total time descending
    using Row = std::pair<std::string, const Entry*>;
    std::vector<Row> rows;
    for (auto& [name, e] : entries_) rows.push_back({name, &e});
    std::sort(rows.begin(), rows.end(),
              [](const Row& a, const Row& b) {
                  return a.second->total_sec > b.second->total_sec;
              });

    // Column widths
    size_t name_w = 6;
    for (auto& [n, _] : rows) name_w = std::max(name_w, n.size());

    std::cout << "\n=== Timing Report ===\n";
    std::cout << std::left  << std::setw(name_w + 2) << "Region"
              << std::right << std::setw(7) << "Calls"
              << std::setw(12) << "Total(ms)"
              << std::setw(12) << "Avg(ms)" << "\n";
    std::cout << std::string(name_w + 33, '-') << "\n";

    for (auto& [name, ep] : rows) {
        double total_ms = ep->total_sec * 1000.0;
        double avg_ms   = (ep->calls > 0) ? total_ms / ep->calls : 0.0;
        std::cout << std::left  << std::setw(name_w + 2) << name
                  << std::right << std::setw(7) << ep->calls
                  << std::fixed << std::setprecision(2)
                  << std::setw(12) << total_ms
                  << std::setw(12) << avg_ms << "\n";
    }
    std::cout << "=====================\n\n";
}

// RAII Scope
Timer::Scope::Scope(const char* name) : name_(name) {
    Timer::get().start(name_);
}
Timer::Scope::~Scope() {
    Timer::get().stop(name_);
}
