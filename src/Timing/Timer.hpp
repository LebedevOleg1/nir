#pragma once
#include <string>
#include <unordered_map>
#include <chrono>

// ============================================================================
// Timer — named timing regions with call count and cumulative time.
//
// Usage:
//   Timer::get().start("euler_flux");
//   step_gpu();
//   Timer::get().stop("euler_flux");
//   Timer::get().report();
//
// RAII scope:
//   { Timer::Scope t("halo_exchange"); do_halo_exchange(); }
//
// On GPU: call cudaDeviceSynchronize() before stop() to get accurate times.
// For deep GPU profiling use nsight / nvprof instead.
// ============================================================================
class Timer {
public:
    static Timer& get();  // Global singleton

    void start(const char* region);
    void stop(const char* region);
    void reset();

    // Print ASCII table: region | calls | total_ms | avg_ms
    void report(int mpi_rank = 0) const;

    // RAII helper: starts on construction, stops on destruction
    struct Scope {
        explicit Scope(const char* name);
        ~Scope();
    private:
        const char* name_;
    };

private:
    struct Entry {
        int    calls        = 0;
        double total_sec    = 0.0;
        std::chrono::high_resolution_clock::time_point t0;
        bool   running      = false;
    };

    std::unordered_map<std::string, Entry> entries_;

    Timer() = default;
};
