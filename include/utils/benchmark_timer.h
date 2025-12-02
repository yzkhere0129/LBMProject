/**
 * @file benchmark_timer.h
 * @brief High-resolution timing utilities for CUDA benchmarks
 *
 * Provides both CPU wall-clock timing and GPU event-based timing
 * for accurate performance measurement.
 */

#pragma once

#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>

namespace lbm {
namespace utils {

/**
 * @brief CUDA event-based timer for accurate GPU timing
 *
 * Uses cudaEvent for precise measurement of GPU kernel execution time.
 * Accounts for asynchronous execution properly.
 */
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
    }

    /// Returns elapsed time in milliseconds (blocks until GPU is done)
    float elapsedMs() {
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

    /// Returns elapsed time in seconds
    float elapsedSec() {
        return elapsedMs() / 1000.0f;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

/**
 * @brief CPU wall-clock timer using std::chrono
 */
class WallTimer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;

    void start() {
        start_ = Clock::now();
    }

    void stop() {
        stop_ = Clock::now();
    }

    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(stop_ - start_).count();
    }

    double elapsedSec() const {
        return std::chrono::duration<double>(stop_ - start_).count();
    }

private:
    TimePoint start_;
    TimePoint stop_;
};

/**
 * @brief Named timer collection for profiling multiple code sections
 */
class TimerPool {
public:
    struct TimerEntry {
        double total_ms = 0.0;
        int count = 0;
        double min_ms = 1e30;
        double max_ms = 0.0;
    };

    void start(const std::string& name) {
        timers_[name].start();
    }

    void stop(const std::string& name) {
        timers_[name].stop();
        double elapsed = timers_[name].elapsedMs();

        auto& entry = entries_[name];
        entry.total_ms += elapsed;
        entry.count++;
        entry.min_ms = std::min(entry.min_ms, elapsed);
        entry.max_ms = std::max(entry.max_ms, elapsed);
    }

    /// Record a section with RAII
    struct ScopedTimer {
        TimerPool& pool;
        std::string name;

        ScopedTimer(TimerPool& p, const std::string& n) : pool(p), name(n) {
            pool.start(name);
        }
        ~ScopedTimer() {
            pool.stop(name);
        }
    };

    ScopedTimer scoped(const std::string& name) {
        return ScopedTimer(*this, name);
    }

    /// Print timing summary
    void print(std::ostream& os = std::cout) const {
        os << "\n=============== Timing Summary ===============\n";
        os << std::setw(25) << "Section"
           << std::setw(12) << "Total(ms)"
           << std::setw(10) << "Calls"
           << std::setw(12) << "Avg(ms)"
           << std::setw(12) << "Min(ms)"
           << std::setw(12) << "Max(ms)"
           << "\n";
        os << std::string(83, '-') << "\n";

        double total_all = 0.0;
        for (const auto& [name, entry] : entries_) {
            double avg = entry.count > 0 ? entry.total_ms / entry.count : 0.0;
            os << std::setw(25) << name
               << std::setw(12) << std::fixed << std::setprecision(2) << entry.total_ms
               << std::setw(10) << entry.count
               << std::setw(12) << std::fixed << std::setprecision(3) << avg
               << std::setw(12) << std::fixed << std::setprecision(3) << entry.min_ms
               << std::setw(12) << std::fixed << std::setprecision(3) << entry.max_ms
               << "\n";
            total_all += entry.total_ms;
        }

        os << std::string(83, '-') << "\n";
        os << std::setw(25) << "TOTAL"
           << std::setw(12) << std::fixed << std::setprecision(2) << total_all
           << "\n";
        os << "==============================================\n";
    }

    const TimerEntry& get(const std::string& name) const {
        static TimerEntry empty;
        auto it = entries_.find(name);
        return it != entries_.end() ? it->second : empty;
    }

    void reset() {
        entries_.clear();
    }

private:
    std::map<std::string, WallTimer> timers_;
    std::map<std::string, TimerEntry> entries_;
};

/**
 * @brief Benchmark results structure
 */
struct BenchmarkResults {
    // Domain info
    int nx, ny, nz;
    int num_steps;
    size_t total_cells;

    // Performance metrics
    double total_time_sec;
    double kernel_time_sec;
    double io_time_sec;
    double mlups;                  // Million Lattice Updates Per Second
    double bandwidth_gb_s;         // Memory bandwidth GB/s

    // Memory usage
    size_t peak_memory_bytes;
    size_t allocated_fields;

    // Accuracy metrics
    float max_temperature;
    float min_temperature;
    float melt_pool_depth_um;
    float melt_pool_width_um;
    float energy_balance_error_pct;

    // Time history
    std::vector<float> time_history_us;
    std::vector<float> tmax_history;
    std::vector<std::vector<float>> probe_history;

    /// Compute MLUPS from domain and timing
    void computeMLUPS() {
        double compute_time = total_time_sec - io_time_sec;
        if (compute_time > 0) {
            mlups = (static_cast<double>(total_cells) * num_steps) / (compute_time * 1e6);
        }
    }

    /// Print summary
    void print(std::ostream& os = std::cout) const {
        os << "\n=============== BENCHMARK RESULTS ===============\n";
        os << "Configuration:\n";
        os << "  Domain: " << nx << " x " << ny << " x " << nz << " cells\n";
        os << "  Total cells: " << total_cells << "\n";
        os << "  Time steps: " << num_steps << "\n";
        os << "\n";

        os << "Performance:\n";
        os << "  Total wall time: " << std::fixed << std::setprecision(2)
           << total_time_sec << " s\n";
        os << "  Kernel time: " << std::fixed << std::setprecision(2)
           << kernel_time_sec << " s ("
           << (100.0 * kernel_time_sec / total_time_sec) << "%)\n";
        os << "  I/O time: " << std::fixed << std::setprecision(2)
           << io_time_sec << " s (excluded from MLUPS)\n";
        os << "  MLUPS: " << std::fixed << std::setprecision(1) << mlups << "\n";
        os << "  Bandwidth: " << std::fixed << std::setprecision(2)
           << bandwidth_gb_s << " GB/s\n";
        os << "\n";

        os << "Memory:\n";
        os << "  Peak GPU memory: " << std::fixed << std::setprecision(1)
           << (peak_memory_bytes / (1024.0 * 1024.0)) << " MB\n";
        os << "  Allocated fields: " << allocated_fields << "\n";
        os << "\n";

        os << "Accuracy:\n";
        os << "  Temperature range: " << std::fixed << std::setprecision(1)
           << min_temperature << " - " << max_temperature << " K\n";
        os << "  Melt pool depth: " << std::fixed << std::setprecision(1)
           << melt_pool_depth_um << " um\n";
        os << "  Melt pool width: " << std::fixed << std::setprecision(1)
           << melt_pool_width_um << " um\n";
        os << "  Energy balance error: " << std::fixed << std::setprecision(2)
           << energy_balance_error_pct << " %\n";
        os << "=================================================\n";
    }

    /// Write results to CSV
    void writeCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open " << filename << " for writing\n";
            return;
        }

        // Header
        file << "# Benchmark Results\n";
        file << "# Domain: " << nx << "x" << ny << "x" << nz << "\n";
        file << "# Steps: " << num_steps << "\n";
        file << "# MLUPS: " << mlups << "\n";
        file << "#\n";

        // Time series data
        file << "time_us,T_max_K";
        for (size_t i = 0; i < probe_history.size(); i++) {
            file << ",probe" << i;
        }
        file << "\n";

        for (size_t t = 0; t < time_history_us.size(); t++) {
            file << time_history_us[t] << "," << tmax_history[t];
            for (size_t i = 0; i < probe_history.size(); i++) {
                if (t < probe_history[i].size()) {
                    file << "," << probe_history[i][t];
                }
            }
            file << "\n";
        }

        file.close();
    }
};

/**
 * @brief Get current GPU memory usage
 */
inline size_t getGPUMemoryUsage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem - free_mem;
}

/**
 * @brief Get GPU device information
 */
inline void printGPUInfo(std::ostream& os = std::cout) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    os << "\n=============== GPU Information ===============\n";
    os << "Device: " << props.name << "\n";
    os << "Compute Capability: " << props.major << "." << props.minor << "\n";
    os << "Total Memory: " << (props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0))
       << " GB\n";
    os << "Memory Bandwidth: " << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1e6)
       << " GB/s (theoretical)\n";
    os << "SM Count: " << props.multiProcessorCount << "\n";
    os << "Max Threads per Block: " << props.maxThreadsPerBlock << "\n";
    os << "===============================================\n";
}

} // namespace utils
} // namespace lbm
