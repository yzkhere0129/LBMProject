#ifndef THERMAL_DIAGNOSTICS_H
#define THERMAL_DIAGNOSTICS_H

#include <cuda_runtime.h>

/**
 * @brief Diagnostic data structure for thermal transport analysis
 *
 * Purpose: Quantify relative contributions of advection vs diffusion
 * to understand why T_internal >> T_surface
 */
struct ThermalDiagnostics {
    // Accumulated transport metrics (over all hot cells)
    double total_advection_magnitude;   // |u·∇T|
    double total_diffusion_magnitude;   // |∇²T|
    double total_velocity_magnitude;    // |u|
    double total_temperature_gradient;  // |∇T|

    // Statistics
    int hot_cell_count;        // Cells with T > 3000 K
    double max_advection;      // Peak advection rate
    double max_diffusion;      // Peak diffusion rate
    double max_velocity;       // Peak velocity
    double max_temperature;    // Peak temperature

    // Spatial info for peak temperature location
    int peak_temp_x, peak_temp_y, peak_temp_z;

    __host__ __device__ void reset() {
        total_advection_magnitude = 0.0;
        total_diffusion_magnitude = 0.0;
        total_velocity_magnitude = 0.0;
        total_temperature_gradient = 0.0;
        hot_cell_count = 0;
        max_advection = 0.0;
        max_diffusion = 0.0;
        max_velocity = 0.0;
        max_temperature = 0.0;
        peak_temp_x = peak_temp_y = peak_temp_z = -1;
    }

    /**
     * @brief Compute diagnostics ratios
     * @return Advection/Diffusion ratio (>1 = advection-dominant, <1 = diffusion-dominant)
     */
    double getAdvectionDiffusionRatio() const {
        if (total_diffusion_magnitude < 1e-10) return 0.0;
        return total_advection_magnitude / total_diffusion_magnitude;
    }

    /**
     * @brief Get average Peclet number over hot cells
     * Pe = |u·L| / α
     * For L = dx, Pe = |u|·dx / α
     */
    double getAveragePeclet(double dx, double alpha) const {
        if (hot_cell_count == 0) return 0.0;
        double avg_velocity = total_velocity_magnitude / hot_cell_count;
        return (avg_velocity * dx) / alpha;
    }

    void print(double dx, double alpha) const {
        printf("\n");
        printf("=== Thermal Transport Diagnostics ===\n");
        printf("Hot cells (T>3000K): %d\n", hot_cell_count);
        printf("Peak temperature: %.1f K at (%d,%d,%d)\n",
               max_temperature, peak_temp_x, peak_temp_y, peak_temp_z);
        printf("\n");
        printf("Transport magnitudes (accumulated):\n");
        printf("  Advection |u·∇T|:  %.3e K/s\n", total_advection_magnitude);
        printf("  Diffusion |α·∇²T|: %.3e K/s\n", total_diffusion_magnitude);
        printf("  Velocity |u|:      %.3e m/s\n", total_velocity_magnitude / hot_cell_count);
        printf("  Grad |∇T|:         %.3e K/m\n", total_temperature_gradient / hot_cell_count);
        printf("\n");
        printf("Dimensionless numbers:\n");
        printf("  Advection/Diffusion ratio: %.3f ", getAdvectionDiffusionRatio());
        if (getAdvectionDiffusionRatio() > 10.0) {
            printf("(ADVECTION-DOMINANT)\n");
        } else if (getAdvectionDiffusionRatio() > 0.1) {
            printf("(MIXED)\n");
        } else {
            printf("(DIFFUSION-DOMINANT - PROBLEM!)\n");
        }
        printf("  Average Peclet number: %.3f\n", getAveragePeclet(dx, alpha));
        printf("\n");
        printf("Peak values:\n");
        printf("  Max advection: %.3e K/s\n", max_advection);
        printf("  Max diffusion: %.3e K/s\n", max_diffusion);
        printf("  Max velocity:  %.3e m/s\n", max_velocity);
        printf("======================================\n");
    }
};

#endif // THERMAL_DIAGNOSTICS_H
