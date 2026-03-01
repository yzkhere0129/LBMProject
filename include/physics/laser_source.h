/**
 * @file laser_source.h
 * @brief Laser heat source model for additive manufacturing simulations
 *
 * Implements Gaussian beam profile with volumetric absorption for
 * laser-material interaction in LPBF/DED processes.
 */

#ifndef LBM_LASER_SOURCE_H
#define LBM_LASER_SOURCE_H

#include <cuda_runtime.h>
#include <cmath>

namespace lbm {
namespace physics {

/**
 * @class LaserSource
 * @brief Models laser heat source with Gaussian intensity distribution
 *
 * Physical model:
 * - Surface intensity: I(r) = (2P)/(πw₀²) * exp(-2r²/w₀²)
 * - Volumetric heat: q(x,y,z) = η * I(r) * β * exp(-β*z)
 *
 * Where:
 * - P: laser power [W]
 * - w₀: beam radius at 1/e² intensity [m]
 * - η: absorptivity (material dependent)
 * - β: extinction coefficient [1/m]
 * - r: radial distance from beam center
 */
class LaserSource {
public:
    // Laser parameters (SI units)
    float power;              ///< Laser power [W]
    float spot_radius;        ///< Beam radius at 1/e² intensity [m]
    float absorptivity;       ///< Material absorptivity [0-1]
    float penetration_depth;  ///< Optical penetration depth [m]

    // Position and motion
    float x0, y0, z0;         ///< Current beam center position [m]
    float vx, vy;             ///< Scan velocity [m/s]

    // Derived parameters
    float beta;               ///< Extinction coefficient [1/m]
    float intensity_factor;   ///< Pre-computed 2P/(πw₀²)
    float gaussian_factor;    ///< Pre-computed -2/w₀²

    /**
     * @brief Construct laser source with physical parameters
     * @param P Laser power [W]
     * @param w0 Beam radius [m]
     * @param eta Absorptivity [0-1]
     * @param delta Penetration depth [m]
     */
    __host__ __device__ LaserSource(float P = 100.0f,
                                    float w0 = 50e-6f,
                                    float eta = 0.35f,
                                    float delta = 10e-6f)
        : power(P), spot_radius(w0), absorptivity(eta), penetration_depth(delta),
          x0(0.0f), y0(0.0f), z0(0.0f), vx(0.0f), vy(0.0f) {
        updateDerivedParameters();
    }

    /**
     * @brief Update pre-computed parameters
     */
    __host__ __device__ void updateDerivedParameters() {
        beta = 1.0f / penetration_depth;
        intensity_factor = 2.0f * power / (M_PI * spot_radius * spot_radius);
        gaussian_factor = -2.0f / (spot_radius * spot_radius);
    }

    /**
     * @brief Set beam center position
     */
    __host__ __device__ void setPosition(float x, float y, float z) {
        x0 = x; y0 = y; z0 = z;
    }

    /**
     * @brief Set scan velocity
     */
    __host__ __device__ void setScanVelocity(float vx_new, float vy_new) {
        vx = vx_new; vy = vy_new;
    }

    /**
     * @brief Compute surface intensity at position (x,y)
     * @return Intensity [W/m²]
     */
    __host__ __device__ float computeIntensity(float x, float y) const {
        float dx = x - x0;
        float dy = y - y0;
        float r2 = dx * dx + dy * dy;

        // Cutoff for numerical stability (10 beam radii)
        if (r2 > 100.0f * spot_radius * spot_radius) {
            return 0.0f;
        }

        return intensity_factor * expf(gaussian_factor * r2);
    }

    /**
     * @brief Compute volumetric heat source at position (x,y,z)
     * @return Heat source [W/m³]
     */
    __host__ __device__ float computeVolumetricHeatSource(float x, float y, float z) const {
        // Surface intensity
        float I = computeIntensity(x, y);

        // Volumetric absorption with Beer-Lambert law
        // Note: z should be depth below surface (positive downward)
        if (z < 0.0f || z > 10.0f * penetration_depth) {
            return 0.0f;
        }

        return absorptivity * I * beta * expf(-beta * z);
    }

    /**
     * @brief Update position based on scan velocity
     * @param dt Time step [s]
     */
    __host__ __device__ void updatePosition(float dt) {
        x0 += vx * dt;
        y0 += vy * dt;
    }

    /**
     * @brief Get total absorbed power
     * @return Absorbed power [W]
     */
    __host__ __device__ float getTotalAbsorbedPower() const {
        return power * absorptivity;
    }

    /**
     * @brief Set laser parameters
     */
    __host__ __device__ void setParameters(float P, float w0, float eta, float delta) {
        power = P;
        spot_radius = w0;
        absorptivity = eta;
        penetration_depth = delta;
        updateDerivedParameters();
    }
};

/**
 * @class ScanPath
 * @brief Abstract base class for laser scan path planning
 */
class ScanPath {
public:
    virtual ~ScanPath() = default;

    /**
     * @brief Get position at time t
     * @param t Time [s]
     * @param[out] x X position [m]
     * @param[out] y Y position [m]
     */
    virtual void getPosition(float t, float& x, float& y) = 0;

    /**
     * @brief Get velocity at time t
     * @param t Time [s]
     * @param[out] vx X velocity [m/s]
     * @param[out] vy Y velocity [m/s]
     */
    virtual void getVelocity(float t, float& vx, float& vy) = 0;
};

/**
 * @class LinearScan
 * @brief Linear scan path with constant velocity
 */
class LinearScan : public ScanPath {
private:
    float x_start, y_start;  ///< Start position [m]
    float x_end, y_end;      ///< End position [m]
    float scan_speed;        ///< Scan speed [m/s]
    float total_time;        ///< Total scan time [s]

public:
    LinearScan(float x0, float y0, float x1, float y1, float v)
        : x_start(x0), y_start(y0), x_end(x1), y_end(y1), scan_speed(v) {
        float dx = x_end - x_start;
        float dy = y_end - y_start;
        float distance = sqrtf(dx * dx + dy * dy);
        total_time = distance / scan_speed;
    }

    void getPosition(float t, float& x, float& y) override {
        if (t <= 0) {
            x = x_start; y = y_start;
        } else if (t >= total_time) {
            x = x_end; y = y_end;
        } else {
            float s = t / total_time;  // Normalized progress [0,1]
            x = x_start + s * (x_end - x_start);
            y = y_start + s * (y_end - y_start);
        }
    }

    void getVelocity(float t, float& vx, float& vy) override {
        if (t > 0 && t < total_time) {
            vx = (x_end - x_start) / total_time;
            vy = (y_end - y_start) / total_time;
        } else {
            vx = 0.0f; vy = 0.0f;
        }
    }
};

/**
 * @class RasterScan
 * @brief Raster (zigzag) scan pattern
 */
class RasterScan : public ScanPath {
private:
    float x_min, y_min;      ///< Scan region min corner [m]
    float x_max, y_max;      ///< Scan region max corner [m]
    float hatch_spacing;     ///< Distance between scan lines [m]
    float scan_speed;        ///< Scan speed [m/s]
    int num_lines;           ///< Number of scan lines

public:
    RasterScan(float xmin, float ymin, float xmax, float ymax,
               float hatch, float v)
        : x_min(xmin), y_min(ymin), x_max(xmax), y_max(ymax),
          hatch_spacing(hatch), scan_speed(v) {
        num_lines = static_cast<int>((y_max - y_min) / hatch_spacing) + 1;
    }

    void getPosition(float t, float& x, float& y) override {
        float line_time = (x_max - x_min) / scan_speed;
        int line_idx = static_cast<int>(t / line_time);
        float t_in_line = fmodf(t, line_time);

        if (line_idx >= num_lines) {
            x = (num_lines % 2 == 0) ? x_min : x_max;
            y = y_max;
            return;
        }

        y = y_min + line_idx * hatch_spacing;

        // Zigzag pattern: even lines go right, odd lines go left
        if (line_idx % 2 == 0) {
            x = x_min + t_in_line * scan_speed;
        } else {
            x = x_max - t_in_line * scan_speed;
        }
    }

    void getVelocity(float t, float& vx, float& vy) override {
        float line_time = (x_max - x_min) / scan_speed;
        int line_idx = static_cast<int>(t / line_time);

        if (line_idx >= num_lines) {
            vx = 0.0f; vy = 0.0f;
            return;
        }

        vy = 0.0f;  // No y-velocity during line scan
        vx = (line_idx % 2 == 0) ? scan_speed : -scan_speed;
    }
};

// CUDA kernel declarations

/**
 * @brief Compute laser heat source distribution on GPU
 * @param heat_source Output heat source array [W/m³]
 * @param laser Laser parameters
 * @param dx,dy,dz Grid spacing [m]
 * @param nx,ny,nz Grid dimensions
 */
__global__ void computeLaserHeatSourceKernel(
    float* heat_source,
    const LaserSource laser,
    float dx, float dy, float dz,
    int nx, int ny, int nz
);

/**
 * @brief Compute total laser energy in domain (for conservation check)
 * @param heat_source Heat source array [W/m³]
 * @param dx,dy,dz Grid spacing [m]
 * @param nx,ny,nz Grid dimensions
 * @return Total power [W]
 */
float computeTotalLaserEnergy(
    const float* heat_source,
    float dx, float dy, float dz,
    int nx, int ny, int nz
);

} // namespace physics
} // namespace lbm

#endif // LBM_LASER_SOURCE_H
