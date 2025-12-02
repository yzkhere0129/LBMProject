/**
 * @file surface_initialization.h
 * @brief 表面初始化工具 - 用于生成不规则表面和粉末床
 *
 * 功能:
 * - 随机粗糙表面生成
 * - 粉末床层模拟
 * - 提升LPBF仿真真实感
 */

#ifndef LBM_UTILS_SURFACE_INITIALIZATION_H
#define LBM_UTILS_SURFACE_INITIALIZATION_H

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

namespace lbm {
namespace utils {

/**
 * @brief 表面类型枚举
 */
enum class SurfaceType {
    FLAT,        // 完美平面
    ROUGH,       // 随机粗糙表面
    POWDER_BED,  // 粉末床层
    SINUSOIDAL   // 正弦波表面（测试用）
};

/**
 * @brief 粗糙表面参数
 */
struct RoughSurfaceParams {
    float amplitude = 5.0e-6f;      // 粗糙度幅值（米）
    float wavelength = 20.0e-6f;    // 特征波长（米）
    float randomness = 0.3f;        // 随机性强度（0-1）
    unsigned int seed = 42;         // 随机种子
};

/**
 * @brief 粉末床参数
 */
struct PowderBedParams {
    float particle_size = 15.0e-6f;      // 粉末粒径（米）
    float size_std_dev = 3.0e-6f;        // 粒径标准差（米）
    float packing_density = 0.6f;        // 堆积密度（0-1）
    float surface_height = 0.9f;         // 表面高度（相对域高度）
    float layer_thickness = 30.0e-6f;    // 粉末层厚度（米）
    unsigned int seed = 42;
};

/**
 * @brief 生成表面高度场（主机端）
 *
 * @param surface_heights 输出: 每个(x,y)位置的表面高度 [nx*ny]
 * @param nx, ny 网格尺寸
 * @param Lx, Ly, Lz 域物理尺寸（米）
 * @param type 表面类型
 * @param rough_params 粗糙表面参数
 * @param powder_params 粉末床参数
 */
void generateSurfaceHeights(
    float* surface_heights,
    int nx, int ny,
    double Lx, double Ly, double Lz,
    SurfaceType type,
    const RoughSurfaceParams& rough_params = RoughSurfaceParams(),
    const PowderBedParams& powder_params = PowderBedParams()
);

/**
 * @brief 根据表面高度场初始化温度/液相分数
 *
 * 设置策略:
 * - 表面以下: 固态（温度=T_init, 液相分数=0）
 * - 表面以上: 空气/真空（不计算或温度=T_init）
 *
 * @param d_temperature 设备端温度场 [nx*ny*nz]
 * @param d_liquid_fraction 设备端液相分数 [nx*ny*nz]
 * @param surface_heights 表面高度场 [nx*ny]
 * @param T_init 初始温度（K）
 * @param nx, ny, nz 网格尺寸
 * @param dx, dy, dz 网格间距（米）
 */
__global__
void initializeFieldsFromSurface(
    float* d_temperature,
    float* d_liquid_fraction,
    const float* d_surface_heights,
    float T_init,
    int nx, int ny, int nz,
    float dx, float dy, float dz
);

/**
 * @brief CUDA核函数：生成随机粗糙表面
 */
__global__
void generateRoughSurfaceKernel(
    float* d_surface_heights,
    int nx, int ny,
    float dx, float dy,
    float base_height,
    RoughSurfaceParams params
);

/**
 * @brief CUDA核函数：生成粉末床表面
 */
__global__
void generatePowderBedKernel(
    float* d_surface_heights,
    int nx, int ny,
    float dx, float dy,
    float Lz,
    PowderBedParams params
);

/**
 * @brief 辅助函数：简单随机数生成（设备端）
 */
__device__ inline
float deviceRandom(unsigned int seed, int x, int y) {
    // 简单的伪随机数（基于位运算）
    unsigned int hash = seed;
    hash = hash * 1103515245 + 12345;
    hash ^= (x * 2654435761U);
    hash = hash * 1103515245 + 12345;
    hash ^= (y * 2654435761U);
    hash = hash * 1103515245 + 12345;
    return (float)(hash & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

/**
 * @brief 辅助函数：2D Perlin噪声（简化版）
 */
__device__ inline
float perlinNoise2D(float x, float y, unsigned int seed) {
    // 简化的Perlin噪声
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);

    float fx = x - x0;
    float fy = y - y0;

    // 双线性插值
    float v00 = deviceRandom(seed, x0, y0);
    float v10 = deviceRandom(seed, x0+1, y0);
    float v01 = deviceRandom(seed, x0, y0+1);
    float v11 = deviceRandom(seed, x0+1, y0+1);

    // Smoothstep插值
    float u = fx * fx * (3.0f - 2.0f * fx);
    float v = fy * fy * (3.0f - 2.0f * fy);

    float v0 = v00 * (1-u) + v10 * u;
    float v1 = v01 * (1-u) + v11 * u;

    return v0 * (1-v) + v1 * v;
}

}} // namespace lbm::utils

#endif // LBM_UTILS_SURFACE_INITIALIZATION_H
