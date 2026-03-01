/**
 * @file simulation_config.h
 * @brief 仿真配置数据结构 - 支持从配置文件加载参数
 *
 * @deprecated This configuration system is NOT connected to MultiphysicsSolver.
 * Use MultiphysicsConfig in multiphysics_solver.h instead.
 * This file is retained only for apps/run_simulation.cu compatibility.
 *
 * 设计目标:
 * - 统一管理所有仿真参数
 * - 支持YAML/JSON配置文件
 * - 参数验证和默认值
 * - 便于序列化和复现
 */

#ifndef LBM_CONFIG_SIMULATION_CONFIG_H
#define LBM_CONFIG_SIMULATION_CONFIG_H

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <stdexcept>

namespace lbm {
namespace config {

/**
 * @brief 域配置
 */
struct DomainConfig {
    int nx = 80;
    int ny = 80;
    int nz = 40;

    double Lx = 160e-6;  // 米
    double Ly = 160e-6;
    double Lz = 80e-6;

    double dx() const { return Lx / nx; }
    double dy() const { return Ly / ny; }
    double dz() const { return Lz / nz; }
};

/**
 * @brief 时间配置
 */
struct TimeConfig {
    double dt = 5e-10;        // 秒
    int n_steps = 8000;
    int output_interval = 100;
    int print_interval = 100;
};

/**
 * @brief 材料配置
 */
struct MaterialConfig {
    std::string type = "Ti6Al4V";  // "Ti6Al4V", "SS316L", "Inconel718", "Custom"

    // 自定义材料属性（当type="Custom"时使用）
    double density = 4420.0;           // kg/m³
    double cp_solid = 670.0;           // J/(kg·K)
    double cp_liquid = 831.0;
    double k_solid = 7.0;              // W/(m·K)
    double k_liquid = 33.0;
    double T_solidus = 1878.0;         // K
    double T_liquidus = 1923.0;
    double latent_heat = 365000.0;     // J/kg
    double viscosity = 5e-7;           // m²/s
    double beta_thermal = 9e-6;        // K⁻¹
};

/**
 * @brief 激光配置
 */
struct LaserConfig {
    bool enabled = true;

    double power = 1200.0;             // W
    double spot_radius = 50e-6;        // 米
    double absorption = 0.35;          // 吸收率
    double penetration_depth = 15e-6;  // 穿透深度

    // 位置 (x, y, z)
    std::array<double, 3> position = {80e-6, 80e-6, 0.0};

    // 时间控制（激光脉冲功能）
    double turn_on_time = 0.0;         // 秒，激光开启时间
    double turn_off_time = -1.0;       // 秒，激光关闭时间（-1表示永不关闭）

    // 扫描路径（未来功能）
    bool moving = false;
    double scan_speed = 0.0;           // m/s
};

/**
 * @brief 物理模块配置
 */
struct PhysicsConfig {
    // 各模块开关
    bool thermal_enabled = true;
    bool phase_change_enabled = true;
    bool fluid_enabled = true;
    bool buoyancy_enabled = true;
    bool darcy_enabled = true;
    bool marangoni_enabled = false;  // Phase 6

    // 热传导
    double thermal_alpha = 1.0e-5;   // m²/s

    // 流体
    double fluid_viscosity = 5e-7;   // m²/s

    // 浮力
    std::array<double, 3> gravity = {0.0, 0.0, 9.81};  // m/s²
    double beta_thermal = 9e-6;                         // K⁻¹
    double T_ref = -1.0;  // -1表示使用熔点，否则为具体温度
    double force_magnitude = 2e-2;  // 格子单位

    // Darcy阻尼
    double darcy_constant = 15.0;  // kg/(m³·s)

    // Marangoni（Phase 6）
    double surface_tension_coeff = 0.0;  // N/m·K
};

/**
 * @brief 边界条件配置
 */
struct BoundaryConfig {
    // 热边界
    std::string thermal_type = "adiabatic";  // adiabatic, periodic, fixed, radiation
    double thermal_value = 300.0;            // K (for fixed BC)

    // 辐射边界条件 (Stefan-Boltzmann radiation)
    bool enable_radiation_bc = false;        // 是否启用辐射边界条件
    double emissivity = 0.35;                // 发射率 (Ti6Al4V typical: 0.3-0.4)
    double stefan_boltzmann = 5.67e-8;       // Stefan-Boltzmann constant [W/(m²·K⁴)]
    double ambient_temperature = 300.0;      // 环境温度 [K]

    // 流体边界
    std::string fluid_type = "no_slip";      // no_slip, free_slip, periodic
};

/**
 * @brief 初始条件配置
 */
struct InitialConfig {
    double temperature = 300.0;  // K
    std::array<double, 3> velocity = {0.0, 0.0, 0.0};  // m/s
};

/**
 * @brief 输出配置
 */
struct OutputConfig {
    std::string format = "vtk";  // vtk, hdf5
    std::vector<std::string> fields = {
        "temperature",
        "liquid_fraction",
        "velocity",
        "phase_state"
    };
    bool save_final_only = false;
};

/**
 * @brief 完整仿真配置
 */
class SimulationConfig {
public:
    // 基本信息
    std::string name = "Laser Melting Simulation";
    std::string output_dir = "visualization_output";

    // 各部分配置
    DomainConfig domain;
    TimeConfig time;
    MaterialConfig material;
    LaserConfig laser;
    PhysicsConfig physics;
    BoundaryConfig boundary;
    InitialConfig initial;
    OutputConfig output;

    /**
     * @brief 从配置文件加载
     * @param filename 配置文件路径
     * @return 配置对象
     */
    static SimulationConfig loadFromFile(const std::string& filename);

    /**
     * @brief 保存到文件
     * @param filename 输出文件路径
     */
    void saveToFile(const std::string& filename) const;

    /**
     * @brief 打印配置摘要
     */
    void printSummary() const;

    /**
     * @brief 验证配置合法性
     * @throw std::runtime_error 如果配置无效
     */
    void validate() const;

    /**
     * @brief 获取预设配置
     * @param preset_name 预设名称
     * @return 配置对象
     */
    static SimulationConfig getPreset(const std::string& preset_name);

private:
    /**
     * @brief 从简单键值对加载（简化版YAML解析）
     */
    void loadFromKeyValue(const std::string& content);
};

}} // namespace lbm::config

#endif // LBM_CONFIG_SIMULATION_CONFIG_H
