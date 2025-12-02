/**
 * @file simulation_config.cpp
 * @brief 仿真配置实现
 */

#include "config/simulation_config.h"
#include "physics/material_properties.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstring>

namespace lbm {
namespace config {

// 辅助函数：去除字符串首尾空白
static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\n\r");
    return str.substr(first, (last - first + 1));
}

// 辅助函数：分割字符串
static std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim(token));
    }
    return tokens;
}

/**
 * @brief 简单的键值对解析器（支持基本YAML语法子集）
 */
SimulationConfig SimulationConfig::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开配置文件: " + filename);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    SimulationConfig cfg;
    cfg.loadFromKeyValue(content);
    cfg.validate();

    return cfg;
}

void SimulationConfig::loadFromKeyValue(const std::string& content) {
    std::istringstream iss(content);
    std::string line;
    std::string current_section;

    while (std::getline(iss, line)) {
        line = trim(line);

        // 跳过空行和注释
        if (line.empty() || line[0] == '#') continue;

        // 检测section（以:结尾且无=号）
        if (line.back() == ':' && line.find('=') == std::string::npos) {
            current_section = line.substr(0, line.size() - 1);
            continue;
        }

        // 解析 key = value
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq_pos));
        std::string value = trim(line.substr(eq_pos + 1));

        // 去除引号
        if (value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.size() - 2);
        }

        // 根据section和key设置值
        if (current_section == "simulation") {
            if (key == "name") name = value;
            else if (key == "output_dir") output_dir = value;
        }
        else if (current_section == "domain") {
            if (key == "nx") domain.nx = std::stoi(value);
            else if (key == "ny") domain.ny = std::stoi(value);
            else if (key == "nz") domain.nz = std::stoi(value);
            else if (key == "Lx") domain.Lx = std::stod(value);
            else if (key == "Ly") domain.Ly = std::stod(value);
            else if (key == "Lz") domain.Lz = std::stod(value);
        }
        else if (current_section == "time") {
            if (key == "dt") time.dt = std::stod(value);
            else if (key == "n_steps") time.n_steps = std::stoi(value);
            else if (key == "output_interval") time.output_interval = std::stoi(value);
        }
        else if (current_section == "material") {
            if (key == "type") material.type = value;
        }
        else if (current_section == "laser") {
            if (key == "enabled") laser.enabled = (value == "true" || value == "1");
            else if (key == "power") laser.power = std::stod(value);
            else if (key == "spot_radius") laser.spot_radius = std::stod(value);
            else if (key == "absorption") laser.absorption = std::stod(value);
            else if (key == "penetration_depth") laser.penetration_depth = std::stod(value);
            else if (key == "turn_on_time") laser.turn_on_time = std::stod(value);
            else if (key == "turn_off_time") laser.turn_off_time = std::stod(value);
        }
        else if (current_section == "physics") {
            if (key == "thermal") physics.thermal_enabled = (value == "true" || value == "1");
            else if (key == "phase_change") physics.phase_change_enabled = (value == "true" || value == "1");
            else if (key == "fluid") physics.fluid_enabled = (value == "true" || value == "1");
            else if (key == "buoyancy") physics.buoyancy_enabled = (value == "true" || value == "1");
            else if (key == "darcy") physics.darcy_enabled = (value == "true" || value == "1");
            else if (key == "marangoni") physics.marangoni_enabled = (value == "true" || value == "1");
            else if (key == "force_magnitude") physics.force_magnitude = std::stod(value);
            else if (key == "darcy_constant") physics.darcy_constant = std::stod(value);
        }
        else if (current_section == "boundary") {
            if (key == "thermal_type") boundary.thermal_type = value;
            else if (key == "thermal") boundary.thermal_type = value;  // Alias
            else if (key == "fluid_type") boundary.fluid_type = value;
            else if (key == "fluid") boundary.fluid_type = value;  // Alias
            else if (key == "enable_radiation_bc") boundary.enable_radiation_bc = (value == "true" || value == "1");
            else if (key == "emissivity") boundary.emissivity = std::stod(value);
            else if (key == "stefan_boltzmann") boundary.stefan_boltzmann = std::stod(value);
            else if (key == "ambient_temperature") boundary.ambient_temperature = std::stod(value);
        }
        else if (current_section == "initial") {
            if (key == "temperature") initial.temperature = std::stod(value);
        }
    }
}

void SimulationConfig::validate() const {
    // 验证域大小
    if (domain.nx <= 0 || domain.ny <= 0 || domain.nz <= 0) {
        throw std::runtime_error("域网格尺寸必须为正数");
    }
    if (domain.Lx <= 0 || domain.Ly <= 0 || domain.Lz <= 0) {
        throw std::runtime_error("域物理尺寸必须为正数");
    }

    // 验证时间步
    if (time.dt <= 0) {
        throw std::runtime_error("时间步长必须为正数");
    }
    if (time.n_steps <= 0) {
        throw std::runtime_error("步数必须为正数");
    }

    // 验证物理参数
    if (physics.thermal_enabled && physics.thermal_alpha <= 0) {
        throw std::runtime_error("热扩散系数必须为正数");
    }
}

void SimulationConfig::printSummary() const {
    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "   仿真配置摘要\n";
    std::cout << "══════════════════════════════════════════════════════════\n";
    std::cout << "名称: " << name << "\n";
    std::cout << "输出目录: " << output_dir << "\n\n";

    std::cout << "【域配置】\n";
    std::cout << "  网格: " << domain.nx << " × " << domain.ny << " × " << domain.nz << "\n";
    std::cout << "  尺寸: " << domain.Lx*1e6 << " × " << domain.Ly*1e6 << " × " << domain.Lz*1e6 << " μm\n";
    std::cout << "  分辨率: dx = " << domain.dx()*1e6 << " μm\n\n";

    std::cout << "【时间配置】\n";
    std::cout << "  时间步: dt = " << time.dt << " s (" << time.dt*1e9 << " ns)\n";
    std::cout << "  总步数: " << time.n_steps << "\n";
    std::cout << "  总时间: " << time.dt * time.n_steps * 1e6 << " μs\n";
    std::cout << "  输出间隔: 每 " << time.output_interval << " 步\n\n";

    std::cout << "【材料】\n";
    std::cout << "  类型: " << material.type << "\n\n";

    std::cout << "【激光】\n";
    std::cout << "  开启: " << (laser.enabled ? "是" : "否") << "\n";
    if (laser.enabled) {
        std::cout << "  功率: " << laser.power << " W\n";
        std::cout << "  光斑半径: " << laser.spot_radius*1e6 << " μm\n";
        std::cout << "  吸收率: " << laser.absorption << "\n";

        // 时间控制信息
        if (laser.turn_on_time > 0 || laser.turn_off_time > 0) {
            std::cout << "  时间控制:\n";
            if (laser.turn_on_time > 0) {
                std::cout << "    开启时间: " << laser.turn_on_time * 1e6 << " μs\n";
            }
            if (laser.turn_off_time > 0) {
                std::cout << "    关闭时间: " << laser.turn_off_time * 1e6 << " μs\n";
                std::cout << "    照射时长: " << (laser.turn_off_time - laser.turn_on_time) * 1e6 << " μs\n";
            }
        }
    }
    std::cout << "\n";

    std::cout << "【物理模块】\n";
    std::cout << "  热传导:   " << (physics.thermal_enabled ? "✓" : "✗") << "\n";
    std::cout << "  相变:     " << (physics.phase_change_enabled ? "✓" : "✗") << "\n";
    std::cout << "  流体:     " << (physics.fluid_enabled ? "✓" : "✗") << "\n";
    std::cout << "  浮力:     " << (physics.buoyancy_enabled ? "✓" : "✗") << "\n";
    std::cout << "  Darcy阻尼: " << (physics.darcy_enabled ? "✓" : "✗");
    if (physics.darcy_enabled) {
        std::cout << "  (C = " << physics.darcy_constant << ")";
    }
    std::cout << "\n";
    std::cout << "  Marangoni: " << (physics.marangoni_enabled ? "✓" : "✗") << "\n\n";

    std::cout << "【边界条件】\n";
    std::cout << "  热边界: " << boundary.thermal_type << "\n";
    std::cout << "  流体边界: " << boundary.fluid_type << "\n";
    std::cout << "  辐射BC: " << (boundary.enable_radiation_bc ? "✓" : "✗");
    if (boundary.enable_radiation_bc) {
        std::cout << " (ε=" << boundary.emissivity
                  << ", T_amb=" << boundary.ambient_temperature << " K)";
    }
    std::cout << "\n\n";

    std::cout << "【初始条件】\n";
    std::cout << "  温度: " << initial.temperature << " K\n";
    std::cout << "══════════════════════════════════════════════════════════\n\n";
}

void SimulationConfig::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("无法创建配置文件: " + filename);
    }

    file << "# LBM仿真配置文件\n";
    file << "# 自动生成于: " << __DATE__ << " " << __TIME__ << "\n\n";

    file << "simulation:\n";
    file << "  name = \"" << name << "\"\n";
    file << "  output_dir = \"" << output_dir << "\"\n\n";

    file << "domain:\n";
    file << "  nx = " << domain.nx << "\n";
    file << "  ny = " << domain.ny << "\n";
    file << "  nz = " << domain.nz << "\n";
    file << "  Lx = " << domain.Lx << "\n";
    file << "  Ly = " << domain.Ly << "\n";
    file << "  Lz = " << domain.Lz << "\n\n";

    file << "time:\n";
    file << "  dt = " << time.dt << "\n";
    file << "  n_steps = " << time.n_steps << "\n";
    file << "  output_interval = " << time.output_interval << "\n\n";

    file << "material:\n";
    file << "  type = \"" << material.type << "\"\n\n";

    file << "laser:\n";
    file << "  enabled = " << (laser.enabled ? "true" : "false") << "\n";
    file << "  power = " << laser.power << "\n";
    file << "  spot_radius = " << laser.spot_radius << "\n";
    file << "  absorption = " << laser.absorption << "\n";
    file << "  penetration_depth = " << laser.penetration_depth << "\n\n";

    file << "physics:\n";
    file << "  thermal = " << (physics.thermal_enabled ? "true" : "false") << "\n";
    file << "  phase_change = " << (physics.phase_change_enabled ? "true" : "false") << "\n";
    file << "  fluid = " << (physics.fluid_enabled ? "true" : "false") << "\n";
    file << "  buoyancy = " << (physics.buoyancy_enabled ? "true" : "false") << "\n";
    file << "  darcy = " << (physics.darcy_enabled ? "true" : "false") << "\n";
    file << "  marangoni = " << (physics.marangoni_enabled ? "true" : "false") << "\n";
    file << "  force_magnitude = " << physics.force_magnitude << "\n";
    file << "  darcy_constant = " << physics.darcy_constant << "\n\n";

    file << "boundary:\n";
    file << "  thermal_type = \"" << boundary.thermal_type << "\"\n";
    file << "  fluid_type = \"" << boundary.fluid_type << "\"\n";
    file << "  enable_radiation_bc = " << (boundary.enable_radiation_bc ? "true" : "false") << "\n";
    file << "  emissivity = " << boundary.emissivity << "\n";
    file << "  stefan_boltzmann = " << boundary.stefan_boltzmann << "\n";
    file << "  ambient_temperature = " << boundary.ambient_temperature << "\n\n";

    file << "initial:\n";
    file << "  temperature = " << initial.temperature << "\n";
}

SimulationConfig SimulationConfig::getPreset(const std::string& preset_name) {
    SimulationConfig cfg;

    if (preset_name == "ti6al4v_melting" || preset_name == "phase5") {
        // Phase 5 优化后的配置
        cfg.name = "Ti6Al4V激光熔化 - Phase 5";
        cfg.domain.nx = 80;
        cfg.domain.ny = 80;
        cfg.domain.nz = 40;
        cfg.domain.Lx = 160e-6;
        cfg.domain.Ly = 160e-6;
        cfg.domain.Lz = 80e-6;

        cfg.time.dt = 5e-10;
        cfg.time.n_steps = 8000;
        cfg.time.output_interval = 100;

        cfg.material.type = "Ti6Al4V";

        cfg.laser.enabled = true;
        cfg.laser.power = 1200.0;
        cfg.laser.spot_radius = 50e-6;
        cfg.laser.absorption = 0.35;
        cfg.laser.penetration_depth = 15e-6;
        cfg.laser.position = {80e-6, 80e-6, 0.0};

        cfg.physics.thermal_enabled = true;
        cfg.physics.phase_change_enabled = true;
        cfg.physics.fluid_enabled = true;
        cfg.physics.buoyancy_enabled = true;
        cfg.physics.darcy_enabled = true;
        cfg.physics.marangoni_enabled = false;
        cfg.physics.force_magnitude = 2e-2;
        cfg.physics.darcy_constant = 15.0;

        cfg.boundary.thermal_type = "adiabatic";
        cfg.boundary.fluid_type = "no_slip";

        cfg.initial.temperature = 300.0;
    }
    else if (preset_name == "stefan_problem") {
        // Stefan问题基准测试
        cfg.name = "Stefan问题验证";
        cfg.domain.nx = 100;
        cfg.domain.ny = 100;
        cfg.domain.nz = 1;  // 2D

        cfg.physics.thermal_enabled = true;
        cfg.physics.phase_change_enabled = true;
        cfg.physics.fluid_enabled = false;  // 无流动
        cfg.physics.buoyancy_enabled = false;

        cfg.laser.enabled = false;
    }
    else if (preset_name == "thermal_only") {
        // 仅热传导（无相变、无流动）
        cfg.name = "纯热传导测试";
        cfg.physics.thermal_enabled = true;
        cfg.physics.phase_change_enabled = false;
        cfg.physics.fluid_enabled = false;
    }
    else {
        throw std::runtime_error("未知的预设: " + preset_name);
    }

    return cfg;
}

}} // namespace lbm::config
