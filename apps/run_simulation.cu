/**
 * @file run_simulation.cu
 * @brief 通用LBM仿真入口 - 支持配置文件驱动
 *
 * 使用方法:
 *   ./run_simulation config/laser_melting_phase5.cfg
 *   ./run_simulation --preset phase5
 *   ./run_simulation --preset thermal_only
 */

#include <iostream>
#include <string>
#include <memory>
#include <cuda_runtime.h>

#include "config/simulation_config.h"
#include "physics/thermal_lbm.h"
#include "physics/fluid_lbm.h"
#include "physics/laser_source.h"
#include "physics/material_properties.h"
#include "io/vtk_writer.h"
#include "core/lattice_d3q19.h"

using namespace lbm;

// 固体速度约束核函数（与Phase5相同）
__global__ void enforceZeroVelocityInSolid(
    float* ux, float* uy, float* uz,
    const float* liquid_fraction,
    float threshold, int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;

    if (liquid_fraction[id] < threshold) {
        ux[id] = 0.0f;
        uy[id] = 0.0f;
        uz[id] = 0.0f;
    }
}

// 力缩放核函数
__global__ void scaleForceArrayKernel(
    float* fx, float* fy, float* fz,
    float scale, int num_cells)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_cells) return;
    fx[id] *= scale;
    fy[id] *= scale;
    fz[id] *= scale;
}

void printUsage(const char* prog_name) {
    std::cout << "用法:\n";
    std::cout << "  " << prog_name << " <config_file>           # 从配置文件加载\n";
    std::cout << "  " << prog_name << " --preset <name>        # 使用预设配置\n\n";
    std::cout << "预设:\n";
    std::cout << "  phase5          - Ti6Al4V激光熔化 (Phase 5优化参数)\n";
    std::cout << "  thermal_only    - 纯热传导测试\n";
    std::cout << "  stefan_problem  - Stefan问题基准测试\n\n";
    std::cout << "示例:\n";
    std::cout << "  " << prog_name << " config/laser_melting_phase5.cfg\n";
    std::cout << "  " << prog_name << " --preset phase5\n";
}

int main(int argc, char** argv) {
    // 解析命令行参数
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    config::SimulationConfig cfg;

    try {
        std::string arg1 = argv[1];
        if (arg1 == "--preset" && argc >= 3) {
            std::string preset_name = argv[2];
            cfg = config::SimulationConfig::getPreset(preset_name);
            std::cout << "✓ 加载预设配置: " << preset_name << "\n";
        } else if (arg1 == "-h" || arg1 == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            cfg = config::SimulationConfig::loadFromFile(arg1);
            std::cout << "✓ 加载配置文件: " << arg1 << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "配置加载失败: " << e.what() << "\n";
        return 1;
    }

    // 打印配置摘要
    cfg.printSummary();

    // 提取配置参数
    int nx = cfg.domain.nx;
    int ny = cfg.domain.ny;
    int nz = cfg.domain.nz;
    int num_cells = nx * ny * nz;

    double dx = cfg.domain.dx();
    double dy = cfg.domain.dy();
    double dz = cfg.domain.dz();
    double dt = cfg.time.dt;

    // 初始化D3Q19格子常数
    core::D3Q19::initializeDevice();

    // 获取材料属性
    physics::MaterialProperties mat;
    if (cfg.material.type == "Ti6Al4V") {
        mat = physics::MaterialDatabase::getTi6Al4V();
    } else if (cfg.material.type == "SS316L") {
        mat = physics::MaterialDatabase::get316L();
    } else if (cfg.material.type == "Inconel718") {
        mat = physics::MaterialDatabase::getInconel718();
    } else {
        std::cerr << "不支持的材料: " << cfg.material.type << "\n";
        return 1;
    }

    float T_ref = cfg.physics.T_ref;
    if (T_ref < 0) {
        // 使用熔点
        T_ref = 0.5f * (mat.T_solidus + mat.T_liquidus);
    }

    std::cout << "初始化物理模块...\n";

    // 1. 热求解器（总是需要）
    std::unique_ptr<physics::ThermalLBM> thermal;
    if (cfg.physics.thermal_enabled) {
        thermal = std::make_unique<physics::ThermalLBM>(
            nx, ny, nz, mat, cfg.physics.thermal_alpha, cfg.physics.phase_change_enabled
        );
        thermal->initialize(cfg.initial.temperature);
        std::cout << "  ✓ 热传导模块\n";
    }

    // 2. 流体求解器（可选）
    std::unique_ptr<physics::FluidLBM> fluid;
    if (cfg.physics.fluid_enabled) {
        // CRITICAL FIX: Use physical viscosity instead of lattice viscosity
        // FluidLBM constructor now handles unit conversion internally using dt and dx
        // For Ti6Al4V liquid: nu_physical ≈ 4.5e-7 m²/s
        // Kinematic viscosity: nu = mu / rho
        float nu_physical = mat.mu_liquid / mat.rho_liquid;

        fluid = std::make_unique<physics::FluidLBM>(
            nx, ny, nz,
            nu_physical,                     // Physical kinematic viscosity [m²/s]
            mat.rho_liquid,                  // Density [kg/m³]
            physics::BoundaryType::WALL,     // x: wall
            physics::BoundaryType::WALL,     // y: wall
            physics::BoundaryType::PERIODIC, // z: periodic
            static_cast<float>(dt),          // Time step for unit conversion
            static_cast<float>(dx)           // Lattice spacing for unit conversion
        );
        fluid->initialize(mat.rho_liquid, 0.0f, 0.0f, 0.0f);  // Initially at rest
        std::cout << "  ✓ 流体模块 (ν_physical = " << nu_physical << " m²/s)\n";
    }

    // 3. 激光源（可选）
    std::unique_ptr<LaserSource> laser;
    if (cfg.laser.enabled) {
        laser = std::make_unique<LaserSource>(
            cfg.laser.power,
            cfg.laser.spot_radius,
            cfg.laser.absorption,
            cfg.laser.penetration_depth
        );
        laser->setPosition(
            cfg.laser.position[0],
            cfg.laser.position[1],
            cfg.laser.position[2]
        );
        std::cout << "  ✓ 激光热源 (P = " << cfg.laser.power << " W)\n";
    }

    // 分配浮力场（如果需要）
    float *d_fx = nullptr, *d_fy = nullptr, *d_fz = nullptr;
    if (cfg.physics.buoyancy_enabled && fluid) {
        cudaMalloc(&d_fx, num_cells * sizeof(float));
        cudaMalloc(&d_fy, num_cells * sizeof(float));
        cudaMalloc(&d_fz, num_cells * sizeof(float));
        std::cout << "  ✓ 浮力模块\n";
    }

    // 分配激光热源场（如果需要）
    float *d_heat_source = nullptr;
    if (laser) {
        cudaMalloc(&d_heat_source, num_cells * sizeof(float));
        cudaMemset(d_heat_source, 0, num_cells * sizeof(float));
    }

    // 设置CUDA网格和块
    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x,
              (ny + block.y - 1) / block.y,
              (nz + block.z - 1) / block.z);

    // 创建输出目录
    system(("mkdir -p " + cfg.output_dir).c_str());

    std::cout << "\n开始时间推进...\n";
    std::cout << "──────────────────────────────────────────────────\n";

    for (int step = 0; step <= cfg.time.n_steps; ++step) {
        double current_time = step * dt;  // 当前物理时间（秒）

        // 1. 激光加热（带时间控制）
        if (laser && thermal) {
            bool laser_on = true;

            // 检查激光时间窗口
            if (cfg.laser.turn_on_time > 0 && current_time < cfg.laser.turn_on_time) {
                laser_on = false;  // 还未到开启时间
            }
            if (cfg.laser.turn_off_time > 0 && current_time >= cfg.laser.turn_off_time) {
                laser_on = false;  // 已过关闭时间
            }

            if (laser_on) {
                computeLaserHeatSourceKernel<<<grid, block>>>(
                    d_heat_source, *laser, dx, dy, dz, nx, ny, nz
                );
                cudaDeviceSynchronize();
                thermal->addHeatSource(d_heat_source, dt);
            }
            // 激光关闭时不添加热源，自然冷却
        }

        // 2. 热传导
        if (thermal) {
            thermal->collisionBGK();
            thermal->streaming();

            if (cfg.boundary.thermal_type == "adiabatic") {
                thermal->applyBoundaryConditions(2, cfg.initial.temperature);
            }

            thermal->computeTemperature();
        }

        // 3. 浮力（如果启用）
        if (cfg.physics.buoyancy_enabled && fluid && thermal) {
            float g = cfg.physics.gravity[2];  // Z方向重力
            float beta = cfg.physics.beta_thermal;

            fluid->computeBuoyancyForce(
                thermal->getTemperature(), T_ref, beta,
                0.0f, 0.0f, g,
                d_fx, d_fy, d_fz
            );

            // 力缩放
            float force_scale = cfg.physics.force_magnitude / 500.0f;
            int block_size = 256;
            int grid_size = (num_cells + block_size - 1) / block_size;
            scaleForceArrayKernel<<<grid_size, block_size>>>(
                d_fx, d_fy, d_fz, force_scale, num_cells
            );
        }

        // 4. Darcy阻尼（如果启用）
        if (cfg.physics.darcy_enabled && fluid && thermal && d_fx && d_fy && d_fz) {
            fluid->applyDarcyDamping(
                thermal->getLiquidFraction(),
                cfg.physics.darcy_constant,
                d_fx, d_fy, d_fz
            );
        }

        // 5. 流体演化
        if (fluid) {
            if (d_fx && d_fy && d_fz) {
                fluid->collisionBGK(d_fx, d_fy, d_fz);
            } else {
                fluid->collisionBGK();
            }
            fluid->streaming();
            fluid->computeMacroscopic();

            // 固体速度约束
            if (thermal) {
                int block_size = 256;
                int grid_size = (num_cells + block_size - 1) / block_size;
                enforceZeroVelocityInSolid<<<grid_size, block_size>>>(
                    fluid->getVelocityX(),
                    fluid->getVelocityY(),
                    fluid->getVelocityZ(),
                    thermal->getLiquidFraction(),
                    0.05f,
                    num_cells
                );
            }
        }

        // 输出
        if (step % cfg.time.output_interval == 0) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/output_%06d.vtk",
                     cfg.output_dir.c_str(), step);

            // 分配主机内存
            float* h_temperature = new float[num_cells];
            float* h_liquid_fraction = new float[num_cells];
            float* h_phase_state = new float[num_cells];
            float* h_ux = new float[num_cells];
            float* h_uy = new float[num_cells];
            float* h_uz = new float[num_cells];

            // 从设备拷贝数据
            if (thermal) {
                thermal->copyTemperatureToHost(h_temperature);
                if (cfg.physics.phase_change_enabled) {
                    thermal->copyLiquidFractionToHost(h_liquid_fraction);

                    // 计算相态
                    for (int i = 0; i < num_cells; ++i) {
                        float fl = h_liquid_fraction[i];
                        if (fl < 0.01f) {
                            h_phase_state[i] = 0.0f;  // 固态
                        } else if (fl > 0.99f) {
                            h_phase_state[i] = 2.0f;  // 液态
                        } else {
                            h_phase_state[i] = 1.0f;  // 糊状区
                        }
                    }
                } else {
                    // 无相变，全部初始化为0
                    for (int i = 0; i < num_cells; ++i) {
                        h_liquid_fraction[i] = 0.0f;
                        h_phase_state[i] = 0.0f;
                    }
                }
            }

            if (fluid) {
                fluid->copyVelocityToHost(h_ux, h_uy, h_uz);
            } else {
                // 无流体，速度为0
                for (int i = 0; i < num_cells; ++i) {
                    h_ux[i] = 0.0f;
                    h_uy[i] = 0.0f;
                    h_uz[i] = 0.0f;
                }
            }

            // 写入VTK文件
            io::VTKWriter::writeStructuredGridWithVectors(
                filename,
                h_temperature,
                h_liquid_fraction,
                h_phase_state,
                nullptr,  // fill_level not used
                h_ux, h_uy, h_uz,
                nx, ny, nz,
                dx, dy, dz
            );

            // 清理主机内存
            delete[] h_temperature;
            delete[] h_liquid_fraction;
            delete[] h_phase_state;
            delete[] h_ux;
            delete[] h_uy;
            delete[] h_uz;

            if (step % (cfg.time.output_interval * 10) == 0) {
                std::cout << "  步数 " << step << " / " << cfg.time.n_steps
                          << "  (输出: " << filename << ")\n";
            }
        }
    }

    std::cout << "──────────────────────────────────────────────────\n";
    std::cout << "\n✓ 仿真完成！\n";
    std::cout << "  输出目录: " << cfg.output_dir << "/\n";
    std::cout << "  文件数量: " << (cfg.time.n_steps / cfg.time.output_interval + 1) << "\n\n";

    // 清理
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_fz) cudaFree(d_fz);
    if (d_heat_source) cudaFree(d_heat_source);

    return 0;
}
