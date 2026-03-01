# CMake generated Testfile for 
# Source directory: /home/yzk/LBMProject/tests/integration
# Build directory: /home/yzk/LBMProject/build_check/tests/integration
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(EvaporationEnergyBalance "/home/yzk/LBMProject/build_check/tests/integration/test_evaporation_energy_balance")
set_tests_properties(EvaporationEnergyBalance PROPERTIES  LABELS "integration;evaporation;energy;week1;critical" TIMEOUT "120" _BACKTRACE_TRIPLES "/home/yzk/LBMProject/tests/integration/CMakeLists.txt;39;add_test;/home/yzk/LBMProject/tests/integration/CMakeLists.txt;0;")
add_test(SubstrateTemperatureReduction "/home/yzk/LBMProject/build_check/tests/integration/test_substrate_temperature_reduction")
set_tests_properties(SubstrateTemperatureReduction PROPERTIES  LABELS "integration;substrate;temperature;week1;critical" TIMEOUT "300" _BACKTRACE_TRIPLES "/home/yzk/LBMProject/tests/integration/CMakeLists.txt;71;add_test;/home/yzk/LBMProject/tests/integration/CMakeLists.txt;0;")
add_test(LaserHeatingIntegrationSimplified "/home/yzk/LBMProject/build_check/tests/integration/test_laser_heating_simplified")
set_tests_properties(LaserHeatingIntegrationSimplified PROPERTIES  LABELS "integration;physics" TIMEOUT "120" _BACKTRACE_TRIPLES "/home/yzk/LBMProject/tests/integration/CMakeLists.txt;117;add_test;/home/yzk/LBMProject/tests/integration/CMakeLists.txt;0;")
add_test(LaserMeltingIntegration "/home/yzk/LBMProject/build_check/tests/integration/test_laser_melting")
set_tests_properties(LaserMeltingIntegration PROPERTIES  LABELS "integration;physics;phase_change" TIMEOUT "180" _BACKTRACE_TRIPLES "/home/yzk/LBMProject/tests/integration/CMakeLists.txt;152;add_test;/home/yzk/LBMProject/tests/integration/CMakeLists.txt;0;")
add_test(StefanProblemValidation "/home/yzk/LBMProject/build_check/tests/integration/test_stefan_1d")
set_tests_properties(StefanProblemValidation PROPERTIES  LABELS "integration;physics;phase_change;validation" TIMEOUT "300" _BACKTRACE_TRIPLES "/home/yzk/LBMProject/tests/integration/CMakeLists.txt;185;add_test;/home/yzk/LBMProject/tests/integration/CMakeLists.txt;0;")
add_test(PoiseuilleFlowFluidLBM "/home/yzk/LBMProject/build_check/tests/integration/test_poiseuille_flow_fluidlbm")
set_tests_properties(PoiseuilleFlowFluidLBM PROPERTIES  LABELS "integration;fluid;validation;fluidlbm" TIMEOUT "180" _BACKTRACE_TRIPLES "/home/yzk/LBMProject/tests/integration/CMakeLists.txt;218;add_test;/home/yzk/LBMProject/tests/integration/CMakeLists.txt;0;")
add_test(ThermalFluidCouplingIntegration "/home/yzk/LBMProject/build_check/tests/integration/test_thermal_fluid_coupling")
set_tests_properties(ThermalFluidCouplingIntegration PROPERTIES  LABELS "integration;coupling;thermal;fluid;convection" TIMEOUT "300" _BACKTRACE_TRIPLES "/home/yzk/LBMProject/tests/integration/CMakeLists.txt;251;add_test;/home/yzk/LBMProject/tests/integration/CMakeLists.txt;0;")
add_test(LaserMeltingConvectionIntegration "/home/yzk/LBMProject/build_check/tests/integration/test_laser_melting_convection")
set_tests_properties(LaserMeltingConvectionIntegration PROPERTIES  LABELS "integration;coupling;laser;melting;convection;phase5" TIMEOUT "300" _BACKTRACE_TRIPLES "/home/yzk/LBMProject/tests/integration/CMakeLists.txt;284;add_test;/home/yzk/LBMProject/tests/integration/CMakeLists.txt;0;")
add_test(RecoilSurfaceDepressionIntegration "/home/yzk/LBMProject/build_check/tests/integration/test_recoil_surface_depression")
set_tests_properties(RecoilSurfaceDepressionIntegration PROPERTIES  LABELS "integration;recoil;keyhole;phase6" TIMEOUT "180" _BACKTRACE_TRIPLES "/home/yzk/LBMProject/tests/integration/CMakeLists.txt;320;add_test;/home/yzk/LBMProject/tests/integration/CMakeLists.txt;0;")
subdirs("multiphysics")
