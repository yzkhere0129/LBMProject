# Validation Framework Implementation Checklist

**For the test developer implementing the comprehensive validation strategy**

---

## Phase 1: Regression Test Suite (Weeks 1-2)

**Goal:** Build 10 automated regression tests with pass/fail criteria
**Duration:** 2 weeks
**Deliverable:** 5-minute test suite running before every commit

---

### Week 1: Critical Tests (Tests 1-3)

#### Test 1: Baseline 150W (CRITICAL)

- [ ] **Day 1-2: Implementation**
  - [ ] Create file: `tests/regression/test_baseline_150W.cu`
  - [ ] Copy config from grid convergence medium grid
  - [ ] Set up test fixture with GoogleTest
  - [ ] Implement simulation runner
  - [ ] Extract metrics: T_max, v_max, E_error, M_error
  - [ ] Add assertions with ±5% tolerance
  - [ ] Test runtime: confirm < 15 sec on RTX 3050

- [ ] **Day 2: Integration**
  - [ ] Add to `tests/CMakeLists.txt`:
    ```cmake
    add_cuda_test(test_baseline_150W regression/test_baseline_150W.cu)
    set_tests_properties(test_baseline_150W PROPERTIES
        LABELS "regression;critical"
        TIMEOUT 30
    )
    ```
  - [ ] Build: `make test_baseline_150W`
  - [ ] Run: `./tests/regression/test_baseline_150W`
  - [ ] Verify: PASS with current code

- [ ] **Day 2: Documentation**
  - [ ] Add reference values to test file header
  - [ ] Document expected results in test README
  - [ ] Update this checklist with actual runtime

- [ ] **Completed:** _______ (date) by _______ (name)

---

#### Test 2: Stability 500 Steps (CRITICAL)

- [ ] **Day 3: Implementation**
  - [ ] Create file: `tests/regression/test_stability_500step.cu`
  - [ ] Same config as Test 1, extend to 500 steps
  - [ ] Add NaN detection: `assert(!isnan(T_max))`
  - [ ] Add divergence check: `assert(T_max < 10000.0f)`
  - [ ] Add velocity bound: `assert(v_max < 1000.0f)`
  - [ ] Add energy drift check: `assert(|dE/dt| < 0.001)`
  - [ ] Test runtime: confirm < 20 sec

- [ ] **Day 3: Integration**
  - [ ] Add to CMakeLists.txt with CRITICAL label
  - [ ] Build and verify PASS
  - [ ] Check memory usage: `nvidia-smi` during run

- [ ] **Day 3: Documentation**
  - [ ] Document what "stability" means (no NaN, no divergence)
  - [ ] Add failure diagnosis guide

- [ ] **Completed:** _______ (date) by _______ (name)

---

#### Test 3: Energy Conservation (CRITICAL)

- [ ] **Day 4: Implementation**
  - [ ] Create file: `tests/regression/test_energy_conservation.cu`
  - [ ] Config: Zero laser, uniform T=2000K, adiabatic BC
  - [ ] Compute E_total = Σ(ρ·cp·T·volume)
  - [ ] Run 100 steps (no heat source)
  - [ ] Check E_final ≈ E_initial (within 1e-5)
  - [ ] Test runtime: confirm < 5 sec

- [ ] **Day 4: Integration**
  - [ ] Add to CMakeLists.txt
  - [ ] Build and verify PASS
  - [ ] Run 5 times, check reproducibility

- [ ] **Day 4: Documentation**
  - [ ] Explain why this test is critical (thermodynamic consistency)
  - [ ] Document acceptable tolerance (roundoff error)

- [ ] **Completed:** _______ (date) by _______ (name)

---

### Week 2: High Priority Tests (Tests 4-5)

#### Test 4: Pure Conduction (HIGH)

- [ ] **Day 5: Analytical Solution**
  - [ ] Implement error function solution:
    ```cpp
    float T_analytical(float x, float t, float T_left, float T_right, float alpha) {
        float T_avg = (T_left + T_right) / 2.0f;
        float dT = (T_left - T_right) / 2.0f;
        float arg = (x - 0.5f) / sqrt(4 * alpha * t);
        return T_avg + dT * erf(arg);
    }
    ```

- [ ] **Day 5: Implementation**
  - [ ] Create file: `tests/regression/test_pure_conduction.cu`
  - [ ] Set up 1D problem: 100×1×1 grid
  - [ ] Initial: step function T(x<50)=2000K, T(x≥50)=300K
  - [ ] Run 10 steps, dt=1e-7 s
  - [ ] Compare to analytical at x=40, 50, 60
  - [ ] Compute L2 error: sqrt(Σ(T_sim - T_analytical)²/N)
  - [ ] Assert: L2_error < 5%

- [ ] **Day 6: Integration and Testing**
  - [ ] Add to CMakeLists.txt with HIGH label
  - [ ] Run convergence test (vary dx, check error decreases)
  - [ ] Document expected error vs dx

- [ ] **Completed:** _______ (date) by _______ (name)

---

#### Test 5: Static Droplet (HIGH)

- [ ] **Day 6: Implementation**
  - [ ] File already exists: `tests/integration/test_static_droplet.cu`
  - [ ] Refactor to add automated checks:
    ```cpp
    // Laplace pressure: ΔP = 2σ/R
    float P_inside = measure_pressure(droplet_interior);
    float P_outside = measure_pressure(droplet_exterior);
    float dP_measured = P_inside - P_outside;
    float dP_expected = 2.0f * sigma / radius;
    EXPECT_NEAR(dP_measured, dP_expected, 0.1f * dP_expected);
    ```
  - [ ] Add spurious current check:
    ```cpp
    float v_max = measure_max_velocity();
    EXPECT_LT(v_max, 0.01f);  // < 1 cm/s
    ```

- [ ] **Day 7: Integration**
  - [ ] Move to `tests/regression/` directory
  - [ ] Update CMakeLists.txt, add HIGH label
  - [ ] Verify PASS

- [ ] **Completed:** _______ (date) by _______ (name)

---

### Week 2: Medium/Low Tests (Tests 6-10)

#### Test 6: Grid Convergence (MEDIUM)

- [ ] **Day 8: Adapt Existing Script**
  - [ ] File exists: `tests/validation/test_grid_convergence.sh`
  - [ ] Add automated Richardson extrapolation:
    ```bash
    p=$(python3 -c "import math; print(math.log((T_c - T_m)/(T_m - T_f)) / math.log(2))")
    if [ $(echo "$p > 0.5" | bc) -eq 1 ]; then echo "PASS"; else echo "FAIL"; fi
    ```
  - [ ] Document KNOWN ISSUE: p < 0 on RTX 3050 (hardware limited)
  - [ ] Mark as INFORMATIONAL (don't block on failure)

- [ ] **Day 8: Integration**
  - [ ] Add to regression suite with MEDIUM label
  - [ ] Add note: "Expected to fail on limited hardware"
  - [ ] Document: "Run on cloud GPU (AWS P3) for publication"

- [ ] **Completed:** _______ (date) by _______ (name)

---

#### Test 7: Marangoni Benchmark (MEDIUM)

- [ ] **Day 9: Implementation**
  - [ ] Create file: `tests/regression/test_marangoni_benchmark.cu`
  - [ ] Set up 2D channel with thermal gradient
  - [ ] Compute analytical estimate: v ≈ (dσ/dT)·∇T·L/μ
  - [ ] Run simulation, measure v_max
  - [ ] Assert: v_max within ±30% of analytical

- [ ] **Day 9: Integration**
  - [ ] Add to CMakeLists.txt with MEDIUM label
  - [ ] Build and test

- [ ] **Completed:** _______ (date) by _______ (name)

---

#### Test 8: Power Scaling (LOW)

- [ ] **Day 10: Implementation**
  - [ ] Create file: `tests/regression/test_power_scaling.cu`
  - [ ] Run Test 1 at P = 100W, 150W, 200W
  - [ ] Check monotonic: T_max(200W) > T_max(150W) > T_max(100W)
  - [ ] Check scaling: 0.8 < d(log T)/d(log P) < 1.2

- [ ] **Day 10: Integration**
  - [ ] Add to CMakeLists.txt with LOW label

- [ ] **Completed:** _______ (date) by _______ (name)

---

#### Test 9: Extreme Power (LOW)

- [ ] **Day 10: Implementation**
  - [ ] Create file: `tests/regression/test_extreme_power.cu`
  - [ ] Run Test 1 with P = 300W (2× baseline)
  - [ ] Check: No crash, no NaN, T_max < 5000K

- [ ] **Completed:** _______ (date) by _______ (name)

---

#### Test 10: Checkpoint Restart (LOW)

- [ ] **Day 10: Implementation**
  - [ ] Create file: `tests/regression/test_checkpoint_restart.cu`
  - [ ] Run to 150 steps, save checkpoint
  - [ ] Restart, run to 300 steps
  - [ ] Compare to single 300-step run
  - [ ] Assert: T_max matches within roundoff

- [ ] **Completed:** _______ (date) by _______ (name)

---

### Week 2: Suite Integration

- [ ] **Day 11: CMake Integration**
  - [ ] Update `tests/CMakeLists.txt`:
    ```cmake
    # Create regression test target
    add_custom_target(run_regression_tests
        COMMAND ${CMAKE_CTEST_COMMAND} -L regression --output-on-failure
        DEPENDS test_baseline_150W test_stability_500step ...
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    ```
  - [ ] Test: `make run_regression_tests`
  - [ ] Verify: All 10 tests run in < 3 minutes

- [ ] **Day 12: Documentation**
  - [ ] Create `tests/regression/README.md`
  - [ ] Document each test: purpose, acceptance criteria, runtime
  - [ ] Add troubleshooting guide

- [ ] **Day 12: Validation**
  - [ ] Run full suite 3 times, check consistency
  - [ ] Measure total runtime (target: < 5 min)
  - [ ] Update reference values if needed

- [ ] **Phase 1 Complete:** _______ (date) by _______ (name)

---

## Phase 2: CI Pipeline (Weeks 3-4)

**Goal:** Automated testing on every commit
**Duration:** 2 weeks
**Deliverable:** GitHub Actions workflow + self-hosted GPU runner

---

### Week 3: Local Pre-Commit Hook

- [ ] **Day 13: Create Pre-Commit Script**
  - [ ] Create file: `scripts/pre_commit_hook.sh`
    ```bash
    #!/bin/bash
    set -e
    cd /home/yzk/LBMProject/build
    echo "Building..."
    make -j8 > /dev/null || exit 1
    echo "Running critical tests..."
    ctest -L critical --output-on-failure || exit 1
    echo "✓ All checks passed"
    ```
  - [ ] Make executable: `chmod +x scripts/pre_commit_hook.sh`
  - [ ] Test manually: `./scripts/pre_commit_hook.sh`

- [ ] **Day 13: Install Git Hook**
  - [ ] Link to git hooks:
    ```bash
    ln -sf ../../scripts/pre_commit_hook.sh .git/hooks/pre-commit
    ```
  - [ ] Test: Make dummy change, try to commit
  - [ ] Verify: Hook runs before commit completes

- [ ] **Day 13: Documentation**
  - [ ] Document in README: "Tests run automatically on commit"
  - [ ] Add bypass option: `git commit --no-verify` (for emergencies)

- [ ] **Completed:** _______ (date) by _______ (name)

---

### Week 3: GitHub Actions Setup

- [ ] **Day 14: Self-Hosted Runner Setup**
  - [ ] On GPU machine (RTX 3050):
    ```bash
    mkdir -p ~/actions-runner && cd ~/actions-runner
    curl -o actions-runner-linux-x64.tar.gz -L \
      https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
    tar xzf ./actions-runner-linux-x64.tar.gz
    ```
  - [ ] Configure (follow GitHub instructions):
    ```bash
    ./config.sh --url https://github.com/USERNAME/LBMProject --token TOKEN
    ```
  - [ ] Test: `./run.sh`
  - [ ] Verify: Runner shows up in GitHub Settings > Actions > Runners

- [ ] **Day 14: Install as Service**
  - [ ] Install service:
    ```bash
    sudo ./svc.sh install
    sudo ./svc.sh start
    ```
  - [ ] Check status: `sudo ./svc.sh status`
  - [ ] Verify: Runner shows "Listening for Jobs"

- [ ] **Completed:** _______ (date) by _______ (name)

---

### Week 3-4: GitHub Actions Workflow

- [ ] **Day 15: Create Workflow File**
  - [ ] Create `.github/workflows/ci.yml`:
    ```yaml
    name: LBM-CUDA CI

    on:
      push:
        branches: [main, develop]
      pull_request:
        branches: [main]

    jobs:
      build_and_test:
        runs-on: self-hosted

        steps:
          - name: Checkout code
            uses: actions/checkout@v3

          - name: Build
            run: |
              mkdir -p build
              cd build
              cmake -DCMAKE_BUILD_TYPE=Release ..
              make -j8

          - name: Run unit tests
            run: |
              cd build
              ctest -L unit --output-on-failure

          - name: Run regression tests
            run: |
              cd build
              ctest -L regression --output-on-failure

          - name: Performance check
            run: |
              cd build
              ./tests/regression/test_baseline_150W | tee runtime.log
              RUNTIME=$(grep "Runtime" runtime.log | awk '{print $2}')
              if [ "$RUNTIME" -gt 20 ]; then
                echo "WARNING: Performance regression (${RUNTIME}s > 20s)"
              fi
    ```

- [ ] **Day 15: Test Workflow**
  - [ ] Commit and push workflow file
  - [ ] Check GitHub Actions tab
  - [ ] Verify: Workflow triggers on push
  - [ ] Debug any failures

- [ ] **Day 16: Add Status Badge**
  - [ ] Add to README.md:
    ```markdown
    ![CI Status](https://github.com/USERNAME/LBMProject/workflows/LBM-CUDA%20CI/badge.svg)
    ```
  - [ ] Verify badge shows correct status

- [ ] **Completed:** _______ (date) by _______ (name)

---

### Week 4: Nightly Build

- [ ] **Day 17: Create Nightly Script**
  - [ ] Create file: `scripts/nightly_build.sh`
    ```bash
    #!/bin/bash

    # Pull latest
    cd /home/yzk/LBMProject
    git pull origin main

    # Build
    cd build
    make clean
    make -j8

    # Run all tests (including slow validation tests)
    ctest --output-on-failure

    # Generate report
    python3 ../scripts/generate_test_report.py

    # Email if failures
    if [ $? -ne 0 ]; then
        mail -s "LBM-CUDA Nightly Build FAILED" team@example.com < test_report.txt
    fi
    ```

- [ ] **Day 17: Set Up Cron Job**
  - [ ] Edit crontab: `crontab -e`
  - [ ] Add line:
    ```
    0 2 * * * /home/yzk/LBMProject/scripts/nightly_build.sh >> /home/yzk/lbm_nightly.log 2>&1
    ```
  - [ ] Test manually: `./scripts/nightly_build.sh`
  - [ ] Wait until 2 AM, check log: `tail -f ~/lbm_nightly.log`

- [ ] **Day 18: Email Notifications**
  - [ ] Install mailutils: `sudo apt install mailutils`
  - [ ] Configure SMTP (if needed)
  - [ ] Test: `echo "Test" | mail -s "Test" your.email@example.com`
  - [ ] Update script to send reports

- [ ] **Completed:** _______ (date) by _______ (name)

---

### Week 4: Test Reporting

- [ ] **Day 19: Test Report Generator**
  - [ ] Create file: `scripts/generate_test_report.py`
    - Parse CTest XML output
    - Generate markdown report
    - Include pass/fail summary
    - List failed tests with diagnostics
    - Add performance metrics
  - [ ] Test: `python3 scripts/generate_test_report.py`
  - [ ] Verify output: `test_results/latest_report.md`

- [ ] **Day 20: Dashboard (Optional)**
  - [ ] Create file: `scripts/generate_test_dashboard.py`
    - Generate HTML dashboard
    - Embed plots (test history, performance trends)
    - Include latest test results
  - [ ] Test: `python3 scripts/generate_test_dashboard.py`
  - [ ] Open: `xdg-open test_results/dashboard.html`

- [ ] **Day 20: History Tracking**
  - [ ] Create CSV files:
    - `test_results/history.csv` (date, commit, pass/fail counts)
    - `test_results/performance.csv` (date, runtime per test)
  - [ ] Update scripts to append to history
  - [ ] Generate trend plots

- [ ] **Phase 2 Complete:** _______ (date) by _______ (name)

---

## Phase 3: Ongoing Validation (Weeks 5+)

**Goal:** Maintain test suite as code evolves
**Duration:** Ongoing
**Deliverable:** Tests for every new feature

---

### For Each New Feature

- [ ] **Before Coding:**
  - [ ] Write acceptance criteria (quantitative)
  - [ ] Identify validation metrics
  - [ ] Set tolerance thresholds

- [ ] **During Development:**
  - [ ] Write test first (TDD)
  - [ ] Implement feature
  - [ ] Run test, debug until PASS
  - [ ] Run full regression suite

- [ ] **Before Merging:**
  - [ ] Add test to regression suite (if critical)
  - [ ] Update CMakeLists.txt
  - [ ] Document test in README
  - [ ] Update reference values (if baseline changed)
  - [ ] Run CI pipeline, verify PASS

- [ ] **After Merging:**
  - [ ] Monitor nightly builds for regressions
  - [ ] Update documentation if behavior changed

---

## Validation Milestones

### Milestone 1: Regression Suite Operational (End of Week 2)

- [ ] 10 regression tests implemented
- [ ] All tests running in < 5 minutes
- [ ] CMake targets configured
- [ ] Documentation complete

**Success Criteria:**
- `make run_regression_tests` → All PASS
- No manual intervention needed
- Clear failure diagnostics

---

### Milestone 2: CI Pipeline Active (End of Week 4)

- [ ] GitHub Actions workflow configured
- [ ] Self-hosted GPU runner operational
- [ ] Pre-commit hook installed
- [ ] Nightly builds running
- [ ] Test reports generated automatically

**Success Criteria:**
- PR triggers automated tests
- Email sent on nightly failures
- Status badge shows correct state
- Team can see test history

---

### Milestone 3: Feature Validation (Weeks 5-8)

For each improvement (substrate BC, variable μ, etc.):

- [ ] Validation test written
- [ ] Test PASS before feature merge
- [ ] Regression suite updated
- [ ] Documentation updated

**Success Criteria:**
- Each improvement has quantitative validation
- No regressions introduced
- Metrics tracked over time

---

## Resources Needed

### Hardware
- [ ] GPU machine for self-hosted runner (RTX 3050 or better)
- [ ] AWS account for cloud GPU (optional, for fine grid convergence)

### Software
- [ ] CMake 3.18+
- [ ] CUDA 11.0+
- [ ] GoogleTest (auto-downloaded by CMake)
- [ ] Python 3.8+ with packages: pandas, matplotlib, numpy

### Access
- [ ] GitHub repository access (for Actions setup)
- [ ] Server access for nightly builds
- [ ] Email/Slack for notifications

---

## Troubleshooting

### Common Issues

**Problem:** Tests fail with "CUDA out of memory"
- Solution: Reduce grid size in test configs
- Alternative: Run tests sequentially (-j1)

**Problem:** GitHub Actions runner offline
- Check: `sudo systemctl status actions.runner`
- Fix: `sudo systemctl start actions.runner`

**Problem:** Test passes locally, fails in CI
- Check CUDA version: `nvcc --version`
- Check GPU: `nvidia-smi` on runner
- Compare: May be hardware differences

**Problem:** Nightly build email not received
- Test mail: `echo "test" | mail -s "test" your@email.com`
- Check spam folder
- Verify SMTP configuration

---

## Sign-Off

### Phase 1: Regression Suite
- [ ] Implemented by: _______________ Date: _______
- [ ] Reviewed by: _______________ Date: _______
- [ ] Tested by: _______________ Date: _______

### Phase 2: CI Pipeline
- [ ] Implemented by: _______________ Date: _______
- [ ] Reviewed by: _______________ Date: _______
- [ ] Tested by: _______________ Date: _______

### Phase 3: Ongoing Validation
- [ ] Process documented: Yes/No
- [ ] Team trained: Yes/No
- [ ] Handoff complete: Yes/No

---

**Document:** Implementation Checklist for Validation Framework
**Version:** 1.0
**Date:** 2025-11-19
**Estimated Effort:** 4 weeks (full-time developer)
**Status:** Ready for implementation
