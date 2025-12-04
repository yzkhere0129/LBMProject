# LBM Multiphysics Test Documentation Index

## Quick Navigation

This index provides quick access to all test-related documentation for the LBM multiphysics simulation framework.

---

## Primary Documentation (NEW)

### 1. TEST_SUITE_SUMMARY.md
**Location**: `/home/yzk/LBMProject/TEST_SUITE_SUMMARY.md`

**Size**: 16 KB (544 lines)

**Purpose**: Executive summary of the entire test suite

**Contents**:
- Overview of 111 tests across 8 categories
- Quick access commands
- Test category summaries
- Critical test suite (29 tests)
- Success criteria
- Common workflows
- Test statistics

**Use When**: You need a high-level overview or quick reference

---

### 2. TEST_GUIDE.md
**Location**: `/home/yzk/LBMProject/docs/TEST_GUIDE.md`

**Size**: 61 KB (2624 lines)

**Purpose**: Comprehensive test suite documentation

**Contents**:
- Detailed description of all 111 tests
- Physics validated by each test
- How to run each test
- Pass criteria and tolerances
- Expected runtime
- Interpreting results
- Adding new tests
- Troubleshooting guide

**Use When**: You need detailed information about a specific test or test category

**Key Sections**:
- Quick Start (commands)
- Test Organization
- VOF Solver Tests (19 tests)
- Marangoni Effect Tests (11 tests)
- Thermal Solver Tests (14 tests)
- Fluid LBM Tests (12 tests)
- Phase Change Tests (8 tests)
- Multiphysics Coupling Tests (25 tests)
- Energy Conservation Tests (8 tests)
- Stability & Validation Tests (14 tests)
- Running Tests
- Interpreting Results
- Adding New Tests

---

### 3. TEST_QUICK_REFERENCE.md
**Location**: `/home/yzk/LBMProject/docs/TEST_QUICK_REFERENCE.md`

**Size**: 13 KB (432 lines)

**Purpose**: Quick command reference for daily use

**Contents**:
- Quick commands for all test categories
- Test patterns and filters
- Pre-commit/pre-push workflows
- Common failure diagnosis
- Pass criteria summary
- Build commands

**Use When**: You need to quickly run tests during development

**Key Sections**:
- Quick Commands
- Test Categories at a Glance
- Critical Tests (MUST PASS)
- Test by Physics Module
- Test by Speed
- Common Workflows
- Failure Diagnosis

---

### 4. TEST_MATRIX.md
**Location**: `/home/yzk/LBMProject/docs/TEST_MATRIX.md`

**Size**: 20 KB (356 lines)

**Purpose**: Complete test-to-physics mapping matrix

**Contents**:
- Comprehensive table of all tests
- Test type (U/I/V/R), speed (F/M/S), priority (C/H/N)
- Physics equations validated
- Testing methods
- Pass criteria
- Test dependencies
- Coverage summary

**Use When**: You need to understand what physics a specific test validates, or find tests for a specific physical phenomenon

**Key Sections**:
- Test Matrix Legend
- VOF Solver Matrix (19 tests)
- Marangoni Effect Matrix (11 tests)
- Thermal Solver Matrix (14 tests)
- Fluid LBM Matrix (12 tests)
- Phase Change Matrix (8 tests)
- Multiphysics Coupling Matrix (25 tests)
- Energy Conservation Matrix (8 tests)
- Stability & Validation Matrix (14 tests)
- Test Coverage Summary
- Critical Test Dependencies
- Physics Equations Tested

---

## Test Runner Script

### RUN_ALL_PHYSICS_TESTS.sh
**Location**: `/home/yzk/LBMProject/RUN_ALL_PHYSICS_TESTS.sh`

**Size**: 13 KB (338 lines)

**Purpose**: Automated test suite runner

**Features**:
- Runs all 111 tests in organized categories
- Colored output (pass/fail/skip)
- Progress tracking
- Detailed logging
- Summary report generation
- Failed test tracking
- Timeout handling

**Usage**:
```bash
cd /home/yzk/LBMProject
./RUN_ALL_PHYSICS_TESTS.sh
```

**Output**:
- Console: Colored progress and summary
- Report: `/home/yzk/LBMProject/test_reports/test_summary_YYYYMMDD_HHMMSS.txt`
- Failed tests: `/home/yzk/LBMProject/test_reports/failed_tests_YYYYMMDD_HHMMSS.txt`

---

## Existing Test Documentation

### 5. VOF_TEST_SUITE_SUMMARY.md
**Location**: `/home/yzk/LBMProject/tests/VOF_TEST_SUITE_SUMMARY.md`

**Size**: 12 KB

**Purpose**: Detailed VOF test suite documentation (80% → 95% coverage)

**Contents**:
- VOF test files created
- Interface advection tests
- Validation tests (analytical solutions)
- Evaporation coupling tests
- Test coverage analysis
- Physics equations tested

**Use When**: You need deep dive into VOF solver testing

---

### 6. STABILITY_TESTS_README.md
**Location**: `/home/yzk/LBMProject/tests/STABILITY_TESTS_README.md`

**Size**: 12 KB

**Purpose**: Thermal stability regression test documentation

**Contents**:
- Background on stability issues
- Stability fixes implemented
- Test descriptions (flux limiter, temp bounds, omega reduction)
- Running stability tests
- Debugging failed tests
- Acceptance criteria

**Use When**: Debugging thermal stability issues

---

### 7. WEEK1_TEST_QUICK_REFERENCE.md
**Location**: `/home/yzk/LBMProject/tests/WEEK1_TEST_QUICK_REFERENCE.md`

**Size**: 8.5 KB

**Purpose**: Quick reference for Week 1 fix tests

**Contents**:
- Evaporation fix tests
- Substrate cooling BC tests
- Quick run commands

---

### 8. TEST_ARCHITECTURE.md
**Location**: `/home/yzk/LBMProject/docs/TEST_ARCHITECTURE.md`

**Size**: 35 KB (1092 lines)

**Purpose**: Test framework architecture and design

**Contents**:
- Test organization philosophy
- Test framework structure
- Best practices
- Test patterns
- Infrastructure

**Use When**: Designing new tests or understanding test framework

---

### 9. TEST_CASES_ENERGY_VERIFICATION.md
**Location**: `/home/yzk/LBMProject/docs/TEST_CASES_ENERGY_VERIFICATION.md`

**Size**: 9.2 KB

**Purpose**: Energy verification test cases

---

### 10. TEST_E_EXECUTIVE_SUMMARY.md
**Location**: `/home/yzk/LBMProject/docs/TEST_E_EXECUTIVE_SUMMARY.md`

**Size**: 12 KB

**Purpose**: Test E executive summary (specific test campaign)

---

### 11. TEST_E_FAILURE_ANALYSIS.md
**Location**: `/home/yzk/LBMProject/docs/TEST_E_FAILURE_ANALYSIS.md`

**Size**: 20 KB

**Purpose**: Failure analysis for Test E

---

## Documentation Usage Guide

### For Daily Development
**Read**: `TEST_QUICK_REFERENCE.md`
- Quick commands
- Pre-commit workflow
- Common patterns

### For Understanding a Specific Test
**Read**: `TEST_GUIDE.md` → Find test category → Read test description

### For Finding Tests for a Physics Phenomenon
**Read**: `TEST_MATRIX.md` → Search for physics equation → Find relevant tests

### For High-Level Overview
**Read**: `TEST_SUITE_SUMMARY.md`

### For Adding New Tests
**Read**:
1. `TEST_ARCHITECTURE.md` (design patterns)
2. `TEST_GUIDE.md` (section: "Adding New Tests")

### For Debugging Test Failures
**Read**:
1. `TEST_QUICK_REFERENCE.md` (section: "Failure Diagnosis")
2. `TEST_GUIDE.md` (section: "Interpreting Results")
3. `STABILITY_TESTS_README.md` (if thermal stability issue)

### For Running Full Test Suite
**Run**: `./RUN_ALL_PHYSICS_TESTS.sh`
**Read**: `TEST_SUITE_SUMMARY.md` for interpretation

---

## Quick Command Reference

### Run All Tests
```bash
cd /home/yzk/LBMProject
./RUN_ALL_PHYSICS_TESTS.sh
```

### Pre-Commit (< 30s)
```bash
cd /home/yzk/LBMProject/build
ctest -R "flux_limiter|temperature_bounds|omega_reduction" --output-on-failure
```

### Pre-Push (< 2min)
```bash
ctest -R "high_pe_stability|test_vof_mass_conservation" --output-on-failure
```

### Run by Category
```bash
ctest -R "vof" --output-on-failure           # VOF tests
ctest -R "marangoni" --output-on-failure     # Marangoni tests
ctest -R "thermal" --output-on-failure       # Thermal tests
ctest -R "fluid" --output-on-failure         # Fluid tests
ctest -R "energy" --output-on-failure        # Energy tests
```

### Run Critical Tests Only
```bash
ctest -L critical --output-on-failure
```

---

## Test Suite Statistics

| Category | Tests | Documentation |
|----------|-------|---------------|
| VOF Solver | 19 | TEST_GUIDE.md, VOF_TEST_SUITE_SUMMARY.md |
| Marangoni | 11 | TEST_GUIDE.md |
| Thermal | 14 | TEST_GUIDE.md, STABILITY_TESTS_README.md |
| Fluid LBM | 12 | TEST_GUIDE.md |
| Phase Change | 8 | TEST_GUIDE.md |
| Multiphysics | 25 | TEST_GUIDE.md |
| Energy | 8 | TEST_GUIDE.md, TEST_CASES_ENERGY_VERIFICATION.md |
| Stability | 14 | TEST_GUIDE.md, STABILITY_TESTS_README.md |
| **TOTAL** | **111** | **11 documents** |

---

## Documentation Hierarchy

```
Level 1: Executive Summary
    └─ TEST_SUITE_SUMMARY.md (16 KB)
         ├─ Overview of all 111 tests
         ├─ Quick access commands
         └─ Test categories

Level 2: Quick Reference (Daily Use)
    └─ TEST_QUICK_REFERENCE.md (13 KB)
         ├─ Quick commands
         ├─ Common workflows
         └─ Failure diagnosis

Level 3: Comprehensive Guide
    └─ TEST_GUIDE.md (61 KB)
         ├─ Detailed test descriptions
         ├─ Physics validated
         ├─ Running tests
         ├─ Interpreting results
         └─ Adding new tests

Level 4: Reference Tables
    └─ TEST_MATRIX.md (20 KB)
         ├─ Complete test matrix
         ├─ Test-physics mapping
         └─ Dependencies

Level 5: Specialized Documentation
    ├─ VOF_TEST_SUITE_SUMMARY.md (12 KB)
    ├─ STABILITY_TESTS_README.md (12 KB)
    ├─ TEST_ARCHITECTURE.md (35 KB)
    └─ Other specialized docs
```

---

## File Locations Summary

### Root Directory
- `/home/yzk/LBMProject/RUN_ALL_PHYSICS_TESTS.sh` - Test runner script
- `/home/yzk/LBMProject/TEST_SUITE_SUMMARY.md` - Executive summary

### Documentation Directory
- `/home/yzk/LBMProject/docs/TEST_GUIDE.md` - Comprehensive guide
- `/home/yzk/LBMProject/docs/TEST_QUICK_REFERENCE.md` - Quick reference
- `/home/yzk/LBMProject/docs/TEST_MATRIX.md` - Test matrix
- `/home/yzk/LBMProject/docs/TEST_ARCHITECTURE.md` - Architecture
- `/home/yzk/LBMProject/docs/TEST_DOCUMENTATION_INDEX.md` - This file

### Tests Directory
- `/home/yzk/LBMProject/tests/VOF_TEST_SUITE_SUMMARY.md` - VOF tests
- `/home/yzk/LBMProject/tests/STABILITY_TESTS_README.md` - Stability tests
- `/home/yzk/LBMProject/tests/WEEK1_TEST_QUICK_REFERENCE.md` - Week 1 tests

### Reports Directory
- `/home/yzk/LBMProject/test_reports/test_summary_*.txt` - Test reports
- `/home/yzk/LBMProject/test_reports/failed_tests_*.txt` - Failed test logs

---

## Contact & Support

For test-related questions:

1. **Quick questions**: Check `TEST_QUICK_REFERENCE.md`
2. **Test details**: Check `TEST_GUIDE.md`
3. **Physics mapping**: Check `TEST_MATRIX.md`
4. **Stability issues**: Check `STABILITY_TESTS_README.md`
5. **Architecture/design**: Check `TEST_ARCHITECTURE.md`

---

## Updates

**Version 1.0** - 2025-12-02
- Created comprehensive test suite documentation
- 111 tests across 8 categories
- 4 new primary documentation files
- Automated test runner script
- Complete test-physics mapping

**Total Documentation**: 11 files, ~160 KB, ~6800 lines

---

**End of Index**
