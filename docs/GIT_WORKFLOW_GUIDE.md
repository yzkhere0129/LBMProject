# Git 工作流程完全指南
## 面向科学计算/CFD项目的版本控制实践

---

## 目录

1. [基础概念](#1-基础概念)
2. [日常工作流](#2-日常工作流)
3. [版本回退](#3-版本回退)
4. [分支管理](#4-分支管理)
5. [协作开发](#5-协作开发)
6. [常见问题](#6-常见问题)
7. [最佳实践](#7-最佳实践)

---

## 1. 基础概念

### 1.1 什么是Git？

Git是一个分布式版本控制系统,专为管理源代码历史而设计。在科学计算项目中,Git帮助你:
- **追踪代码变化**: 记录每次修改的内容、时间和原因
- **协作开发**: 多人同时工作而不相互干扰
- **回退错误**: 轻松恢复到之前的工作状态
- **实验新想法**: 使用分支安全地测试新算法

### 1.2 三个工作区域

```
┌─────────────────────────────────────────────────────────────┐
│  工作目录 (Working Directory)                                │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  你实际编辑文件的地方                                   │  │
│  │  - src/solver.cu                                       │  │
│  │  - config/simulation.yaml                              │  │
│  │  - tests/test_lbm.cpp                                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                      │                                       │
│                      │ git add                               │
│                      ↓                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  暂存区 (Staging Area / Index)                         │  │
│  │  准备提交的文件快照                                     │  │
│  │  - 已修改的 solver.cu                                   │  │
│  │  - 新增的测试文件                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                      │                                       │
│                      │ git commit                            │
│                      ↓                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  本地仓库 (.git directory)                             │  │
│  │  永久保存的项目历史                                     │  │
│  │  [commit 1] → [commit 2] → [commit 3] → HEAD          │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 核心概念解释

- **工作目录**: 你的项目文件夹,包含所有源码、配置、数据文件
- **暂存区**: 临时存储你想要提交的改动
- **本地仓库**: `.git`文件夹,包含完整的项目历史
- **远程仓库**: GitHub/GitLab上的副本,用于团队协作
- **提交(Commit)**: 项目的一个快照,包含改动、作者、时间和说明
- **分支(Branch)**: 独立的开发线,用于隔离不同的工作
- **HEAD**: 当前所在分支的指针

---

## 2. 日常工作流

### 2.1 完整的开发循环

```bash
# 1. 查看当前状态
git status

# 2. 查看具体改动
git diff                    # 工作区 vs 暂存区
git diff --staged           # 暂存区 vs 上次提交

# 3. 添加文件到暂存区
git add src/solver.cu                    # 添加单个文件
git add src/                             # 添加整个目录
git add -u                               # 添加所有已跟踪的修改文件
git add -A                               # 添加所有改动(新增+修改+删除)

# 4. 提交改动
git commit -m "fix: correct thermal boundary condition in LBM solver"

# 5. 查看提交历史
git log --oneline --graph --decorate --all
```

### 2.2 实际工作示例

**场景**: 修复了Marangoni对流的计算错误

```bash
# 步骤1: 检查当前状态
$ git status
On branch feature/marangoni-fix
Changes not staged for commit:
  modified:   src/physics/MarangoniForce.cu
  modified:   tests/test_marangoni.cpp

Untracked files:
  docs/MARANGONI_BUG_REPORT.md

# 步骤2: 查看代码改动
$ git diff src/physics/MarangoniForce.cu
-    float sigma_gradient = (sigma_right - sigma_left) / (2.0f * dx);
+    float sigma_gradient = computeSurfaceTensionGradient(T, pos);

# 步骤3: 分阶段提交
# 先提交代码修复
git add src/physics/MarangoniForce.cu tests/test_marangoni.cpp
git commit -m "fix: correct Marangoni force calculation with proper gradient

- Replace simple finite difference with accurate surface tension gradient
- Add temperature-dependent surface tension coefficient
- Fixes issue #42: incorrect flow direction in thermal gradient"

# 再提交文档
git add docs/MARANGONI_BUG_REPORT.md
git commit -m "docs: add bug report for Marangoni force issue"

# 步骤4: 查看提交历史
$ git log --oneline -3
a1b2c3d docs: add bug report for Marangoni force issue
e4f5g6h fix: correct Marangoni force calculation with proper gradient
i7j8k9l feat: implement evaporation cooling model
```

### 2.3 常用状态检查命令

```bash
# 查看简洁状态
git status -s

# 输出示例:
# M  src/solver.cu          # 已修改,已暂存
#  M config/params.yaml     # 已修改,未暂存
# ?? output/results.vtk     # 未跟踪文件

# 查看详细的提交历史
git log --stat              # 显示每次提交的文件变化统计
git log -p                  # 显示每次提交的具体改动
git log --since="2 weeks"   # 最近两周的提交

# 查看特定文件的历史
git log --follow src/physics/ThermalLBM.cu
git log -p src/physics/ThermalLBM.cu    # 包含改动内容

# 查看谁修改了某行代码
git blame src/solver.cu
```

---

## 3. 版本回退

### 3.1 回退场景分类

```
回退类型决策树:

改动在哪里?
├─ 仅在工作区(未git add)
│   └─> 使用 git restore <file> 或 git checkout -- <file>
│
├─ 已在暂存区(已git add,未commit)
│   └─> 使用 git restore --staged <file> 或 git reset HEAD <file>
│
├─ 已提交到本地(已commit,未push)
│   ├─> 想保留改动: git reset --soft HEAD~1
│   ├─> 想放弃改动: git reset --hard HEAD~1
│   └─> 想修改提交信息: git commit --amend
│
└─ 已推送到远程(已push)
    ├─> 回退最近提交: git revert <commit>
    └─> 回退多个提交: 创建新分支或联系团队
```

### 3.2 详细回退方法

#### 3.2.1 撤销工作区改动

```bash
# 场景: 修改了solver.cu,但想丢弃所有改动
git restore src/solver.cu              # Git 2.23+新语法
# 或
git checkout -- src/solver.cu          # 旧语法

# 恢复所有工作区文件
git restore .

# 场景: 误删除了文件,想恢复
git restore tests/test_thermal.cpp
```

#### 3.2.2 撤销暂存区

```bash
# 场景: 不小心把大型输出文件加入暂存区
git restore --staged output/large_result.vtk    # Git 2.23+
# 或
git reset HEAD output/large_result.vtk          # 旧语法

# 取消所有暂存
git restore --staged .
# 或
git reset HEAD
```

#### 3.2.3 回退提交(未推送)

```bash
# 场景1: 提交信息写错了,想重新写
git commit --amend -m "fix: correct buoyancy force sign error"

# 场景2: 忘记添加某个文件到上次提交
git add forgotten_file.cu
git commit --amend --no-edit         # 不修改提交信息

# 场景3: 想回退到上一个提交,保留改动
git reset --soft HEAD~1
# 现在改动又回到暂存区,可以重新组织提交

# 场景4: 想回退到上一个提交,丢弃所有改动
git reset --hard HEAD~1
# ⚠️  警告: 改动将永久丢失!

# 场景5: 回退到更早的提交
git reset --hard a1b2c3d           # 回退到特定commit
git reset --hard HEAD~3            # 回退3个提交
```

#### 3.2.4 回退提交(已推送)

```bash
# 使用revert创建新提交来撤销改动(推荐)
git revert a1b2c3d                 # 撤销特定提交
git revert HEAD                    # 撤销最新提交
git revert HEAD~2..HEAD            # 撤销最近3个提交

# Revert示例
$ git log --oneline
a1b2c3d (HEAD) feat: add laser scanning
e4f5g6h fix: thermal boundary bug
i7j8k9l feat: implement VOF

$ git revert a1b2c3d
# 创建新提交,撤销a1b2c3d的改动

$ git log --oneline
m9n0o1p (HEAD) Revert "feat: add laser scanning"
a1b2c3d feat: add laser scanning
e4f5g6h fix: thermal boundary bug
```

### 3.3 紧急救援

```bash
# 场景: 误执行了git reset --hard,想恢复
# Git保留了操作历史(reflog)

# 1. 查看所有操作历史
git reflog
# 输出:
# a1b2c3d HEAD@{0}: reset: moving to HEAD~1
# e4f5g6h HEAD@{1}: commit: fix thermal bug
# i7j8k9l HEAD@{2}: commit: add new feature

# 2. 恢复到误操作前
git reset --hard HEAD@{1}
# 或
git reset --hard e4f5g6h

# reflog保留约90天的历史
```

---

## 4. 分支管理

### 4.1 分支策略

```
main (生产代码)
├─ develop (开发分支)
│   ├─ feature/marangoni-convection (功能开发)
│   ├─ feature/laser-scanning (功能开发)
│   └─ bugfix/thermal-boundary (Bug修复)
└─ hotfix/critical-nan-issue (紧急修复)
```

### 4.2 基本分支操作

```bash
# 创建并切换到新分支
git checkout -b feature/evaporation-model
# 等同于:
# git branch feature/evaporation-model
# git checkout feature/evaporation-model

# 查看所有分支
git branch                  # 本地分支
git branch -r               # 远程分支
git branch -a               # 所有分支

# 切换分支
git checkout develop
git checkout main

# 删除分支
git branch -d feature/old-feature      # 安全删除(已合并)
git branch -D feature/abandoned        # 强制删除(未合并)

# 重命名分支
git branch -m old-name new-name
```

### 4.3 功能开发完整流程

```bash
# 场景: 开发新的蒸发冷却模型

# 1. 从develop创建功能分支
git checkout develop
git pull origin develop                 # 确保是最新代码
git checkout -b feature/evaporation-cooling

# 2. 开发工作(可能多次提交)
# ... 编辑代码 ...
git add src/physics/EvaporationModel.cu
git commit -m "feat: implement basic evaporation physics"

# ... 继续开发 ...
git add src/physics/EvaporationModel.cu tests/test_evaporation.cpp
git commit -m "feat: add evaporation cooling energy sink"

git add docs/EVAPORATION_IMPLEMENTATION.md
git commit -m "docs: document evaporation model parameters"

# 3. 保持与develop同步(重要!)
git checkout develop
git pull origin develop
git checkout feature/evaporation-cooling
git merge develop                       # 或使用rebase(见下文)

# 4. 完成开发,准备合并
git checkout develop
git merge feature/evaporation-cooling
git push origin develop

# 5. 清理分支
git branch -d feature/evaporation-cooling
git push origin --delete feature/evaporation-cooling
```

### 4.4 Merge vs Rebase

```bash
# Merge(保留完整历史)
git checkout develop
git merge feature/laser-scanning

# 优点: 保留所有提交历史,冲突解决简单
# 缺点: 历史图谱复杂,有额外的merge提交

# 提交历史示例:
#     D---E---F  feature/laser-scanning
#    /         \
# A---B---C-----G  develop

# Rebase(线性历史)
git checkout feature/laser-scanning
git rebase develop

# 优点: 干净的线性历史,容易追踪
# 缺点: 改写历史(不要rebase已推送的提交!)

# 提交历史示例:
# A---B---C---D'---E'---F'  feature/laser-scanning
#                           develop

# 何时用Merge?
# - 功能分支合并回主分支
# - 多人协作的分支
# - 想保留分支开发历史

# 何时用Rebase?
# - 更新本地功能分支
# - 清理个人提交历史
# - 创建线性项目历史
```

### 4.5 Bug修复流程

```bash
# 紧急Bug: NaN值导致模拟崩溃

# 1. 从main创建hotfix分支
git checkout main
git checkout -b hotfix/nan-in-temperature-field

# 2. 修复bug
# ... 编辑 src/physics/ThermalLBM.cu ...
git add src/physics/ThermalLBM.cu
git commit -m "fix: add NaN check in thermal collision operator

- Add isnan() validation before temperature update
- Clamp extreme values to physical bounds
- Fixes #89: simulation crash with high laser power"

# 3. 测试修复
# ... 运行测试 ...
git add tests/test_thermal_stability.cpp
git commit -m "test: add NaN detection test case"

# 4. 合并到main和develop
git checkout main
git merge hotfix/nan-in-temperature-field
git push origin main

git checkout develop
git merge hotfix/nan-in-temperature-field
git push origin develop

# 5. 清理
git branch -d hotfix/nan-in-temperature-field
```

---

## 5. 协作开发

### 5.1 远程仓库基础

```bash
# 查看远程仓库
git remote -v
# 输出示例:
# origin  https://github.com/user/LBMProject.git (fetch)
# origin  https://github.com/user/LBMProject.git (push)

# 添加远程仓库
git remote add origin https://github.com/user/LBMProject.git
git remote add upstream https://github.com/original/LBMProject.git

# 修改远程仓库URL
git remote set-url origin git@github.com:user/LBMProject.git

# 删除远程仓库
git remote remove upstream
```

### 5.2 推送和拉取

```bash
# 首次推送分支
git push -u origin feature/new-model
# -u 设置upstream,之后只需git push

# 日常推送
git push                              # 推送当前分支
git push origin main                  # 推送到特定远程分支

# 拉取更新
git pull                              # 拉取并合并
git pull --rebase                     # 拉取并rebase(推荐)

# 拉取详解(pull = fetch + merge)
git fetch origin                      # 仅下载,不合并
git merge origin/develop              # 手动合并

# 查看远程分支
git branch -r
git remote show origin
```

### 5.3 冲突解决

```bash
# 场景: 合并时出现冲突

$ git merge feature/laser-model
Auto-merging src/config/SimulationParams.h
CONFLICT (content): Merge conflict in src/config/SimulationParams.h
Automatic merge failed; fix conflicts and then commit the result.

# 1. 查看冲突文件
$ git status
Unmerged paths:
  both modified:   src/config/SimulationParams.h

# 2. 编辑冲突文件
# 文件内容示例:
<<<<<<< HEAD
float laser_power = 200.0f;  // 你的改动
=======
float laser_power = 150.0f;  // 他人的改动
>>>>>>> feature/laser-model

# 3. 手动解决冲突
# 删除冲突标记,保留正确的代码:
float laser_power = 200.0f;  // 使用更高功率

# 4. 标记为已解决
git add src/config/SimulationParams.h

# 5. 完成合并
git commit -m "merge: resolve laser power conflict, use 200W"

# 取消合并(如果搞砸了)
git merge --abort
```

### 5.4 冲突解决策略

```bash
# 策略1: 使用mergetool(图形化工具)
git mergetool
# 配置mergetool:
git config --global merge.tool vimdiff
git config --global merge.tool meld      # Linux
git config --global merge.tool kdiff3    # 跨平台

# 策略2: 选择一方的改动
git checkout --ours src/config/params.h     # 使用我们的版本
git checkout --theirs src/config/params.h   # 使用他们的版本
git add src/config/params.h

# 策略3: 查看冲突来源
git log --merge                           # 查看冲突相关提交
git diff                                  # 查看冲突细节
```

### 5.5 协作最佳实践

```bash
# 每日工作流程

# 早晨: 同步最新代码
git checkout develop
git pull --rebase origin develop
git checkout feature/my-feature
git rebase develop

# 工作中: 频繁小提交
git add -p                                # 交互式添加改动片段
git commit -m "refactor: extract thermal kernel"

# 晚上: 推送工作进度
git push origin feature/my-feature

# 合并前: 清理提交历史
git rebase -i HEAD~5                      # 交互式rebase
# 可以squash(合并)、reword(改写)、drop(删除)提交
```

---

## 6. 常见问题

### 问题1: 不小心提交了大文件

```bash
# 症状: push被拒绝,提示文件过大
$ git push
error: file output/simulation_100GB.vtk exceeds GitHub's file size limit of 100 MB

# 解决方案1: 从最后一次提交移除
git reset --soft HEAD~1
git restore --staged output/simulation_100GB.vtk
echo "output/*.vtk" >> .gitignore
git add .gitignore
git commit -m "chore: ignore large VTK output files"

# 解决方案2: 已推送多次,需要重写历史(危险!)
# 使用git-filter-repo工具
pip install git-filter-repo
git filter-repo --path output/simulation_100GB.vtk --invert-paths
git push --force origin main              # 需要团队协调
```

### 问题2: 误删除了分支

```bash
# 使用reflog恢复
git reflog
# 找到分支的最后一次提交,例如 a1b2c3d

# 重建分支
git checkout -b recovered-branch a1b2c3d
```

### 问题3: 提交到了错误的分支

```bash
# 场景: 应该在feature分支提交,却提交到了develop

# 在develop分支:
git log --oneline -1                      # 记下错误提交的hash: a1b2c3d

# 回退develop
git reset --hard HEAD~1

# 切换到正确分支并应用
git checkout feature/correct-branch
git cherry-pick a1b2c3d
```

### 问题4: 想临时保存工作切换分支

```bash
# 使用stash
git stash                                 # 保存当前工作
git stash save "WIP: thermal model refactor"  # 带说明保存

# 切换分支处理紧急任务
git checkout hotfix/critical-bug
# ... 修复bug ...

# 回到原分支,恢复工作
git checkout feature/thermal-model
git stash pop                             # 应用并删除stash
# 或
git stash apply                           # 仅应用,保留stash

# 查看stash列表
git stash list
# stash@{0}: WIP: thermal model refactor
# stash@{1}: WIP: laser scanning implementation

# 应用特定stash
git stash apply stash@{1}

# 清理stash
git stash drop stash@{0}                  # 删除特定stash
git stash clear                           # 删除所有stash
```

### 问题5: 合并后发现引入了Bug

```bash
# 使用bisect二分查找有问题的提交
git bisect start
git bisect bad                            # 当前版本有bug
git bisect good a1b2c3d                   # 已知好的版本

# Git会自动切换到中间提交
# 测试当前版本
./build/tests
# 如果有bug:
git bisect bad
# 如果正常:
git bisect good

# 重复直到找到第一个坏提交
# 结束bisect
git bisect reset
```

### 问题6: 想修改历史提交信息

```bash
# 修改最近一次提交
git commit --amend

# 修改更早的提交(例如最近3个提交中的某个)
git rebase -i HEAD~3

# 交互式界面示例:
# pick a1b2c3d feat: add laser model
# reword e4f5g6h fix: thermal bug          # 改为reword
# pick i7j8k9l docs: update README

# 保存后会打开编辑器让你修改e4f5g6h的提交信息
```

### 问题7: 想拆分一个大提交

```bash
# 回到提交之前
git reset HEAD~1

# 交互式添加不同部分
git add -p src/physics/Thermal.cu
# 选择要添加的改动片段(y/n/s)

git commit -m "fix: correct thermal diffusion"

git add src/physics/Fluid.cu
git commit -m "feat: add viscosity model"
```

### 问题8: 远程分支已删除,本地还有

```bash
# 清理本地对远程分支的引用
git fetch --prune

# 或在拉取时自动清理
git pull --prune
```

### 问题9: 想查看某个文件的历史版本

```bash
# 查看文件在特定提交时的内容
git show a1b2c3d:src/solver.cu

# 恢复文件到历史版本
git checkout a1b2c3d -- src/solver.cu
```

### 问题10: 想合并特定提交到其他分支

```bash
# Cherry-pick特定提交
git checkout target-branch
git cherry-pick a1b2c3d                   # 应用单个提交
git cherry-pick e4f5g6h i7j8k9l           # 应用多个提交
git cherry-pick a1b2c3d..i7j8k9l          # 应用提交范围
```

---

## 7. 最佳实践

### 7.1 提交信息规范

遵循约定式提交(Conventional Commits):

```
<类型>(<范围>): <简短描述>

<详细描述>

<footer>
```

#### 类型(必需)

- **feat**: 新功能
- **fix**: Bug修复
- **docs**: 文档变更
- **refactor**: 代码重构(不改变功能)
- **perf**: 性能优化
- **test**: 添加或修改测试
- **chore**: 构建过程或辅助工具变动
- **style**: 代码格式(不影响代码含义)

#### 实际示例

```bash
# 好的提交信息
git commit -m "feat(thermal): implement radiation boundary condition

Add Stefan-Boltzmann radiation heat loss at domain boundaries:
- Emissivity-dependent heat flux calculation
- Non-linear boundary condition solver
- Validated against analytical solution

Refs #123"

git commit -m "fix(lbm): correct streaming step for D3Q19 scheme

The streaming kernel was accessing incorrect neighbor indices
for diagonal directions (indices 14-18). Updated index mapping
based on standard D3Q19 lattice structure.

Fixes #156"

git commit -m "perf(cuda): optimize thermal collision kernel

- Reduce global memory access by 40%
- Use shared memory for temporary storage
- Measured 2.3x speedup on RTX 3090

Benchmark results in docs/PERFORMANCE.md"

# 不好的提交信息(避免)
git commit -m "update code"               # 太模糊
git commit -m "fix bug"                   # 没说什么bug
git commit -m "changes"                   # 毫无信息
```

### 7.2 .gitignore配置

针对CUDA/CFD项目的完整`.gitignore`:

```bash
# 编译输出
build/
*.o
*.obj
*.exe
*.out
*.a
*.so
*.dll
*.dylib

# CUDA编译文件
*.cubin
*.fatbin
*.ptx

# CMake
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
Makefile
*.cmake

# 仿真输出(大文件!)
output/*.vtk
output/*.vtu
output/*.pvd
results/**/*.h5
results/**/*.dat
*.csv

# 但保留示例文件
!output/example.vtk
!docs/images/*.png

# IDE配置
.vscode/
.idea/
*.swp
*.swo
*~

# Python(如果用于后处理)
__pycache__/
*.pyc
.ipynb_checkpoints/

# 日志和临时文件
*.log
tmp/
temp/

# OS文件
.DS_Store
Thumbs.db

# 机密信息
config/credentials.yaml
*.key
*.pem
```

### 7.3 标签管理

```bash
# 轻量标签(指向提交的指针)
git tag v1.0.0

# 附注标签(推荐,包含完整信息)
git tag -a v1.0.0 -m "Release version 1.0.0: First stable release

Features:
- Thermal LBM solver with phase change
- Marangoni convection
- Evaporation cooling
- Laser heat source with scanning

Validated against experimental data from:
King et al. (2015), Metall. Trans. A, 46(6):2735-2749"

# 给历史提交打标签
git tag -a v0.9.0 a1b2c3d -m "Beta release"

# 查看标签
git tag
git tag -l "v1.*"                        # 筛选标签

# 查看标签详情
git show v1.0.0

# 推送标签到远程
git push origin v1.0.0                   # 单个标签
git push origin --tags                   # 所有标签

# 删除标签
git tag -d v0.9.0                        # 删除本地
git push origin --delete v0.9.0          # 删除远程

# 检出特定版本
git checkout v1.0.0
```

### 7.4 提交频率与粒度

```bash
# 原则: 小步快跑,原子提交

# 好的做法:
# 提交1: 实现核心算法
git add src/physics/MarangoniKernel.cu
git commit -m "feat: implement Marangoni kernel"

# 提交2: 添加测试
git add tests/test_marangoni.cpp
git commit -m "test: add Marangoni force unit tests"

# 提交3: 更新文档
git add docs/PHYSICS_MODELS.md
git commit -m "docs: document Marangoni implementation"

# 不好的做法:
# 一次提交包含功能、测试、文档、重构、Bug修复...
git add .
git commit -m "update marangoni and fix some bugs"
```

### 7.5 分支命名约定

```bash
# 功能开发
feature/evaporation-cooling
feature/laser-scanning
feature/gpu-optimization

# Bug修复
bugfix/nan-temperature
bugfix/boundary-condition-error
fix/memory-leak-in-solver

# 紧急修复
hotfix/critical-crash
hotfix/data-corruption

# 实验性工作
experiment/new-lbm-scheme
experiment/ml-acceleration

# 文档
docs/user-manual
docs/api-reference

# 版本发布
release/v1.0.0
release/v1.1.0
```

### 7.6 代码审查检查清单

提交前自检:

```bash
# 1. 代码质量
□ 代码已格式化(clang-format)
□ 没有调试用的print语句
□ 没有注释掉的代码
□ 变量命名清晰有意义

# 2. 功能验证
□ 本地测试通过
□ 新功能有单元测试
□ 文档已更新

# 3. Git相关
□ 提交信息描述清晰
□ 没有包含大文件(VTK, HDF5)
□ .gitignore已更新
□ 一次提交只做一件事

# 自动检查脚本
#!/bin/bash
# pre-commit-check.sh

echo "Running pre-commit checks..."

# 检查大文件
large_files=$(find . -type f -size +10M -not -path "./.git/*")
if [ -n "$large_files" ]; then
    echo "Error: Large files detected:"
    echo "$large_files"
    exit 1
fi

# 运行测试
make test
if [ $? -ne 0 ]; then
    echo "Error: Tests failed"
    exit 1
fi

# 代码格式检查
clang-format --dry-run --Werror src/**/*.cu
if [ $? -ne 0 ]; then
    echo "Error: Code formatting issues"
    exit 1
fi

echo "All checks passed!"
```

### 7.7 团队协作指南

```bash
# 工作流程约定

# 1. 主分支保护
# main: 仅通过PR合并,需要code review
# develop: 日常开发,定期合并到main

# 2. Pull Request流程
# a. 创建功能分支
git checkout -b feature/new-model

# b. 开发并推送
git push -u origin feature/new-model

# c. 在GitHub创建PR
#    - 填写PR模板
#    - 关联相关Issue
#    - 请求审查者

# d. 代码审查
#    - 至少一人approve
#    - CI测试通过
#    - 无merge冲突

# e. 合并后清理
git checkout develop
git pull
git branch -d feature/new-model
git push origin --delete feature/new-model

# 3. 同步分支
# 每天开始工作前
git checkout develop
git pull --rebase origin develop

# 功能分支定期同步
git checkout feature/my-feature
git rebase develop
```

### 7.8 性能和存储优化

```bash
# 定期清理仓库
git gc --aggressive --prune=now

# 查看仓库大小
git count-objects -vH

# 找出大文件
git rev-list --objects --all | grep "$(git verify-pack -v .git/objects/pack/*.idx | sort -k 3 -n | tail -10 | awk '{print$1}')"

# 使用Git LFS管理大文件
git lfs install
git lfs track "*.vtk"
git lfs track "*.h5"
git add .gitattributes
git commit -m "chore: add LFS tracking for large output files"

# 查看LFS文件
git lfs ls-files
```

---

## 附录A: 快速参考卡片

### 常用命令速查

```bash
# 基础操作
git init                    # 初始化仓库
git clone <url>             # 克隆仓库
git status                  # 查看状态
git add <file>              # 添加文件
git commit -m "msg"         # 提交
git push                    # 推送
git pull                    # 拉取

# 分支
git branch                  # 列出分支
git branch <name>           # 创建分支
git checkout <name>         # 切换分支
git checkout -b <name>      # 创建并切换
git merge <branch>          # 合并分支
git branch -d <name>        # 删除分支

# 历史
git log                     # 查看历史
git log --oneline           # 简洁历史
git log --graph             # 图形化历史
git diff                    # 查看改动
git show <commit>           # 查看提交

# 撤销
git restore <file>          # 撤销工作区
git restore --staged <f>    # 撤销暂存
git reset --soft HEAD~1     # 软回退
git reset --hard HEAD~1     # 硬回退
git revert <commit>         # 反向提交

# 远程
git remote -v               # 查看远程
git fetch                   # 获取远程
git push origin <branch>    # 推送分支
git pull origin <branch>    # 拉取分支
```

### Git配置建议

```bash
# 用户信息
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# 编辑器
git config --global core.editor "vim"

# 默认分支名
git config --global init.defaultBranch main

# 颜色输出
git config --global color.ui auto

# 别名(提高效率)
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'restore --staged'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual 'log --oneline --graph --decorate --all'

# 使用示例
git st                      # = git status
git visual                  # = git log --oneline --graph --decorate --all
```

---

## 附录B: 故障排除

### 常见错误信息

```bash
# 错误1: "Your branch is ahead of 'origin/main' by X commits"
# 含义: 本地有未推送的提交
# 解决: git push origin main

# 错误2: "Your branch is behind 'origin/main' by X commits"
# 含义: 远程有新提交
# 解决: git pull origin main

# 错误3: "Please commit your changes or stash them before you merge"
# 含义: 有未提交的改动
# 解决:
git stash
git merge <branch>
git stash pop

# 错误4: "fatal: refusing to merge unrelated histories"
# 含义: 两个仓库没有共同历史
# 解决: git pull origin main --allow-unrelated-histories

# 错误5: "error: failed to push some refs"
# 含义: 远程分支有新提交
# 解决:
git pull --rebase
git push
```

---

## 结语

Git是强大的工具,但也需要实践来熟练掌握。记住:

1. **频繁提交**: 小步前进,容易回退
2. **清晰描述**: 好的提交信息是未来的自己的感谢信
3. **保持同步**: 及早发现冲突,及早解决
4. **勇于实验**: 有Git在,不用害怕犯错
5. **团队协作**: 沟通比技术更重要

遇到问题时,记住三个救命稻草:
- `git status`: 了解当前状态
- `git log`: 查看历史
- `git reflog`: 终极后悔药

Happy coding!

---

**文档版本**: v1.0  
**最后更新**: 2025-12-02  
**适用项目**: LBMProject (CUDA/CFD Simulation Platform)
