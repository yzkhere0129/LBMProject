## Code Philosophy

Write code that is **concise, elegant, efficient, and in good taste**. Avoid over-design and cleverness.

## Core Principles

1. **Incremental progress over big bangs** - Make small changes that compile and pass tests
2. **Learn from existing code** - Study the codebase before implementing. Match existing patterns and style
3. **Pragmatic over dogmatic** - Adapt to project reality, not theoretical ideals
4. **Clear intent over clever code** - Be boring and obvious. If it needs explanation, it's too complex

## Simplicity Rules

- Single responsibility per function/class
- Avoid premature abstractions
- Choose the boring, proven solution
- Delete code whenever possible

## Modular Architecture (积木式开发)

- **独立插拔**: 每个物理模块可独立实例化、测试、替换，不依赖其他模块
- **经典 benchmark 验收**: 每个模块以对应的经典解析/benchmark 问题通过作为完成标准
- **最小接口**: 模块间仅通过标量/向量场指针耦合，不暴露内部数据结构
- **自下而上验证**: L1(格子) → L3(求解器) → L4(耦合)，低层通过后才测高层
- **合成输入测试**: 用解析构造的场(速度/温度)驱动模块，不依赖其他求解器输出

## Quality Standards

- **High performance** - Consider algorithmic complexity and resource usage
- **Efficiency** - Optimize for common paths, minimize allocations
- **Good taste** - Code should feel natural and idiomatic to its language

## Working Style

- **Be honest** - Don't claim tests passed when they didn't. Don't overstate accomplishments
- **Be practical** - Ship working code over perfect code
- **Be humble** - Let the code speak for itself

## Environment setup
- you can use env-setup-specialist subagent to set up the developing environment