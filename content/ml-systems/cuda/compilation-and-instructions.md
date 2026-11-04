---
title: PTX、SASS 与编译流程
category: cuda
tags: [PTX, SASS, ptxas, cubin]
description: CUDA C++ · Triton · CuTe DSL · ptxas · cubin · AOT / JIT
updated: 2026-11-04
---
# PTX、SASS 与 kernel 编译流程

CUDA C++、Triton 和 CuTe DSL 可以表达 GPU kernel。它们采用不同的编译流程，但常见的 NVIDIA 后端都会生成 PTX，再生成目标 GPU 的机器码。**PTX 是虚拟指令表示，SASS 是原生机器指令的汇编表示；GPU 执行的是机器码。**

线程怎样组织、工作怎样放入 SM，见 [执行层级与硬件映射](execution-hierarchy.md)。本页解释这些线程执行的代码怎样生成。计算库、kernel 工具与模型框架的关系见 [GPU 软件栈与工具](../gpu/software-stack.md)。

## 1. 源代码、IR、PTX 与机器码

| 层次 | 表达什么 | 示例或用途 |
| --- | --- | --- |
| Kernel 源代码 | 开发者编写的计算、访存和协作逻辑 | CUDA C++、Triton、CuTe DSL |
| IR（Intermediate Representation） | 编译器内部的中间表示 | 保存程序结构，执行优化，再转换到下一层 |
| PTX（Parallel Thread Execution） | NVIDIA 的虚拟指令集及其文本表示 | 描述线程的算术、访存、控制流和同步操作 |
| 原生机器码 | 目标 GPU 能执行的二进制指令 | 通常放在 cubin 中 |
| SASS | 原生机器指令的汇编表示 | 用反汇编工具阅读最终指令 |

```text
Kernel 源代码
    ↓ 编译器的中间表示与优化
PTX
    ↓ 针对目标 GPU 编译
CUBIN：包含原生机器码的 CUDA 二进制文件
    ├── 加载到 GPU，启动 kernel
    └── 反汇编 → SASS 文本，供开发者查看
```

SASS 文本不是 cubin 后面必需生成的可执行产物。查看 SASS 是分析二进制的一种方式；正常运行只需要相应机器码。

来源：[CUDA 平台：编译、Cubins 与 Fatbins](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)。

## 2. PTX 把底层操作与具体机器编码分开

假设源代码表达浮点加法：

```cpp
float c = a + b;
```

对应的 PTX 算术指令可以类似下面这样。这里省略函数、变量声明和数据加载，只展示加法：

```ptx
add.f32 %f3, %f1, %f2;
```

`add` 表示加法，`.f32` 表示 32 位浮点类型。指令把 `%f1 + %f2` 的结果放入 `%f3`。这些名字表示虚拟寄存器；后端还需要分配实际寄存器。

PTX 提供跨 NVIDIA GPU 代际的公共编译目标。对于各目标都支持的操作，编译器可以把同一份 PTX 转换成不同目标的机器码。一条 PTX 指令不保证对应一条原生指令。

PTX 同样有版本和硬件能力要求。模块通过 `.version` 指定 PTX 版本，通过 `.target` 声明目标能力。新硬件专属指令不能据此在旧 GPU 上运行；架构专属目标也有额外的兼容限制。

PTX 描述线程的操作，grid、block、warp 描述执行分组，SM 是执行工作的硬件。PTX 文档把 CUDA thread block 称为 CTA（Cooperative Thread Array）。

来源：[PTX 的设计目标](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#goals-of-ptx)、[PTX 的 target 指令](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ptx-module-directives-target)。

## 3. SASS 随架构演进，也保留基础指令

原生指令集会随硬件能力变化，但不是每一代都全面更换指令。NVIDIA 的 Ampere、Hopper、Blackwell 指令表都列有下面的基本操作：

| SASS 指令 | 操作 |
| --- | --- |
| `FADD` | FP32 加法 |
| `FMUL` | FP32 乘法 |
| `FFMA` | FP32 乘加 |

新硬件也会增加指令。例如 Hopper 在保留 `HMMA` 矩阵乘加指令的同时，增加了 `HGMMA`，支持跨 warpgroup 的矩阵乘加。这类变化涉及线程协作与操作数组织，不能仅按指令名字理解。

**指令读起来相似，不等于二进制兼容。** 例如，为 Ampere `sm_86` 生成的 cubin 不能直接在 Hopper `sm_90` 上加载运行。兼容的 PTX 则可以针对目标 GPU 重新编译。兼容性还取决于 PTX 版本、目标能力和驱动支持。

来源：[NVIDIA 原生指令表](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-reference)、[CUDA 二进制兼容规则](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#binary-compatibility)。

## 4. CUDA C++、Triton、CuTe DSL 的编译路径

下面只画 GPU 设备代码的常见路径。CUDA C++ 文件还可能包含 CPU 端代码，`nvcc` 会组织相应的主机编译步骤。具体中间阶段和选项可能随工具版本变化。

```text
CUDA C++ kernel
    → CUDA 编译器前端与优化
    → PTX
    → ptxas
    → cubin

Triton kernel（NVIDIA 后端）
    → Triton IR（TTIR）
    → Triton GPU IR（TTGIR）
    → LLVM IR
    → PTX
    → ptxas
    → cubin

CuTe DSL kernel
    → MLIR 中间表示与 CuTe / NVVM 转换
    → PTX
    → PTX Compiler 库
    → cubin
```

这些路径都经过 PTX，但开发者不必手动调用后面的每个工具。

| 编写方式 | 谁组织编译 | PTX 之后的常见路径 |
| --- | --- | --- |
| CUDA C++ | `nvcc` 编译驱动组织工具链 | 调用 `ptxas` 生成目标二进制 |
| Triton | Triton 编译器组织中间阶段 | 当前 NVIDIA 后端通过子进程调用 `ptxas` |
| CuTe DSL | CuTe DSL 的编译组件组织转换 | 当前官方文档说明使用 PTX Compiler 库 |

Triton 的 NVIDIA 后端在 `add_stages` 中注册 TTIR、TTGIR、LLVM IR、PTX 和 cubin 阶段。`make_ptx` 生成 PTX，`make_cubin` 再调用 `ptxas`。CuTe DSL 提供 `--ptxas-options` 选项，但文档把选项接收方写为 PTX Compiler 库。因此，选项带有 ptxas 名字，不代表一定启动了同名的外部程序。

Triton 和 CuTe DSL 使用 Python 语法表达 kernel，不要求先转换成 CUDA C++ 再调用 `nvcc`。GPU 上的每次乘加也不是 Python 解释器逐行执行的结果。

来源：[NVCC 文档](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)、[Triton NVIDIA 后端源码](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py)、[CuTe DSL 编译选项](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.html)。源码链接指向移动的 main 分支，精确阶段以所用版本为准。

## 5. ptxas 处理实际寄存器和原生指令

`ptxas` 的输入是 PTX，输出是目标 GPU 的二进制。这个阶段还会优化程序、分配实际寄存器、选择原生指令并安排指令执行顺序。它不是把每一行 PTX 换成固定的一行 SASS。

PTX Compiler API 则以库函数提供 PTX 编译能力。应用传入 PTX 字符串，取得编译后的二进制，再使用 CUDA Driver API 加载。外部 `ptxas` 和 PTX Compiler 库承担相同类别的工作，但接口和调用方式不同。

三类 kernel 最后使用 NVIDIA 后端，不代表它们生成相同的程序。PTX 生成之前，kernel 源码和前端优化已经影响分块与数据布局。访存和同步安排也受源码和前端优化影响。后端在这些输入与目标约束下继续优化。

来源：[PTX Compiler API](https://docs.nvidia.com/cuda/ptx-compiler-api/index.html)、[NVCC ptxas 选项](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ptxas-options)。

## 6. 运行时编译不只指驱动 JIT

AOT（Ahead-of-Time）表示运行前编译；JIT（Just-in-Time）表示运行时按需要编译。JIT 描述编译时机，不指定必须使用哪个编译工具。

| 情况 | 程序运行时发生什么 |
| --- | --- |
| 已有适用的 cubin | 加载机器码，启动 kernel |
| Triton / CuTe DSL 首次编译一种配置 | DSL 编译器生成目标代码，通常缓存结果供后续调用复用 |
| 应用携带 PTX，由驱动 JIT | 驱动把兼容的 PTX 编译成目标 GPU 机器码，再加载运行 |

因此，Triton 在运行时调用 `ptxas` 编译 kernel，也属于 JIT；这不等于把 PTX 交给驱动完成 JIT。CuTe DSL 通过编译库生成 cubin，也是另一条运行时编译路径。

已经编译且满足复用条件的 kernel，可以直接使用缓存结果。不能把每次 Python 函数调用都算成重新编译。驱动 JIT 也有自己的编译缓存。

来源：[CUDA JIT 与编译缓存](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#just-in-time-compilation)。

## 7. 阅读编译产物时关注什么

| 想理解的问题 | 优先查看的层次 |
| --- | --- |
| 计算公式、数据分块和协作逻辑 | Kernel 源码 |
| 编译器如何表达布局和变换程序 | 对应编译阶段的 IR |
| 生成了哪些虚拟算术、访存或同步操作 | PTX |
| 目标程序最终使用哪些原生指令 | Cubin 的 SASS 反汇编 |
| 这些指令实际花了多少时间 | 对应硬件上的运行与测量 |

`nvdisasm` 和 `cuobjdump` 可以查看 cubin 的原生指令。PTX 与 SASS 能帮助解释代码生成，不能单独证明一个 kernel 的实际延迟或吞吐。本文的流程和指令片段用于说明语义，没有编译、运行或性能验证。

来源：[NVIDIA CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)。
