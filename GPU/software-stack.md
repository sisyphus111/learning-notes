# GPU 软件栈：计算库、kernel 开发与编译工具

cuBLAS、cuDNN 提供可调用的计算实现。CUTLASS、CuTe DSL 和 Triton 帮助开发者编写 kernel。编译器生成 GPU 指令，运行时和驱动负责加载程序、提交工作与管理资源。它们承担不同职责，也会在同一个模型中组合使用。

本页覆盖 LLM 推理和 GPU kernel 开发中常见的名称。设备与用户态/内核态的关系见 [驱动、运行时与框架](driver-runtime-framework.md)。计算图与编译流程见 [计算图、编译与 CUDA Graph](graphs-and-compilation.md)。Tensor 和模型层的基础见 [神经网络基础](../PyTorch/neural-networks.md)。

数值类型与计算模式见 [浮点格式、混合精度与量化](precision-and-quantization.md)：FP16、BF16、TF32、FP8、E8M0 和块缩放格式不能只按名字中的 bit 数混同。

## 1. 从一层 Linear 放置各个工具

计算 `y = x @ weight.T + bias` 时，开发者有两条常见路径：

```text
调用现有实现：
PyTorch / 推理框架
    → 选择 cuBLAS、cuBLASLt 或其他实现
    → 启动 GPU kernel

开发实现：
CUDA C++ / CUTLASS / CuTe DSL / Triton
    → 相应编译管线
    → 生成 GPU kernel
    → 接入模型并启动
```

kernel 是运行在 GPU 上的函数。库可以提供事先编译的 kernel，也可以在运行时生成或编译特定实例。一个模型通常混用现成库和定制 kernel；这些名称不构成必须逐层经过的固定调用链。

## 2. 调用计算的库

| 名称 | 软件类型 | 主要用途 |
| --- | --- | --- |
| cuBLAS | 基础线性代数库 | 矩阵乘法、矩阵向量乘法、向量运算 |
| cuBLASLt | 灵活的矩阵乘法库接口 | 选择布局、输入/累加类型、算法与 workspace；支持部分输出融合 |
| cuDNN | 深度学习计算库 | 卷积、归一化、attention，以及支持的融合计算 |
| cuSPARSE | 稀疏线性代数库 | 稀疏矩阵乘法与其他稀疏运算 |
| cuSPARSELt | 结构化稀疏矩阵乘法库 | 例如使用支持的 2:4 格式与 Sparse Tensor Core |
| cuSOLVER | 线性代数求解库 | 线性方程求解、QR、Cholesky、SVD |
| cuFFT | 快速傅里叶变换库 | 时域与频域之间的变换 |
| cuRAND | 随机数生成库 | 在 GPU 上生成随机数 |
| cuTENSOR | 多维 Tensor 计算库 | Tensor contraction、维度置换与部分逐元素计算 |

BLAS 是 Basic Linear Algebra Subprograms。GEMM 是 General Matrix Multiply，常见接口表达 `D = alpha * op(A) @ op(B) + beta * C`。库根据矩阵尺寸、布局、硬件和精度约束选择实现；库内部也可能使用 Tensor Core。[cuBLAS 文档](https://docs.nvidia.com/cuda/cublas/index.html)

cuBLASLt 中的 Lt 来自 lightweight。它提供更灵活的矩阵乘法描述和算法选择接口，不表示计算性能低于 cuBLAS。epilogue 指主乘加结束后的输出处理，例如缩放、加 bias、激活和类型转换。具体可融合内容由接口和配置决定。

cuDNN 的范围不只包含 CNN 卷积。cuDNN 还提供 attention 实现，并通过图接口表达支持的计算组合。frontend 是构建这些操作图的接口层，backend 提供执行引擎与底层接口。[cuDNN 概览](https://docs.nvidia.com/cudnn/overview)

稀疏计算还需要匹配存储格式。dense Tensor 中出现零值，不会自动触发 cuSPARSELt 的结构化稀疏路径。调用方需要准备符合要求的压缩矩阵和元数据。[cuSPARSELt 文档](https://docs.nvidia.com/cuda/cusparselt/index.html)

数学计算库的其他入口见 [CUDA 库目录](https://docs.nvidia.com/cuda/)，多维收缩接口见 [cuTENSOR 文档](https://docs.nvidia.com/cuda/cutensor/latest/index.html)。

## 3. CUTLASS 的具体软件形态

**CUTLASS 是 NVIDIA 开源的 kernel 开发库。** 开发者在 CUDA C++ 程序中包含 CUTLASS 头文件，使用模板类型和组件组合出 kernel。编译器再把生成的实现编译到程序或扩展模块中。CUTLASS 也提供示例、测试、profiler 和可调用算子接口。

“体系”只是描述项目覆盖范围的泛称。判断软件类型时，把 CUTLASS 记为“用于开发和复用高性能矩阵 kernel 的软件库”更准确。C++ 模板、CuTe 布局规则和 kernel 编程接口都是这个库的组成部分；CuTe DSL 是项目提供的一种 Python 编程方式。[CUTLASS 官方定义](https://docs.nvidia.com/cutlass/latest/overview.html)

### 3.1 在 C++ 程序中怎样使用它

下面用经典的 `device::Gemm` 接口说明软件形态。代码是接口片段，省略 GPU 内存分配和状态检查，未编译或运行。设 M=N=K=1024，A/B 为已分配的 FP16 GPU 指针，C/D 为 FP32 GPU 指针，四张矩阵均按行连续存储。

```cpp
#include <cutlass/gemm/device/gemm.h>

using RowMajor = cutlass::layout::RowMajor;
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, RowMajor,       // A 的类型与布局
    cutlass::half_t, RowMajor,       // B 的类型与布局
    float, RowMajor,                 // C/D 的类型与布局
    float,                          // 累加类型
    cutlass::arch::OpClassTensorOp,  // 使用 TensorOp 计算组件
    cutlass::arch::Sm80>;            // 目标架构类别，例如 A100

Gemm gemm;
Gemm::Arguments args(
    {M, N, K},
    {A, K}, {B, N}, {C, N}, {D, N},
    {1.0f, 0.0f});                  // D = A @ B，beta=0
cutlass::Status status = gemm(args);
```

这段代码仍是 CUDA C++。`Gemm<...>` 通过 C++ 模板参数选择输入类型、布局、累加方式和硬件计算组件。`args` 提供运行时尺寸、指针、stride 和缩放参数。`gemm(args)` 调用生成的实现并启动 GPU 工作。完整程序还要检查支持条件、编译目标和返回状态。

就像 `std::vector` 是 C++ 库提供的模板类型，`cutlass::gemm::device::Gemm` 也是库提供的模板类型。开发者用 C++ 语法实例化它；模板参数和运行时参数分别表达 kernel 的构造选择与本次计算的数据。

经典 GEMM 接口的调用方式可对照 [CUTLASS basic GEMM 示例](https://github.com/NVIDIA/cutlass/blob/main/examples/00_basic_gemm/basic_gemm.cu)。不同代际的 CUTLASS 接口并不完全相同，上面的片段用于辨认“库怎样进入 C++ 程序”。

### 3.2 CUTLASS 允许组合什么

| 可组合内容 | 影响 |
| --- | --- |
| block / warp / instruction tile | 把输出区域与归约工作分配给线程和矩阵指令 |
| 数据布局与 copy 组件 | 决定地址映射、线程访问和片上数据移动 |
| MMA 组件 | 选择矩阵乘加的基本操作 |
| pipeline | 协调加载、计算和缓冲区复用 |
| epilogue | 组织输出缩放、融合、转换和写回 |

因此，使用 CUTLASS 可以复用已有组合，也可以替换其中的组件。开发者不必从零重写每条加载指令和每种 MMA，但仍需要理解布局、分块、资源占用与同步。

### 3.3 CuTe 与 CuTe DSL

CuTe 是 CUTLASS 中描述 Tensor 和线程映射的核心抽象之一。一个 CuTe Tensor 把数据位置与 layout 结合起来；layout 表达逻辑坐标怎样映射到地址，或数据怎样分配给线程。MMA atom 和 copy atom 表达基本计算、复制操作，再由 tiled operation 把基本操作铺到更大的区域。

CuTe DSL 沿用这些概念，以 Python 语法编写 kernel。DSL 是 Domain-Specific Language，即面向特定领域的语言。其编译管线把程序转换成 GPU 代码；Python 用于表达程序，不意味着 GPU 上的每次乘加由 Python 解释器执行。[CuTe DSL 概览](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)

```text
CUTLASS 项目
    ├── CUDA C++ 模板和 kernel 组件
    │       └── CuTe：Tensor、layout、copy、MMA 等抽象
    └── CuTe DSL：在 Python 编程方式中表达这些抽象
```

## 4. 编写 kernel 的其他工具与基础组件

| 名称 | 类型 | 具体职责 |
| --- | --- | --- |
| CUDA C++ | C++ 的 GPU 编程扩展 | 表达线程、block、shared memory、同步与设备函数 |
| Triton | kernel 语言与编译器体系 | 用块级 Tensor 操作表达加载、归约、矩阵乘法与写回 |
| CUB | CUDA 并行基础算法库 | 提供 device/block/warp 级 reduction、scan、排序组件 |
| Thrust | C++ 并行算法库 | 提供并行算法，例如 sort、transform、reduce |
| libcu++ / libcudacxx | CUDA C++ 标准库支持 | 提供原子操作、同步与标准库设施 |
| CCCL | CUDA Core Compute Libraries 项目 | 组织基础库，包括 CUB、Thrust、libcu++ |

例如，Triton 程序通过 `tl.dot(a_tile, b_tile, acc)` 表达矩阵块乘加，编译器根据目标和约束选择指令。CUB 则能作为 CUDA C++ kernel 内的组件，例如让一个 block 合作求和。两者都帮助开发 kernel，但接口粒度不同。[Triton 文档](https://triton-lang.org/main/index.html)、[CCCL 项目](https://github.com/NVIDIA/cccl)

## 5. 编译工具与指令表示

| 名称 | 它是什么 | 输入、输出或职责 |
| --- | --- | --- |
| nvcc | CUDA C++ 编译工具链入口 | 组织 host/device 代码编译 |
| ptxas | PTX 编译工具 | 把 PTX 转换为特定 GPU 架构的机器代码 |
| NVRTC | 运行时编译库 | 在程序运行过程中编译 CUDA 源码 |
| nvJitLink | 运行时链接库 | 链接运行时生成或提供的设备代码 |
| PTX | 虚拟指令集及其表示 | 表达加载、算术和矩阵操作，仍需转换为目标机器代码 |
| SASS | GPU 机器指令的汇编表示 | 反汇编具体 GPU 程序时查看的指令 |
| LLVM | 编译器基础设施 | 提供中间表示、优化与代码生成能力 |
| MLIR | 编译器基础设施 | 组织不同抽象层次的中间表示及其转换 |

IR 是 Intermediate Representation，即编译器内部的中间表示。源代码、算子图、PTX 和机器指令位于不同抽象层次。Triton、CuTe DSL 使用自己的编译管线，不要求先生成 CUDA C++ 再调用 nvcc。编译与链接工具入口见 [CUDA 文档](https://docs.nvidia.com/cuda/)。

Tensor Core 执行特定矩阵指令。库或编译器先选择、生成使用这些指令的 kernel；硬件按指令执行。普通标量乘加循环不会因为恰好实现了矩阵乘法，就在运行时自动转成 Tensor Core 算法。

## 6. 模型编译与推理框架

| 名称 | 所处位置 | 主要职责 |
| --- | --- | --- |
| PyTorch | Tensor 与模型框架 | 表达模型、调用算子、管理参数与自动求导 |
| torch.compile | PyTorch 编译入口 | 组织图捕获、后端编译与编译结果复用 |
| TorchDynamo | 图捕获组件 | 从 Python 执行中提取 Tensor 计算区域及 guards |
| TorchInductor | 编译后端 | 优化图，生成 kernel 或调用计算库 |
| AOTAutograd | 自动求导的编译组件 | 在需要梯度的路径中准备前向和反向图供后端编译 |
| TensorRT | 推理优化与运行时工具 | 优化并执行模型推理 |
| TensorRT-LLM | LLM 推理工具与运行时 | 组织 LLM 执行、kernel 与分布式推理优化 |
| vLLM、SGLang | LLM 推理与服务框架 | 管理请求、batch、KV cache、模型与多卡执行 |

`torch.compile`、Dynamo、Inductor 是入口与组件的关系。CUDA Graph 是执行机制，既可单独使用，也可与编译结果组合。详细过程见 [计算图、编译与 CUDA Graph](graphs-and-compilation.md)。官方入口见 [PyTorch 编译文档](https://docs.pytorch.org/docs/2.9/torch.compiler.html)、[TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html)、[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)。

## 7. LLM 专用 kernel 项目

| 名称 | 主要内容 |
| --- | --- |
| FlashAttention | attention 算法及其 kernel 实现，利用分块与数据复用减少中间存储访问 |
| FlashInfer | 面向推理的 kernel 库，覆盖的计算包括 attention、KV cache 操作与采样 |
| DeepGEMM | 针对特定精度、布局和场景优化的 GEMM kernel 库 |
| FlashMLA | 高性能 attention kernel 项目，尤其围绕 Multi-head Latent Attention |

“FlashAttention”描述计算算法与实现，“CuTe DSL”描述 kernel 的编写方式。一份 FlashAttention 实现可以使用 CuTe DSL；上层框架再调用这份实现。其他实现也可以选择不同的 kernel 编程工具。[FlashAttention](https://github.com/Dao-AILab/flash-attention)、[FlashInfer](https://docs.flashinfer.ai/)、[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[FlashMLA](https://github.com/deepseek-ai/FlashMLA)

## 8. 通信、执行与分析工具

| 名称 | 具体职责 |
| --- | --- |
| NCCL | GPU 间集合通信，例如 all-reduce、all-gather、reduce-scatter |
| CUDA Runtime API | 分配内存、管理 stream/event、启动 kernel |
| CUDA Driver API | 提供的接口包括 context、设备模块加载与 kernel 启动 |
| CUDA Toolkit | 包含编译器、开发库、头文件与工具的开发工具包 |
| NVIDIA Driver | 支持系统与应用使用 GPU 的驱动软件 |
| CUDA Graph | 保存 CUDA 工作及依赖，实例化后重复提交执行 |
| Nsight Systems / nsys | 查看 CPU、GPU、通信与等待的整体时间线 |
| Nsight Compute / ncu | 分析具体 kernel 的指令、访存和执行指标 |
| Compute Sanitizer | 检查 GPU 非法访存、部分竞争与同步错误 |

TP 的 Row Parallel Linear 可以先调用矩阵 kernel 产生部分和，再调用 NCCL all-reduce 合并贡献。推理框架负责把两步放进模型执行流程。计算库、通信库和上层框架分别承担计算、传输和组织职责。[NCCL 概览](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html)

CUDA Runtime 与 Driver API 都提供用户态调用接口，Driver API 不等于内核态驱动模块。Toolkit 与 Driver 也有各自的版本和安装来源。具体接口关系见 [CUDA Runtime / Driver API](https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html)，工具入口见 [CUDA 文档目录](https://docs.nvidia.com/cuda/)。

本页是软件职责与接口的教学说明。C++ 片段没有经过 CUDA 编译；没有执行 kernel、数值测试或性能测量。
