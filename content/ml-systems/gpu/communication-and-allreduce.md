---
title: MPI 与 AllReduce 算法
category: gpu
tags: [MPI, AllReduce, Ring, Tree]
description: Ring / Tree · Recursive Doubling · One-shot / Two-shot · 通信量与依赖
updated: 2026-11-04
---
# 通信基础：MPI 与 AllReduce 算法

AllReduce 的性能取决于数据怎样流动、谁计算部分和，以及下一步要等待谁。Ring、Tree、Recursive Doubling、One-shot 和 Two-shot 都能得到相同的数学结果，但通信量、重复归约和依赖深度不同。

本文先用同一组四卡数据走通 Ring 与 Tree，再展开 Recursive Doubling、One-shot / Two-shot，最后用统一口径比较通信成本，并连接 TP / EP 的实际张量布局。各节数值均为教学推导，不是硬件实测。

## 1. 接口、语义、算法与硬件分别规定什么

AllReduce 规定通信结束后各方得到什么结果。Ring、Tree 规定怎样交换数据以实现这个结果。MPI 定义进程通信接口；Open MPI、MPICH 是具体实现。NCCL 则提供面向 NVIDIA GPU 的通信库。

| 层次 | 回答的问题 | 例子 |
| --- | --- | --- |
| 操作语义 | 各方提供什么，最后各方得到什么 | AllReduce、ReduceScatter、AllGather |
| 通信算法 | 数据分几步、沿什么逻辑路径交换 | Ring、Tree、Recursive Doubling |
| 接口标准与通信库 | 应用怎样调用，谁组织实际执行 | MPI、Open MPI、MPICH、NCCL |
| 协议与传输框架 | 消息、远程访问怎样映射到可用传输 | UCX、网络插件 |
| 互联与传输能力 | 字节经过哪些链路，设备能访问哪些内存 | PCIe、NVLink、InfiniBand、RoCE、GPUDirect RDMA |

同一个 AllReduce 可以由不同库、用不同算法、经过不同互联完成。逻辑上的 Ring 不要求物理设备连成环，Tree 的父子关系也不等于网络交换机的物理连接。

MPI 或 NCCL 的调用方通常只指定数据、归约操作和参与组。库根据消息大小、rank 数和拓扑选择实现。Open MPI 的 tuned component 就支持按 communicator 大小和消息大小选择算法。[Open MPI 算法选择](https://docs.open-mpi.org/en/main/tuning-apps/collectives/tuned.html)

## 2. MPI 的来源与使用范围

MPI 是 Message Passing Interface，即消息传递接口。它面向分布式内存并行程序：不同进程拥有独立地址空间，通过显式通信交换计算数据。

MPI 由并行计算社区共同制定，主要组织是 MPI Forum。1992 年 4 月的 Williamsburg 研讨会启动标准化工作。Jack Dongarra、Rolf Hempel、Tony Hey、David Walker 在同年 11 月提出早期草案，之后由来自厂商、大学与实验室的参与者共同完善。MPI 1.0 于 1994 年 5 月发布。[MPI 原始标准的背景](https://www.mcs.anl.gov/research/projects/mpi/mpi-standard/mpi-report-1.0/node2.html)、[MPI 标准版本记录](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report-book.pdf)

MPI 最初服务于超级计算机和 CPU 集群，远早于今天的 LLM。不同厂商原先提供的通信接口互不兼容，统一接口让并行程序可以移植到不同系统。

例如天气模拟把地理网格分给四个进程：

```text
┌──────────┬──────────┐
│ rank 0   │ rank 1   │
├──────────┼──────────┤
│ rank 2   │ rank 3   │
└──────────┴──────────┘
```

每个进程计算本区域的温度、气压和风速。计算区域边界时，需要邻居的数据，因此每轮交换边界。程序还可以用 AllReduce(MAX) 得到全区域的最大误差，用 Broadcast 分发参数，用 Gather 汇集结果。以上工作可以全部由 CPU 完成。

现代 CUDA-aware MPI 实现可以接受 GPU buffer，应用不必先手动复制到 CPU。具体支持哪些操作、是否使用 GPUDirect RDMA，取决于实现、版本、传输组件与硬件。接受 GPU 指针不等于保证所有路径都绕过 host memory。[Open MPI 的 CUDA 支持](https://docs.open-mpi.org/en/main/tuning-apps/accelerators/cuda.html)

## 3. process、rank 与 communicator

| 对象 | 含义 |
| --- | --- |
| process | 实际运行程序的进程，拥有自己的地址空间 |
| rank | 进程在某个通信组内的编号 |
| communicator | 参与组及其通信上下文，用于隔离消息与通信操作 |
| root | 某些操作指定的结果接收者或数据来源，使用组内 rank 表示 |

一个进程使用一张 GPU 是常见映射，但 rank 本身不是 GPU 硬件编号。相同进程在不同子组内还可能拥有不同编号。`MPI_COMM_WORLD` 包含一次普通 MPI 初始化中的初始进程；程序也可以建立 TP 或 DP 子组。

MPI 包含点对点 Send/Recv、集合通信、非阻塞操作和远程内存访问。例如 `MPI_Allreduce` 是阻塞接口；`MPI_Iallreduce` 返回 request，调用方之后等待或查询完成。非阻塞不自动保证通信与计算实现重叠，还取决于依赖、进度机制和资源。[MPI AllReduce 接口](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Allreduce.3.html)

下面是 API 语义片段，假设 MPI 已初始化并获得 `rank`：

```cpp
float local = rank + 1;
float result;

MPI_Allreduce(
    &local, &result,
    1, MPI_FLOAT, MPI_SUM,
    MPI_COMM_WORLD
);
// 若组内有 rank 0、1、2、3，每个 result 都是 10。
```

接口中的 `count=1` 表示一个元素。对于数组，归约作用于不同 rank 的对应位置。阻塞接口返回时，本进程可以使用结果；这不意味着所有进程在同一时刻返回。

## 4. 固定 AllReduce 的输入输出

后文假设同一个组内有四个 rank，归约操作为 SUM。每个 rank 提供四个数：

```text
rank 0：[   1,    2,    3,    4]
rank 1：[  10,   20,   30,   40]
rank 2：[ 100,  200,  300,  400]
rank 3：[1000, 2000, 3000, 4000]
```

AllReduce 后，每个 rank 都得到 `[1111, 2222, 3333, 4444]`。若 rank p 的输入为 X_p，结果为 Y，则：

$$
Y[i]=\sum_{p=0}^{P-1}X_p[i]
$$

输入、输出的逻辑 shape 相同。求和沿 rank 这个分布式维度发生，不是对单个本地 tensor 的所有元素求和。SUM 不自动除以 rank 数。

需要区分三个相关操作：

| 操作 | 以上例子的结果 |
| --- | --- |
| AllReduce | 每个 rank 都得到完整 `[1111,2222,3333,4444]` |
| ReduceScatter | rank 0 得到 `[1111]`，rank 1 得到 `[2222]`，依此类推 |
| AllGather | 每人提供一份数据，所有人得到按 rank 顺序拼接的结果，不求和 |

先 ReduceScatter，再 AllGather，可以实现 AllReduce。这个关系描述输入输出语义，不要求实际执行时调用两个独立 API。[NCCL 集合通信语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)

### 统一字节数、轮次与归约计算的口径

| 符号 | 本页含义 |
| --- | --- |
| P | 参与 AllReduce 的 rank 数；后续 GPU 例子假设一卡一 rank |
| L | 每个 rank 的完整输入元素数 |
| s | 每个通信元素的字节数；BF16 为 2 |
| N = Ls | 每个 rank 的完整输入字节数，不是全系统输入之和 |
| V_send,rank | 一张卡累计发送的跨卡有效 payload 字节数 |
| V_total | 全系统累计发送量，每次传输只计一次 |

除非另行说明，以下模型使用软件点对点传输，等长分片，不计同步标志、协议头、重传和本地 buffer 拷贝。对称算法的每卡发送量与接收量相同；全双工下，不能把两者相加后直接除以单向带宽来估计时间。单树的各卡流量可能不同，平均值不代表每卡实际值。

“通信次数”也有三个层次：

| 层次 | 具体含义 |
| --- | --- |
| 框架 collective 次数 | 一个模型层调用几次 AllReduce / AllGather |
| 算法依赖轮次或阶段 | 一次 collective 内，有多少次必须等待前一批数据的推进 |
| GPU kernel 数量 | 库实际启动几个 kernel |

一次 Ring AR 的多轮交换可以在同一 kernel 内完成；Two-shot 的两个阶段也不必对应两个 kernel。Direct 算法的一个阶段可能访问所有 peer，而 Ring 一轮通常只访问环上的邻居，不能把不同算法的“1 阶段”直接赋予同一个延迟。

## 5. Ring 的分块与 ReduceScatter

把四个 rank 排成逻辑环，每个人向下一位发送、从上一位接收：

```text
rank 0 → rank 1 → rank 2 → rank 3 → rank 0
```

把每份输入切成 P 个等长块。先令每块只有一个数，以便手算：

```text
rank 0：[a₀ | a₁ | a₂ | a₃]
rank 1：[b₀ | b₁ | b₂ | b₃]
rank 2：[c₀ | c₁ | c₂ | c₃]
rank 3：[d₀ | d₁ | d₂ | d₃]

第 j 块的最终结果：Rⱼ = aⱼ + bⱼ + cⱼ + dⱼ
```

### 跟踪第 0 块的贡献

一种合法的分工是让第 0 块最终在 rank 0 归约完成：

```text
rank 1：b₀
    → rank 2：b₀ + c₀
    → rank 3：b₀ + c₀ + d₀
    → rank 0：a₀ + b₀ + c₀ + d₀ = R₀
```

代入数值，三轮相邻传递形成 `10 → 110 → 1110 → 1111`。接收者将上游部分和加上自己的贡献，再交给下一位。

其他块同时沿环推进，不需要等第 0 块完成。下面列出每轮在各 rank 新形成的部分和，未列出的块仍可能保留旧值：

| ReduceScatter 轮次 | rank 0 | rank 1 | rank 2 | rank 3 |
| --- | --- | --- | --- | --- |
| 第 1 轮 | 第 2 块：3003 | 第 3 块：44 | 第 0 块：110 | 第 1 块：2200 |
| 第 2 轮 | 第 1 块：2202 | 第 2 块：3033 | 第 3 块：444 | 第 0 块：1110 |
| 第 3 轮 | 第 0 块：1111 | 第 1 块：2222 | 第 2 块：3333 | 第 3 块：4444 |

三轮结束后，每个 rank 拥有一块已经包含所有贡献的结果。块的归属取决于所选 Ring 调度，其他实现可以采用不同的循环编号。

### rank 1、2、3 的第 0 个位置是否也改了

若采用“收到部分和后，原地更新该块，再发送”的教学实现，三轮 ReduceScatter 后，第 0 块可以保留如下内容：

| rank | 初值 | 此时该位置的值 | 已包含的贡献 |
| --- | --- | --- | --- |
| rank 1 | 10 | 10 | rank 1；它在这条路径中只发送初值 |
| rank 2 | 100 | 110 | rank 1、2 |
| rank 3 | 1000 | 1110 | rank 1、2、3 |
| rank 0 | 1 | 1111 | 全部 rank |

因此，rank 2、3 确实参与计算并形成中间部分和，rank 1 的这个位置则可以保持初值。ReduceScatter 完成后，只有各 rank 负责的最终块具有完整结果保证。

真实实现不必将这些部分和写回原输入。它可以使用寄存器、通信缓冲区或输出区，并把归约后的值直接发给下一跳。独立 ReduceScatter API 的输出只包含本 rank 的结果分片，不能依赖其他输入位置留下什么值。上表说明一种原地实现，不能当作所有库的存储保证。

## 6. Ring 的 AllGather 阶段

ReduceScatter 后的有效结果为：

```text
rank 0：R₀
rank 1：R₁
rank 2：R₂
rank 3：R₃
```

AllGather 继续沿相同方向传播完整块，不再累加。每轮把上一轮收到的块发给下一位：

| 阶段 | rank 0 已拥有的完整块 | rank 1 | rank 2 | rank 3 |
| --- | --- | --- | --- | --- |
| 开始 | R₀ | R₁ | R₂ | R₃ |
| 第 1 轮 | R₀、R₃ | R₀、R₁ | R₁、R₂ | R₂、R₃ |
| 第 2 轮 | R₀、R₂、R₃ | R₀、R₁、R₃ | R₀、R₁、R₂ | R₁、R₂、R₃ |
| 第 3 轮 | 全部 | 全部 | 全部 | 全部 |

例如 R₀ 经过 `rank 0 → rank 1 → rank 2 → rank 3`。如果此前保留了第 0 块的中间部分和，完整 R₀ 到达后，会写入该块的最终输出位置。

经典 Ring 因而需要 P−1 轮 ReduceScatter，加 P−1 轮 AllGather。实际 kernel 可以分段流水化，并融合两个阶段交界处的接收、归约、存储和发送。逻辑轮次不等于 CPU launch 次数，也不要求每轮设置全组 barrier。[NCCL Ring 实现](https://github.com/NVIDIA/nccl/blob/master/src/device/all_reduce.h)

## 7. Ring 的通信量与延迟

令 N 是每个 rank 的完整输入字节数，P 是 rank 数。假设 N 可等分，每轮每个 rank 发送 N/P 字节：

$$
V_{\mathrm{send,rank}}=2(P-1)\frac{N}{P}
=2\frac{P-1}{P}N
$$

每个 rank 的接收量相同。发送量和接收量分别计数，不能把二者相加后称为单向链路的传输量。

四个 rank、每人输入 100 MiB 时，每人累计发送 150 MiB、接收 150 MiB。这里是算法流量推导，没有测量实际网络带宽。当 P 增大时，发送量接近 2N，但通信轮数随 P 线性增长。

用一个简化模型说明两项成本：令 α 为每轮固定延迟，b 为有效单向带宽。假设各环边并行、发送与接收可以重叠，暂不计归约计算和链路争用：

$$
T_{\mathrm{ring}}\approx2(P-1)\alpha
+2\frac{P-1}{P}\frac{N}{b}
$$

第一项来自依赖轮次，第二项来自传输量。小消息容易受第一项影响；大消息更容易受第二项影响。这里的 α 不是“每轮都由 CPU launch 一个 kernel”的开销，它还包含消息协议与推进的固定成本。

Ring 的优点是块分工和每个 rank 的流量较均衡，容易持续利用环上的链路。缺点是依赖轮次多，较慢的链路或 rank 会影响后续推进。逻辑环若多次穿过同一条受限网络链路，均衡的逻辑流量也可能形成物理拥塞。

## 8. Tree：向上 Reduce，向下 Broadcast

基础单树可以采用下面的逻辑父子关系：

```text
             rank 0
            /      \
        rank 1     rank 2
                      \
                      rank 3
```

### 向上归约

叶子向父节点发送数据。父节点将子节点的贡献与自己的输入相加，再向上传递：

```text
第 1 轮：rank 1 → rank 0，形成 X₀ + X₁
         rank 3 → rank 2，形成 X₂ + X₃

第 2 轮：rank 2 → rank 0
         rank 0 得到 X₀ + X₁ + X₂ + X₃
```

单看第一个元素，rank 0 得到 `1+10=11`，rank 2 得到 `100+1000=1100`，rank 0 再得到 `11+1100=1111`。完整向量的其他位置采用相同操作。rank 1、3 提交原始输入后，不再担任向上归约的代表，也没有自动获得相应部分和。

### 向下广播

root 得到完整结果后，把结果交给子节点，再由子节点继续转发：

```text
第 3 轮：rank 0 → rank 2，传递完整 Y
第 4 轮：rank 0 → rank 1，rank 2 → rank 3，传递完整 Y
```

这里选择了反向展开的广播调度，便于与后文 Recursive Doubling 对照，不是唯一合法调度。这个基础模型是 Reduce 加 Broadcast。它沿树向上合并完整向量，而 Ring 让各 rank 分担不同块的最终归约。实际 Tree 可以分段，在下游已处理前一段时继续处理下一段。

平衡二叉树的高度约为 h=ceil(log₂P)，向上、向下的依赖深度合计约为 2h。例如 P=1024 时，经典 Ring 有 2046 轮，平衡树约有 20 层。层数差异不等于固定的实测加速倍数。

## 9. 基础单树的带宽与负载问题

叶子只需向父节点发送和接收结果。内部节点还要接收多个子节点的数据、归约、向上传递，再把最终结果分发给多个子节点。这些工作可能争用同一张网卡、GPU 访存带宽或互联链路。

未分段的基础树在上层只剩少数活跃边，部分参与者需要等待。分段流水线可以缓解这种现象，但不能仅凭 log₂P 的高度就认定没有带宽问题。

仍使用 α、b、N。对最基础的未分段模型，若每层传递完整 N 字节，并理想化地假设同层链路可独立并行：

$$
T_{\mathrm{tree,basic}}\approx2h\left(\alpha+\frac{N}{b}\right)
$$

这个模型说明完整消息沿关键路径经过多层的成本。它忽略内部节点的并发接收争用，也没有描述分段、多树和归约广播重叠。不能直接拿它预测现代 NCCL Tree。

### 相同的全系统流量，不同的关键路径

基础 Tree 不必比 Ring 发送更多的全系统总字节。在前述点对点模型下：

| 算法 | 全系统累计发送量 |
| --- | --- |
| Ring | P × 2(P−1)N/P = 2(P−1)N |
| 基础 Tree | P−1 条树边，每条上下各传 N，共 2(P−1)N |

总字节相同，仍可能耗时不同。关键在于流量由哪些 rank 和链路承担，哪些传递必须依次完成。Ring 的每 rank 流量较均衡，单树的内部节点承担更多转发。

若上下行采用同一棵树，节点 r 连接 d_r 条树边，则每条相邻边在两个方向各传 N 字节：

$$
V_{\mathrm{send},r}=V_{\mathrm{recv},r}=d_rN
$$

有两个孩子和一个父节点的内部节点，发送与接收各为 3N；叶子各为 N。在上面的四卡树中，rank 0、2 各发送 2N，rank 1、3 各发送 N，全系统为 6N。平均每卡 1.5N，但没有一张卡恰好发送 1.5N。

这些统计不包含协议头、重传与共享物理链路的重复负载。multicast 或硬件归约需要按实际传输机制另行计数。

## 10. Double Binary Tree 怎样改善单树

把数据分成两半，分别交给两棵互补的树：

```text
前半部分 → Tree A
后半部分 → Tree B

让 Tree A 的内部节点尽量在 Tree B 中做叶子，反之亦然。
```

一棵树中负责转发的 rank，在另一棵树中少做转发。两棵树配合分段流水线，分担内部节点的压力，同时保留较低的依赖深度。

下面给出一个四卡互补示意，不代表具体机器的 NCCL 拓扑：

```text
Tree A：前半张量 N/2        Tree B：后半张量 N/2

        rank 0                     rank 3
       /      \                   /      \
    rank 1   rank 2             rank 2   rank 1
                \                          \
                rank 3                     rank 0
```

每张卡在一棵树中连接两条边，在另一棵树中连接一条边。因此，每卡累计发送、接收各为 `2 × N/2 + N/2 = 1.5N`。全系统发送量仍为 6N，但四张卡的工作更均衡。“Double”指两棵树各处理一部分数据，每棵树内部仍包含向上归约和向下广播。

NCCL 的 double binary tree 就采用这个方向。因此，优化后的 Tree 也可以同时追求低延迟和高带宽。“Ring 带宽高、Tree 延迟低”适合提示主要矛盾，不能当成所有实现的性能定律。[NVIDIA Double Binary Tree 设计](https://developer.nvidia.com/blog/massively-scale-deep-learning-training-nccl-2-4/)

Tree 也不要求先把全系统所有原始输入堆到 root。内部节点可以先归约，再向上传递一个相同大小的部分和。root 的问题主要是接收、归约和分发压力，不是必须保存 P 份原始 tensor。

## 11. Ring 与 Tree 的取舍

| 维度 | 经典 Ring | 优化后的 Tree |
| --- | --- | --- |
| 依赖深度 | 约 2(P−1)，随 P 线性增长 | 平衡树通常随 log P 增长 |
| 工作分担 | 分块归约，每个 rank 的流量较均衡 | 需要分段、多树或其他安排平衡内部节点 |
| 小消息 | 固定延迟容易占主导 | 较低的依赖深度通常有利 |
| 大消息 | 容易形成持续传输的流水线 | 也能获得高带宽，取决于具体实现 |
| 拓扑映射 | 需要安排好的环顺序和 channels | 需要安排好的父子关系与树 |
| 慢 rank / 慢链路 | 影响环上的推进 | 影响相应子树和汇总路径 |
| 故障 | 普通 collective 可能失败或等待 | 不因为是树就自动具有容错能力 |

消息很小且 rank 很多时，减少轮次通常更重要。消息很大时，流量均衡与有效带宽更重要。两者的交叉点受设备、协议、消息形状和并发工作影响，没有通用字节阈值。

真实机器还可能采用分层算法：先在快速的本地互联内归约或切分，再跨节点交换。得到结果后，再在节点内分发。Ring、Tree 也不是全部候选；Recursive Doubling、递归折半配合倍增、直接交换，以及硬件辅助路径，都可能参与选择。

改变归约顺序还可能改变浮点舍入结果。数学上的求和一致，不代表不同算法和配置一定逐 bit 相同。

## 12. 放回 GPU、LLM 与 CUDA Graph

| 场景 | 通信需求 |
| --- | --- |
| TP Row Parallel Linear | 合并各 GPU 对同一输出位置产生的部分和 |
| 输出继续保持分片 | 可以考虑 ReduceScatter，避免立即复制完整结果 |
| 数据并行训练 | 合并各 rank 的梯度，缩放规则由训练语义决定 |
| CPU 科学计算 | 归约误差、统计量或分布式数值计算结果 |

在同一个模型中，decode 的 TP payload 往往比大批量 prefill 小，因此更容易受固定延迟影响。大批量 prefill 更可能显露带宽成本，但仍要看 tensor shape、rank 数和拓扑。

MPI 实现与 NCCL 都可以组织 AllReduce，但各自的执行与同步接口不同。NCCL 的普通 host API 将工作提交到 CUDA stream；CPU 调用返回不代表 GPU 已完成通信。同 stream 消费结果可以依赖 stream 顺序，跨 stream 消费需要 event/wait。在通信完成前，不能覆盖仍在使用的发送 buffer 或释放其存储。[NCCL Stream 语义](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)

CUDA Graph 可以捕获受支持的通信工作，但不会消除通信本身。各 rank 仍需启动匹配的图，数据仍需传输与归约。具体捕获规则见 [多 GPU CUDA Graph](graphs-and-compilation.md#多-gpu-每个-rank-的图包含本地计算和通信工作)。

profiler 中一段很长的通信工作，也可能包含等待其他 rank 的时间。分析原因时，需要结合各 rank 的到达时间、消息大小、实际算法与拓扑。只看一个 NCCL kernel 的持续时间，不能认定网络带宽不足。

本页的轮次、流量和数值表是教学推导。MPI 片段用于说明接口语义，没有运行 MPI、NCCL、GPU 通信或性能测试。

## 13. Recursive Doubling：每轮交换完整部分和

Recursive Doubling 让每张卡都保留当前完整部分和，每轮与一个伙伴双向交换。双方的贡献集合互不重叠，相加后，已经覆盖的 rank 数翻倍。以下先假设 P 是 2 的幂。

仍用 A、B、C、D 表示四张卡的完整输入，每份都是 N 字节：

```text
rank 0: A = [   1,    2,    3,    4]
rank 1: B = [  10,   20,   30,   40]
rank 2: C = [ 100,  200,  300,  400]
rank 3: D = [1000, 2000, 3000, 4000]
```

第一轮，伙伴编号的最低位不同：

```text
rank 0 ↔ rank 1       双方都得到 A+B = [11,22,33,44]
rank 2 ↔ rank 3       双方都得到 C+D = [1100,2200,3300,4400]
```

第二轮，伙伴编号的第二个二进制位不同：

```text
rank 0 ↔ rank 2       双方都得到 (A+B)+(C+D)
rank 1 ↔ rank 3       双方都得到 (A+B)+(C+D)
```

此时四张卡都得到 `[1111,2222,3333,4444]`，不需要额外广播。第二轮发送的是第一轮形成的部分和，不是原始输入。下面是概念伪代码。exchange 表示交换本轮更新前的数据，本地发送、接收完成后才能修改部分和；它不是可直接运行的通信 API：

```text
partial = copy(local_input)
for j = 0 .. log2(P)-1:
    peer = rank XOR (1 << j)
    remote_partial = exchange(peer, partial)
    partial = partial + remote_partial
output = partial
```

每轮开始时，自己和伙伴的部分和必须覆盖互不重叠的 rank 集合；否则会重复计入贡献。MPICH 实现通过逐轮增大的 mask 和 XOR 伙伴满足这个条件。[MPICH Recursive Doubling](https://github.com/pmodels/mpich/blob/main/src/mpi/coll/allreduce/allreduce_intra_recursive_doubling.c)

### 翻倍的是贡献数，不是消息长度

```text
已经覆盖的输入 rank 数：1 → 2 → 4 → 8
每份部分和的字节数：   N → N → N → N
```

因此，每卡的通信量、加法次数分别是：

$$
R_{\mathrm{RD}}=\log_2P,\qquad
V_{\mathrm{send,rank}}=N\log_2P,\qquad
A_{\mathrm{rank}}=L\log_2P
$$

四卡每卡发送 2N；八卡每卡发送 3N。这个 AllReduce 不应与 AllGather 中逐轮扩大已收集数据范围的 recursive doubling 混淆。

在每轮固定延迟为 α、有效单向带宽为 b、每元素归约时间为 γ 的简化模型中：

$$
T_{\mathrm{RD}}\approx\log_2P\left(\alpha+\frac{N}{b}+L\gamma\right)
$$

这个串行相加的模型用于说明轮次、字节和归约成本，不包含实际 kernel 的通信计算重叠。非 2 的幂的 P 需要额外处理；MPICH 会先把部分 rank 配对合并，再在 2 的幂大小的活动组中运行，最后把结果返还。

## 14. Tree 与 Recursive Doubling：相似的归约依赖，不同的结果所有权

两者都逐层合并不同 rank 的贡献。区别在于：单树向上归约通常每组只保留一个继续归约的代表；Recursive Doubling 交换后双方都保留合并结果，并继续参与。

| 四卡例子的第一轮后 | 单树 Reduce | Recursive Doubling |
| --- | --- | --- |
| rank 0 | A+B，继续向上归约 | A+B，继续下一轮 |
| rank 1 | 已提交 B，等待最终结果 | A+B，继续下一轮 |
| rank 2 | C+D，继续向上归约 | C+D，继续下一轮 |
| rank 3 | 已提交 D，等待最终结果 | C+D，继续下一轮 |

单树第二轮后只有 root 拥有完整 Y，还要向下广播。Recursive Doubling 第二轮后，每张卡都已拥有 Y。若只从某张卡的结果向后追溯依赖，Recursive Doubling 也可以展开成归约树；但整个集群并不是沿一棵固定树先上行、再下行。

### 用全系统流量看重复工作的代价

四卡单树向上有 3 次单向 N 字节传输，向下又有 3 次，共 6N。Recursive Doubling 每轮有两对双向交换，即 4 次单向传输；两轮共 8N。

| 指标 | 单树 Reduce + Broadcast | Recursive Doubling |
| --- | --- | --- |
| 结果形成方式 | 先在 root 完整，再分发 | 最后一轮所有 rank 都得到完整结果 |
| 全系统累计发送量 | 2(P−1)N | PN log₂P |
| 全系统加法次数 | (P−1)L | PL log₂P |
| 每卡负载 | 由树中的位置决定 | 对称 |
| 依赖轮次 | 平衡树为 O(log P)，包含上行和下行 | log₂P |

常见未分段二项树调度的上行、下行各有 log₂P 轮；Recursive Doubling 用更多通信和重复归约换掉独立广播。真实 Tree 的层数、分段和流水线由具体实现决定，不应把 `2 log₂P` 直接当作 NCCL kernel 的耗时。

## 15. Rabenseifner：递归折半归约，再倍增收集

Recursive Doubling 每轮交换完整 N 字节，因此总量为 N log₂P。Rabenseifner 用分块减少总量：先以 recursive halving 完成 ReduceScatter，再以 recursive doubling 完成 AllGather。

以 P=8 为例，ReduceScatter 每轮只保留自己后续负责的部分，把另一部分交给伙伴归约。每卡发送量依次是：

```text
ReduceScatter：N/2 → N/4 → N/8
AllGather：   N/8 → N/4 → N/2
```

ReduceScatter 结束时，每卡只有 N/8 字节的最终分块；AllGather 再让每卡获得完整结果。这里 AllGather 的数据范围会增长，而第 13 节的 Recursive Doubling AllReduce 始终交换完整大小的部分和。

$$
R_{\mathrm{Rab}}=2\log_2P,\qquad
V_{\mathrm{send,rank}}=2\frac{P-1}{P}N
$$

八卡时为 6 轮、每卡发送 1.75N；Ring 是 14 轮、同样发送 1.75N。字节相同而逻辑轮次更少，不保证实际时间更短：伙伴选择、拓扑映射、物理链路争用和流水线会改变每轮的有效带宽。这是对算法结构的比较，不表示某个 NCCL 配置一定采用该实现。[MPICH Rabenseifner 实现及成本模型](https://github.com/pmodels/mpich/blob/main/src/mpi/coll/allreduce/allreduce_intra_reduce_scatter_allgather.c)

## 16. One-shot：每卡直接读取所有输入，计算完整输出

本页的 One-shot / Two-shot 指推理框架中常见的 direct peer-access AllReduce，不是 MPI 统一规定的算法名。下面采用 remote read 描述；使用 remote write 的实现可以保持相同的分工，但 buffer 与同步协议会不同。

One-shot 中，每张卡都读取所有原始输入，独立计算完整 Y。它不需要等待其他卡形成 A+B 之类的中间结果，只需等待对方的原始输入就绪。

| 执行计算的卡 | 本地输入 | 远程读取 | 本地最终输出 |
| --- | --- | --- | --- |
| rank 0 | A | B、C、D | A+B+C+D |
| rank 1 | B | A、C、D | A+B+C+D |
| rank 2 | C | A、B、D | A+B+C+D |
| rank 3 | D | A、B、C | A+B+C+D |

例如 rank 0 的线程计算第 0 个位置时，读取 `1、10、100、1000`，累加得到 1111，再写入本地输出。各线程处理不同输出位置。下面是概念伪代码，不是同步协议的完整实现：

```text
等待所有需要的 peer 输入就绪且可见
parallel_for i in 全部输出位置:
    acc = input[rank 0][i]
    for peer = 1 .. P-1:
        acc += input[peer][i]
    local_output[i] = acc
在所有读者完成前，保持源 buffer 有效且不被覆盖
```

GPU kernel 可以通过 peer pointer 直接发出远程加载，按元素或向量包边读取、边在寄存器中累加。不要求先把 P 份完整输入复制到本地临时区。vLLM 的 `cross_device_reduce_1stage` 遍历全部输出位置并执行跨卡归约；FlashInfer 的 MNNVL One-shot 也让每卡执行本地完整归约。[vLLM Custom AllReduce](https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh)、[FlashInfer MNNVL 策略](https://docs.flashinfer.ai/generated/flashinfer.comm.trtllm_mnnvl_ar.trtllm_mnnvl_allreduce.html)

### 一个阶段不等于一条消息，也不等于没有等待

每卡读取其他 P−1 卡的完整输入，因此：

$$
V_{\mathrm{send,rank}}=V_{\mathrm{recv,rank}}=(P-1)N,\qquad
A_{\mathrm{rank}}=(P-1)L
$$

四卡每卡接收 3N、做 3L 次加法；八卡为 7N、7L。所有 rank 都独立归约相同输出，存在重复计算。

One-shot 减少的是“等待远端中间归约结果后，再开始下一轮”的串行依赖。输入发布、peer 就绪、可见性和 buffer 生命周期仍需正确同步。访问 P−1 个 peer 的链路也会争用，不能把“一个阶段”理解成所有数据瞬时到达。

## 17. Two-shot：分工归约输出，再交换最终分块

Two-shot 将完整输出分成 P 个区间，每张卡只负责一个区间的归约。第一阶段完成 ReduceScatter，第二阶段完成 AllGather。

### 第一阶段：每卡只算自己负责的位置

四卡例子中，每个区间先只放一个元素：

```text
rank 0 读取 A[0]、B[0]、C[0]、D[0] → tmp₀ = [1111]
rank 1 读取 A[1]、B[1]、C[1]、D[1] → tmp₁ = [2222]
rank 2 读取 A[2]、B[2]、C[2]、D[2] → tmp₂ = [3333]
rank 3 读取 A[3]、B[3]、C[3]、D[3] → tmp₃ = [4444]
```

每张卡从三个 peer 各读一个 N/4 字节的区间，本地输入不计跨卡传输：

$$
V_1=3N/4
$$

虽然实现可以没有显式的 scatter 调用，但结果已经符合 ReduceScatter 语义：对应位置跨所有 rank 求和，完整结果的不同分块留在不同 rank。

### 第二阶段：读取已经归约好的分块

```text
rank 0：本地 tmp₀，读取 tmp₁、tmp₂、tmp₃
rank 1：读取 tmp₀，本地 tmp₁，读取 tmp₂、tmp₃
rank 2：读取 tmp₀、tmp₁，本地 tmp₂，读取 tmp₃
rank 3：读取 tmp₀、tmp₁、tmp₂，本地 tmp₃

每卡按位置写出 [1111,2222,3333,4444]
```

这一步只收集最终分块，不再数值相加。每卡从其他三卡各读 N/4 字节，第二阶段也是 3N/4。一般化后：

$$
V_{\mathrm{send,rank}}=V_{\mathrm{recv,rank}}
=2\frac{P-1}{P}N,\qquad
A_{\mathrm{rank}}=\frac{P-1}{P}L
$$

归约工作只覆盖本卡负责的 L/P 个输出位置，每个位置需要 P−1 次加法。全系统对每个位置归约一次，再分发结果；One-shot 则在 P 张卡上各算一次。

### 阶段间的依赖与 buffer 生命周期

```text
等待原始输入就绪且可见
parallel_for i in 本卡负责的输出区间:
    local_tmp[i] = sum(input[peer][i] for peer in 所有 ranks)
发布本卡结果区间已就绪

for owner in 所有 ranks:
    等待 owner 的结果区间就绪且可见
    将 tmp[owner] 复制到本地输出的对应区间
```

真实实现可以按 chunk 发布结果并流水化消费，但读取某个分块前，owner 必须完成该分块的归约，并正确保证跨设备可见性。临时结果也必须保持有效，直到相关读者完成。

Two-shot 不规定恰好两个 kernel，也不规定恰好两个全组 barrier。已阅读的 vLLM `cross_device_reduce_2stage` 在同一个 kernel 内包含归约、阶段间同步和收集；其具体线程映射与可见性约束不能用上述伪代码替代。[vLLM Two-stage 实现](https://github.com/vllm-project/vllm/blob/main/csrc/custom_all_reduce.cuh)

## 18. 用同一口径比较通信、计算与固定成本

以下公式假设等长分片；对数算法另假设 P 为 2 的幂。加法次数是归约所需的标量加法数量，不包含地址计算、同步和类型转换。

| 算法 | 依赖轮次 / 阶段 | 每卡发送量 | 每卡加法次数 |
| --- | --- | --- | --- |
| Ring | 2(P−1) 轮 | 2(P−1)N/P | (P−1)L/P |
| Recursive Doubling | log₂P 轮 | N log₂P | L log₂P |
| Rabenseifner | 2 log₂P 轮 | 2(P−1)N/P | (P−1)L/P |
| Direct One-shot | 一个主要数据交换阶段 | (P−1)N | (P−1)L |
| Direct Two-shot | 两个主要数据阶段 | 2(P−1)N/P | (P−1)L/P |
| 单树 Reduce + Broadcast | O(log P)，包含上行、下行 | 节点 r 为 d_rN；平均 2(P−1)N/P | 不均衡；全系统 (P−1)L |

每卡接收量在这些模型中等于相应发送量。对于单树，有 c_r 个孩子的节点承担 c_rL 次加法。Direct 的一个阶段访问多个 peer，与 Ring 的邻居交换一轮不是同一种工作单位。

### 八卡、每卡输入 16 MiB 的手算

| 算法 | 每卡发送 | 每卡接收 | 全系统累计发送 |
| --- | --- | --- | --- |
| Ring | 28 MiB | 28 MiB | 224 MiB |
| Recursive Doubling | 48 MiB | 48 MiB | 384 MiB |
| Rabenseifner | 28 MiB | 28 MiB | 224 MiB |
| One-shot | 112 MiB | 112 MiB | 896 MiB |
| Two-shot | 28 MiB | 28 MiB | 224 MiB |
| 单树 | 平均 28 MiB，各卡不等 | 平均 28 MiB，各卡不等 | 224 MiB |

One-shot 与 Two-shot 的通信量比例为：

$$
\frac{V_{\mathrm{one}}}{V_{\mathrm{two}}}
=\frac{(P-1)N}{2(P-1)N/P}=\frac P2
$$

四卡相差 2 倍，八卡相差 4 倍；归约加法次数则相差 P 倍。P=2 时两者有效通信量相同，Two-shot 的额外阶段并不会带来字节数收益。

### 小消息为何可能选择通信量更多的算法

One-shot 可以在读取原始输入后直接写出本地最终结果。Two-shot 要发布并消费中间归约分块，多出阶段依赖和临时区读写。对小 payload，固定成本可能比节省的传输时间更大。

用局部线性近似表达这种权衡：

$$
T_{\mathrm{one}}\approx a_1+b_1N,\qquad
T_{\mathrm{two}}\approx a_2+b_2N
$$

常见趋势是 `a₁ < a₂`、`b₁ > b₂`：One-shot 固定成本小，但随 payload 增长更快。在这些前提成立且线性近似有效的区间内，交叉点为 `N* = (a₂−a₁)/(b₁−b₂)`。a 的单位是时间，b 的单位是时间/字节。

这不是可直接用于配置的通用阈值。b 包含实际链路、访存和归约吞吐的影响，a 也受同步、调度、融合与 rank 到达时间影响。FlashInfer 将 MNNVL One-shot 定位为较小 payload 的低延迟路径，将 Two-shot 定位为较大 payload 的吞吐路径；阈值仍依赖实际实现与硬件。[FlashInfer AllReduce 策略](https://docs.flashinfer.ai/generated/flashinfer.comm.trtllm_mnnvl_ar.trtllm_mnnvl_allreduce.html)

### algbw、busbw 与物理流量

NCCL-tests 的 AllReduce 带宽口径为：

$$
\mathrm{algbw}=\frac Nt,\qquad
\mathrm{busbw}=\mathrm{algbw}\cdot2\frac{P-1}{P}
$$

这里 N 是每个 rank 的完整输入字节数。busbw 使用标准通信量因子归一化，不是逐条读取 NVLink 或网卡计数器得到的物理流量。尤其不能将这个系数当作 One-shot、硬件 multicast 或 in-network reduction 的真实流量证明。[NCCL-tests 性能指标定义](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

跨节点可以采用分层算法，NVLS 可以使用 NVLink SHARP 硬件能力。分析这些路径时，要分别统计 GPU 注入量、接收量与 fabric 内部流量，不能直接照搬纯软件 peer-to-peer 模型。[NCCL 算法与 NVLS 配置](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)

## 19. 从 AllReduce 成本回到 TP / EP

并行方案先决定权重和 token 的分布，再决定本地 GEMM shape 与通信张量。比较 TP / EP 时，必须固定全局工作量，并追踪哪些布局转换位于关键路径。GEMM 内部的 tiling、register tiling 和 split-K 见 [GEMM 计算与优化](../CUDA/gemm.md)，这里专门计算通信。

### TP：每卡计算减少，但输出部分和的大小不随 P 等比例下降

取一个普通两层 MLP，隐藏维 H=4096，中间维 F=16384，M 个 token。忽略 bias，使用逐元素激活 φ，TP8 的前向路径是：

```text
每卡相同输入 X [M,4096]
    ↓ column-parallel
本卡 W1_r [4096,2048]
Z_r = φ(X @ W1_r)             [M,2048]
    ↓ row-parallel
本卡 W2_r [2048,4096]
Y_partial_r = Z_r @ W2_r      [M,4096]
    ↓ AllReduce(SUM)
每卡完整 Y                    [M,4096]
```

第一层输出是不同列，激活后可以直接交给匹配的第二层权重分片。第二层输出是相同位置的不同部分和，需要归约。这个普通 MLP 每卡 FLOPs 约为 `4MHF/P`，AR 输入字节数却是：

$$
N=sMH
$$

BF16 下，M=1 时 N=8 KiB；M=128 时 N=1 MiB。TP8 Ring 的每卡发送量分别为 14 KiB 和 1.75 MiB，One-shot 则分别为 56 KiB 和 7 MiB。

增加 TP 时，每卡计算和权重读取可以减少，但同一批 token 的 `[M,H]` 部分和不会自动再除以 P。小 payload 的固定通信成本可能逐渐占据关键路径；较大 payload 还要比较算法的流量项。经典 Megatron dense Transformer 前向在 attention 输出和 MLP 输出各有一次 AR，但 sequence parallel、融合和其他布局会改变实际 collective，不能把两次 AR 当作所有模型的固定规则。[Megatron TP 原始设计](https://research.nvidia.com/labs/adlr/MegatronLM/)

Row-parallel TP 与单卡 split-K 都把矩阵乘法的 K 维求和拆开，之后合并部分和；前者跨 GPU 合并，后者在单卡内合并。它们共享数学分解，不共享同步与传输成本。

### EP：通信是路由 token，而每个 expert 的 batch 由全局路由决定

纯 EP 把完整 experts 放到不同卡上。常见路径是：

```text
源 GPU 的 token
    → dispatch 到 expert 所在 GPU
    → 按 expert 分组并执行 grouped GEMM
    → combine 返回源 GPU
    → 按路由权重合并 top-k 输出
```

假设全局有 T 个 token、E 个 experts，每 token 选择 k 个 experts，路由均衡，则每 expert 的输入行数约为 `M_e = Tk/E`。固定 T、E、k 时，增加 EP 卡数不会自动增加 M_e，只会改变每卡持有的 expert 数和路由位置。

例如 T=1024、E=64、k=2、EP8，每 expert 平均接到 32 个 token，每卡持有 8 个 experts。Grouped GEMM 可以统一调度这些任务，但不同 expert 使用不同权重，不能把它们的权重复用当成同一份权重上的 M=256 GEMM。[Megatron MoE 与 Grouped GEMM](https://docs.nvidia.com/megatron-core/developer-guide/0.15.0/user-guide/features/moe.html)

令每卡原有 t 个 token，hidden size 为 H，输入与输出通信精度均为 s 字节，远程 expert assignment 比例为 f。假设各卡工作均衡，每个 assignment 单独传输，且没有目标卡去重、压缩或 padding。此时每卡平均发送量近似为：

$$
V_{\mathrm{dispatch}}\approx tkHsf,\qquad
V_{\mathrm{dispatch+combine}}\approx2tkHsf
$$

同一个 token 的多个目标 expert 可能位于同一卡，实际 dispatcher 可以合并传输；metadata、量化、分层转发、负载不均衡也会改变成本。EP 的 dispatch / combine 是 All-to-All 类数据重分布，不能直接套用 AR 的 `2(P−1)/P` 系数。

### 判断真实瓶颈时要记录什么

| 需要记录的对象 | 对性能判断的作用 |
| --- | --- |
| 每卡 token 数、每 expert 的 M_e、权重分片 shape | 确认本地计算工作量和负载是否相同 |
| 通信张量的 shape、dtype 与归约语义 | 明确 N，并判断是否必须复制完整结果 |
| collective 次数、实际算法、rank 组和拓扑 | 区分固定成本、有效流量和物理拥塞 |
| 各 rank 的输入就绪时间、通信完成时间与消费者 | 区分数据传输和等待慢 rank，定位无法隐藏的通信 |
| 融合前后的中间张量读写和 launch | 区分网络流量收益与本地 HBM / 调度收益 |

例如 AR 与 residual / RMSNorm 融合，可以减少中间结果的本地读写和 launch，但不意味着跨卡字节数一定减少。通信与计算重叠也要求存在可独立推进的工作，并且两者不会因争用 SM、HBM 或链路而抵消收益。

本页的算法字节数、加法次数和 TP / EP 示例只建立可手算的基线。实际 latency、吞吐和算法切换点，需要在固定 shape、dtype、rank 数与拓扑下测量，并结合多 rank 时间线核对。
