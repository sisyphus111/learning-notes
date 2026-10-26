# 通信基础：MPI 与 Ring / Tree AllReduce

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
          /
      rank 3
```

### 向上归约

叶子向父节点发送数据。父节点将子节点的贡献与自己的输入相加，再向上传递：

```text
第 1 层：rank 3 → rank 1，形成 X₁ + X₃
         rank 2 → rank 0，形成 X₀ + X₂

第 2 层：rank 1 → rank 0
         rank 0 得到 X₀ + X₁ + X₂ + X₃
```

单看第一个元素，rank 1 得到 `10+1000=1010`，rank 0 先得到 `1+100=101`，再得到 `101+1010=1111`。完整向量的其他位置采用相同操作。

### 向下广播

root 得到完整结果后，把结果交给子节点，再由子节点继续转发：

```text
第 1 层：rank 0 → rank 1、rank 2
第 2 层：rank 1 → rank 3
```

这个基础模型是 Reduce 加 Broadcast。它沿树向上合并完整向量，而 Ring 让各 rank 分担不同块的最终归约。实际 Tree 可以分段，在下游已处理前一段时继续处理下一段。

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

这些统计不包含协议头、重传与共享物理链路的重复负载。multicast 或硬件归约需要按实际传输机制另行计数。

## 10. Double Binary Tree 怎样改善单树

把数据分成两半，分别交给两棵互补的树：

```text
前半部分 → Tree A
后半部分 → Tree B

让 Tree A 的内部节点尽量在 Tree B 中做叶子，反之亦然。
```

一棵树中负责转发的 rank，在另一棵树中少做转发。两棵树配合分段流水线，分担内部节点的压力，同时保留较低的依赖深度。

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
