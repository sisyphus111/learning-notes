# 案例：Qwen3-32B 在 H100 上的 TP、GEMM 与 roofline

## 1. 分析对象、结论与证据范围

以 Qwen3-32B dense Transformer、H100 SXM 80 GB 为例，分析 BF16 输入与权重、FP32 累加的推理路径。先讨论一个 FFN 投影，再扩展到完整 FFN、attention 和 decode step。

**TP 通过多卡同时提供计算能力与显存带宽，降低同一批工作的完成时间。GEMM 通过多行输入复用权重，降低每行的摊销成本。** TP 分片不自动改善单卡 kernel 效率，新增通信与较小的工作规模可能抵消收益。

模型和硬件尺寸来自官方资料。流量、FLOPs 与时间来自公式推导。CTA/warp/register 配置是可核算的教学设计。未执行 H100 benchmark，也没有确认 cuBLAS、SGLang 或其他后端为这些 shape 实际选择的 kernel。

基础材料：[Transformer 运行全过程](transformer-runtime.md)、[GEMM 分块与流水线](../CUDA/gemm.md)、[通信与 AllReduce](../GPU/communication-and-allreduce.md)。

## 2. 固定模型尺寸与矩阵方向

| Qwen3-32B 参数 | 数值 |
| --- | ---: |
| 总参数量 | 约 32.8B |
| Transformer 层数 | 64 |
| hidden size H | 5120 |
| FFN intermediate size F | 25600 |
| Q heads / KV heads | 64 / 8 |
| head dimension | 128 |

Q 的投影宽度是 $64\times128=8192$，K/V 各为 $8\times128=1024$。不能由 hidden size 5120 推断 Q 的输出一定也是 5120。[模型卡][model]、[官方配置][config]

以下权重按数学上的右乘方向书写。PyTorch Linear 实际常保存 `[out_features, in_features]`，计算时使用转置；逻辑方向不要求每次物理转置权重。

$$
G=XW_g,\quad U=XW_u,\quad Z=\operatorname{SiLU}(G)\odot U,\quad Y=ZW_d.
$$

```text
X：       [M, 5120]
Wg / Wu： [5120, 25600]
G/U/Z：   [M, 25600]
Wd：      [25600, 5120]
Y：       [M, 5120]
```

M 是本次 Linear 一起处理的 token 行数。单请求单 token decode 通常 `M=1`；32 个请求各生成一个 token 可形成 `M=32`；prefill 的 1024 个已知 token 可形成 `M=1024`。

同一请求未来尚未生成的 token 依赖当前输出，不能直接合进当前 GEMM。batch、prefill、部分 speculative verification 才能提供已知的多行输入。

一个 up 权重含 $5120\times25600=131072000$ 个参数。BF16 每元素 2 bytes：

$$
|W_u|=262144000\text{ bytes}=262.144\text{ MB}=250\text{ MiB}.
$$

即使只算一行输入，仍要使用这整份权重。本文 MB/GB/TB 采用十进制，MiB/GiB/KiB 采用二进制。

## 3. H100 资源与理想 roofline

| H100 SXM 资源 | 本页采用的数值 |
| --- | ---: |
| BF16 dense Tensor Core 峰值 | 约 989 TFLOP/s |
| HBM 带宽 | 3.35 TB/s |
| SM 数量 | 132 |
| L2 cache | 50 MB |
| register file / SM | 65536 个 32-bit registers，即 256 KiB |
| shared memory / SM 上限 | 228 KiB |
| 最大驻留 warps / SM | 64 |

官网 BF16 的约 1979 TFLOP/s 标有 sparsity，dense 分析不能直接使用该数值。本页使用约一半的 dense 峰值。每 block 可用 shared memory、寄存器分配粒度和其他资源还有额外约束。[H100 规格][h100]、[架构与 SM 数][hopper-architecture]、[资源限制][hopper-tuning]

对于 $C[M,N]=A[M,K]B[K,N]$，主要计算量为 $F_{ops}=2MNK$，FMA 的乘法与加法各计一次 FLOP。若 A/B 各从 HBM 读取一次、BF16 C 写出一次，且不读旧 C：

$$
D_{min}=2(MK+KN+MN)\text{ bytes},\qquad I=\frac{F_{ops}}{D_{min}}=\frac{MNK}{MK+KN+MN}.
$$

I 的单位为 FLOP/byte。该流量是假设充分复用的理想边界；实际 HBM 流量可能受重复加载、cache 和额外工作影响。

$$
P_{roof}(I)=\min(P_{peak},B_{HBM}I),\qquad T\ge\max\left(\frac{F_{ops}}{P_{peak}},\frac{D_{HBM}}{B_{HBM}}\right).
$$

本页 H100 的理论转折点为：

$$
I_{ridge}=\frac{989\times10^{12}}{3.35\times10^{12}}\approx295\text{ FLOP/byte}.
$$

```text
吞吐上限
    ↑                  ┌────────── 计算峰值
    |                ／
    |              ／   左侧：HBM 供给约束
    |            ／     右侧：计算峰值约束
    └────────────────────────────→ 算术强度
                    约 295 FLOP/byte
```

图只说明分段关系，坐标和斜率不按比例绘制。真实 roofline 还受形状、指令与片上资源限制，不是只把 FLOPs 除以峰值。[NVIDIA GEMM 性能分析][gemm-performance]

## 4. 从 GEMV 到 GEMM：复用同一份权重

固定 up projection 的 `K=5120, N=25600`。M 相对 K/N 较小时，权重占主要流量：

$$
D\approx2KN,\qquad F_{ops}=2MKN,\qquad I\approx M\text{ FLOP/byte}.
$$

因此增加 M，能让权重读取服务更多 token。这里是“输入行数提高复用”。每个权重不必在同一个 CTA 内服务全部 M 行。CTA 内复用与 L2 跨 CTA 复用共同影响实际流量。

| M | I，FLOP/byte | 计算下界，μs | HBM 下界，μs | 两者最大值，μs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.00 | 0.27 | 78.27 | 78.27 |
| 32 | 31.76 | 8.48 | 78.84 | 78.84 |
| 128 | 124.27 | 33.93 | 80.60 | 80.60 |
| 512 | 457.14 | 135.71 | 87.64 | 135.71 |
| 1024 | 825.81 | 271.42 | 97.03 | 271.42 |

这些数字统一使用 989 TFLOP/s 与理想 BF16 流量，不含 launch、padding、实际利用率和 epilogue。M=1 对应 GEMV，不能假设它达到 Tensor Core 峰值。若采用 FP32 CUDA Core FMA，按约 67 TFLOP/s 计算，算术下界约 3.9 μs。这个数仍小于约 78.3 μs 的权重读取项。

M=128 时，FLOPs 是 M=1 的 128 倍，理想总流量却只从约 262 MB 增到 270 MB。它在这个模型下仍偏 HBM 受限；M=512 才转向计算受限。不能把所有 GEMM 都称为 compute-bound。

若连续执行 128 次 GEMV，且每次都重新从 HBM 读权重，读取项合计约 10 ms。合成一个 M=128 GEMM，理想模型约 80.6 μs。这是不同复用假设下的推导，不是实测加速比。

### 延迟和每行摊销成本

$$
\text{行吞吐}=M/T_{kernel},\qquad\text{每行摊销成本}=T_{kernel}/M.
$$

`80.6 μs / 128 ≈ 0.63 μs` 是每行的摊销成本。batch 内每个请求仍需等待整个约 80.6 μs 的工作，不能说单请求延迟只有 0.63 μs。收集 batch 的等待也可能增加端到端延迟。

## 5. TP4 怎样切完整 Transformer block

采用常见的 replicated hidden state、column-parallel → row-parallel 组织。四张卡在每层开始持有相同的 `[M,5120]` 输入。

| 线性层 | TP1 的逻辑权重 | TP4 每卡权重 | 分工 |
| --- | --- | --- | --- |
| Q | `[5120,8192]` | `[5120,2048]` | 16 个 Q heads / rank |
| K、V，各一个 | `[5120,1024]` | `[5120,256]` | 2 个 KV heads / rank |
| attention 输出投影 | `[8192,5120]` | `[2048,5120]` | 输出完整形状的部分和 |
| gate、up，各一个 | `[5120,25600]` | `[5120,6400]` | FFN 特征分片 |
| down | `[25600,5120]` | `[6400,5120]` | 输出完整形状的部分和 |

Q/K/V 按匹配的 GQA 分组切分，各卡本地完成自己 heads 的 attention。输出投影后的部分和通过 AllReduce 得到完整 hidden state，再做残差与后续 norm。FFN 中则是：

```text
相同 X [M,5120]
    ├→ rank 0：gate/up → SwiGLU → down 部分和 ─┐
    ├→ rank 1：gate/up → SwiGLU → down 部分和 ─┤
    ├→ rank 2：gate/up → SwiGLU → down 部分和 ─┼→ AllReduce → Y [M,5120]
    └→ rank 3：gate/up → SwiGLU → down 部分和 ─┘
```

每卡 Z 为 `[M,6400]`。SwiGLU 是逐元素计算，所以 gate/up 后不必先 AllGather 出 `[M,25600]`。down 的各分片直接消费本地 Z，得到 `[M,5120]` 部分和：

$$
Y=\sum_{p=0}^{3}Z_pW_{d,p}.
$$

**M 不除以 TP 数。** 每张卡处理同一批 token，只负责其中一部分特征或归约贡献。标准布局每层 forward 在 attention 输出、FFN 输出各做一次 AllReduce。sequence parallel、融合归约或其他布局可以改变通信形式，需按实际执行路径分析。[Megatron TP 的列切分与行切分][megatron]

## 6. TP 的本地计算收益与算术强度

up 在 TP4 每卡计算 `[M,5120] @ [5120,6400]`。每卡权重为 65.536 MB，FLOPs 也变成原来的四分之一。四张卡同时执行，墙钟时间由最慢 rank 决定，不将四卡时间相加。

| M | TP1 投影理想模型，μs | TP4 每卡投影理想模型，μs |
| ---: | ---: | ---: |
| 1 | 78.27 | 19.57 |
| 128 | 80.60 | 20.44 |
| 1024 | 271.42 | 67.86 |

小 M 下主要减少每卡权重读取；大 M 下可同时利用多卡计算能力。但 TP 不自动提高每卡 I。column parallel 的强度为：

$$
I_p=\frac{2MK(N/p)}{2(MK+K(N/p)+M(N/p))}.
$$

M=128 时，TP1 为 124.27 FLOP/byte，TP4 为 122.49 FLOP/byte，反而略低：每卡仍读取完整 X，但只算四分之一输出列。理想每卡时间缩短，主要来自工作量减少与资源并行。

## 7. AllReduce 怎样抵消延迟收益

FFN 的归约输出是 `[M,5120]`。假设使用 BF16 作为通信 dtype，payload 为：

$$
S=2M\times5120\text{ bytes}.
$$

以 Ring 为示例，p 个 rank 的每卡发送量为 $2(p-1)S/p$；TP4 为 `1.5S`。H100 的 NVLink 900 GB/s 是双向总量，用单向发送量估计时采用 450 GB/s 的理论单向值，不能直接代入 900。

$$
T_{AR}\approx\alpha_{AR}+\frac{1.5S}{B_{net}}.
$$

$\alpha_{AR}$ 表示启动、协议和同步的固定项，需要测量。实际 NCCL 可能选择不同算法，其有效带宽也低于硬件峰值。通信若采用 FP32，payload 和这里也不同。

先把三个投影作为独立 GEMM 相加，暂不计 SwiGLU：

| M | TP1 三个投影，μs | TP4 本地三个投影，μs | Ring 纯带宽项，μs |
| ---: | ---: | ---: | ---: |
| 1 | 234.81 | 58.71 | 0.034 |
| 128 | 241.80 | 61.33 | 4.37 |
| 1024 | 814.26 | 203.57 | 34.95 |

`0.034 μs` 不是实际 AllReduce 延迟，小消息通常由固定开销主导。上述 down 投影与 up 的 K/N 对调，故在该简化流量模型下具有相同的时间项。真实 gate/up 常融合为一次 GEMM，不能把表当成完整 FFN 的测量。

$$
T_{FFN,TP4}\approx T_{local\ matmuls}+T_{SwiGLU}+\alpha_{AR}+T_{AR,data}+T_{other}.
$$

down 与消费它的归约有数据依赖，不能无条件将两者完全重叠。完整 64 层还有 attention、残差、norm、KV 访问、LM head 和调度。理想下界的比值也不是实际 speedup 上限或保证。

TP 的条件收益是：本地计算/搬运省下的时间，大于通信、同步及形状效率损失。单卡矩阵很小、TP 度数过高或互联较慢时，这个条件可能不成立。

## 8. kernel 与 CTA：一个具体的分块设计

对 `M=1024` 的 up projection，设计教学 tile：`BM=128, BN=128, BK=64`。这是普通非 persistent 调度示例，不代表库的实际算法。

```text
整个 GEMM：[1024,5120] @ [5120,25600]
一个 CTA：负责 C 的 [128,128] 输出 tile
一次 K 迭代：A[128,64] @ B[64,128]，累加到同一个 C tile
```

每轮 A/B 各 16 KiB，共 32 KiB。FP32 C 累加状态为 $128\times128\times4=64$ KiB，但它分布在线程寄存器中，不要加到 shared memory 分配里。

完整 K 循环有 $5120/64=80$ 轮。每轮：

$$
F_{tile}=2\times128\times128\times64=2097152\text{ FLOP}.
$$

相对将这对输入 tile 装入 shared memory 的 32768 bytes，强度为 64 FLOP/byte。A 元素可服务 128 个输出列，B 元素可服务 128 个输入行。

这是 tile 装载边界的强度，不是整个 GEMM 的 HBM 强度，也不是 WGMMA 在 shared memory 上实际读取字节的完整统计。不同 MMA 子块可能重读操作数；跨 CTA 的 L2 复用又可能减少 HBM 流量。分析时必须固定所讨论的存储层级。[CUTLASS 的分层 GEMM][cutlass]

### shared memory 与 warp 之间的数据交接

普通 shared memory 物理上来自 SM，逻辑上按 CTA 分配，同一 CTA 的多个 warp 可以共同访问。它不是“每个 warp 的独立缓存”。跨 CTA 共享则需要 cluster/DSMEM 等额外机制，不是本例的默认路径。

使用两个缓冲槽时，A/B 数据共占约 64 KiB：

```text
producer：将下一对 A/B tile 搬入空闲槽
                  ↓ 数据到达通知
consumer：读取就绪槽，完成本轮 MMA
                  ↓ 消费完成通知
producer：允许覆盖旧槽
```

TMA 可以由选出的线程发起，搬运绕开普通数据寄存器中转。buffer 复用必须等待 consumer 使用完；只确认 TMA 完成还不够。bank conflict、swizzle 和多级缓冲的基础见 [GEMM 笔记](../CUDA/gemm.md)。

## 9. warpgroup 与每个 thread 的 accumulator

示意 CTA 使用 4 个 producer warps 和 4 个 consumer warps，共 256 threads。consumer 的四个 warp 构成一个 128-thread warpgroup；这不是说 TMA 必须由 128 threads 一起搬运。

BF16 WGMMA 的一个形状是 `m64n128k16`：

```text
[64,16] @ [16,128] → [64,128] 的 FP32 累加结果
```

warpgroup 的 128 threads 集体参与。为了覆盖本例 `[128,128]` 输出，一个 warpgroup 可以先后更新两个 64 行子块；BK=64 又分四个 K=16 片段。每轮有 $2\times4=8$ 次 warpgroup MMA，80 轮共 640 次。这是集体指令次数，不是每个线程独立计算一整块矩阵。[WGMMA 数据流与线程协作][wgmma]

只计算 C accumulator，整个 tile 有 16384 个 FP32 值，分给 128 个 consumer threads：

$$
16384/128=128\text{ 个 FP32 accumulator/thread}.
$$

也就是每线程仅输出累加器就需要 128 个 32-bit registers。这些状态跨 80 轮持续保留，最后才写出 BF16 输出，避免每轮把部分和写回全局内存。

线程还需要地址、descriptor、循环状态和 epilogue 临时值，实际寄存器需求高于 accumulator 数。Hopper 的 WGMMA 累加结果位于寄存器，不是 Blackwell SM100 的 TMEM。

## 10. register file 怎样限制驻留与吞吐

假设实现支持按角色分配寄存器，预算为 producer 每线程 32、consumer 每线程 160，则每 CTA 需要：

$$
128\times32+128\times160=24576\text{ 个 32-bit registers}=96\text{ KiB}.
$$

单看寄存器总量，每 SM 的 CTA 数上界为：

$$
\lfloor65536/24576\rfloor=2.
$$

shared memory 单看 64 KiB 数据槽可给出 $\lfloor228/64\rfloor=3$ 的粗略上界。在这组预算中，register file 更早限制到最多两个 CTA。两个 CTA 共 16 warps，相对 64 个最大驻留 warps，为 25% 的理论 warp occupancy。

实际还受分配粒度、SM 子分区、barrier、epilogue scratch 和编译配置限制。不同角色预算需要正确的 launch/寄存器配置，可能涉及 `setmaxnreg`。不能拿源代码里实际使用的变量数量直接计算 occupancy。

register file 物理上属于 SM，但每个 thread 的寄存器是私有逻辑状态。另一个 warp 不能随意读取你的局部变量。warp 内可用 shuffle；跨 warp 交换通常使用 shared memory 和相应同步。

更大 tile 增加复用，也增加 accumulator 和临时寄存器，可能减少驻留或引起 spill。spill 的 local memory 访问会经过缓存/全局内存层级。反过来，25% occupancy 也不等于 25% Tensor Core 利用率：足够的独立工作与流水线可能已能持续供给计算。

## 11. GEMV 的 thread/warp 分工为何不同

M=1 时，$y[n]=\sum_k x[k]W[k,n]$。假设权重物理保存为 `[N,K]`、K 连续，可让一个 warp 负责 4 个输出。lane t 处理 `t, t+32, t+64, ...` 的 K 位置，每线程约处理 160 个 K 位置，为 4 个输出保留 4 个 FP32 部分和。最后做 warp 内归约。

相邻 lanes 访问同一权重行的相邻 K 元素，便于合并访存。实际实现可进一步打包 BF16、向量化加载。这个设计不需要不同 warp 合并同一个输出；若沿 K 在多个 warp/CTA 间分工，就需额外归约。

| 项目 | GEMV 教学设计 | GEMM 教学设计 |
| --- | --- | --- |
| 权重服务的输入行 | 1 | CTA 内 128 |
| 每线程 accumulator | 示例为 4 个 | consumer 示例为 128 个 |
| 主要工作 | 并行点积与归约 | WGMMA tile 累加 |
| shared memory 用途 | 可复用 x 或交换部分和 | A/B tile 复用与 MMA 操作数准备 |
| 关键限制 | 权重流量、访存并发、启动 | 复用、Tensor Core 供给、寄存器与流水线 |

GEMV 可以有很多 CTA，不是因为只有一个向量就只能用一个线程。它缺少的是“同一权重服务多行输入”的复用。库也可能采用窄矩阵 Tensor Core 路径，需将 padding 与有效计算利用率纳入分析。

## 12. TP 怎样改变 CTA 数与 K 循环

继续使用 `BM=BN=128, BK=64`，固定 M=128：

| 投影 | 配置 | 每卡 CTA 数 | 每 CTA 的 K 轮数 |
| --- | --- | ---: | ---: |
| up | TP1：K=5120，N=25600 | 200 | 80 |
| up | TP4：K=5120，N=6400 | 50 | 80 |
| down | TP1：K=25600，N=5120 | 40 | 400 |
| down | TP4：K=6400，N=5120 | 40 | 100 |

column parallel 缩短 N，CTA 变少而 K 循环不变；row parallel 缩短 K，CTA 数可以不变，但每个 CTA 更早完成。若 CTA tile 不变，每线程 accumulator 也不会自动变成四分之一。

H100 有 132 个 SM，50 个 CTA 无法让所有 SM 同时各驻留一个 CTA。不过不能直接将 HBM 带宽乘 `50/132`。部分 SM 仍可能发出足够多的访存请求，计算覆盖率与带宽利用率需要分别检查。

缩小 tile 可增加 CTA，但损失复用。例如把 BN 从 128 降到 32，CTA 数增加 4 倍，同一 tile 装载边界的强度变成：

$$
I_{stage}=\frac{2B_MB_NB_K}{2(B_MB_K+B_KB_N)}:\quad64\to25.6\text{ FLOP/byte}.
$$

库还可选择 split-K、persistent 或交换矩阵方向。fused gate/up 的 N 又是两个输出拼接后的宽度，CTA 数与单个投影不同。表格用于解释机制，不能作为 cuBLAS 的实际 launch geometry。

## 13. 系统吞吐、单请求延迟与显存容量

同一个 decode step 的系统输出吞吐近似为“本步生成 token 数 / step 时间”。用户的 inter-token latency 接近 step 时间，而不是 step 时间除以 batch size。prefill 的 TTFT、decode ITL 与单 kernel 耗时也需分开。

若有四张 H100，应比较四卡 TP 与四个单卡副本，而不能只比较 TP4 和一张卡。低并发下 TP 让单请求使用多卡；总 batch 有限时，TP 保留 M，可避免把 batch 分给副本后降低复用。高并发、单卡已能形成高效大 GEMM 时，多副本避免 TP 通信，可能提供更高系统吞吐。

TP 也改变容量。BF16 权重约 65.6 GB，还需 KV、激活与 workspace。该模型每个历史 token 的全部 64 层 KV 为：

$$
2_{K/V}\times64\times8\times128\times2=262144\text{ bytes}=256\text{ KiB}.
$$

128 个请求、每个 4096-token 上下文，仅 KV 就约 128 GiB。因此本页 M=128 可用于单层算子或 prefill 分析，不能默认这种长上下文 decode batch 能装进一张 H100。TP4 按 KV heads 分片后，每 rank 对应约 64 KiB/历史 token，还会影响可服务的 batch。

模型级结果还包括 KV 读取、attention 计算、norm、采样与调度。单个 FFN 投影的 4 倍理想缩短，不等于完整 Transformer 的 4 倍端到端加速。

## 14. 怎样复核这个案例

数值复算脚本：[qwen3_h100_tp_case.py](https://github.com/sisyphus111/learning-notes/blob/main/scripts/cases/qwen3_h100_tp_case.py)。仅使用 Python 标准库计算数学模型，不导入 Torch、不访问 GPU。

```sh
.venv/bin/python scripts/cases/qwen3_h100_tp_case.py
```

实际验证应固定模型、batch、上下文、dtype、TP、GPU 时钟与互联拓扑。先用 Nsight Systems 确认 GEMM、通信和 host gap 的关键路径，再对相关 kernels 用 Nsight Compute 检查：

| 层级 | 所需证据 |
| --- | --- |
| 请求 / step | TTFT、ITL、输出 tokens/s、相同资源下副本对照 |
| kernel | 实际 shape、算法、耗时、有效 FLOP/s |
| 通信 | payload、算法、固定开销、有效带宽、暴露的等待 |
| HBM / L2 | 实际字节数、命中、读写带宽 |
| CTA / SM | tile、grid、驻留、SM 覆盖和尾部 |
| warp / shared | MMA 供给、访存与 barrier 等待、shared 流量、bank conflict |
| thread / register | registers/thread、角色分配、spill 与临时值 |

TP8 的 up 权重分片约 32.8 MB，小于 H100 的 50 MB L2。反复测同一分片可能变成热缓存测试；完整模型轮流访问各层时未必如此。比较 TP1/TP4/TP8 要说明缓存状态，不能将 cache 命中差异误算成纯 HBM 带宽扩展。

## 15. 来源与适用范围

模型配置、硬件规格与机制的资料基准日期为 2026-10-28。表格是理想模型，寄存器预算是教学配置，均不代表实测。可先用它形成假设，再对实际后端与机器验证。

- [Qwen3-32B 模型卡][model]与[配置][config]：模型和矩阵尺寸。
- [H100 产品规格][h100]：最终产品算力、带宽与互联，注意 dense / sparse 区别。
- [Hopper 架构][hopper-architecture]与[Tuning Guide][hopper-tuning]：SM、register file、shared memory 与异步机制。
- [NVIDIA GEMM 性能背景][gemm-performance]：FLOPs、算术强度、tiling 与性能边界。
- [CUTLASS Efficient GEMM][cutlass]与[WGMMA 指南][wgmma]：CTA/warpgroup 分层、存储路径。
- [Megatron TP][megatron]与[NCCL collective 语义][nccl]：列/行切分组合和输出归约。Ring 流量推导见本仓库的[通信笔记](../GPU/communication-and-allreduce.md)。

[model]: https://huggingface.co/Qwen/Qwen3-32B
[config]: https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json
[h100]: https://www.nvidia.com/en-us/data-center/h100/
[hopper-architecture]: https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
[hopper-tuning]: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
[gemm-performance]: https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
[cutlass]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
[wgmma]: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/wgmma_programming.html
[megatron]: https://research.nvidia.com/labs/adlr/MegatronLM/
[nccl]: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html
