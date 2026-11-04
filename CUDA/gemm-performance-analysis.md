# GEMM 性能分析方法：从宏观约束到微观执行

分析 GEMM 时，宏观看计算量、数据量和依赖约束；微观看这些工作怎样分配到 CTA、warp、指令和存储层级。**矩阵 shape 提供并行与复用机会，kernel 决定这些机会能实现多少。**

[GEMM 计算与优化](gemm.md) 解释 tiling、内存布局、Tensor Core 与流水线的具体机制。本文把这些机制连接成分析方法，回答三个问题：GEMM 为什么可能比多次 GEMV 更高效，TP 为什么可能加速，以及“计算和 HBM 都没跑满”还可能受什么限制。

本文的时间、流量和 wave 数都是明确假设下的手算，不是 GPU 实测。示意代码用于解释依赖和分工，不是可直接运行的 kernel。

## 1. 固定工作负载，再区分需求与供给

全文采用数学布局：

$$
Y_{M\times N}=X_{M\times K}W_{K\times N}
$$

| 符号 | 在线性层中的含义 |
| --- | --- |
| $M$ | 本次一起计算的 token 数 |
| $K$ | 输入维度，也是求和维度 |
| $N$ | 输出维度 |
| $s$ | 每个输入和输出元素的字节数；本文数值例子采用 BF16，$s=2$ |
| $B_M,B_N,B_K$ | CTA 的输出 tile 大小，以及每次处理的 K 分块长度 |

实际框架可能保存 `W[N,K]` 并计算 `X @ W.T`，分析时要把数学维度与物理 stride 分开。普通 decode 的 M 常接近本轮请求数，prefill 的 M 是本次处理的 token 总数。投机验证中一条请求可以贡献多个 token。

一次乘加计为 2 FLOPs，主矩阵乘法的有效计算量约为：

$$
F=2MKN
$$

若输入各读一次、输出写一次，不读取旧输出，也不计 scale、workspace 和后处理，理想数据量为：

$$
Q_{\mathrm{ideal}}=s(MK+KN+MN)
$$

这描述工作需求。硬件的计算吞吐和带宽描述供给能力。若实际 HBM 传输量是 $Q_{\mathrm{HBM}}$，相应吞吐上限可写为：

$$
P_{\mathrm{attained}}\leq
\min\left(P_{\mathrm{peak}},\ B_{\mathrm{HBM}}\frac{F}{Q_{\mathrm{HBM}}}\right)
$$

其中 $P_{\mathrm{peak}}$ 必须匹配使用的精度、指令和稠密/稀疏路径。这个上限假设有足够并行工作和有效的数据供应，**不能由此直接断言实际运行一定是 compute-bound 或 HBM-bandwidth-bound**。短 kernel、指令依赖、同步、L2/shared memory 供给和无效计算都可能让实际吞吐低于这个上限。

来源：[NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)。

## 2. 并行度不足：有工作，不代表此刻能执行

### 2.1 地址依赖怎样让显存带宽闲置

先看一个纯粹的依赖例子。假设只有一个线程沿链表式索引读取，数据均未命中 cache：

```cpp
// 下一次读取的地址，要等上一次读取返回后才能确定。
for (int i = 0; i < 1000; ++i) {
    index = next[index];
}
```

```text
读取 next[index0] → 等待 → 得到 index1
读取 next[index1] → 等待 → 得到 index2
读取 next[index2] → 等待 → 得到 index3
```

假设一次请求传输 32 bytes，延迟为 500 ns，而且只能有一个请求在途，那么这条链的传输吞吐最多约为：

$$
32\ \mathrm{bytes}/500\ \mathrm{ns}=64\ \mathrm{MB/s}
$$

这两个数是教学假设。即使硬件带宽达到 TB/s 级，单条链也无法持续提交请求；计算工作又很少，计算单元同样闲置。限制来自 **memory latency 与独立请求不足**，并非 HBM 带宽已经耗尽。

如果同时执行很多条独立链，就有机会覆盖等待。因此，“串行依赖长”本身不能决定性能；要看是否还有其他独立工作可以执行。

### 2.2 GEMM 的 K 很长，也可能只有一个 CTA

考虑：

$$
X[64,65536]\times W[65536,128]\rightarrow Y[64,128]
$$

计算量约为 1.07 GFLOPs。若一个 CTA 负责一个 $64\times128$ 输出 tile，且没有 split-K，整个 kernel 只有一个 CTA。采用 $B_K=32$ 时，这个 CTA 要完成 2048 个 K 分块的累加。

```text
整个输出只对应一个 CTA
          ↓
一个 SM 执行漫长的 K 循环
          ↓
其他 SM 没有分配到工作
```

即使 GPU 有 128 个 SM，也只有一个 SM 承担这段计算。全卡计算吞吐无法达到峰值，一个 CTA 通常也不足以产生用满 HBM 带宽的请求数量。

GEMM 的限制与 pointer chasing 不同：输入地址通常可以提前确定，加载可以预取；K 求和也可以拆成部分和。Split-K 把不同 K 段分给多个 CTA，增加并行任务，最后再合并输出。代价是额外存储、归约和同步；改变累加顺序还可能改变浮点舍入结果。

### 2.3 全卡 CTA 足够，还要看每个 SM 的 ready work

普通的“一 CTA 对应一输出 tile”实现中，不含 split-K 的 CTA 数为：

$$
N_{\mathrm{CTA}}=\left\lceil M/B_M\right\rceil
\left\lceil N/B_N\right\rceil
$$

这是第一层并行度。Persistent kernel 可以用较少 CTA 连续领取多个 tile，此时要分别看启动 CTA 数、工作 tile 数和调度方式，不能混用它们。

还需要检查：

| 层次 | 需要回答的问题 | 主要影响因素 |
| --- | --- | --- |
| 全卡任务 | 是否有足够多且均衡的工作？ | 输出 tiles、split-K、grouped GEMM、尾部 wave |
| SM 内执行 | 是否持续有指令可发射？ | Resident warps、独立 accumulator、操作数就绪、同步 |
| 数据路径 | 数据能否及时送到消费者？ | 复用、coalescing、cache、shared memory 带宽与 bank conflict |
| 计算路径 | 指令是否高效完成有效工作？ | Tensor Core 路径、tile 填充率、地址计算、epilogue |

**Occupancy 表示驻留的 warp 比例，不表示这些 warp 此刻都能执行。** 所有 warp 同时等待数据时，高 occupancy 也不能保证高吞吐。有足够指令级并行和软件流水线时，低 occupancy 也可能高效运行。

来源：[CUDA Best Practices：Occupancy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy)、[CUTLASS：线程块分工与并行归约](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)。

## 3. GEMM 比多次 GEMV 更高效：复用怎样发生

### 3.1 从一个权重服务多个 token，到一次 CTA 加载

固定线性层：

$$
Y[M,8192]=X[M,8192]W[8192,8192]
$$

比较 M=1 和 M=64。只观察同一个 K 分块和同一组输出列，取 $B_K=32,B_N=128$。两种工作都需要同一个权重 tile：

$$
W_{\mathrm{tile}}[32,128],\qquad
32\times128\times2=8\ \mathrm{KiB}
$$

```text
M=1：
X [1,32]  × W [32,128] → 累加到 Y [1,128]

M=64，CTA 覆盖 64 个 token：
X [64,32] × W [32,128] → 累加到 Y [64,128]
```

| 一个 K 分块 | M=1 | M=64 |
| --- | ---: | ---: |
| 权重 tile | 8 KiB | 8 KiB |
| 输入 tile | 64 B | 4 KiB |
| 有效 FLOPs | 8,192 | 524,288 |
| 输入与权重加载字节数 | 8,256 B | 12,288 B |

计算增加了 64 倍，这次输入与权重的加载量却只增加约 1.49 倍。CTA 先把同一块 W 放到 shared memory。负责不同 token 行的线程或 warp 共同使用它，分别更新各自的输出累加器。

```text
一次加载 W tile [32,128]
            ↓
      CTA shared memory
       ↙     ↓      ↘
    token 0 token 1 … token 63
       ↓     ↓          ↓
   各自独立的输出 accumulator
```

64 次独立 GEMV 每次都需要重新取得这块权重，是否命中 cache 要另行分析。GEMM 则明确提供了在一个 CTA 内跨 token 复用的机会。它不是把 64 次 GEMV 的 launch 简单放在一起。

### 3.2 Register tiling 同时增加复用与独立计算链

再缩小到两个 token、两个输出列，在固定 k 上计算：

```cpp
// SIMT register tile 示意：同一线程持有四个独立的输出累加器。
acc00 = fma(x0, w0, acc00);
acc01 = fma(x0, w1, acc01);
acc10 = fma(x1, w0, acc10);
acc11 = fma(x1, w1, acc11);
```

`w0`、`w1` 各服务两个 token；`x0`、`x1` 各服务两个输出列。四个输入值完成四次 FMA，同时四个 accumulator 提供独立的计算链。相比只有一个 `acc` 的连续更新，调度器有更多机会覆盖指令延迟。

这解释了 register tiling 的两个作用：减少 shared memory 到 registers 的重复读取，以及增加 instruction-level parallelism（ILP）。代价是 register pressure，可能减少驻留 warp，严重时会 spill。

Tensor Core 也以矩阵片段执行乘加，但 fragment 的分布和指令依赖取决于架构，不能把这段单线程 SIMT 示意直接当成 MMA 的线程布局。更多有效 M 行通常有助于填满矩阵指令的 tile；小 M 时，即使执行 Tensor Core 指令，也可能只有少数行是有效计算。

### 3.3 Batch、总 latency 与每 token 成本

假设输入和权重各读取一次、输出写入一次，且各元素均占 s 字节，理想算术强度为：

$$
I_{\mathrm{ideal}}=\frac{2MKN}{s(MK+KN+MN)}
\approx\frac{2M}{s}
$$

权重字节占主导时，BF16 下近似为 M FLOP/byte。这个近似依赖权重只读取一次的假设，不能直接当成实际 HBM 算术强度。增大 batch 提供更多复用机会与输出工作，能实现多少收益还取决于 CTA 分块、缓存和并行度。

这不保证一个大 batch 的总 latency 小于一个 token 的 latency，也不保证请求的排队时间下降。比较时要固定指标：单轮时间、每 token 平均成本、系统吞吐与用户端延迟并不相同。

来源：[NVIDIA Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)、[CUDA：Shared Memory in Matrix Multiplication](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-in-matrix-multiplication-c-ab)。

### 3.4 超过一个 CTA 的 token 数后，增大 M 还有收益吗

**普通 tiled GEMM 的 shared memory 权重复用通常限于一个 CTA；超过它负责的 token 数后，吞吐仍可能提高，但收益不再来自同一个 CTA 内复用次数的增加。** 以下固定每个 CTA 计算 $B_M=64,B_N=128$ 的输出 tile，沿 K 分块累加；不讨论 cluster 协作等特殊实现。

把 M 从 64 增加到 128，固定输出通道 0～127，看两个 CTA 的分工：

```text
CTA A：token  0～63  × 输出通道 0～127
CTA B：token 64～127 × 输出通道 0～127

每个 K 分块中：
                       ┌→ CTA A 的 shared memory → 64 个 token
HBM → L2 中的权重 tile ─┤
                       └→ CTA B 的 shared memory → 另外 64 个 token
```

两个 CTA 各自加载相同的权重 tile，各自的那份仍只服务 64 个 token。CTA B 不会自动使用 CTA A 的 shared memory。但是，若 CTA B 请求时数据仍在 L2，它就不必重新从 HBM 读取。**两次向 shared memory 加载，不等于两次从 HBM 读取。** 上图表示可能的 L2 命中路径，不保证所有权重都能如此复用。

| 观察层级 | M 从 64 增加到 128，tile 配置固定 |
| --- | --- |
| 单个 CTA 内的权重复用 | 每份权重仍服务 64 个 token |
| 所有 CTA 的权重加载请求量 | 随 M 方向的 CTA 数增加 |
| HBM 实际权重读取量 | 取决于 L2 命中，不一定同比增加 |

L2 复用取决于工作集大小、缓存竞争和 CTA 调度顺序。相同权重的请求在时间上接近时，可能在被逐出前命中；不要求整个权重矩阵同时驻留 L2。CUTLASS 的 threadblock rasterization 就会考虑不同 CTA 的数据局部性。[CUTLASS 分层 GEMM 与调度](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#threadblock-rasterization)

即使没有额外的缓存收益，增大 M 也可能增加同时工作的 SM 数。普通输出分块、不使用 split-K 时：

$$
\#\mathrm{CTA}=\left\lceil\frac{M}{B_M}\right\rceil
\left\lceil\frac{N}{B_N}\right\rceil
$$

以 $N=8192$、$B_M=64$、$B_N=128$ 为例，假设 GPU 有 80 个 SM，且这个 kernel 每个 SM 同时驻留一个 CTA：

| M | CTA 数 | 按每轮最多 80 个 CTA 的简化分组 |
| ---: | ---: | --- |
| 64 | 64 | 64 |
| 128 | 128 | 80 + 48 |
| 256 | 256 | 80 + 80 + 80 + 16 |
| 320 | 320 | 80 + 80 + 80 + 80 |

这里的 wave 是分析并行容量的分组，不表示硬件在每轮之间执行全局同步。M=64 时任务数不足以覆盖 80 个 SM；增加 M 可以提供更多并行工作，但最后一轮是否填满仍影响平均利用率。这就是 wave quantization，因而吞吐不保证随 M 单调上升。表格是教学模型，不是实测时间。[NVIDIA Tile / Wave Quantization](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

还可以推导一个明确的边界。假设 M 是 $B_M$ 的整数倍，tile 配置固定，无跨 CTA 权重缓存命中，每个权重元素占 s 字节，且只统计权重流量。每组 $B_M$ 个 token 都需要从 HBM 重新取得完整 W，因此：

$$
Q_W=\frac{M}{B_M}KNs,\qquad
\frac{F}{Q_W}=\frac{2MKN}{(M/B_M)KNs}=\frac{2B_M}{s}
$$

**M 被约掉了。** BF16、$B_M=64$ 时，这个只计权重的比值是 64 FLOP/byte；继续增加 M 不会提高它。完整算术强度还需计入输入、输出等流量，不能把这个比值当成完整 HBM 算术强度。

因此，增加 M 的收益需要分别判断：tile 尚未填满时，有效行数增加可以减少空算；超过 tile 后，L2 复用、更多 SM 并行以及 launch 等固定开销的摊薄仍可能提高吞吐。实际库还可能选择更大的 tile 或不同 kernel，此时上述固定配置的假设也发生变化。当复用和硬件利用率都已饱和，继续增加 M 通常主要增加总工作量，吞吐趋于平台；整次 GEMM 的延迟并不会因此持续下降。

## 4. Arithmetic intensity 必须对应具体存储层级

上一节 CTA 的输入加载强度为：

$$
I_{\mathrm{tile}}=\frac{2B_MB_NB_K}{s(B_MB_K+B_KB_N)}
=\frac{2B_MB_N}{s(B_M+B_N)}
$$

代入 $B_M=64,B_N=128,B_K=32,s=2$，得到约 42.7 FLOP/byte。这里的字节数是当前 CTA 输入 tile 的加载量，忽略输出写回，不能直接当作整个 GEMM 的 HBM 流量。

原因在于数据沿不同路径被复用：

```text
HBM → L2 → CTA 的 shared memory → registers / MMA 操作数
       │              │                    │
    跨 CTA 复用    CTA 内跨线程复用     同一操作数参与多次计算
```

这是理解经典分层 GEMM 的示意。具体 Tensor Core 架构可能直接消费 shared memory 操作数，或使用 TMEM 保存累加器；实际数据路径见 [GEMM 计算与优化](gemm.md)。

不同 CTA 可能重复加载同一份 X，重复请求可能命中 L2；shared memory 中的值又可能被多个 warp 读取；register tiling 继续减少 shared memory 读取。因此需要区分：

$$
I_{\mathrm{HBM}}=F/Q_{\mathrm{HBM}},\qquad
I_{\mathrm{L2}}=F/Q_{\mathrm{L2}},\qquad
I_{\mathrm{SMEM}}=F/Q_{\mathrm{SMEM}}
$$

每一级的流量必须与该级带宽配对。输入预先驻留 cache 时，HBM 流量可能小于冷启动时的理想输入字节数；重复加载、workspace 和额外事务又可能增加流量。

**HBM 与 Tensor Core 都没跑满，不足以证明 CTA 数量不足。** Shared memory bank conflict、L2 供给、地址计算、指令发射或同步也可能成为约束。需要沿具体数据路径判断是谁在等待谁。

## 5. Tiling 与流水线怎样互相制约

增大输出 tile 可以提高输入复用，但也减少整个问题的 tile 数，增加 accumulator 和 shared memory 用量。因而“先保证并行度，再独立选择算术强度”只是分析顺序，不是两个彼此独立的调参步骤。

例如固定同一问题，把输出 tile 从 $64\times64$ 增大到 $128\times128$：

- 忽略输出写回时，BF16 tile 输入加载强度从 32 增至 64 FLOP/byte。
- 每个 tile 的输出 accumulator 元素数增至 4 倍。
- 对整除的 M、N，输出 tile 数降至原来的四分之一。
- 固定 $B_K$ 时，单级输入缓冲大小增至 2 倍。

微观改动同时降低了重复搬运、增加了资源需求，并减少了可分配任务。必须检查这些变化中哪一项影响当前瓶颈。

相比之下，$B_K$ 在输入复用公式中被约掉，增大它不会直接提高这个模型中的算术强度；它主要改变循环次数、同步频率、指令组织和缓冲容量。

流水线改变的是时间上的重叠：

```text
串行：
加载 0 → 等待 → 计算 0 → 加载 1 → 等待 → 计算 1

流水线稳态：
计算 tile 0，同时加载 tile 1
计算 tile 1，同时加载 tile 2
```

如果没有重叠，一轮时间可能接近加载与计算时间之和；流水线充分时，稳态一轮时间才可能接近两者中的较大值。增加 stages 提供更多在途工作，但增加 shared memory 占用，并存在启动和排空成本。短 K 循环未必能充分摊薄这些成本。

来源：[CUTLASS：Pipelining](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html#pipelining)、[NVIDIA：Tile / Wave Quantization](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dimension-quantization-effects)。

## 6. TP 为什么快：把 tile 或 K 工作分给更多 GPU

Batching 增大 M，让同一份权重服务更多 token。普通 TP 通常不增加 M，而是让多张 GPU 合作处理同一批 token，增加可用的总计算资源和总 HBM 带宽。

### 6.1 Column-parallel：每卡计算更少的输出 tiles

考虑：

$$
X[1024,8192]W[8192,8192]\rightarrow Y[1024,8192]
$$

取 $B_M=64,B_N=128,B_K=32$。单卡共有：

$$
N_{\mathrm{CTA}}=(1024/64)(8192/128)=1024
$$

假设每张 GPU 有 128 个 SM，该 kernel 每个 SM 同时驻留一个 CTA，各 CTA 工作量和执行时间近似相同。单卡需要 8 个 wave。

Column-parallel TP4 按输出列切权重：

$$
W_r[8192,2048],\qquad Y_r[1024,2048]
$$

每卡仍持有完整 X，但只计算 256 个输出 tiles，即 2 个 wave。

| 每卡工作 | 单卡 | Column-parallel TP4 |
| --- | ---: | ---: |
| CTA 输出 tile | 64×128 | 64×128 |
| 每个 CTA 的 K 分块数 | 256 | 256 |
| 输出 CTA 数 | 1024 | 256 |
| 简化模型中的 wave 数 | 8 | 2 |

一个 CTA 的加载、MMA 和累加工作没有改变。**原来一张 GPU 分 8 批执行的工作，现在由四张 GPU 各执行 2 批。** 若本地效率保持接近，忽略其他开销时，GEMM 时间可能接近四分之一。

### 6.2 Row-parallel：每卡遍历更短的 K 段

按 K 切分时，每卡持有匹配的输入和权重分片：

$$
X_r[1024,2048]W_r[2048,8192]\rightarrow Y_r[1024,8192]
$$

输出 tile 数仍为 1024，但每个 CTA 的 K 分块数从 256 降至 64。不同卡的 $Y_r$ 是相同输出位置的不同贡献，最后需要求和：

$$
Y=\sum_r Y_r
$$

若各卡都需要完整 Y，可通过 AllReduce 完成；其他输出布局可能采用 ReduceScatter。经典 MLP 把 column-parallel 与 row-parallel 连接起来，使第一层的输出分片直接成为第二层的输入分片，然后归约第二层输出。

来源：[Megatron 的 Column / Row Parallel 设计](https://research.nvidia.com/labs/adlr/MegatronLM/)。

### 6.3 小 batch decode：分摊权重读取

BF16 权重 $W[8192,8192]$ 为 128 MiB，TP4 后每卡只持有 32 MiB：

```text
单卡：一个 HBM 系统读取 128 MiB 权重
TP4：四个 HBM 系统同时各读取 32 MiB 权重
```

这要求权重确实需要从 HBM 读取，而且各卡 kernel 能产生足够的访存并行度。M=1 时，TP4 后每卡仍处理同一个 token；TP 没有把它自动变成大 M 的 GEMM。

权重字节占主导时，每卡主计算量和权重字节数近似一起缩小：

$$
F_r\approx2MKN/p,\qquad Q_{W,r}\approx sKN/p
$$

因此 $F_r/Q_{W,r}\approx2M/s$，这个近似下的算术强度与 p 无关。加速来自更多硬件同时服务这份工作。激活重复读取、输出部分和及通信会让完整成本偏离这个近似。

### 6.4 TP 的加速边界

在前面的 column-parallel 示例中，如果 M 从 1024 降至 64，单卡只有 64 个输出 CTA，TP4 后每卡只有 16 个。对于假设的 128-SM GPU，工作分得更细，却更难用满每张卡。

Row-parallel 缩短 K 后，也可能增加流水线启动、排空和 epilogue 的相对占比。通信还有固定延迟、数据搬运、同步等待及资源竞争。只有不在关键路径上、被独立工作覆盖的通信时间，才可能隐藏。

分析 TP 时需同时列出每卡 GEMM shape、tile 数、K 循环长度、权重字节和通信 payload，不能只把 FLOPs 除以 p。具体硬件推导见 [Qwen3-32B / H100 TP 案例](../Attention/qwen3-h100-tp-case.md)，MoE 的不同专家 batch 与数据路由见 [EP 与 TP](../Attention/moe-ep-tp.md)。

## 7. 宏观与微观的对应表

| 微观改变 | 宏观上改变什么 | 需要同时检查的代价 |
| --- | --- | --- |
| 增大 $B_M,B_N$ | 更多片上复用，减少单位 FLOP 的搬运 | Tile 数下降，片上资源占用和边界浪费增加 |
| Register tiling | 减少 shared memory 读取，增加独立累加链 | Register pressure、驻留 warp 数与 spilling |
| Split-K | 增加并行任务，缩短每份 K 归约 | Workspace、额外归约、同步与数值顺序 |
| 多阶段预取 | 增加在途请求，覆盖加载延迟 | 缓冲空间、启动与排空成本 |
| 改善访存布局 | 减少额外事务，提高数据供给效率 | 重排流量、地址计算与消费者布局 |
| 提高 MMA tile 填充率 | 减少无效计算，提高有效吞吐 | 指令与 tile 选择受 shape、精度约束 |
| 增大 batch | 增加权重复用机会和独立输出 | 总工作量与可能的排队时间增加 |
| 增加 TP | 增加总 SM/HBM 资源，减少每卡工作 | 本地 GEMM 变小，跨卡通信增加 |

“有足够多的任务”与“任务此刻可以执行”之间，还有资源驻留、操作数就绪和指令依赖。优化就是在这些约束下，把数学上的并行性与复用机会转化为可持续的执行。

## 8. 拿到一个 GEMM 时的分析顺序

1. **固定合同与指标。** 记录 M/N/K、dtype、累加类型、stride、转置、后处理和正确性容差。明确比较单次 latency、有效 FLOPs 吞吐、每 token 成本还是端到端延迟。
2. **计算需求。** 算主 FLOPs，列出输入、输出、scale、workspace 和额外 kernel；区分冷/热 cache，给出理想字节数与实际流量可能不同的原因。
3. **展开工作映射。** 看输出 tile 数、split-K、每 CTA 的 K 循环、wave 和尾部；persistent/grouped 调度还需区分 CTA 与任务数量。
4. **检查持续供给。** 看 resident 与 ready warps、独立 accumulator、在途加载、pipeline stages 和 barrier。不要把 occupancy 当作利用率。
5. **定位资源限制。** 区分 HBM、L2、shared memory、指令发射与矩阵计算流水线；检查边界无效计算和地址/转换/epilogue 开销。
6. **用最小对照验证原因。** 固定其他条件，只改变 tile、split-K 或 stages 中的一项，验证正确性并测量完整调用成本；TP 同时检查各 rank 和关键路径上的通信。

例如，“HBM 和 Tensor Core 都没跑满”只是一条观测。CTA 很少、ready warp 不足、shared memory 受限、短 kernel 的固定开销，都可能产生这个现象。先提出可区分的机制，再用时间线、计数器与受控对照验证，避免用一个利用率数字直接命名瓶颈。

## 9. 继续阅读

- [GEMM 计算与优化](gemm.md)：CTA/register tiling、布局、异步流水线、Split-K 与融合。
- [CUDA 执行层级与硬件映射](execution-hierarchy.md)：grid、CTA、warp、SM 与驻留资源。
- [Qwen3-32B / H100 TP 案例](../Attention/qwen3-h100-tp-case.md)：把方法用于模型 shape、硬件预算与通信。
- [MoE FFN：EP 与 TP](../Attention/moe-ep-tp.md)：专家 token 数、grouped GEMM 和数据路由。
- [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)：吞吐、带宽与延迟约束。
- [NVIDIA Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)：矩阵形状、算术强度和 tile/wave quantization。
- [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)：分层并行、片上复用、软件流水线和并行归约。
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)：访存效率、occupancy 与指令级并行。
