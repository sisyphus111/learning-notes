---
title: MoE FFN：EP 与 TP
category: attention
tags: [MoE, EP, TP, AllReduce]
description: expert GEMM · A2A / AllReduce · token 布局 · 负载均衡 · 显存与混合并行
updated: 2026-11-04
---
# MoE FFN：EP 与 TP 的通信、GEMM 和部署取舍

## 1. 固定工作量，区分专家分布与 token 分布

**EP 将不同 experts 分给不同 GPU；TP 将同一个 expert 的矩阵分给多张 GPU。** 前者保留完整 expert 的矩阵形状，但通常要交换 token。后者共同计算每个激活专家，矩阵更窄，最后归约部分和。

本页采用 DeepSeek-V3 routed experts 的代表尺寸，暂不计 shared expert、router 的算术和其他 Transformer 层。GPU 数和 batch 是教学设定，不是模型报告的部署配置。所有数字为推导，没有执行 CUDA benchmark。

| 符号 | 数值 | 含义 |
| --- | ---: | --- |
| P | 8 | 比较使用相同的 8 张 GPU |
| E | 256 | routed expert 总数 |
| k | 8 | 每 token 选择的不同 experts 数 |
| H | 7168 | hidden size |
| F | 2048 | 单 expert 的 intermediate size |
| T | 1024 | 整个并行组中的唯一 token 数，不计 TP 副本 |

均衡路由时，单个 expert 平均收到 $M_e=Tk/E=32$ 条输入。[DeepSeek-V3 配置依据][v3]

| | 纯 EP8 | 纯 TP8 |
| --- | --- | --- |
| 每卡持有权重 | 32 个完整 experts | 全部 256 个 experts 各自的 1/8 |
| 单 expert 的本地 F | 2048 | 256 |
| 典型输入布局 | 每卡 T/P 个不同 token | 各卡复制相同的 T 个 token |
| 本地 token-expert 关系数，均衡时 | Tk/P=1024 | Tk=8192，但每条只算 1/P 的 FFN |
| 典型通信 | dispatch + combine | 输出 AllReduce |

这张表同时描述了权重布局与典型 token 布局，两者并非永远绑定。第 7 节会统一输入输出边界，再比较通信。[TensorRT-LLM MoE 并行定义][trt-guide]

基础材料：[dense FFN 的 row/column TP](qwen3-h100-tp-case.md)、[GEMM 分块](../cuda/gemm.md)、[Ring / Tree AllReduce](../gpu/communication-and-allreduce.md)。

## 2. MoE FFN 的计算与权重存储

一个 SwiGLU expert 的计算为：

$$
E_e(x)=\left[\operatorname{SiLU}(xW_{g,e})\odot(xW_{u,e})\right]W_{d,e}.
$$

token t 的 routed 输出为：

$$
y_t=\sum_{e\in\operatorname{TopK}(t)}g_{t,e}E_e(x_t).
$$

稀疏选择发生在 expert 维度。被选中的 expert 内部通常仍执行 dense GEMM，且路由权重不能任意移到非线性之前。

单 expert 参数量约为 3HF。BF16 下，两种策略的每卡 routed 权重存储都是：

$$
\frac EP\,3HF\times2=E\,\frac{3HF}{P}\times2=2.625\text{ GiB}.
$$

负载均衡时，两种策略的每卡主要计算量同为：

$$
\frac{6TkHF}{P}\approx90.19\text{ GFLOP}.
$$

这个相等关系不代表性能相同。EP 的每卡工作量还取决于实际路由；TP/EP 的 GEMM 形状、访存和同步不同。存储全部专家权重，也不表示每一步都读取所有权重：小 batch 可能只激活部分 experts。

## 3. expert grouped GEMM 的形状

按 expert 聚集输入后，第一组 GEMM 可以合并 gate/up，第二组 GEMM 做 down。权重按数学上的右乘方向写，物理存储可采用转置布局。

| 步骤 | EP8 中一个本地 expert | TP8 中一个 expert 的分片 |
| --- | --- | --- |
| gate/up | `[32,7168] @ [7168,4096]` | `[32,7168] @ [7168,512]` |
| SwiGLU 后 | `[32,2048]` | `[32,256]` |
| down | `[32,2048] @ [2048,7168]` | `[32,256] @ [256,7168]` |
| 每卡 expert 分组数 | 32 | 256 |

**固定同一批全局 token 后，EP 与 TP 的单 expert M 都是 32。** EP 保留完整 F，不会自动把 M 变成 256。Grouped GEMM 统一组织不同专家的任务，但每个专家仍使用自己的权重。不能合成一个所有 token 共用权重的普通 GEMM。

若局部输入、权重、输出都用 BF16，按各自读写一次的理想流量，$I=MNK/(MK+KN+MN)$：

| GEMM | EP，FLOP/byte | TP，FLOP/byte |
| --- | ---: | ---: |
| gate/up | 31.61 | 29.99 |
| down | 31.37 | 28.33 |

TP 的输入或输出中有一侧没有随矩阵计算量同比缩小，因此局部强度略降。两者 M 都小，不能因为 EP 矩阵更宽就认定它必然 compute-bound。实际缓存、量化、融合会改变字节数。[Grouped GEMM 与布局][deepgemm]

## 4. CTA、warp 与寄存器会怎样变化

设所有 experts 恰好各有 32 行，每 expert 需要一个 M tile，输出 N tile 为 128，K tile 为 64。该设定只演示工作量分解，不代表实际 kernel 的最优配置。

| 步骤 | EP 每卡 CTA 数 | TP 每卡 CTA 数 | EP / TP 的每 CTA K 轮数 |
| --- | ---: | ---: | ---: |
| gate/up | 32×32=1024 | 256×4=1024 | 112 / 112 |
| down | 32×56=1792 | 256×56=14336 | 32 / 4 |

gate/up 的 CTA 数可以相同：TP 每个 expert 更窄，但本卡需要处理更多 expert 分组。down 则变成更多、更短的任务，总乘加量没有改变，但 pipeline 填充/排空、调度和 epilogue 的占比可能增加。

相同 CTA 输出 tile 对应相同的 accumulator 容量，TP 不会自动把每线程寄存器减少到 1/P。若库改小 tile，寄存器和 shared memory 可能下降，但复用也随之变化。小 M 还会出现行方向 padding；例如实际采用 BM=64 时，32 条有效行只占半个 tile，EP/TP 都可能承担这类代价。

实际 grouped kernel 可以使用 persistent 调度、不同 tile 或融合。更多 CTA 不自动意味着更慢，更少 CTA 也不自动意味着更快。需要结合有效吞吐、驻留、访存与关键路径判断。

## 5. TP 的通信：先合并本地专家贡献，再归约一次

纯 TP 中，各卡拥有一致的 token 和路由结果。对 token t、expert e，down 产生部分和 $E_{e,r}(x_t)$：

$$
y_t=\sum_e g_{t,e}\sum_rE_{e,r}(x_t)=\sum_r\left(\sum_e g_{t,e}E_{e,r}(x_t)\right).
$$

因此可以先在本地完成路由加权和按 token 汇总，再对 `[T,H]` 做一次 AllReduce：

```text
各 expert 的本地 down 部分和
        ↓ 本地乘路由权重、按原 token 相加
partial_y [T,H]
        ↓ AllReduce
完整 y [T,H]
```

不需要为 top-8 的 8 个 experts 分别做 8 次 AllReduce。求和可以交换顺序，依赖的是 down 后这些操作的线性性质；不能把同样的交换应用到 SwiGLU 之前。

TP 仍需 router、按 expert 整理输入和恢复结果。没有跨卡 token dispatch，不表示没有本地重排成本。

## 6. EP 的 A2A：不等长的 dispatch 与 combine

典型 token-owned EP 的流程为：

```text
每卡 T/P 个不同 token
        ↓ dispatch：去目标 expert 所在卡
本地完整 expert 的 grouped GEMM
        ↓ combine：结果返回原 token 所在卡
每卡恢复自己的 T/P 个输出
```

这是两次逻辑交换，不一定对应两次 NCCL AllToAll API 调用。DeepEP 或融合 kernel 可以使用不同的传输实现。各 rank 间数据长度随路由变化，计数、地址、有效范围和返回映射也有成本。

### 一个明确的发送量模型

假设路由均匀，token 的来源均衡，输入输出通信精度相同，每元素 s bytes。暂不去重同 token 的多个 expert 目标，每条远程 token-expert 关系发送一次 H 维输入、返回一次 H 维输出。远程比例为 $1-1/P$，每卡平均发送量为：

$$
V_{EP}\approx2\frac TP k\left(1-\frac1P\right)Hs.
$$

复制输入/输出的 TP 使用 Ring AllReduce 时，每卡发送量为：

$$
V_{TP}\approx2\frac{P-1}{P}THs,\qquad \frac{V_{EP}}{V_{TP}}\approx\frac{k}{P}.
$$

本例 k=P=8、s=2，两者均为 25.690112 MB/卡。统计的是每卡发送字节，不包含接收，也不是整个网络的汇总流量。

这个结果不能推广成固定延迟比例。A2A 与 AllReduce 的协议、固定开销、拥塞、归约计算和拓扑不同；NCCL 也未必选择 Ring。更不能无条件声称 A2A 的流量总比 AllReduce 小或大。

### 去重与目标侧合并怎样改变结果

同一 token 的多个 experts 在同一目标 rank 时，输入可以只发送一次，再在目标侧展开。计算结束后，也可以先将该 token 的本地 expert 贡献加权相加，再返回一个 H 维结果。

两端都采用这些优化时，数据量更接近“每 token 涉及多少个不同的远程 rank”，不再直接正比于 k。仅 dispatch 去重、combine 仍逐 expert 返回时，则需分别计量两个方向。部分 FlashInfer EP 路径明确在目标侧先合并本地贡献。[FlashInfer EP 数据流][flashinfer]

路由偏斜、不同输入输出精度、节点级转发和本地目标都会进一步改变流量。只看 top-k，不能推断实际网络带宽需求。

## 7. 统一边界：EP 不必永远绑定 A2A

上面的自然布局不同：EP 为 token 分散输入/输出，TP 为 token 复制输入/输出。公平比较需计入前后层给出的布局。

若让 TP 从分散输入开始，并最终回到分散输出，可以使用：

```text
AllGather 输入 → TP expert 计算 → ReduceScatter 输出部分和
```

在输入与输出同为 `[T,H]`、同通信 dtype 的理想 Ring 模型下，AG+RS 发送量等于一次 AR 的发送量。但它们位于计算两侧，启动和重叠条件不同。

反过来，若 attention 已提供复制的全部 token，EP 可以采用：

```text
每卡从完整输入中选择本地 experts 的 token
    → 计算完整 experts
    → 本地按 token 汇总
    → AllReduce
```

这条 EP 路线不必再跨卡 dispatch 输入。如果输入原来分散，则获得这份完整输入仍有成本。Megatron 支持不同 dispatcher，例如 AllGather 和 AllToAll；专家权重布局与 token 搬运方案是两个选择。[Megatron MoE dispatcher][megatron]

## 8. 负载不均衡与小 batch

若 E7 很热，EP 中拥有 E7 的 rank 可能形成长尾。其他 rank 即使提前算完，整层也可能等待它。TP 则将 E7 的矩阵分片分给所有 ranks，共同承担该专家的工作。

TP 内不同 experts 的 M 依然不均匀，但在复制输入和相同切分下，各 rank 看到同一组路由任务。它通常不会产生“热点专家只归某一张卡”的同类负载偏斜。

EP 层延迟应关注最慢 rank，而非平均 FLOPs：

$$
T_{layer}\sim\max_r T_r.
$$

若全局只有一个 token、top-8，EP8 不保证八个目标 experts 恰好每卡一个，可能有卡空闲、有卡承担多个。TP8 则让所有卡参与被激活 experts 的分片计算。但分片过小也会增加相对固定开销，不能据此断言 TP 必然更快。

动态专家复制与 EPLB 可以帮助 EP 分担热点，同时付出权重复制、显存和路由重映射成本。训练时还需维护副本权重与梯度一致性。这类复制不等于完全取消 token dispatch/combine。

## 9. 显存、融合与混合并行

| 项目 | 需要比较的实际成本 |
| --- | --- |
| 权重 | 同 P、均匀划分时，纯 EP 与纯 TP 的 routed 参数存储可以相同 |
| 激活 | TP 将 F 分片，但多个 rank 处理相同 token；EP 接收本地专家输入，热点 rank 的峰值可能更高 |
| 部分输出 | 若逐 expert 物化 down 输出，TP 在多个 rank 写同形状部分和；融合加权汇总可减少中间写出 |
| 通信缓冲 | EP 的容量、计数、padding、路由 metadata；TP 的 AR 或 AG/RS 存储 |
| 量化与布局 | scale、转换、专家行对齐、打包与反向重排 |
| 融合与重叠 | 通信和计算争用 SM/HBM；函数调用减少不等于通信消失 |
| KV cache | 由 attention 并行和缓存布局决定，不能由 FFN 的 EP/TP 名称推断 |

固定同一批 token，假设每条 token-expert 关系的 H 维 down 输出都完整物化。TP 全组会产生 P 份部分结果，EP 全组只产生一份专家结果。这解释了 TP 可能增加的中间流量；真实 kernel 可通过融合避免全部物化。

EP×TP 混合时，先将 experts 分给 EP 组，再在组内将每个 expert 切给 TP ranks。设 $P=P_{EP}P_{TP}$，每卡负责 $E/P_{EP}$ 个专家的权重分片，单 expert 本地 intermediate 为 $F/P_{TP}$。它在矩阵形状、负载分担和通信范围之间折中，仍需根据 attention 端点布局与实际拓扑评估。

## 10. 什么场景优先考察哪种方案

| 场景 | EP 的主要机会与风险 | TP 的主要机会与风险 |
| --- | --- | --- |
| 极小 batch decode | 有效专家少，卡可能空闲；dispatch/combine 固定开销明显 | 可让全部卡合作，仍受小 kernel 和归约固定成本限制 |
| 大 batch / prefill | M 增大、完整 F 有利于 GEMM，通信可有重叠空间 | F 变小，更多短任务与部分输出，大 payload 归约也有成本 |
| 路由偏斜 | 专家所在 rank 形成尾部，可研究 EPLB | 热点专家计算由 TP ranks 分担 |
| 细粒度、小 F experts | 保留完整 expert 形状 | 继续切 F 会使 N/K 很小，可能增加对齐浪费 |
| 跨节点 | A2A 目标分散，需分析拥塞和慢 rank | 频繁归约也可能昂贵，需研究分层和混合布局 |

这些是搜索方向，不是固定部署规则。至少记录唯一 token 数 T、每专家 $M_e$ 分布与最慢 rank。还需记录 kernel 形状及耗时、实际通信字节、重排成本和暴露的通信时间。

大 batch 或更宽矩阵不自动保证高吞吐，fused kernel 也不自动消除等待。应在相同 GPU 数、精度、请求负载和缓存条件下比较端到端延迟与吞吐。[NVIDIA MoE 性能指南][trt-guide]

## 11. 证据范围与来源

本页采用 2026-10-29 查阅的官方文档，模型尺寸、算法语义和数学成本分开陈述。算术强度与通信字节是教学模型，不是测量结果，也没有据此承诺 EP/TP 的固定加速比。

- [DeepSeek-V3 技术报告][v3]：routed expert 的 H/F/E/k 尺寸依据。
- [TensorRT-LLM MoE Performance Guide][trt-guide]：EP、TP、混合并行的工作分配及部署条件。
- [Megatron MoE][megatron]：专家分布、dispatcher、grouped GEMM 与通信重叠。
- [FlashInfer MoE EP 设计][flashinfer]：dispatch、目标侧专家汇总、combine 与融合执行。
- [DeepGEMM 接口][deepgemm]：不同 M、固定 N/K 的 expert grouped GEMM 与布局。

[v3]: https://arxiv.org/html/2412.19437v2
[trt-guide]: https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/moe-perf-guide.html
[megatron]: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md
[flashinfer]: https://github.com/flashinfer-ai/flashinfer/blob/main/docs/design_docs/moe_ep_architecture.md
[deepgemm]: https://github.com/deepseek-ai/DeepGEMM#interfaces
