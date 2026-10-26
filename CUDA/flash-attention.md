# FlashAttention 1–4：从 online softmax 到 Ampere、Hopper 与 Blackwell

## 1. 固定 attention 的计算与维度

FlashAttention 计算 scaled dot-product attention。它接收已经投影好的 Q、K、V，输出 attention 结果。生成 Q/K/V 的 Linear、RoPE 和最终输出投影是相邻步骤，不全部包含在下面的公式中。

先看一个 batch、一个 head，并省略 dropout：

$$
S=\tau QK^T+M,\qquad P=\operatorname{softmax}_{\text{每行}}(S),\qquad O=PV,\qquad \tau=1/\sqrt d.
$$

| 符号 | 形状 | 维度含义 |
| --- | --- | --- |
| Q | $[N_q,d]$ | $N_q$ 个 query，每个 query 有 d 个特征 |
| K | $[N_k,d]$ | $N_k$ 个 key，点积维度为 d |
| V | $[N_k,d_v]$ | 每个 key 对应一个长度为 $d_v$ 的 value |
| S、P | $[N_q,N_k]$ | 每个 query 对所有 key 的分数、概率 |
| O | $[N_q,d_v]$ | 每个 query 对 V 做加权求和后的输出 |
| M | 可广播到 $[N_q,N_k]$ | 允许位置为 0，屏蔽位置为 $-\infty$ |

后面讨论普通 self-attention 时，令 $N_q=N_k=N$、$d_v=d$。加入 batch 和 head 后，逻辑形状可写成 `[B, Hq, Nq, d]` 与 `[B, Hkv, Nk, d]`。MHA 中 `Hq = Hkv`；GQA 中多个 Q heads 共享一个 KV head。

两次矩阵乘法的归约维度不同：

```text
Q @ K.T：[Nq, d]  @ [d, Nk]  → [Nq, Nk]   沿特征 d 求和
P @ V：  [Nq, Nk] @ [Nk, dv] → [Nq, dv]   沿 key/token 位置 Nk 求和
```

**一个输出行依赖许多 KV 行，但不同输出行之间没有 softmax 归一化依赖。** 这个性质会决定 query 分块与 warp 分工。矩阵乘法的基础可先读 [GEMM 计算与优化](gemm.md)。

## 2. 朴素 attention 的显存访问成本

直观实现依次计算 QK、softmax 和 PV。若把 S、P 保存为完整全局 Tensor，就要反复访问 $N_qN_k$ 个中间元素：

```text
读 Q/K → 写 S → 读 S → 写 P → 读 P/V → 写 O
```

以 `B=1, H=32, N=8192` 为例，一个 `[B,H,N,N]` 矩阵有 $2^{31}$ 个元素。按 BF16 存储，仅这一张矩阵就占 4 GiB；按 FP32 存储则占 8 GiB。这是尺寸推导，不表示所有实现都同时分配两张这样的矩阵。

QK 的 Tensor Core 计算可以很快，但随后仍需写出、读入庞大的中间结果。FlashAttention 让一个 score tile 产生后就完成 softmax 更新和 PV 累加，避免把完整 S、P 写入 HBM。

![朴素 attention 与 FlashAttention 的数据流](../assets/flash-attention-dataflow.svg)

图中“片上”包含按实现安排的寄存器、shared memory，以及 SM100 上的 TMEM。论文中的 SRAM 是片上存储分析模型，不能直接等同于 CUDA 的 `__shared__`。

这项改变不减少稠密 attention 必须处理的 query-key 对数，主要计算量仍为 $O(N^2d)$。减少的是中间矩阵的全局存储和访问。“exact attention”指不靠稀疏化或低秩近似改变目标公式。浮点舍入顺序可以不同，不保证与另一实现逐位相同。[FA1 原论文](https://arxiv.org/html/2205.14135v2)

## 3. online softmax：分块后怎样保持归一化正确

### 分子和分母必须使用同一个参考值

固定一个 query，设分数为 $s_1,\ldots,s_N$，value 向量为 $v_1,\ldots,v_N$：

$$
o=\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}=\frac{\sum_j e^{s_j-m}v_j}{\sum_j e^{s_j-m}}.
$$

分子和分母乘了相同因子，比例不变。通常取 $m=\max_j s_j$，使指数的输入不大于 0，避免溢出。

若分别对两个 KV 块做 softmax，再把输出相加，结果会错：每块用了自己的分母。正确做法是保留可以合并的状态。

### 每个 query 保留三个状态

对已处理的 key 集合，维护：

$$
m=\max_j s_j,\qquad \ell=\sum_j e^{s_j-m},\qquad u=\sum_j e^{s_j-m}v_j.
$$

m 和 $\ell$ 各是一个标量；u 是长度为 $d_v$ 的向量。u 是**尚未除以分母的输出分子**，最终输出才是 $o=u/\ell$。

设新块分数为 $s^{\mathrm{new}}$，对应矩阵为 $V^{\mathrm{new}}$：

$$
m'=\max(m,\max s^{\mathrm{new}}),\qquad \alpha=e^{m-m'}.
$$

$$
p^{\mathrm{new}}=e^{s^{\mathrm{new}}-m'},\qquad \ell'=\alpha\ell+\sum_j p_j^{\mathrm{new}},\qquad u'=\alpha u+p^{\mathrm{new}}V^{\mathrm{new}}.
$$

**最大值改变后，旧分母和旧分子必须一起乘 $\alpha$。** 旧状态以 m 为参考，新数据以 $m'$ 为参考；不统一参考值，就不能相加。

### 两块数据的手算

选择便于检查的 logits 和标量 values：

```text
scores = [ln(1), ln(2), ln(3), ln(4)]
values = [10,    20,    30,    40]
```

完整 softmax 的权重为 `[0.1, 0.2, 0.3, 0.4]`，输出为 30。按两个元素一块计算：

| 阶段 | m | 分母 $\ell$ | 分子 u |
| --- | --- | --- | --- |
| 处理前两项 | $\ln2$ | $1/2+1=1.5$ | $10/2+20=25$ |
| 处理后两项 | $\ln4$ | $0.5\times1.5+3/4+1=2.5$ | $0.5\times25+(3/4)\times30+40=75$ |

第二块使参考值从 $\ln2$ 变成 $\ln4$，所以旧状态乘 $\alpha=1/2$。最终 $75/2.5=30$，与整行计算一致。这是手算示例，不是 GPU 性能实验。

### mask 和第一块的边界

等长 causal attention 允许 `key_index <= query_index`。屏蔽位置设为 $-\infty$，指数贡献为 0。位于对角线右上方的整块可以跳过；完全位于允许区域的块不必逐元素检查 mask。

初始化常写成 `m=-inf, l=0, u=0`，但必须处理没有有效元素的块。如果旧 m 和新块最大值都是 `-inf`，直接计算差值会出现 NaN。实际 kernel 会跳过无效块或使用安全初始化；全 masked 行的输出语义也需由接口规定。本文教学代码限定每个 query 至少能看到一个有效 key。

不等长 Q/K 的 causal 对齐规则需要单独确认。decode 的新 token 通常位于历史序列末端，不能直接使用没有位置偏移的左上角三角 mask。

## 4. 从一行递推到一个 attention tile

设一个工作单位负责 $B_r$ 个 query 行，每次加载 $B_c$ 个 KV 行：

| 对象 | 逻辑形状 | 作用 |
| --- | --- | --- |
| $Q_i$ | $[B_r,d]$ | 当前拥有的 query 行 |
| $K_j$、$V_j$ | $[B_c,d]$、$[B_c,d_v]$ | 当前读取的 KV 块 |
| $S_{ij}$ | $[B_r,B_c]$ | 当前块的 query-key 分数 |
| m、$\ell$ | $[B_r]$ | 每个 query 独立的 softmax 状态 |
| U | $[B_r,d_v]$ | 当前 query 块的输出分子累加器 |

```text
持有 Qi 和状态 m / l / U
    ↓
加载 Kj、Vj
    ↓
Sij = scale × Qi @ Kj.T
    ↓
逐行更新 m / l，得到当前块的指数权重 Pij
    ↓
U = alpha[:, None] × U + Pij @ Vj
    ↓
继续下一个 KV 块，最后 Oi = U / l[:, None]
```

这里 Pij 是以当前参考值缩放的指数权重，不一定是整行归一化后的最终概率。处理完所有 KV 块后，最终分母才确定。

例如 `Br=64, Bc=128, d=dv=128`：QK、PV 的输出形状都是 `[64,128]`。前者的 128 表示 key 位置，后者表示 value 特征。**相同的尺寸不代表相同的维度含义。**

这个循环描述现代常见的 query-owned forward。各代实现会改变循环顺序、状态位置和工作分配，不能把它直接标成 FA1 原论文的 Algorithm 1。

## 5. FlashAttention-1：消除完整 S/P 的 HBM 往返

### 原论文怎样组织分块

FA1 的关键组合是 tiling、online softmax 和 backward 重计算。原论文 Algorithm 1 采用外层 KV 块、内层 Q 块的表达：

```text
for 每个 KV 块 j:
    加载 Kj、Vj
    for 每个 Q 块 i:
        加载 Qi、Oi、mi、li
        计算当前 score tile
        合并 softmax 状态并更新 Oi
        写回 Oi、mi、li
```

这个算法仍会读写 Q、O 和行统计量。节省来自避免物化完整 S/P；不意味着“每个输入只读一次，所有输出一直驻留片上”。

FA1 用已经归一化的 o 表示输出。若新块自身的状态为 $\tilde m,\tilde\ell,\tilde u$，令：

$$
m'=\max(m,\tilde m),\quad \alpha=e^{m-m'},\quad \beta=e^{\tilde m-m'}.
$$

合并公式为：

$$
\ell'=\alpha\ell+\beta\tilde\ell,\qquad o'=\frac{\alpha\ell o+\beta\tilde u}{\ell'}.
$$

它与前面的 u 递推在实数算术下等价，但逐轮更新归一化输出，涉及额外非矩阵运算。这为 FA2 留下了优化空间。

### 内存复杂度与 IO 复杂度

| 指标 | 朴素稠密 attention | FlashAttention 的基本改变 |
| --- | --- | --- |
| 主要 FLOPs | $O(N^2d)$ | 仍为 $O(N^2d)$ |
| S/P 全局中间存储 | $O(N^2)$ | 不保存完整 S/P |
| Q/K/V/O 的存储 | $O(Nd)$ | 仍需保存输入与输出 |
| 行统计量 | 由实现安排 | 通常额外 $O(N)$ |

FA1 在片上容量 M 的模型中给出 $\Theta(N^2d^2/M)$ 的 HBM 元素访问量，适用条件为 $d\le M\le Nd$。M 以可存储元素数计，不是 batch size，也不是 FLOPs。该模型说明 tile 复用如何降低 IO，不是所有 shape 与 GPU 的实测流量承诺。[FA1 §3 与 Algorithm 1](https://arxiv.org/html/2205.14135v2)

## 6. backward：不保存 P，梯度怎样计算

以下推导省略 dropout。设上游梯度 $G=\partial\mathcal L/\partial O$，形状与 O 相同。由 $O=PV$ 得到：

$$
dV=P^TG,\qquad dP=GV^T.
$$

softmax 的行内导数为：

$$
D_i=\sum_a G_{ia}O_{ia},\qquad dS_{ij}=P_{ij}(dP_{ij}-D_i).
$$

D 每个 query 行一个标量，shape 为 `[Nq]`。它沿 value 特征 a 求和，不沿 query 行求和。最后：

$$
dQ=\tau\,dS K,\qquad dK=\tau\,dS^T Q.
$$

| 计算 | 形状检查 |
| --- | --- |
| $dV=P^TG$ | $[N_k,N_q][N_q,d_v]\to[N_k,d_v]$ |
| $dP=GV^T$ | $[N_q,d_v][d_v,N_k]\to[N_q,N_k]$ |
| $dQ=\tau dS K$ | $[N_q,N_k][N_k,d]\to[N_q,d]$ |
| $dK=\tau dS^TQ$ | $[N_k,N_q][N_q,d]\to[N_k,d]$ |

forward 可以保存每行 log-sum-exp：

$$
L_i=m_i+\log\ell_i=\log\sum_j e^{S_{ij}}.
$$

backward 按块重新计算 $S_{ij}=\tau Q_iK_j^T+M_{ij}$，再用 $P_{ij}=e^{S_{ij}-L_i}$ 恢复概率块。用额外算术换取不保存完整 P 的显存空间，也减少从 HBM 读回 P 的流量。

因此 backward 常见五类矩阵计算：重算 QK、计算 dP、dV、dQ、dK。它只重算需要的 attention 中间量，不是重跑整个模型 forward。若启用 dropout，还要按一致 RNG 状态重建 mask，并处理相应导数，不能直接套用这里的无 dropout 公式。[FA2 backward 算法](https://arxiv.org/html/2307.08691v1)

## 7. FlashAttention-2：并行度、非 matmul 与 warp 分工

### 把归一化推迟到最后

FA2 保存未归一化的 U，KV 循环中执行第 3 节的更新，最后才除以 $\ell$。这样省去逐块更新完整归一化输出的部分乘除运算。

**FA2 仍然需要 rescaling。** 参考最大值改变时，旧 U 和旧 $\ell$ 仍要乘 $e^{m_{old}-m_{new}}$。省去逐轮最终归一化，不等于取消缩放。

Tensor Core 与普通逐元素、归约、指数运算的吞吐不同。非 matmul 操作的 FLOPs 数少，不表示耗时一定可以忽略。

### 沿 query 行增加 CTA 并行度

不同 query 行有独立的输出与 softmax 状态。FA2 forward 让一个 CTA 持有一个 Q tile，遍历相关 KV tiles，输出自己的 O tile。

```text
CTA 0：Q 的第   0～127 行 → O 的第   0～127 行
CTA 1：Q 的第 128～255 行 → O 的第 128～255 行
CTA 2：Q 的第 256～383 行 → O 的第 256～383 行
```

例如 `B=1, H=8, Nq=8192, Br=128`，按 `(batch, head, query tile)` 划分可产生 $1\times8\times64=512$ 个工作单位。这是示意配置推导，不是同时驻留的 CTA 数；并发驻留还受 SM 数、寄存器和 shared memory 限制。

仅按 batch/head 并行时，小 batch、长序列可能无法提供足够工作。query 分块增加了独立任务。切不同 Q 行不需合并输出；切同一 Q 行的不同 KV 段则需要合并 softmax 状态，两者不能混淆。

### CTA 内从 sliced-K 改为 sliced-Q

用一个 Q tile 和两个 warp 分组说明所有权：

| 分工 | 每组负责什么 | 输出关系 |
| --- | --- | --- |
| sliced-K | 相同 Q 行、不同 KV 片段 | 同一个 O 行有多个部分结果，需要合并 |
| sliced-Q | 不同 Q 行、共享可访问的 K/V | 每组拥有不同 O 行，减少跨 warp 输出合并 |

```text
sliced-K：warp 组 0 → O[同一批行] 的部分和 ─┐
          warp 组 1 → O[同一批行] 的部分和 ─┴→ 合并

sliced-Q：warp 组 0 → O[前半批行]
          warp 组 1 → O[后半批行]
```

减少的是特定的跨 warp 通信与 shared memory 往返。加载 K/V、Tensor Core 协作、行内归约和缓冲区复用仍需同步。

FA2 backward 又有不同所有权：按 KV/column tiles 分工时，CTA 局部累加自己的 dK/dV；不同 KV tiles 对相同 dQ 的贡献仍需汇总。forward 的独立输出行结论不能套到所有梯度上。[FA2 作者讲解](https://crfm.stanford.edu/2023/07/17/flash2.html)、[sliced-Q 图示](https://princeton-nlp.github.io/flash-atttention-2/)

## 8. Ampere、Hopper、Blackwell 改变了什么

下表比较典型高性能路径，以 A100/SM80、H100/SM90、B200/SM100 为例。新架构仍可支持旧指令路径；FlashAttention 版本也不是硬件代号。

| 资源或机制 | Ampere：A100 | Hopper：H100 | Blackwell：B200 |
| --- | --- | --- | --- |
| global → shared | `cp.async` 小段异步复制 | TMA 多维 tile 搬运 | 继续使用 TMA |
| 典型矩阵指令 | `mma.sync` | `wgmma.mma_async` | `tcgen05.mma` |
| 参与或发起单位 | warp，32 threads 协作 | warpgroup，128 threads 协作 | 单线程发起，支持规定的 1-CTA / 2-CTA 模式 |
| MMA 累加结果 | 寄存器 | 寄存器 | TMEM |
| 重点资源 | 寄存器、shared memory、warp 分工 | 异步依赖、寄存器压力、角色分工 | TMEM 交换、softmax 与更快 MMA 的匹配 |

### Ampere：数据搬运已经可以异步

`cp.async` 让 global → shared 复制避免通过普通数据寄存器中转，也能配合缓冲区与计算重叠。FA3 的意义不能概括为“第一次让搬运异步”；Hopper 增加 TMA 和异步 warpgroup MMA，进一步改变工作组织。[Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)

### Hopper：TMA 与 WGMMA 支持新的角色分工

TMA 用 tensor 描述符搬运 tile，减少逐线程组织地址和复制的工作。一名线程可发起搬运，consumer 等完成通知后使用 shared memory 数据。WGMMA 让矩阵计算也以异步操作推进，为独立算术留下调度空间。

producer 负责搬运，consumer 负责计算，双方通过 barrier 交接缓冲区。`setmaxnreg` 可按角色调整寄存器配额，使 producer 少占寄存器、consumer 保留更多状态。[Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)、[FA3 作者文章](https://pytorch.org/blog/flashattention-3/)

### Blackwell SM100：TMEM 改变中间量的存放方式

TMEM（Tensor Memory）是服务 Tensor Core 路径的专用片上存储。`tcgen05` 累加器在 TMEM；相应模式下 operand A 来自 TMEM 或 shared memory，operand B 来自 shared memory。线程通过专门的 TMEM load/store，与寄存器中的逐元素计算交换数据。

```text
QK 累加到 TMEM
    → 读到寄存器，做 max / exp / sum
    → 按下一次 MMA 要求组织 P
    → PV 累加到 TMEM
```

TMA 是搬运机制，TMEM 是存储，职责不同。单线程发起 `tcgen05.mma` 不表示一个 CUDA Core 独自完成矩阵运算。[CUTLASS tcgen05 编程说明](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/mma/tcgen05_programming.html)

### B200 与 RTX 5090 需要区分

本文 TMEM、`tcgen05`、2-CTA 协作说明面向 B200/SM100。RTX 5090 属于 SM120，不能仅凭同属 Blackwell 套用这套路径。CUTLASS 对 SM120 使用扩展的 warp-level `mma.sync` 路线。软件包支持 SM120，不等于它执行了 SM100 kernel。[CUTLASS Blackwell 功能表](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

## 9. FlashAttention-3：在 Hopper 上重叠搬运、GEMM 与 softmax

### producer / consumer 管理实际数据有效期

设 consumer 正在使用 shared memory 的槽 0。producer 可把下一块 K/V 搬进槽 1。consumer 使用某个槽前，搬运必须完成。producer 覆盖某个槽前，上一轮 consumer 必须使用完。

TMA 与 WGMMA 都是异步机制。“指令已经发出”不表示数据已经可读或计算已经完成。barrier、等待和阶段计数是实现的一部分。

### softmax 与 GEMM 的重叠来自不同工作状态

同一 tile 仍有严格依赖：

```text
QK_j 完成 → 得到 S_j → softmax_j 产生 P_j → PV_j
```

不同 consumer warpgroups 可以交错工作：一组处理 softmax，另一组做 GEMM。也可以在同一 warpgroup 内保留跨迭代状态，让不同 KV tiles 的 GEMM 与 softmax 重叠。

![Hopper 与 Blackwell attention 的角色、依赖和存储](../assets/flash-attention-pipeline.svg)

图示是论文的代表性数据流，不是实测时间线，也不覆盖所有配置。例如当前 FA3 的部分模式会让 P 经由 shared memory。FA3 理想软件流水线与实际指令调度也有区别。Algorithm 2 描述的重叠意图，在 Appendix B.2 的 SASS 分析中没有完全按原样实现。学习时先看依赖，再看生成代码和测量，不能把伪代码排列当成硬件时间线。

更多 pipeline stages 需要保存更多 score、概率和累加状态，增加寄存器压力。FA3 的三阶段尝试没有自动胜过两阶段方案。这里的 GEMM/softmax 软件流水级数，也不等于 shared memory 缓冲槽数量。[FA3 §3 与 Appendix B](https://arxiv.org/html/2407.08608v2)

### FP8 路线还需要数值与布局配合

FA3 的低精度路径利用 FP8 Tensor Core，同时处理量化误差与布局：

- 对 Q/K/V 按块缩放，避免一个全局 scale 被局部 outlier 主导。
- 量化前对 Q、K 做相同的正交混合，使大值分散到多个坐标。
- 按 PV 指令要求安排 V 的布局，以及从 score 累加器到 P 操作数的重排。

设正交变换矩阵为 R，满足 $RR^T=I$，则：

$$
(QR)(KR)^T=QRR^TK^T=QK^T.
$$

这说明相同的正交混合在精确算术下保持点积，不需要借助特征值理解。FA3 使用带随机符号的 Hadamard 变换分散 outlier。随后 FP8 量化仍有误差，不能把低精度路径说成与 BF16 逐位相同。

布局同样重要：FP32 accumulator 的元素分布，不一定满足下一次 FP8 MMA 的 operand layout。论文涉及 V 的片上转置、寄存器重排与 byte permutation。TMA 的寻址或 swizzle 并不自动替代任意布局转换。[FA3 低精度方法](https://arxiv.org/html/2407.08608v2#S3.SS3)

### backward 有自己的分工

多个 KV tiles 对同一 dQ 产生贡献。FA3 backward 讨论专门的 dQ writer，将梯度写回与矩阵计算分离，避免写回争用阻塞其他工作。forward 的最优分工不必等于 backward。

官方 Hopper 实现按分页形式、dtype 和布局选择不同路径，例如部分 paged-KV 路径不采用 TMA。具体配置应从 [Hopper mainloop](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp) 的 dispatch 和模板条件继续追踪。

## 10. FlashAttention-4 forward：匹配更快 MMA 与 softmax 吞吐

### 瓶颈随硬件变化移动

FA4 论文重点分析 B200/SM100：Tensor Core 吞吐增长快于部分非矩阵执行单元和片上数据通路。矩阵计算耗时缩短后，指数、重缩放与 shared memory 往返更容易限制 attention。

因此“换上更快 Tensor Core”不足以让完整 attention 按相同比例加速。FA4 同时改变 softmax 安排和矩阵中间量的存储路径。下面以论文 BF16 主线为基线；FP8/FP4 扩展在后面单列。[FA4 论文](https://arxiv.org/html/2603.05451v1)

### 两个 query tiles 交错推进

作者介绍的典型 forward CTA 处理两个 Q tiles，每个 tile 有 128 个 query tokens。它们有独立的 softmax 状态和 O 累加器，可以交错安排 MMA、softmax 和 correction。

| 角色或步骤 | 工作与数据位置 |
| --- | --- |
| producer | 用 TMA 准备 Q/K/V 的 shared memory 数据 |
| MMA 发起者 | 提交 `tcgen05`，将 QK 或 PV 结果累加到 TMEM |
| softmax warpgroup | 从 TMEM 读取 score 到寄存器，做 max、exp、sum，将 P 按所需布局写回 TMEM |
| correction warpgroup | 必要时重缩放旧输出累加状态，并参与最终输出处理 |

PV 可以直接从 TMEM 读取 P 作为 operand A，V 从 shared memory 提供。这减少了 P 再经过 shared memory 的中转，但 TMEM 与寄存器间仍有实际移动。

两个 query tiles 为重叠提供了空间，但所有阶段同时运行也可能争用资源。它们的指数阶段可能竞争同一执行资源，作者因此还协调两组 softmax 的推进。tile、warp 数和阶段数是配置选择，不是所有 FA4 的统一定义。[FA4 作者 forward 讲解](https://tridao.me/blog/2026/flash4/#forward-pass-new-softmax-pipelining-with-conditional-rescaling)、[SM100 forward 源码](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/flash_fwd_sm100.py)

### conditional rescaling：暂时保留旧参考值

标准 online softmax 在最大值变化时重缩放旧 U。SM100 的 U 位于 TMEM，修正它涉及读出、逐元素乘法与写回，不能只按“一次乘法”估算成本。

关键观察是：分子和分母中的指数都可对 score 使用同一个偏移，相当于分子、分母同乘 $e^{-c}$。这个参考值不必每轮等于真正最大值。使用 c 表示当前参考值，避免把它始终称作 max。

采用自然指数记号时，一种自洽的条件更新为：

$$
c'=\begin{cases}\max(c,\max s^{\mathrm{new}}),&\max s^{\mathrm{new}}-c>\theta,\\c,&\text{其他情况}.\end{cases}
$$

$$
\alpha=e^{c-c'},\quad p=e^{s^{\mathrm{new}}-c'},\quad \ell'=\alpha\ell+\sum p,\quad u'=\alpha u+pV.
$$

保留 c 时，$\alpha=1$，旧分子与分母不用重缩放。新块仍以同一个 c 计算，所以比值保持正确。第一次有效 tile 需初始化 c，不能把 `-inf` 当成普通有限参考值处理。

**错误写法是更新 c，却跳过旧 u 的缩放。** 那会把两套指数参考值直接相加。

官方实现通常使用 `exp2`，将分数乘 $\log_2 e$ 后，在以 2 为底的域中计算。典型 BF16 路径的 `rescale_threshold=8` 是该域的差值，对应因子 $2^8=256$。改用自然指数时，同等阈值应为 $8\ln2$。

例如当前参考值为 3，新块最大值为 5，在阈值 8 的 exp2 域中可保持参考值 3，最大指数为 $2^{5-3}=4$。之后新块最大值变成 13，增长超过阈值，就将参考值改成 13，并将旧状态乘 $2^{3-13}$。

保留较旧参考值会允许部分指数大于 1，必须考虑 dtype 动态范围、累计值范围与误差。BF16 的设置不能原样套到 FP8。源码 `SoftmaxSm100` 在跳过修正时确实保留旧参考值，不同精度路径使用不同条件。[SoftmaxSm100 源码](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/softmax.py)

### 软件指数：将部分工作分流到 FMA

指数运算使用专门的指令路径，例如 `MUFU.EX2`。当指数吞吐限制 softmax 时，FA4 将部分指数计算交给普通算术单元。

可把这类近似理解为将 $x=n+f$ 分成整数与小范围小数：

$$
2^x=2^n2^f,\qquad 2^f\approx a_0+f(a_1+f(a_2+\cdots)).
$$

整数部分通过指数相关操作重建，小数部分使用多项式与 FMA。真实取整范围、系数和特殊值处理应按源码确定，不能将这个概念式直接当成标准 `exp` 的替代实现。

这仍在计算指数近似，并非省去 softmax，也不是用 Tensor Core 做所有逐元素指数。硬件指数与软件指数可混合，代价是额外指令、寄存器与近似误差。比例需要按架构和 tile 调节，不能推广成所有 Blackwell 都应使用软件 exp。[FA4 作者 softmax 说明](https://tridao.me/blog/2026/flash4/)

## 11. FlashAttention-4 backward：布局、TMEM 与 2-CTA

### 五个 GEMM 之间怎样复用中间量

第 6 节梯度公式不变，FA4 改变 tile 朝向、存储和流水线。例如将前两项组织成：

$$
S^T=\tau KQ^T+M^T,\qquad dP^T=VG^T.
$$

softmax 恢复与逐元素导数随后生成适合后续计算的 $P^T$、$dS^T$：

$$
dV=P^TG,\qquad dK=\tau dS^TQ.
$$

相应实现让这些中间块以 operand-A 所需布局留在 TMEM，减少 shared memory 中转。转置后，每个元素仍要使用对应 query 行的 L 和 D；物理布局改变不代表 softmax 改为沿错误的轴归一化。

TMEM 容量有限。score、概率和梯度块的有效期需要交错安排，释放后的区域才能复用。这里优化片上数据流，不是把所有中间 Tensor 永久留在 TMEM。

### 2-CTA 是同一 GPU 内的协作

SM100 支持规定的 CTA pair 协作 MMA。两 CTA 位于同一 cluster 的配对关系中，使用两个 SM 的资源，分担操作数准备和访问。它不是两张 GPU 的 Tensor Parallel，也不是所有 `tcgen05` 都必须用两个 CTA。

FA4 backward 借此组织更多矩阵工作，缓解 shared memory 的数据供给压力。不能简化为“一个 CTA 只算 dK，另一个只算 dV”。

对论文使用的 cooperative MMA tile，A 与累加结果沿 MMA 的 M 维在两个 CTA 中分布；operand B 沿 MMA 的 N 维分担准备。每个 CTA 在自己的 shared memory 准备一半 B，硬件联合消费，从而减少每个 SM 需要提供的操作数数据量。

这里 M/N/K 是当前这一次矩阵乘法的轴，不始终对应 attention 的 query/key 轴。计算 $KQ^T$ 时，M 是 key 位置、N 是 query 位置；计算 $dS K$ 时，M 变为 query 位置、归约 K 才是 key 位置。

### dQ 的归约轴导致额外交换

对 dK/dV，不同 KV 行的输出可分别拥有；但 dQ 要对 KV 位置求和。同样的 tile 划分，到了 dQ 这一步会改变归约关系。

作者的 2-CTA 路线通过 cluster 内 distributed shared memory 交换部分 dS。交换后，每个 CTA 拥有一部分 query 行所需的完整配对 KV 范围，再计算相应 dQ。先合并两个配对块的贡献，可以减少向全局 dQ 做原子累加的次数。其他 KV 分块的贡献仍需正确归约。

用两个 KV 范围 $J_0,J_1$ 表达同一个 query 行的贡献：

$$
dQ_{\mathrm{pair}}=\tau\bigl(dS[:,J_0]K[J_0,:]+dS[:,J_1]K[J_1,:]\bigr).
$$

交换前，两边按 KV 范围分别持有贡献。交换后，一边拥有前半 query 行的两个 KV 范围；另一边拥有后半 query 行的两个 KV 范围。相较论文对应的 1-CTA 分解，配对内先求和使全局 dQ 原子归约次数减半。[作者的 2-CTA 说明](https://tridao.me/blog/2026/flash4/#2-cta-backward-pass-reducing-shared-memory-traffic-and-global-atomic-adds)

不同迭代的 dP、dS 与 dQ 交错推进，可以隐藏交换延迟。确定性模式还要固定累加顺序，例如用 semaphore 协调写回；这不表示整个 backward 串行执行。[FA4 backward 设计](https://arxiv.org/html/2603.05451v1#S3.SS2)、[SM100 backward 源码](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/flash_bwd_sm100.py)

### 变长与 causal 工作还需要调度

causal mask 和变长 batch 使不同 tile 的有效 KV 或 Q 循环长度不同。即使每个 tile 内部很快，最后剩下一两个长任务，也会让其他 SM 提前空闲。

FA4 的相应调度策略使用 LPT（Longest Processing Time），优先推进估计耗时较长的工作，以减少尾部等待。确定性 causal backward 还涉及 SPT（Shortest Processing Time）与 CTA swizzling。它们重排任务索引，使执行顺序更适合规定的累加顺序，减少 semaphore 等待。

调度取决于 causal/local、确定性和 kernel 配置。不能把“persistent、LPT、SPT”同时当作所有 FA4 调用固定启用的特征。应同时查看 kernel 的 `tile_scheduler` 选择与具体调度器实现。[FA4 作者调度说明](https://tridao.me/blog/2026/flash4/)

## 12. 四代之间的关系与适用边界

| 版本 | 主要问题 | 代表性改变 |
| --- | --- | --- |
| FA1 | 完整 S/P 的 HBM 存储和往返 | tiling、online softmax、backward 重计算 |
| FA2 | IO 改善后，并行度和工作划分仍不充分 | 延迟归一化、query-tile 并行、sliced-Q |
| FA3 | Hopper 异步硬件需要新的流水线 | TMA/WGMMA、warp specialization、跨工作重叠、FP8 数值与布局 |
| FA4 | SM100 MMA 吞吐增长快于部分配套路径 | TMEM 数据流、conditional rescaling、软件指数、backward 2-CTA |

### 论文代际与软件包覆盖范围

FA1/FA2 不仅能在 Ampere 上运行，FA3 主要面向 Hopper。FA4 论文重点讨论 SM100，但 CuTe DSL 软件包的覆盖范围随版本扩展，不等于一套 SM100 kernel 在所有 GPU 上运行。

本文源码快照为官方仓库 `98c1cee0ed59307833841cdbf567fe336ceec13a`，核对日期为 2026-10-26。该快照 [interface.py](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/interface.py) 包含不同架构的 dispatch。SM120 使用不同于 SM100 的 MMA 路径，不能用 B200 的 TMEM 流水线解释所有调用。

CuTe DSL 是编写和编译 GPU kernel 的工具。Python 风格代码会生成 GPU 实现，并非 Python 解释器逐元素计算 attention。

### FA4 与 FP8 / FP4 后续工作

FA4 基础论文以 BF16 路线分析主要瓶颈。后续低精度扩展继续处理 MXFP8/FP4 的量化、scale 布局和精度问题。它们与 FA4 分块和流水线有关，但不是版本号必然包含的统一行为。

例如 [Low Precision FlashAttention-4](https://pytorch.org/blog/low-precision-flash-attention-4-end-to-end-block-scaled-attention-for-blackwell/) 讨论端到端 block-scaled attention。理解它之前，可先读 [浮点格式与量化](../GPU/precision-and-quantization.md)，再追踪 QK、PV 各自的输入、累加与输出精度。硬件支持一种格式，不代表上层调用自动采用它。

### prefill 与 decode 的瓶颈不同

prefill 有许多 Q 行，Q tiles 可复用 KV 并提供并行工作。单 token decode 每个请求通常只有一个 Q 行，query 分块空间小，更容易受 KV cache 读取带宽和 batch/head 并行度限制。

decode 可以沿 KV 方向切分，让多个 CTA 计算同一 Q 的不同 KV 段，再合并 softmax 状态。这常被称为 split-KV / Flash-Decoding 路线，需要部分输出合并，与 FA2 sliced-Q 的独立行分工不同。不能把长序列 prefill 的论文加速比当作任意 decode 的加速比。

### GQA、分页、变长与稀疏是不同维度

| 概念 | 改变了什么 | 与 FlashAttention 的关系 |
| --- | --- | --- |
| GQA / MQA | 多个 Q heads 共享 KV head | 改变 head 映射与复用，仍可使用分块 attention |
| Paged KV cache | 页表组织 KV 的存储地址 | kernel 按页寻址，可与分块 attention 组合 |
| 变长 batch | 各请求 Q/KV 长度不同 | 需要 offsets、有效长度、边界处理和调度 |
| sliding window / block sparse | 只访问部分 query-key 对 | 改变 attention 图样，不等于稠密 FA 自己变成线性复杂度 |
| dropout | 对概率应用随机 mask 和缩放 | backward 需重建一致随机状态 |
| FlashInfer plan/run | 准备调度与元数据，再执行 kernel | 属于库和执行组织层，不是 FA1–4 的后续版本号 |

实际 attention 常同时涉及这些维度。核对实现时，应查具体 dtype、head dimension、mask、forward/backward 和架构分支，仅确认“用了 FlashAttention”不足以说明完整行为。

## 13. 教学参考实现：从数学对应到 kernel

下面以 CUDA Tensor 写法表达数学。代码没有复现 warp 分工、TMA、TMEM 或高性能流水线，Python 循环会产生多次 kernel 提交。两个教学函数用 `no_grad` 只演示 forward。训练的显存收益需要专门的重计算 backward；直接让 autograd 记录 Python 分块循环，会保留各轮中间量。它们用于阅读和小尺寸对照，不作为性能实现。

### 普通 attention：显式生成 S/P

```python
import math
import torch

@torch.no_grad()
def dense_attention(q, k, v):
    # 单个 batch/head：[Nq,d]、[Nk,d]、[Nk,dv]。
    scores = (q.float() @ k.float().T) / math.sqrt(q.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v.float()).to(q.dtype)
```

### 按 KV 分块的 online softmax

此函数使用 FP32 状态，省略 dropout 和自定义 mask。causal 分支仅支持等长 self-attention，并从左到右处理 KV 块。因此，每个 query 在第一块至少有一个有效 key。

```python
@torch.no_grad()
def tiled_attention(q, k, v, block_q=64, block_k=128, causal=False):
    assert q.ndim == k.ndim == v.ndim == 2
    assert q.device == k.device == v.device
    assert q.shape[1] == k.shape[1] and k.shape[0] == v.shape[0]
    assert k.shape[0] > 0 and block_q > 0 and block_k > 0
    nq, d = q.shape
    nk, dv = v.shape
    assert not causal or nq == nk
    out = torch.empty((nq, dv), device=q.device, dtype=q.dtype)

    for q0 in range(0, nq, block_q):
        qi = q[q0:q0 + block_q].float()
        rows = qi.shape[0]
        m = torch.full((rows,), -torch.inf, device=q.device, dtype=torch.float32)
        l = torch.zeros((rows,), device=q.device, dtype=torch.float32)
        u = torch.zeros((rows, dv), device=q.device, dtype=torch.float32)

        for k0 in range(0, nk, block_k):
            kj = k[k0:k0 + block_k].float()
            vj = v[k0:k0 + block_k].float()
            scores = (qi @ kj.T) / math.sqrt(d)
            if causal:
                q_pos = torch.arange(q0, q0 + rows, device=q.device)
                k_pos = torch.arange(k0, k0 + kj.shape[0], device=q.device)
                scores = scores.masked_fill(k_pos[None, :] > q_pos[:, None], -torch.inf)

            m_new = torch.maximum(m, scores.amax(dim=-1))
            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new[:, None])
            l = alpha * l + p.sum(dim=-1)
            u = alpha[:, None] * u + p @ vj
            m = m_new

        out[q0:q0 + rows] = (u / l[:, None]).to(q.dtype)
    return out
```

`[:, None]` 给每个 query 行的标量增加一个轴，使它按行缩放 `[rows, dv]` 或 `[rows, Bc]`。每轮只保留一个 score tile，不保存完整 `[Nq,Nk]` 矩阵。

即便递推一致，不同 matmul 精度模式、概率转换和累加顺序也会产生浮点差异。比较结果应使用与 dtype、shape 相适应的误差指标，不能要求所有实现逐位相等。

### PyTorch SDPA 与官方 FA4 的布局区别

```python
import torch.nn.functional as F
from flash_attn.cute import flash_attn_func

# 假设 q/k/v 是已准备好的连续 CUDA BF16 Tensor。
# 官方 FlashAttention 常用：[B, N, H, D]。
o_fa4 = flash_attn_func(q, k, v, causal=True)

# PyTorch SDPA 常用：[B, H, N, D]。
o_sdpa = F.scaled_dot_product_attention(
    q.transpose(1, 2),
    k.transpose(1, 2),
    v.transpose(1, 2),
    dropout_p=0.0,
    is_causal=True,
).transpose(1, 2)
```

此接口对照限定等长 MHA。GQA、变长和 paged KV 需要各自参数。SDPA 按环境和输入选择 backend，调用它不代表运行了 FA3/FA4；`torch.compile` 也不是指定 FlashAttention 代际的开关。[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)、[官方 FA4 接口](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/README.md#flashattention-4-cutedsl)

## 14. 论文与源码阅读入口

| 材料 | 重点内容 |
| --- | --- |
| [FA1：2205.14135](https://arxiv.org/abs/2205.14135) | Algorithm 1 循环顺序、online softmax、IO 模型与重计算 |
| [FA2：2307.08691](https://arxiv.org/abs/2307.08691) | 非 matmul FLOPs、query 并行、sliced-Q、backward 所有权 |
| [FA3：2407.08608](https://arxiv.org/abs/2407.08608) | TMA/WGMMA、ping-pong、跨迭代流水线、FP8 布局；附录的实际调度分析 |
| [FA4：2603.05451](https://arxiv.org/abs/2603.05451) | softmax 瓶颈、conditional rescaling、软件指数、backward 2-CTA |
| [FA4 interface.py](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/interface.py) | 输入布局、参数检查、架构和配置 dispatch |
| [FA4 softmax.py](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/softmax.py) | 参考最大值、阈值单位、指数、行统计量 |
| [FA4 SM100 forward](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/flash_fwd_sm100.py) | 加载、TMEM 中 S/P/O、warp 角色、barrier |
| [FA4 SM100 backward](https://github.com/Dao-AILab/flash-attention/blob/98c1cee0ed59307833841cdbf567fe336ceec13a/flash_attn/cute/flash_bwd_sm100.py) | 转置 tile、TMEM 有效期、DSMEM 交换、dQ 汇总与确定性 |

阅读 kernel 时，按一块数据追踪：从哪里加载、由哪个角色拥有、何时完成、由谁消费、何时允许覆盖。再追踪一个输出元素的归约范围，确认各 KV 块怎样进入同一个结果。通过这些对应关系，理解指令和 buffer 怎样实现数学公式。

辅助资料索引：[KernelWiki `wiki/kernels/flash-attention-4.md`](https://github.com/mit-han-lab/KernelWiki/blob/52b8eb06f95ca9b6a1906200b1153e3e21d89c32/wiki/kernels/flash-attention-4.md)，页面 ID 为 `kernel-flash-attention-4`，标注置信级别为 `source-reported`。该索引用于定位资料；算法细节和软件支持范围以本页链接的原论文与固定版本源码为准。

本页解释性能机制，不提供本机跑分。文档、图示和数学检查不能代替 CUDA kernel 的正确性、吞吐或端到端验证。论文峰值只适用于其 GPU、dtype、shape、版本和计时方式，不能用一个 TFLOP/s 数字代表所有 attention 工作负载。
