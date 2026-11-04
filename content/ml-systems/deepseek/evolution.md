---
title: V2–V4 架构演进
category: deepseek
tags: [DeepSeek, MLA, DSA, CSA, MoE]
description: MLA · DSA · CSA / HCA · MoE · mHC · Muon · 缓存与训练
updated: 2026-11-04
---
# DeepSeek：从 V2 的 MLA 到 V4 的压缩注意力

## 1. 几代模型分别改变了什么

| 版本 | 历史怎样表示和读取 | 本代需要理解的其他技术 |
| --- | --- | --- |
| V2 | 每个 token 的 K/V 联合压缩成 latent；仍访问全部历史 | 细粒度 MoE、共享专家 |
| V3 | 延续 MLA；扩大模型与专家池 | 辅助损失自由的负载均衡、MTP、FP8 训练、DualPipe |
| V3.2 | MLA 缓存保留每个 token；indexer 选择主 attention 读取的位置 | DSA、indexer 的训练与低精度计算 |
| V4-Flash / Pro | 多个 token 合成压缩条目；CSA 选择细条目，HCA 读取全部粗条目；保留局部窗口 | shared K=V MQA、分组输出投影、mHC、Muon |

这条路线涉及两个不同的压缩维度：MLA 减少**每条记录的宽度**，CSA/HCA 减少**历史记录的数量**。DSA 决定本次读取哪些记录，不负责永久删除未选中的记录。

```text
V2 / V3：   T 个 token → T 条较窄 latent → 全部历史参与主 attention
V3.2：      T 个 token → T 条较窄 latent → indexer 选 k 条 → 主 attention
V4 CSA：    T 个 token → 约 T/4 条摘要   → indexer 选 k 条 → 主 attention
V4 HCA：    T 个 token → 约 T/128 条摘要 → 全部摘要参与主 attention
                                            ↑
                               CSA/HCA 都另外读取近期原始条目
```

本文聚焦已讨论的 V2、V3、V3.2 和原版 V4。V4.1 的其他结构见[模型家族对照](../attention/model-architectures.md)，不把它的配置混入 V4。下一页[一次前向与全部形状](forward-shapes.md)把本页机制代入 V3、V3.2、V4-Pro 的真实尺寸。

资料核对日期：2026-10-29。配置来自固定 revision 的官方文件，保存在[配置快照](../../../assets/deepseek-configs.json)。本文的形状、算术和复杂度属于静态推导；未运行模型或测量 GPU 性能。

## 2. 配置与参数规模

B 表示十亿参数，T 表示万亿参数。总参数量不是一次前向全部参与计算的参数量，也不是运行时显存需求。MoE 路由、cache、activation、量化 scales 和辅助模块要分别考虑。

| 配置 | V2 | V3 | V3.2 | V4-Flash | V4-Pro |
| --- | ---: | ---: | ---: | ---: | ---: |
| 报告参数规模 | 236B / 21B 激活 | 671B / 37B 激活 | 延续 V3 规模并增加 indexer | 284B / 13B 激活 | 约 1.6T / 49B 激活 |
| 主干层数 | 60 | 61 | 61 | 43 | 61 |
| hidden size | 5120 | 7168 | 7168 | 4096 | 7168 |
| Q heads | 128 | 128 | 128 | 64 | 128 |
| Q latent 宽度 | 1536 | 1536 | 1536 | 1024 | 1536 |
| 主 QK 每头宽度 | 128 + 64 | 128 + 64 | 128 + 64 | 448 + 64 | 448 + 64 |
| V 每头逻辑宽度 | 128 | 128 | 128 | 512，共享 K=V | 512，共享 K=V |
| MLA KV latent 宽度 | 512 | 512 | 512 | 不沿用该结构 | 不沿用该结构 |
| routed experts | 160 | 256 | 256 | 256 | 384 |
| 每 token 选中 routed experts | 6 | 8 | 8 | 6 | 6 |
| shared experts | 2 | 1 | 1 | 1 | 1 |
| 单 expert 中间宽度 | 1536 | 2048 | 2048 | 2048 | 3072 |
| 开头 dense FFN 层数 | 1 | 3 | 3 | 0 | 0 |
| CSA indexer top-k | 无 | 无 | 不适用；DSA 选 2048 个 token | 512 个压缩条目 | 1024 个压缩条目 |
| residual streams | 1 | 1 | 1 | 4 | 4 |

V4 开头的 hash routing 层仍然是 MoE，只是 expert IDs 由 token ID 查表确定。它不是 V3 的前三层 dense FFN。Flash 前两层使用纯滑窗 attention；Pro 前两层使用 HCA，后续再交错安排 CSA/HCA。原版 Pro 和 Flash 不能套用同一张尺寸表。

参数量使用报告的近似口径；不能拿主干规模与包含 MTP 的 checkpoint 参数数直接相减。V3.2 增加了 indexer，因此“延续 V3 规模”不代表每个参数完全相同。

来源：[V2 报告][v2-paper]、[V2 配置][v2-config]、[V3 报告][v3-paper]、[V3 配置][v3-config]、[V3.2 配置][v32-config]、[V4 报告][v4-paper]、[Flash 配置][v4f-config]、[Pro 配置][v4p-config]。

## 3. V2：每个 token 一份 latent，多个 head 使用它

### 3.1 联合压缩的对象是 K/V

以下使用行向量。设一个 token 的 hidden 为 $h_j\in\mathbb R^D$：

$$
c_j=h_jW^{DKV},\qquad c_j\in\mathbb R^{512}.
$$

各 head 用各自的矩阵产生内容 K 和 V：

$$
k^C_{j,h}=c_jU_{K,h},\qquad v_{j,h}=c_jU_{V,h}.
$$

V2/V3 的每头内容 K、V 都是 128 维，所以 $U_{K,h},U_{V,h}$ 都是 `[512,128]`。实际模型还包含 latent 的 RMSNorm；这里先省略以看清线性关系。

```text
一个 token 的 hidden
        ↓ 下投影、归一化
共享 latent c：[512]
        ├─ U_K[head 0] → 内容 K[0]：[128]
        ├─ U_V[head 0] → V[0]：     [128]
        ├─ U_K[head 1] → 内容 K[1]：[128]
        └─ U_V[head 1] → V[1]：     [128]
```

这是模型在训练中学出的表示，不是对任意已有 MHA 缓存做无损压缩。MQA 直接共享 K/V head；MLA 允许不同 head 从同一 latent 得到不同内容 K/V。

### 3.2 decode 不必恢复全部历史 K/V

内容分数利用矩阵乘法的结合律：

$$
q_h^C(k_{j,h}^C)^T
=q_h^CU_{K,h}^Tc_j^T
=(q_h^CU_{K,h}^T)c_j^T.
$$

把 query 从 128 维变换为 512 维，再与历史 latent 点积，就避免了对每个历史位置展开 K。这个变换常称为 K 投影的权重吸收。

Value 一侧利用线性性：

$$
\sum_j p_{h,j}(c_jU_{V,h})
=\left(\sum_jp_{h,j}c_j\right)U_{V,h}.
$$

先对 latent 加权得到 512 维向量，再做每头的 V 投影。不是先把 softmax 权重乘 V 投影矩阵；两者相乘的维度通常就不匹配。

各头 Q 的生成矩阵依然存在。吸收后每个 head 有自己的 query、attention 权重和 latent 加权结果。历史 latent 在 heads 间共享，不意味着所有 heads 的输出相同。

### 3.3 位置编码为什么单独存 64 维

普通 RoPE 的旋转依赖 token 位置，不能把所有位置对应的旋转统一塞进一张固定矩阵。MLA 把内容分支与位置分支拆开：

$$
\ell_{t,h,j}=\alpha\left(
(q^C_{t,h}U_{K,h}^T)c_j^T+q^R_{t,h}(k_j^R)^T
\right).
$$

每层每 token 的核心缓存是：

```text
c_j：[512]，归一化后的联合 KV latent
k_R：[64]，已施加 RoPE 的位置 key
```

这里的 `kv_lora_rank` 是架构里的低秩宽度，不表示额外挂了 PEFT LoRA adapter。吸收也不能任意跨越 RMSNorm、softmax 或其他非线性操作。

[V2 报告][v2-paper]给出 MLA 与 DeepSeekMoE 的设计；更详细的 decode 和 hidden 生命周期推导见[现有 MLA 教学](../attention/model-architectures.md#4-mla压缩每个-token-的表示不压缩-token-数)。

## 4. V3：MLA 不变，把训练与 MoE 扩展到更大规模

V3 继续使用 512 维联合 KV latent 和 64 维位置 key。变化不只是 attention，还包括训练系统和专家路由。

| 技术 | 输入、输出与作用 | 推理时能看到什么 |
| --- | --- | --- |
| DeepSeekMoE | router 为 token 选择专家；专家各自执行 SwiGLU | 每个 MoE 层执行 8 个 routed experts 和 1 个 shared expert |
| 辅助损失自由的负载均衡 | 训练时根据专家负载调整选择偏置 | 偏置影响选谁；混合权重来自原 affinity |
| Multi-Token Prediction，MTP | 训练主干同时学习额外的未来 token 预测目标 | 普通主干生成不必执行 MTP；可用于推测生成 |
| FP8 混合精度训练 | 为适合的矩阵运算选择 FP8 数据与 scales | 权重与 kernel 可使用低精度，但不能理解为所有算子都用 FP8 |
| DualPipe | 安排流水线中的前后向与通信，减少等待 | 是训练流水线调度，不是新增 attention 公式 |

“辅助损失自由”主要指用偏置调节代替强制的专家均衡辅助梯度。V3 仍保留小权重的 sequence-wise balance loss，不能写成“训练完全没有任何均衡辅助损失”。总参数增长也不会使一个 token 执行全部专家。

MTP 模块不等于一次普通 forward 自动生成多个已确认 token。若用它做 speculative decoding，还需要候选生成、主模型验证、接受或拒绝，以及对应的 cache 管理。[V3 报告][v3-paper]

## 5. V3.2 DSA：indexer 选择位置，主 MLA 计算结果

### 5.1 两套 query 各自承担什么

V3.2 的 indexer 有 64 个 query heads，每头 128 维；主 MLA 有 128 个 query heads。Indexer 使用主 Q 下投影后的 latent 产生自己的 query，也保存自己的历史 indexer keys。

```text
当前 hidden
  ├─ 主 Q latent → indexer Q → 扫描 indexer keys → top-2048 位置
  └─ 主 Q latent → 主 MLA Q ────────────────────→ 读取所选 MLA 条目
                                                        ↓
                                              主分数、softmax、V 加权
```

简化后的 indexer 分数：

$$
I_{t,j}=\sum_{a=1}^{64}w_{t,a}\operatorname{ReLU}
\left(q^I_{t,a}(k^I_j)^T\right).
$$

Indexer 分数负责排序，不是最终 attention 权重。选中集合会被主 heads 共用，主 heads 仍各自计算分数。未选中的缓存继续保留，下一 query 仍可能选中它。

### 5.2 选择器怎样学会筛选

DSA 使用主 attention 的分布作为监督，训练 indexer 找到重要位置。先在冻结主干的阶段让 indexer 对齐目标分布，再继续训练稀疏 attention。不能把 top-k 理解为一个无需训练、随机截断历史的位置选择器。[V3.2 报告][v32-paper]

### 5.3 稀疏的范围和代码实现的差别

设历史长度为 T，选择数为 k。单 token decode 的主 attention 从全历史读取变成读取 k 个条目，但 indexer 还要扫描 T 个历史 keys。对于长 prefill，主稀疏 attention 的配对数可接近 $O(Sk)$；全历史扫描的 indexer 仍有二次项。DSA 整体不能直接称为严格线性 attention。

官方教学参考代码有“计算完整 scores，再加 index mask”的分支，便于表达数学语义。高性能稀疏 kernel 可以直接读取选中的位置。阅读源码时，先确认选择集合的数学定义，再看参考代码是否先算全量 scores。实际部署 kernel 是否避免未选条目的计算，需要单独核对。[V3.2 参考实现][v32-code]

## 6. V4：CSA 与 HCA 压缩序列维度

### 6.1 压缩条目怎样生成

设一个压缩组里有 m 个 token，每个先投影出候选向量 $u_j\in\mathbb R^c$ 和 pooling logits $z_j\in\mathbb R^c$。用位置偏置 $b_j$ 后，对每个特征 d 单独沿 token 维归一化：

$$
\alpha_{j,d}=\frac{\exp(z_{j,d}+b_{j,d})}
{\sum_{r\in\mathrm{group}}\exp(z_{r,d}+b_{r,d})},
\qquad
c_{\mathrm{group},d}=\sum_{j\in\mathrm{group}}\alpha_{j,d}u_{j,d}.
$$

例如一个 `[4,512]` 组可变成 `[512]`：第 0 个特征可以主要保留第 1 个 token，第 1 个特征可以主要保留第 3 个 token。每个特征都有自己的 pooling 权重，不是固定平均。

这些权重来自组内 hidden 和位置参数，不依赖未来 query。摘要一旦生成，后续 queries 就可以复用。压缩是有损的模型表示，需要训练配合，不保证能逐 token 解压回原值。

### 6.2 CSA：轻度压缩后选择

CSA 全称 Compressed Sparse Attention，压缩步长为 4。实际用两组投影让相邻块重叠：一个条目最多汇总 8 个 token 的候选信息。相邻条目共享部分来源，最终条目数仍约为 T/4。第一个条目的前邻居为空，按实现处理 padding。

```text
历史 token → 约 T/4 个主压缩条目 [512]
                    ↑
历史 token → 约 T/4 个 indexer 压缩 keys [128]
                    ↓ 当前 query 打分
             top-k 条目编号
                    ↓
           主 attention 读取对应条目
```

Indexer 选择的是压缩条目编号，而不是 V3.2 的原始 token 编号。Pro 选 1024 条，Flash 选 512 条。

### 6.3 HCA：重度压缩后全部读取

HCA 全称 Heavily Compressed Attention。它使用不重叠的 128-token 组，得到约 T/128 个条目，不再用 indexer 截取 top-k。每个 query 读取全部因果可见的压缩条目。

固定压缩率没有让历史变成固定大小状态：T 增长时，T/128 也增长。HCA 仍是压缩条目上的 softmax attention，不等于 recurrent linear attention。

### 6.4 两者都有近期窗口，且必须保持因果性

CSA/HCA 的长程摘要会丢失细节。当前组尚未完成时，也不能读取包含未来 token 的摘要。所以每层额外保留 128 个近期 token 的未压缩条目。

```text
主 attention 的读取集合
= 近期窗口原始条目
+ 本次可见的长程压缩条目
```

两部分共同进入归一化，不能直接把两次独立 softmax 的结果相加。可以分段计算后正确合并 softmax 的统计量，但那是同一数学结果的 kernel 实现。

压缩条目只在所需输入已经到达后变为可见。参考实现使用零基位置 t，以 `floor((t+1)/m)` 控制可用完整组数。位于组末尾的当前 token 已参与前向，因此刚完成的组可以被这个位置读取。比较论文中的不等号前，应先统一一基/零基位置记号。

### 6.5 同一个条目同时用作 K 和 V

V4 的主 attention 使用 shared K=V MQA。设读取集合为 C，忽略 RoPE 和 sink：

$$
P_h=\operatorname{softmax}(q_hC^T/\sqrt{512}),\qquad o_h=P_hC.
$$

这与 V3 的 512 维 latent 缓存宽度碰巧相同，但角色不同。V3 的 512 维 latent 还对应 head-specific 的 K/V 恢复矩阵；V4 的共享条目直接参与主 attention。

每个 Q head 是 512 维，最后 64 维使用 RoPE。同一条目也用于 V，因此输出携带位置分量；参考实现对输出最后 64 维施加当前 query 位置的逆 RoPE。每头另有 attention sink logit b：

$$
p_{h,j}=\frac{\exp(\ell_{h,j})}{\sum_r\exp(\ell_{h,r})+\exp(b_h)}.
$$

Sink 增加分母，不提供一个普通历史 V；实际条目权重之和可以小于 1。

来源：[V4 attention 公式][v4-paper]、[官方参考实现][v4-code]、[Transformers V4 实现说明][v4-hf]。

## 7. 用同一历史长度比较读取与缓存

固定为单请求 decode：已有 P=131072 个历史 token，本次输入 1 个新 token，T=131073。这个新 token 尚未完成下一压缩组。

| 每层、单请求 | V3 | V3.2 | V4-Pro CSA | V4-Pro HCA |
| --- | ---: | ---: | ---: | ---: |
| 长程主缓存条目数 | 131073 | 131073 | 32768 | 1024 |
| 主 attention 访问的长程条目数 | 131073 | 至多 2048 | 至多 1024 | 1024 |
| 独立局部窗口 | 无；已包含在全量集合 | 无独立保留窗口保证 | 128 | 128 |
| Indexer 扫描条目数 | 无 | 131073 | 32768 | 无 |
| 主缓存每条逻辑宽度 | 512 + 64 | 512 + 64 | 512 | 512 |
| 额外 indexer key 宽度 | 无 | 128 | 128 | 无 |

在这个例子里，CSA/HCA 的主 attention 都读 1152 条，但 CSA 先在 32768 条细摘要中筛选；HCA 直接读全部粗摘要。不能据此断言两种层延迟相同。

只按主缓存有效元素、并假设统一 BF16，容量可手算为：

| 类型 | 元素数 | 假设 BF16 的字节数 |
| --- | ---: | ---: |
| V3 / V3.2 主缓存 | `131073 × 576 = 75498048` | `150996096`，约 144 MiB |
| CSA 主缓存，含窗口 | `(32768+128) × 512 = 16842752` | `33685504`，32.125 MiB |
| HCA 主缓存，含窗口 | `(1024+128) × 512 = 589824` | `1179648`，1.125 MiB |

这是便于比较形状的假设值，不是实际部署显存。V3.2 还有 indexer cache；V4 有 indexer cache、压缩器的未完成组状态和重叠组状态。实际 cache 可能混用 FP8/BF16，还包含 scales、页表、分配粒度与容量预留。参考代码也可能在 BF16 tensor 中模拟量化精度，不能把它的分配量当作优化服务的物理格式。

## 8. mHC：同一个 token 的多条残差流

### 8.1 从普通残差到四条流

普通子层忽略 norm 后为 $x'=x+F(x)$。F 可以是 attention 或 FFN。V4 把一个 token 的层间状态从 `[D]` 扩成 `[4,D]`：

$$
X'=BX+C\,F(AX),
\quad A\in\mathbb R^{1\times4},\ B\in\mathbb R^{4\times4},\ C\in\mathbb R^{4\times1}.
$$

```text
X：[4,D] ── A：[1,4] ──→ 子层输入：[1,D] ── F ──→ [1,D]
   │                                                 │
   └── B：[4,4] ──→ 原残差混合：[4,D]       C：[4,1] ──→ [4,D]
                            └──────────── 相加 ────────────┘
                                           ↓
                                      X_next：[4,D]
```

4 是 residual stream 数，不是 batch、token 数或 head 数。Attention/FFN 只接收一次 D 维输入，不因为有 4 条流就执行 4 份完整子层。Attention 前后和 FFN 前后分别有一套混合参数。

### 8.2 混合系数怎样产生

以 Pro 的 D=7168 为例，当前状态展平为 `[28672]`。一个小投影产生 `4+4+16=24` 个动态系数，再叠加可学习的静态偏置和缩放。A、C 使用有界非负变换，B 用 Sinkhorn-Knopp 处理。

B 被限制为双随机矩阵：元素非负，每行和每列都为 1。有限次迭代只近似满足约束。Pro 配置中迭代次数为 20。

只看两个流的一个特征：

$$
\begin{bmatrix}0.8&0.2\\0.2&0.8\end{bmatrix}
\begin{bmatrix}10\\2\end{bmatrix}
=
\begin{bmatrix}8.4\\3.6\end{bmatrix}.
$$

这个例子里，流之间交换信息，但残差混合没有把数值无约束地放大。两项和仍为 12，各项位于原来的 2 到 10 之间。该性质针对固定 B 的残差混合路径。加入子层、动态系数的导数后，不构成整个网络永不失稳的证明。

### 8.3 多条 residual 不等于历史 hidden cache

Decode 时保存的是当前 token 的 `[4,D]` 状态，逐层生成各层 cache 后就可以释放相应临时 activation。下一 token 依靠各层 cache 读取历史，无需重跑历史 FFN。增加残差流会增加 activation 读写与混合开销，但不会要求普通生成保存所有历史层的完整 hidden。

来源：[mHC 原论文][mhc-paper]、[V4 实现][v4-code]。

## 9. Muon：训练时变换更新矩阵

Muon 是 optimizer，与 AdamW 属于同一层面。它作用在 backward 之后：

```text
forward → loss → backward → 梯度矩阵 → 动量与更新变换 → 修改权重
```

设权重 $W\in\mathbb R^{m\times n}$，梯度为 G。省略 Nesterov、weight decay 和额外缩放的示意为：

$$
M_t=\beta M_{t-1}+G_t,\qquad
U_t\approx\operatorname{Ortho}(M_t),\qquad
W_{t+1}=W_t-\eta U_t.
$$

Muon 近似正交化的是**更新矩阵**，不直接强制权重 W 正交。与按元素估计尺度的优化器相比，它利用整个更新矩阵的结构。

用一个理想化的满秩对角例子理解：

$$
M=\begin{bmatrix}100&0\\0&1\end{bmatrix}
\quad\longrightarrow\quad
\operatorname{Ortho}(M)=\begin{bmatrix}1&0\\0&1\end{bmatrix}.
$$

把整个 M 除以一个常数，仍保留 100:1 的强度差。理想正交化调整不同方向的尺度，再用学习率控制整体更新。一般非对角矩阵不能逐元素照搬这个例子，有限次近似也不保证精确得到单位矩阵。

### Newton-Schulz 与 V4 的混合迭代

正交化用一系列矩阵乘法近似完成。归一化初始矩阵后，迭代具有如下形式：

$$
X_{k+1}=aX_k+b(X_kX_k^T)X_k+c(X_kX_k^T)^2X_k.
$$

V4 使用 10 次混合迭代：前 8 次与后 2 次使用不同系数，并使用 Nesterov 和更新 RMS 缩放。Embedding、预测头、RMSNorm，以及 mHC 的静态偏置和 gating factors 仍使用 AdamW。

Muon 增加了 optimizer 阶段的矩阵运算。评估收益应比较达到同等质量的总训练时间，不能直接推断每步更快。推理只读取已经训练好的权重，不执行 Muon。mHC 和 CSA/HCA 则属于 forward，推理仍会执行。

来源：[Muon 作者说明][muon]、[实现仓库](https://github.com/KellerJordan/Muon)、[V4 optimizer 算法][v4-paper]。

## 10. 从 V3.2 到 V4：设计是否更保守

**V4 的 KV 使用方式更直接，但对历史信息的压缩更激进。** 如果“保守”指减少 MLA 中显式的 K/V 上投影与权重吸收，能理解这种判断；如果指架构改动更小，或者更完整地保留历史细节，这个判断就不成立。

本节于 2026-11-03 补充。下面先比较公开结构，再讨论设计取舍；“更直接”“更激进”是基于机制的分析，不是论文给出的能力排名，也不是本机性能测量。

### 10.1 KV 使用方式更直接，但核心计算有连续性

V3.2 的 MLA 在模型定义中，通过共同 latent 生成各 head 的内容 K/V。执行时可以吸收 K 投影，并将 V 投影移到加权聚合之后，从而直接读取 latent。V4 则直接规定：压缩条目兼作主 attention 的 K 和 V。

```text
V3.2，按吸收后的执行方式理解：
逐 token 的共同 latent → indexer 选位置 → 多头打分与 latent 聚合 → 输出映射

V4 CSA：
多个 token 的学习压缩条目 → indexer 选条目 → 与近期窗口共同参与 attention → 分组输出映射

V4 HCA：
更粗的学习压缩条目 → 全部可见条目与近期窗口共同参与 attention → 分组输出映射
```

V3.2 报告明确说明，DSA 使用 MLA 的 **MQA mode**。它的高效核心计算也已经是多个 query heads 读取共同 latent。因此，不能把“未吸收 MLA 的模型定义图”和“V4 的直接执行图”当成同一层面的比较，再由此得出 V4 退回了简单 attention。[V3.2 §2.1 与附录 A][v32-paper]

| 比较对象 | V3.2 的吸收后 MLA | V4 |
| --- | --- | --- |
| 内容打分 | 各 head 的 query 对共享 latent 打分 | 各 head 的 query 对共享压缩条目打分 |
| 内容聚合 | 各 head 对同一 latent cache 加权 | 各 head 对同一读取集合加权 |
| head 差异在哪里 | query、attention 权重及输出映射 | query、attention 权重及分组输出映射 |
| 历史条目的单位 | 每个 token 一份 latent | 多个 token 合成摘要，另保留近期逐 token 条目 |

这张表只对照内容读取主线。两代的位置处理、归一化和参数化也有差别，不能据此把它们认定为同一个函数。V4 仍保留 query 的低秩中间表示，称它“放弃 latent”也不准确。[V4 §2.3][v4-paper]

### 10.2 对历史细节的取舍，V4 反而更大胆

假设很早之前的一段代码包含几个变量名，当前 query 需要其中一个位置的信息。

在 V3.2 中，每个历史位置仍保留自己的 MLA 条目。Indexer 这次没有选中某个位置，下一次仍有机会选择它。单条 latent 本身是压缩表示，但**不同位置的记录仍然分别保存**。

在 V4 中，超出近期窗口的内容主要由学习得到的摘要承载。多个 token 已合并成一条记录，后来的 query 可以读取这条记录，却不能要求这一层再单独取回合并前那个位置的原始 KV 条目。

```text
V3.2：保存逐位置条目，每次读取时选择
V4：  先决定怎样合并位置，再读取或选择合并后的条目
```

这意味着 V4 的压缩器要提前学会保留未来可能用到的信息。HCA 每 128 个 token 形成一个条目，对逐位置访问自由度的约束尤其明显。SWA 保留近期细节，CSA 提供较细的长程摘要，HCA 提供较粗的长程覆盖，三者在不同层和读取分支中协作。

这种组织比 V3.2 更积极地减少历史记录，但不能直接推导“任务效果更差”。摘要可以学到有用信息，前面的层也可以把信息传入后续表示。具体丢失哪些任务所需的细节，仍要通过质量评估判断。

### 10.3 更重视执行成本，是合理解读而非已验证结论

V4 不再要求先定义各 head 的内容 K/V 展开，再靠吸收避免展开。它用直接的 shared K=V 接口组织主 attention，并通过 grouped output projection 控制宽输出的投影成本。[V4 shared-KV 与输出投影][v4-paper]

**我的解读是：设计者把更多重点放在历史条目数量、共享方式和可控的执行成本上。** 这是从机制得到的判断。不能进一步省略验证，声称 V4 的全部 kernels 都更简单或更快：压缩器、indexer、SWA、缓存更新与因果边界也增加了工作。

比较效率应至少区分：

- 为当前 query 搬运多少历史数据；
- 生成、选择和处理摘要需要多少计算；
- 不同 batch、历史长度和 TP 布局下，主要瓶颈是什么；
- 与该成本对应的模型质量是否达到同一要求。

原版 V4 的压缩条目仍随历史增长，共享 KV 也不会自动解决所有 TP 下的缓存复制。不能将“每条记录更少”推导为“所有执行成本都与历史长度无关”。

### 10.4 不能只数矩阵，判断谁的表达能力更强

共享 K/V、分组输出瓶颈都对参数之间的关系施加约束。相对一张完全自由的同形状大输出矩阵，分组投影确实限制了映射，并减少相应成本。

但比较对象是 V3.2 时，结论没有这么直接：V3.2 本身也包含共同 latent、各 head 的投影与输出矩阵之间的因子分解。V4 又同时改变了位置处理、head 输出宽度和序列压缩。仅仅观察“少了一张上投影”或者“用了 MQA”，不足以证明两代模型存在严格的表达能力高低关系。

在所核对的公开报告中，我没有看到隔离这些因素的完整同预算消融：保持其他结构和训练条件一致，只替换这套 attention 参数化，再比较质量与性能。因此，下列判断需要分别保留证据边界：

| “更保守”的具体含义 | 本节判断 |
| --- | --- |
| KV 使用方式更直接，不再突出复杂的吸收技巧 | 基本同意 |
| 更愿意约束结构，换取可控的存储和执行成本 | 部分同意，属于设计解读 |
| 对旧 attention 架构的改动更小 | 不同意，新增了跨 token 压缩与多粒度组合 |
| 对历史细节保留得更完整、更谨慎 | 不同意，逐位置记录被进一步合并 |
| 表达能力或任务效果一定弱于 V3.2 | 证据不足 |

理解这次变化，可以沿着一个问题继续追踪：**除了让每个 token 的历史表示更紧凑，还能怎样决定哪些细节值得长期保存，以及以什么粒度保存？** 这正是 CSA/HCA 与局部窗口组合需要解决的问题。

## 11. 阅读顺序与来源

先把[前向与形状](forward-shapes.md)中的 V3 路径走通，再看 V3.2 增加的 indexer，最后把 V4 的 cache、输出投影与 residual 替换进去。遇到通用概念，可回到 [Transformer 运行全过程](../attention/transformer-runtime.md)、[浮点格式与量化](../gpu/precision-and-quantization.md)和[MoE 所需的通信基础](../gpu/communication-and-allreduce.md)。

本文配置固定在下列 revision，完整 JSON 位于[配置快照](../../../assets/deepseek-configs.json)。固定配置用于复核尺寸，不代表已执行这些模型。

| 模型 | 固定配置 |
| --- | --- |
| V2 | [4461458f][v2-config] |
| V3 | [e815299b][v3-config] |
| V3.2 | [a7e62ac0][v32-config] |
| V4-Flash | [60d8d707][v4f-config] |
| V4-Pro | [b5968e91][v4p-config] |

[v2-paper]: https://arxiv.org/abs/2405.04434
[v3-paper]: https://arxiv.org/html/2412.19437v1
[v32-paper]: https://arxiv.org/html/2512.02556v1
[v4-paper]: https://arxiv.org/html/2606.19348v1
[mhc-paper]: https://arxiv.org/html/2512.24880v2
[muon]: https://kellerjordan.github.io/posts/muon/
[v2-config]: https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/4461458f186c35188585855f28f77af5661ad489/config.json
[v3-config]: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json
[v32-config]: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/config.json
[v4f-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json
[v4p-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json
[v32-code]: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/inference/model.py
[v4-code]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py
[v4-hf]: https://huggingface.co/docs/transformers/v5.20.0/model_doc/deepseek_v4
