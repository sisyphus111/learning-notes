# DeepSeek、Kimi、Qwen：模型结构与注意力变体

## 1. 按版本看结构，按计算角色理解名词

资料核对日期为 **2026-10-28**。本页选择三家公开模型的主要架构路线，覆盖 DeepSeek-V3 到 V4.1、Kimi-K2 到 K3，以及 Qwen3、Next、3.5、3.8 的代表分支。它不是所有 checkpoint、API 别名或后训练版本的目录。

结构结论来自官方技术报告、模型卡与固定 revision 的配置。参数量采用各自公开口径。主干、视觉模块、词表、条件记忆和草稿模块可能分开计数，不能只比较模型名称里的数字。第 14 节保留配置来源，方便以后更新。

理解一个模型，可以分别追踪四条路径：

| 路径 | 核心疑问 | 代表设计 |
| --- | --- | --- |
| token mixing | 当前 token 怎样读取历史？ | GQA、MLA、DSA、CSA、GDN、KDA、QSA |
| channel mixing | 每个 token 怎样变换特征？ | dense FFN、MoE、LatentMoE |
| depth mixing | 不同深度的表示怎样传递？ | 残差加法、mHC、AttnRes、Gated Residual |
| 输入与生成 | 图像怎样进入模型，多个 token 怎样生成？ | ViT/projector、位置编码、MTP、speculative decoding |

**MoE 不规定 attention 用什么公式；dense 也不等于每一层都使用 full attention。** Qwen3.5-27B 就可以同时是 dense FFN 和 GDN/full-attention 混合模型。

![三个系列的代表性结构变化](../assets/model-architecture-routes.svg)

图中按家族分组，框内描述不同计算角色。这些机制不都属于同一个 attention kernel；同系列也可以并存多条结构路线。

## 2. 先建立一张代表版本地图

表中 B 表示十亿参数，T 表示万亿参数；“总量/激活量”不是显存需求比，也不是端到端计算量比。共享专家、视觉模块和辅助参数的口径还需看各节说明。

### DeepSeek

| 代表版本 | 公开参数口径 | 历史读取机制 | FFN 与其他结构 |
| --- | --- | --- | --- |
| V3 / 原版 R1 主干 | 671B / 37B | MLA，全历史 softmax | DeepSeekMoE；R1 的主要差别在后训练 |
| V3.2 | 延续 V3 规模，增加 indexer | MLA + DSA | 学习选择 token，主干仍是 MoE |
| V4-Flash | 284B / 13B | SWA + 交替的 CSA/HCA | MoE、mHC、MTP |
| V4-Pro | 约 1.6T / 49B | SWA + CSA/HCA | 更大宽度与专家池；具体层配置不同于 Flash |
| V4.1-Flash，本文配置为 1017 版本 | 552B 主干，另有 196B Engram；prefill 8B、decode 16B 激活口径 | CED + CSA2 + 层内 SWA | 跨层 KV/索引复用、Single-Pass mHC、独立训练 DSpark |

来源：[V3][ds3-card]、[V3.2 报告](https://arxiv.org/abs/2512.02556)、[V4 模型卡][ds4f-card]、[V4.1 报告](https://arxiv.org/abs/2609.19969)。V4.1 的输入/输出激活量不同，不能统一写成“每 token 16B”。

### Kimi

| 代表版本 | 公开参数口径 | 历史读取机制 | FFN 与其他结构 |
| --- | --- | --- | --- |
| K2 | 约 1T / 32B | MLA | 384 个 routed experts，每 token 选 8 个，另有共享专家 |
| K2.5 | 约 1T / 32B 文本主干 | MLA；加入原生多模态路径 | MoonViT + projector，文本主干仍是 MoE |
| Kimi Linear 48B-A3B | 48B / 约 3B | KDA + MLA | 27 层中 20 个 KDA、7 个 MLA；MoE |
| K3 | 约 2.8T / 104B | 69 个 KDA + 24 个 gated MLA | Stable LatentMoE、Block AttnRes、MoonViT-V2 |

K2.6 配置的主要文本尺寸仍是 61 层、hidden 7168、64 个 attention heads、384 个 routed experts。不能仅凭版本升级，断言它已经切换到 KDA。K3 才是本页重点比较的新混合主干。[K2][k2-card]、[K2.5][k25-card]、[K2.6 配置][k26]、[Kimi Linear][kl-card]、[K3][k3-card]

### Qwen

| 代表版本 | 公开参数口径 | attention 配置 | FFN |
| --- | --- | --- | --- |
| Qwen3-32B | 约 32.8B | 每层 GQA | dense SwiGLU |
| Qwen3-235B-A22B | 235B / 22B | 每层 GQA | 128 experts，选 8 |
| Qwen3-Next-80B-A3B | 80B / 约 3B | 3 个 GDN 层 + 1 个 gated full-attention 层 | 512 routed experts，选 10，另有 shared expert |
| Qwen3.5-27B | 27B | 48 个 GDN + 16 个 gated full-attention 层 | dense |
| Qwen3.5-35B-A3B | 35B / 约 3B | 30 个 GDN + 10 个 gated full-attention 层 | 256 routed experts，选 8，另有 shared expert |
| Qwen3.8-27B | 27B | 保留上述 27B 分支的主要混合形状 | dense |
| Qwen3.8-2.4T-A95B | 约 2.4T / 95B | 69 个 GDN + 23 个 gated full-attention 层 | 512 routed experts，选 10 |
| Qwen3.8-Flash-Next | 125B 主干 / 6B，另有 51B n-gram、4B MTP | 36 个 GDN + 12 个 QSA 层 | 512 routed experts，选 10；另有 GR 与 n-gram embedding |

这些行来自对应的官方模型卡和配置，固定链接见第 14 节。**Qwen3.8-Flash-Next 与 Qwen3.8-27B 不是只改参数量的同构模型。** 前者是公开的实验架构分支；配置里的 `qwen4_exp_text` 也不等于已经确认所有未来 Qwen4 都采用它。

## 3. GQA：共享 KV heads，仍然对历史 token 做 softmax

设 query heads 数为 $H_q$，KV heads 数为 $H_{kv}$，head dimension 为 d。MHA 中二者相等；GQA 让一组 query heads 共用一个 KV head；MQA 则只有一个 KV head。

例如 Qwen3-32B：

```text
hidden：[B,T,5120]
Q：     [B,T,64,128]
K/V：   [B,T,8,128]
每 8 个 Q heads 共用 1 个 KV head
```

Q 的投影宽度为 `64×128=8192`，不必等于 hidden size。普通 GQA 仍保存历史 token 的 K/V；共享 head 减少缓存宽度，不减少历史位置数量。

若 K/V 都使用每元素 b 字节，单层缓存的逻辑大小为：

$$
M_{\mathrm{KV}}=BT H_{kv}(d_k+d_v)b.
$$

Qwen3 的 dense 和 MoE 版本都可以使用 GQA。是否 MoE 取决于 FFN 的路由，而不是这个缓存公式。[Qwen3-32B 配置][q3d]、[Qwen3-235B 配置][q3m]

## 4. MLA：压缩每个 token 的表示，不压缩 token 数

### 用一个 latent 生成多个 head 的内容 K/V

以行向量约定写一个简化 MLA 内容分支：

$$
c_t=h_tW^{DKV},\qquad k^C_{t,i}=c_tW^{UK}_i,\qquad v_{t,i}=c_tW^{UV}_i.
$$

$c_t$ 是跨 heads 共享的低维表示，head-specific 的内容 K/V 由各自 up-projection 得到。DeepSeek-V3 的 `kv_lora_rank=512`，还把位置相关的 RoPE key 分支分开保存，维度为 64。

```text
每个 token 的缓存：
latent c：[512]  +  decoupled RoPE key：[64]
```

这里 `lora_rank` 描述架构中的低秩投影，不表示推理时额外挂了一个 PEFT LoRA adapter。

### 为什么不用先恢复全部 K/V 再计算

内容分数可以重排为：

$$
q_i^C(k_{t,i}^C)^T=(q_i^C(W_i^{UK})^T)c_t^T.
$$

加权 value 也有：

$$
\sum_t p_{i,t}(c_tW_i^{UV})=\left(\sum_t p_{i,t}c_t\right)W_i^{UV}.
$$

这允许将部分投影吸收到 query/output 路径，避免 decode 时逐 token 物化所有 head 的完整 K/V。位置相关分支需要单独处理，RoPE 分支的结构决定了投影能否吸收。

若只按 BF16 的 512+64 个缓存元素算，V3 每层每 token 是 1152 bytes，61 层约 68.6 KiB。这是逻辑表示的尺寸推导；实际还可能使用量化、对齐、page metadata 和其他缓冲区。[DeepSeek-V2 MLA 报告](https://arxiv.org/abs/2405.04434)、[V3 配置][ds3]

MLA 仍为历史中的每个 token 保存一个条目，长度随 T 增长。它也不是在任意已有 MHA 权重外做一次无损压缩：模型本身就是按这种表示训练的。

## 5. DeepSeek：从 MLA 到 DSA、CSA/HCA，再到 CSA2

### V3 / R1：attention 与 MoE 的基础组合

V3 主干有 61 层、hidden size 7168、128 个 query heads；前三层使用 dense FFN，其余配置采用细粒度 routed experts 与 shared expert。routed expert 的 intermediate size 为 2048，不等于整个模型只有一个 2048 宽的 FFN。

原版 DeepSeek-R1 使用 V3 家族主干，强化学习改变了训练与生成行为，不意味着 R1 新增了一种叫“reasoning attention”的算子。R1-Distill-Qwen/Llama 则继承各自学生模型的架构，不能从 R1 名字推断它们都使用 MLA。[V3 配置][ds3]、[R1 官方说明](https://huggingface.co/deepseek-ai/DeepSeek-R1)

### V3.2 的 DSA：先便宜地选位置，再昂贵地读内容

DSA 是 DeepSeek Sparse Attention。在 MLA 主 attention 之前，lightning indexer 用较小的表示为历史位置打分，再选 top-k；主 attention 只读取选中的 KV 条目。

```text
当前 hidden → indexer query
历史 indexer keys → relevance scores → top-k token IDs
                                      ↓
当前主 query + 选中的 MLA KV → sparse softmax → output
```

该配置的 indexer 有 64 个 heads、head dimension 128，`index_topk=2048`。这些是 indexer 尺寸，不能当成主 attention 的 head 数。[V3.2 配置][ds32]

DSA 减少主 attention 的候选集合，但 indexer 自己也有成本。对整段 prefill，若每个 query 都给所有历史 index keys 打分，indexer 仍有二次规模。top-k 本身也不会使缓存自动消失：这一步未选中的 token，下一个 query 仍可能需要。[V3.2 报告](https://arxiv.org/abs/2512.02556)

### V4 的 CSA/HCA：沿序列方向压缩条目

V4 转为 shared-KV MQA 的压缩历史路线，不能继续简单标成“V3 MLA 加了稀疏”。其核心缓存条目在主 attention 中兼作 key 和 value，query 有多个 heads，输出使用 grouped projection 降低投影成本。

| 分支 | 怎么保存历史 | 主 attention 读取什么 |
| --- | --- | --- |
| CSA：Compressed Sparse Attention | 学习压缩相邻 token，使用较低压缩率 | indexer 选出的压缩条目 |
| HCA：Heavily Compressed Attention | 将更多 token 合成一个条目 | 对压缩后的历史做 attention，不采用 CSA 的 top-k 选择 |
| SWA：Sliding-Window Attention | 保留近期局部 KV | 最近窗口内的细节 |

公开配置常见 CSA ratio=4、HCA ratio=128、local window=128。CSA 使用重叠来源做学习压缩，不能理解为每四个 token 简单求平均；HCA 的更高比例压缩采用不同组织。压缩率固定时，HCA 对全部压缩历史的 prefill 仍可有二次项，只是常数降低。

Flash 有 43 层、hidden 4096、64 个主 query heads；Pro 有 61 层、hidden 7168、128 个主 query heads。两者主 head dimension 为 512、共享 KV head 数为 1。开头与尾部存在特定配置，不能用“全模型严格交替”替代真实 `compress_ratios` 列表。[V4 报告](https://arxiv.org/abs/2606.19348)、[Flash 配置][ds4f]、[Pro 配置][ds4p]

### V4.1 的 CED：改变 prompt 经过哪些层

V4.1-Flash 使用 40 层语言主干：20 层 causal encoder，随后 20 层 decoder。上半部分 global KV 可以从 encoder 输出投影得到，不必让每个 prompt token 完整经过全部 decoder 层，才取得 decoder 的历史 global KV。

```text
完整 prompt → 20 层 causal encoder → encoder hidden
                                      ├→ decoder 的 global KV
                                      └→ 后续生成所需表示

新生成 token → encoder + decoder 的计算路径 → logits
```

这里的 encoder 仍然是 causal 的，不是可以任意查看未来的双向 BERT encoder。decoder 的局部 SWA KV 仍与自身层的 hidden 相关，所以实际 prefill 还需要尾部 SWA 处理。论文的 Bounded Replay 只重放有限尾部，并接受其明确说明的近似，不能说“后 20 层对 prompt 完全不用做任何工作”。

CED 解释了官方将激活量区分为 prefill 8B、decode 16B 的原因。552B 是主干口径，另有 196B Engram；这两个数不能混成“总模型只有 552B”。[V4.1 架构报告](https://arxiv.org/abs/2609.19969)、[1017 配置][ds41]

### CSA2：跨层共享 KV 和跨层复用索引是两件事

CSA2 在序列压缩之外，又减少不同层的重复缓存与索引计算：

| 模式 | global KV / indexer K | top-k IDs |
| --- | --- | --- |
| Full | 按指定来源生成 | 本层重新计算 |
| Reindex | 复用前面的缓存 | 用本层 indexer query 重新选择 |
| Reuse | 复用前面的缓存 | 复用前面的选择结果 |

所有模式仍计算本层 query 和本层 SWA 数据，不能把 Reuse 理解为直接复用上一层 attention 输出。hierarchical indexer 还先选候选池，再让后续索引器在较小范围中筛选。

CSA2 同时涉及缓存条目大小、序列压缩、层间复用与 KV 量化。论文的“890 bytes/token”指指定部署口径下的 global KV footprint，不能当作整个请求的总显存公式。SWA、状态、临时量、权重和持久化缓存要另算。[V4.1 CSA2 说明](https://arxiv.org/html/2609.19969v1#S2.SS3)

## 6. Kimi：MLA 主干、Kimi Linear 与 K3 的混合

### K2 / K2.5：专家池扩大，多模态与文本骨架分别看

K2 使用 61 层、hidden 7168、64 个 MLA query heads，KV latent 为 512。相较 V3，主干类名或某些尺寸相近，不代表专家数、head 数和训练方法都相同。K2 有 384 个 routed experts，每 token 选择 8 个，另有 1 个 shared expert，首层使用 dense FFN。

K2.5 的文本主干保留这条 MLA + MoE 路线，视觉输入通过 MoonViT 和 projector 接入。视觉 encoder 的工作不属于语言模型的 MLA cache；图像转成多少 visual tokens，会影响后续 prefill 与缓存负担。[K2][k2-card]、[K2.5][k25-card]

### Kimi Linear：多数层用 KDA，少数层保留 MLA

48B-A3B 版本有 27 层：20 个 KDA、7 个 MLA。其配置显式列出 full-attention 层，而不是用一个完全整齐的 3:1 比例推断所有边界。

KDA 是 Kimi Delta Attention。沿用上一页 `[d_k,d_v]` 的矩阵状态、行向量 q/k/v：

$$
\bar S_t=\operatorname{diag}(a_t)S_{t-1},\qquad S_t=\bar S_t+\beta_t k_t^T(v_t-k_t\bar S_t).
$$

与 scalar-decay Gated DeltaNet 相比，$a_t$ 是沿 key 特征方向的向量，可对不同状态方向使用不同遗忘系数。KDA 写入的是预测误差；不能将它简化成普通 GLA 的“衰减后加外积”。完整推导见 [FLA 笔记](flash-linear-attention.md)。[Kimi Linear 报告](https://arxiv.org/abs/2510.26692)、[固定配置][kl]

### K3：时间、深度、特征宽度三条路径一起改变

K3 有 93 层，配置中是 **69 个 KDA + 24 个 gated MLA**。主体重复 3 个 KDA 接 1 个 MLA，最后还有额外 MLA 层。MLA 层使用 NoPE，并有输出 gate；KDA 的短卷积和递推仍带来顺序与时效信息。NoPE 不能解释成整个模型对顺序不敏感。

K3 的 KDA 还调整了 gate 参数化。单步 log decay 的下界限制了 chunk 内累计衰减的数值范围，让更多局部计算能使用矩阵指令。这里的数值设计与 kernel 效率有关，不能把 Kimi Linear 和 K3 的 KDA 实现视为完全相同。

其余两条路径是：

- **Block AttnRes**：在深度方向选择先前的表示，见第 9 节。
- **Stable LatentMoE**：routed experts 在宽度 3584 的 latent 空间工作，主干 hidden 为 7168；先聚合再 norm/up-project，与全宽 shared experts 合并。

K3 有 896 个 routed experts，每 token 选 16 个，另有 2 个 shared experts。它使用 SiTU-GLU 等稳定化设计，不能把所有专家实现都默认写成未经改动的 SwiGLU。[K3 模型卡][k3-card]、[K3 技术报告](https://arxiv.org/abs/2607.24653)

K3 同时有 recurrent state 和增长的 MLA cache。使用 KDA 不意味着全模型 KV cache 变成常量。公开配置的 `mla_use_nope`、`linear_attn_config` 与 `attn_res_block_size`，比模型名称更能说明实际结构。[K3 配置][k3]

## 7. Qwen：dense/MoE 与 attention 路线是独立选择

### Qwen3：先掌握常规 GQA 主干

Qwen3-32B 是 64 层、hidden 5120、FFN intermediate 25600，64 个 Q heads 共享 8 个 KV heads。Qwen3-235B-A22B 则有 94 层、hidden 4096、128 个专家选 8 个，Q/KV heads 为 64/4。

两者都使用 GQA，但 FFN、尺寸和权重存储差异很大。模型名字中的 A22B 表示公开的激活参数口径，不表示把 235B 权重全部丢掉后只需加载 22B。[32B 配置][q3d]、[235B 配置][q3m]

### Qwen3-Next / 3.5：GDN 与 gated attention 混合

这条路线通常每四层中三个采用 Gated DeltaNet，一个采用 gated full attention。full-attention 层仍使用 GQA；其 output gate 在 attention 输出上做调制，与 GDN 的状态遗忘 gate 不同。

以 Qwen3.5-27B 为例：

| 路径 | 配置 |
| --- | --- |
| 主干 | 64 层，hidden 5120，dense FFN intermediate 17408 |
| full attention | 16 层；Q/KV heads=24/4，head_dim=256 |
| GDN | 48 层；key heads=16，value heads=48，key/value dim=128/128 |

GDN 的 head 数和 state 形状不能从 full-attention 的 24/4 heads 推导。这里每组 key head 对应更多 value heads，状态容量要按实际 value/state heads 统计。

同家族的 35B-A3B 版本改用 MoE，只有 40 层，30 个 GDN + 10 个 full attention；不能按 27B 的层数计算它的缓存。Qwen3.6 的相应 checkpoints 继续使用 Qwen3.5 模型类，也说明版本名称升级不一定改变底层架构。[27B 配置][q35d]、[35B 配置][q35m]、[Transformers Qwen3.5/3.6 实现说明](https://huggingface.co/docs/transformers/v5.15.0/model_doc/qwen3_5)

### Qwen3.8 也分成不同结构分支

3.8-27B 的主要文本配置仍是 64 层、48 GDN + 16 full attention。3.8-2.4T-A95B 使用 92 层、hidden 8192、69 GDN + 23 full attention，专家池为 512、top-k 为 10。二者不能因为同属 3.8，就套用 Flash-Next 的新 QSA 与 GR 解释。[27B 配置][q38d]、[2.4T 配置][q38m]

Flash-Next 有 48 层、hidden 2560，结构为：

```text
12 × [GDN → MoE → GDN → MoE → GDN → MoE → QSA → MoE]
每个子层通过 Gated Residual 读写扩展后的残差状态
另有 n-gram embedding 与 MTP 模块
```

官方将参数拆为 125B 主干、6B 激活，另有 51B n-gram embedding 和 4B MTP。把它统称为“125B 全部参数”，会遗漏部署时需要处理的表与辅助模块。[Flash-Next 模型卡][q38n-card]

### QSA：压缩 indexer，选择原始 token micro-block

QSA 的 lightweight indexer 将相邻 4 个 token 的索引 key 汇成 micro-block 表示。query 给这些块打分，选择块后展开为 token IDs，让主 attention 读取原始 token 的 K/V。

该配置的主 Q/KV heads 为 24/2、head_dim=256；indexer 为 4 个 Q heads 和 1 个 K head、head_dim=128，预算 512 个 blocks，即 2048 tokens。尾部未满块还需按 causal 规则处理。

**QSA 压缩的是用于选择的索引表示；CSA 的主 attention 直接读取压缩 KV 条目。** 两者不是一个简单的改名。压缩 indexer 可以减少打分成本，但也不能据此声称整个 prefill 已无二次项。[Flash-Next 配置][q38n]、[QSA 论文说明](https://arxiv.org/html/2608.30320v1#S2.SS1.SSS2)

配置中的 `layer_types="full_attention"` 可能保留了槽位类别名称，而该槽位内已加入 QSA indexer。因此需要结合 `indexer_*` 字段、论文与实际 dispatch 阅读，不能只统计字符串就断言每层都算 dense attention。

## 8. MoE、共享专家与 LatentMoE 的完整计算流程

### 每个 token 选择一部分 FFN

一种抽象表达是：

$$
y=\sum_{e\in\mathcal E_{\mathrm{shared}}}E_e(x)+\sum_{e\in\operatorname{TopK}(r(x))}w_e(x)E_e(x).
$$

具体模型会改变 router score、归一化、缩放和共享分支。不能从这条示意式推断所有模型采用同一种 softmax router。

```text
输入 token hidden
  → router 选择 expert IDs 与权重
  → 按专家分组 / dispatch
  → 每个专家对收到的 token 执行 FFN
  → 按 router 权重 combine
  → 合并 shared expert 分支
```

在 Expert Parallel 中，dispatch/combine 可能跨 GPU，涉及 AllToAll 或专门的点对点传输。不同专家收到的 token 数不同，形成 grouped GEMM 和负载均衡问题。小 batch 时每个专家的矩阵可能很窄，活跃参数少不保证 Tensor Core 利用率高。

### total、active 与显存分别怎么算

总参数量决定权重容量的主要部分；active 参数量描述某个 token 路径使用的参数。一次 batch 的不同 token 可以路由到不同专家，合起来访问更多权重。实际显存还包含量化 scale、KV/state、临时 buffer 与复制副本。

“模型有 896 个专家、选 16 个”也不能直接算整个模型 FLOPs 为 `16/896`：attention、router、shared experts、投影和词表计算都还在。

### LatentMoE 压缩的是 FFN 的特征通道

K3 将 routed 分支先从 hidden 7168 投影到 latent 3584，再在 latent 空间执行专家、聚合、归一化和升维。shared 分支保留全宽。这可以减少 routed 分支的数据与参数成本，但增加相应投影及融合需求。

它和 MLA 都使用 latent 一词，压缩对象却不同：MLA 面向 attention 的历史 KV 表示；LatentMoE 面向 routed FFN 的输入输出宽度。[K3 Stable LatentMoE](https://arxiv.org/html/2607.24653v1#S2.SS3)

## 9. 深度方向的三类变化：mHC、AttnRes 与 GR

### mHC：扩展残差状态并约束混合

普通残差保存一个 `[D]` 向量，执行 `x + F(x)`。Hyper-Connections 将残差扩成多个分支，先从分支中读出子层输入，再将子层结果写回，并混合原残差分支。

mHC 对残差混合施加约束，使混合矩阵非负，并使每行与每列的和接近 1，以控制深层传播中的放大。它改变的是跨层信息流，不是从 token 历史里选 top-k。Single-Pass mHC 再调整执行组织以便融合，不代表 attention 只需做一遍乘法。[DeepSeek-V4 mHC](https://arxiv.org/html/2606.19348v1#S2.SS2)

### AttnRes：对先前层或块的表示做 attention

普通残差对先前结果累加；AttnRes 让当前层为同一 token 的不同深度表示分配权重。其 attention 的候选轴是**层/块**，不是历史 token。

K3 使用 Block AttnRes：块内保留累积表示，块间选择，降低保存所有中间层输出的成本。配置块大小为 12，93 层的最后一个块不满；这与时间序列上的 chunk size 不是同一个参数。

### Gated Residual：按特征读取、按分支写回

Qwen3.8-Flash-Next 扩展到 4 个 residual branches。数据依赖的逐元素 read gate 决定子层读取什么，分支级 write gate 决定结果如何写回。它没有简单照搬 mHC 的完整分支混合算子，也不等于 AttnRes 在先前层输出之间做 softmax。[AttnRes](https://arxiv.org/html/2607.24653v1#S2.SS2)、[Gated Residual](https://arxiv.org/html/2608.30320v1#S2.SS2)

这些结构会改变 activation 大小、算子融合、pipeline stage 边界和训练反向保存策略。不能只统计 attention kernel 与 FFN kernel，就忽略残差路径的成本。

## 10. 其他注意力与位置编码话题怎样放进来

| 机制 | 选择或改变的对象 | 需要避免的误解 |
| --- | --- | --- |
| causal mask | 哪些时间位置可见 | 与 dense/MoE 无直接等价关系 |
| SWA | 最近窗口内的 KV | 局部层仍可通过多层间接传播更远信息 |
| NSA | 压缩、选择和局部窗口三个分支 | 不能看到 DeepSeek 的 NSA 论文，就认定所有 DeepSeek 发布模型都采用同一 NSA 实现 |
| MoBA | 对历史块路由，选择部分 blocks 做 attention | “Mixture”指选 KV blocks，不是 FFN experts |
| DSA | indexer 选 token | 与 MLA 可组合，索引成本仍需统计 |
| CSA/HCA | 压缩主 KV 的序列条目 | 缓存仍可能随长度增长 |
| QSA | micro-block indexer 选择 token 范围 | 与压缩主 KV 的 CSA 不同 |
| GDN/KDA | 更新固定维度的历史状态 | 混合模型仍可能有增长的 full-attention cache |
| attention sink | 特殊的注意力/归一化设计或现象 | 需分清学习的 sink 参数、保留初始 tokens 与观测到的 sink 行为 |

NSA 是 DeepSeek 的独立研究路线，MoBA 是 Moonshot 的 block-routing 研究。只有论文、配置或源码明确建立关系时，才将它们写成某个 checkpoint 的组成。[NSA](https://arxiv.org/abs/2502.11089)、[MoBA](https://arxiv.org/abs/2502.13189)

### NSA 与 MoBA 各自怎样读取历史

NSA 同时组织三类信息：压缩块提供粗粒度历史，selection 分支读取选中的细粒度 token blocks，sliding window 保留邻近细节。各分支有自己的 attention 输出，再通过门控结合。它的稀疏布局与训练、kernel 组织一起设计，不等于在任意 dense 模型推理时临时删除一些 KV。

MoBA 将 K/V 按序列位置分块，以块级 key 摘要为路由依据，为 query 选择部分块，再在选中的 token 上计算 attention。当前所在块和 causal 边界需要特殊处理。不同块的局部输出合并时，仍须保持正确的 softmax 归一化，不能直接把各块独立 softmax 的输出相加。

两者都需要考虑选择粒度与硬件效率。选单个 token 更灵活，但可能导致离散访存；选连续块更便于合并访问和矩阵 tiling，但也可能多读无关 token。块大小、索引器和选择预算共同决定质量与成本。

### 同样叫 gate，作用位置可能完全不同

| gate | 作用对象 | 例子 |
| --- | --- | --- |
| recurrent forget gate | 旧的矩阵状态 | GDN 的 scalar decay、KDA 的 channel-wise decay |
| attention output gate | 本次 attention 读出的特征 | Qwen gated attention、K3 gated MLA |
| expert router | FFN 分支的选择与加权 | routed MoE |
| residual read/write gate | 跨层残差分支 | Gated Residual |

以列向量写法为例，output gate 可抽象成 $y=W_O\bigl(g(x)\odot\operatorname{Attn}(Q,K,V)\bigr)$。它没有改变“历史 K/V 仍需保存”的事实，也不等于在状态写入前乘遗忘系数。

RoPE 改变 query/key 的位置相关表示。YaRN 调整长上下文位置编码方案，NoPE 表示相关 attention 分支不显式加入位置编码。它们与 MHA/GQA/MLA 的 head 或 latent 组织不是同一维度。多模态模型还可能对时间、高度、宽度使用不同位置轴，需要查看具体 position IDs 与 rotary 配置。

更长的上下文支持范围，不保证短上下文吞吐或长距离检索质量不变。配置上限、训练长度、验证过的质量和部署可容纳长度应分别记录。

## 11. KV cache、recurrent state 与参数记忆分别算账

### 四种缓存压缩可以组合

| 压缩方向 | 例子 | 减少哪部分 |
| --- | --- | --- |
| 每个条目的宽度 | GQA、MLA | KV heads 或 latent 维度 |
| 条目数量 | CSA/HCA | 压缩后的历史位置数 |
| 层间重复 | CSA2 的 KV reuse | 独立保存 global KV 的层数 |
| 每个数的字节数 | FP8/FP4 KV | payload 精度，另有 scale 与对齐成本 |

稀疏选择通常首先减少“每次读多少条目”，不必然减少“总共存多少条目”。recurrent 层则将历史合并为状态，属于另一类表示。

### 一个混合模型的具体尺寸

按 Qwen3.5-27B 配置，假设 full-attention KV 使用 BF16、GDN state 使用 FP32，且暂不计 TP、卷积状态和 allocator 对齐：

$$
M_{\mathrm{full\ KV/token}}=16\times2\times4\times256\times2=65536\ \text{bytes}=64\ \text{KiB}.
$$

GDN 状态按 48 个线性层、每层 48 个 value/state heads 计算：

$$
M_{\mathrm{GDN\ state}}=48\times48\times128\times128\times4=144\ \text{MiB/request}.
$$

因此仅这两项，在 8192-token 历史下约为 `512 MiB + 144 MiB`。这是配置与精度假设下的逻辑推导，不是框架实测。它说明短历史时固定状态也可能占据显著空间，不能说“线性层缓存是 O(1)，所以可以忽略”。

### Engram / n-gram embedding 是参数，不是请求历史缓存

DeepSeek-V4.1 的 Engram、Qwen Flash-Next 的 n-gram tables，按照短 token 组合查表，提供训练得到的条件记忆。它们属于模型参数，需要存储与带宽。它们与请求历史 KV cache、外部文档 RAG 的作用不同。

这类参数可以采用 host-memory 预取或其他分层存储策略，减少常驻 accelerator 容量压力，但不会消除传输和等待。原论文的参数规模与低延迟结论，不能直接变成本地机器的性能保证。

### prefix caching 与 speculative rollback

混合缓存至少要区分 append-only KV、固定 recurrent state、局部窗口和卷积状态。prefix 命中时应恢复对应层和位置的状态；分支生成要独立维护可写状态。验证拒绝 token 时，需要恢复 recurrent/卷积状态，不能只缩短 KV 长度。

跨层共享 KV 还需要知道缓存生产者、消费者和生命周期。复用 top-k IDs 需要确认它们对应同一组 KV 条目。图捕获只复用执行安排，不会自动替系统维护这些语义。

## 12. 多模态、MTP 与训练方法属于哪些层次

### 多模态输入路径

一种常见路径是：

```text
图像/视频 → patch 或视觉特征 → vision encoder → projector/merger
                                                    ↓
文本 → tokenizer → embedding ──────────────→ 混合 token 序列 → 主干
```

Kimi-K2.5 的 MoonViT、K3 的 MoonViT-V2、DeepSeek-V4.1 的 DeepSeek-ViT，以及 Qwen 相关视觉分支，不能仅因最后都进入语言 hidden 就当成同一个 encoder。视觉 token 压缩、位置和注入方式需要读对应配置。

有些大型发布包只提供语言主干或单独发布视觉部分。模型品牌、论文的原生多模态能力和某个 checkpoint 实际包含的模块，必须分别确认；查看 `text_config`、`vision_config` 与权重清单。

### MTP 与 speculative decoding

MTP 是 Multi-Token Prediction，给模型增加预测后续位置的训练目标和辅助模块。推理时可以利用合适的辅助模块做 draft，再由 target 验证；是否启用、接受率和收益取决于具体 serving 实现。

DeepSeek-V4.1 报告将 DSpark 与主干预训练分开：它省去主干预训练阶段的 MTP 模块，再单独训练草稿。公开配置中的 `num_nextn_predict_layers` 等字段需要结合实际模块含义阅读，不能将全部辅助层计入 40 层主干。

K3 报告的训练 MTP 描述与发布主干配置的 `num_nextn_predict_layers=0` 也属于不同观察对象。不能仅据其中一个字段断言论文没用 MTP，或当前包能直接启用草稿。

### reasoning 与 optimizer 不改变名词层次

Thinking、Instruct、RL、distillation 主要描述训练或生成行为，可能共享主干。Muon 是 optimizer，不是 attention 变体。FP8/MXFP4/QAT 描述数值表示与训练部署策略，也不等于全部层、缓存和计算都使用同一种精度。

例如 K3 模型卡描述 MXFP4/MXFP8 的训练部署路线，但配置量化规则对部分模块设有例外。因此不能简单用“总参数×0.5 byte”当作完整权重显存。[K3 配置][k3]、[Qwen Flash-Next 设计报告](https://arxiv.org/abs/2608.30320)

## 13. 从架构追到 kernel 与系统成本

| 架构变化 | 增减的工作 | 应重点检查 |
| --- | --- | --- |
| MLA | latent 投影、吸收后的 attention 路径 | prefill/decode 是否采用不同 kernel，KV 格式与 TP 复制 |
| DSA/QSA | indexer、top-k、稀疏 gather、主 attention | 索引器开销、选择粒度、访存连续性与动态 shapes |
| CSA/HCA | compressor、压缩缓存与局部窗口 | 压缩条目生成时机、尾部、窗口与 position 对齐 |
| CSA2/CED | 跨层缓存依赖、重索引、尾部 SWA 处理 | producer/consumer、prefix 命中及近似 replay 边界 |
| GDN/KDA | state update、chunk 算法、短卷积 | state 带宽、寄存器压力、batch/head 并行度与回滚 |
| MoE | routing、dispatch、grouped GEMM、combine | 专家 token 分布、EP 通信、空/小矩阵与重叠 |
| mHC/AttnRes/GR | 残差读写、归一化、深度归约 | activation 体积、融合与 PP 边界 |
| 条件记忆 | n-gram 查表、参数预取 | host/GPU 传输、命中与预取时机 |

同样叫“3B active”的模型，也可能在上下文读取、state 容量、router 与专家宽度上有不同成本。比较延迟或吞吐时，需要固定模型版本、精度、batch、上下文长度、TP/EP/PP 配置与缓存命中状态。结构分析不能替代测量。

相关基础页：[Transformer 运行全过程](transformer-runtime.md)、[GEMM](../CUDA/gemm.md)、[FlashAttention](../CUDA/flash-attention.md)、[FLA](flash-linear-attention.md)、[通信操作](../GPU/communication-and-allreduce.md)、[CUDA Graph](../GPU/graphs-and-compilation.md)、[精度与量化](../GPU/precision-and-quantization.md)。

## 14. 固定配置来源与阅读方法

阅读时先看 `model_type`、具体类与文本/视觉分支。再看 `layer_types` 或特殊层列表，核对 heads、state/latent dimensions、专家数和辅助层。不能统一用 `hidden_size / num_attention_heads` 推导所有模型的 head_dim。

本页比较的层数均指语言主干。部分列表还包含 MTP/draft 的条目；例如 `compress_ratios` 的长度可能超过 `num_hidden_layers`。Kimi 的 `num_experts_per_token` 与其他实现的 `num_experts_per_tok` 也不能漏读。

以下链接固定到核对时的 revision。配置证明的是公开结构字段；论文解释设计目标，实际源码决定执行，profiler 才能说明选中了哪个 kernel。本页未下载模型权重，未执行模型或性能测试。

仓库同时保存 `assets/model-architecture-configs.json` 的公开配置摘录。它记录核对日期、revision、配置来源与关键字段，便于重新计算层数、head 数和缓存尺寸。

| Checkpoint | 主干层数 / hidden | 配置与模型卡（固定版本） |
| --- | --- | --- |
| deepseek-ai/DeepSeek-V3 | 61 / 7168 | [config][ds3] · [model card][ds3-card] · `e815299b` |
| deepseek-ai/DeepSeek-V3.2 | 61 / 7168 | [config][ds32] · [model card][ds32-card] · `a7e62ac0` |
| deepseek-ai/DeepSeek-V4-Flash | 43 / 4096 | [config][ds4f] · [model card][ds4f-card] · `60d8d707` |
| deepseek-ai/DeepSeek-V4-Pro | 61 / 7168 | [config][ds4p] · [model card][ds4p-card] · `b5968e91` |
| deepseek-ai/DeepSeek-V4.1-Flash-1017 | 40 / 5120 | [config][ds41] · [model card][ds41-card] · `1bae0e10` |
| moonshotai/Kimi-K2-Instruct | 61 / 7168 | [config][k2] · [model card][k2-card] · `fd1984e2` |
| moonshotai/Kimi-K2.5 | 61 / 7168 | [config][k25] · [model card][k25-card] · `4d01dfe0` |
| moonshotai/Kimi-K2.6 | 61 / 7168 | [config][k26] · [model card][k26-card] · `7eb5002f` |
| moonshotai/Kimi-K3 | 93 / 7168 | [config][k3] · [model card][k3-card] · `f831ab66` |
| moonshotai/Kimi-Linear-48B-A3B-Instruct | 27 / 2304 | [config][kl] · [model card][kl-card] · `e1df551a` |
| Qwen/Qwen3-32B | 64 / 5120 | [config][q3d] · [model card][q3d-card] · `9216db57` |
| Qwen/Qwen3-235B-A22B | 94 / 4096 | [config][q3m] · [model card][q3m-card] · `8efa6172` |
| Qwen/Qwen3-Next-80B-A3B-Instruct | 48 / 2048 | [config][qnext] · [model card][qnext-card] · `9c7f2fbe` |
| Qwen/Qwen3.5-27B | 64 / 5120 | [config][q35d] · [model card][q35d-card] · `fc05daec` |
| Qwen/Qwen3.5-35B-A3B | 40 / 2048 | [config][q35m] · [model card][q35m-card] · `59d61f3c` |
| Qwen/Qwen3.8-27B | 64 / 5120 | [config][q38d] · [model card][q38d-card] · `1d4bf0f2` |
| Qwen/Qwen3.8-2.4T-A95B | 92 / 8192 | [config][q38m] · [model card][q38m-card] · `207bd685` |
| Qwen/Qwen3.8-Flash-Next | 48 / 2560 | [config][q38n] · [model card][q38n-card] · `de4b8e4d` |

[ds3]: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json
[ds3-card]: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/README.md
[ds32]: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/config.json
[ds32-card]: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/README.md
[ds4f]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json
[ds4f-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/README.md
[ds4p]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json
[ds4p-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/README.md
[ds41]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/config.json
[ds41-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md
[k2]: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/config.json
[k2-card]: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/fd1984e2b7a3350dbf7305fe73a4ede25c14de50/README.md
[k25]: https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/config.json
[k25-card]: https://huggingface.co/moonshotai/Kimi-K2.5/blob/4d01dfe0332d63057c186e0b262165819efb6611/README.md
[k26]: https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/config.json
[k26-card]: https://huggingface.co/moonshotai/Kimi-K2.6/blob/7eb5002f6aadc958aed6a9177b7ed26bb94011bb/README.md
[k3]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json
[k3-card]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/README.md
[kl]: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/config.json
[kl-card]: https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/e1df551a447157d4658b573f9a695d57658590e9/README.md
[q3d]: https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/config.json
[q3d-card]: https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/README.md
[q3m]: https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json
[q3m-card]: https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/README.md
[qnext]: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7/config.json
[qnext-card]: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/9c7f2fbe84465e40164a94cc16cd30b6999b0cc7/README.md
[q35d]: https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json
[q35d-card]: https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/README.md
[q35m]: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json
[q35m-card]: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md
[q38d]: https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json
[q38d-card]: https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md
[q38m]: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/config.json
[q38m-card]: https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B/blob/207bd685a7e3696cfaff12ded7c6a7ea0f88c996/README.md
[q38n]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json
[q38n-card]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md
