---
title: DeepSeek、Kimi 与 Qwen 架构
category: attention
tags: [MLA, GQA, DSA, CSA, MoE]
description: MLA / GQA · DSA / CSA / QSA · KDA / GDN · MoE · 残差与缓存
updated: 2026-11-04
---
# DeepSeek、Kimi、Qwen：模型结构与注意力变体

快速查找“哪个模型用了什么技术”：见[模型技术总表](model-technology-map.md)。

DeepSeek 的逐代推导已独立整理为[架构演进](../DeepSeek/evolution.md)和[一次前向与张量形状](../DeepSeek/forward-shapes.md)。本页继续保留不同模型家族的横向对照。

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

### 用小尺寸看清联合压缩与缓存对象

先忽略位置编码与 norm，取 hidden dimension D=8、两个 heads、每 head 的 K/V dimension 为 4、latent dimension r=2。沿用本节的行向量约定：

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| 当前层的 token hidden $x_j$ | `[1,8]` | 这个 token 进入当前 attention 的特征 |
| $W^{DKV}$ | `[8,2]` | 所有 heads 共用的下投影权重 |
| $c_j=x_jW^{DKV}$ | `[1,2]` | 每层、每 token 各一份的 latent |
| $W^{UK}_h$、$W^{UV}_h$ | 各 `[2,4]` | head h 的内容 K/V 上投影权重 |
| $k_{j,h}$、$v_{j,h}$ | 各 `[1,4]` | 这个 token 在该 head 下的逻辑 K/V |

```text
                         ┌→ UK,0 → head 0 的 K
                         ├→ UV,0 → head 0 的 V
token hidden → latent c ─┤
                         ├→ UK,1 → head 1 的 K
                         └→ UV,1 → head 1 的 V
```

展开后的两个 heads 共含 16 个 K/V 元素，但这个教学结构只需缓存 2 个 latent 元素。上投影矩阵是共享模型权重，不随每个 token 复制存储。“联合”表示多个 heads 的 K 和 V 都从同一份 latent 生成，并非所有 heads 得到相同 K/V。

这同时限制了表示空间：16 个展开值由 2 个 latent 值决定。它是一种训练时采用的参数化结构，不是对任意现成 MHA 的 16 个数提供无损压缩。实际 MLA 还会处理 norm 和位置分支，不能用这个小例子直接估算真实模型压缩比。

### K 侧：变换当前 query，不逐条展开历史 key

固定一个 head，令 $\bar q_h=q_h(W_h^{UK})^T$。则 $\bar q_h$ 的形状为 `[1,r]`，内容分数就是 $\bar q_hc_j^T$。每个 head 使用自己的上投影矩阵变换自己的 q，批量实现时也要保持这一对应关系。

用数值检查这条恒等式：

$$
c=[2,1],\qquad W^{UK}=\begin{bmatrix}1&0&1&2\\0&1&1&-1\end{bmatrix},\qquad q=[1,2,0,1].
$$

先展开时，$k=cW^{UK}=[2,1,3,3]$，$qk^T=7$。先变换 query 时，$\bar q=q(W^{UK})^T=[3,1]$，$\bar qc^T=7$。两个路径得到相同内容分数，而后者不用生成 k。

多头 query 的功能仍然存在。简化地写 $q_h=xW^Q_h$，可将两个线性映射合并为 $\bar W^Q_h=W^Q_h(W_h^{UK})^T$，直接生成 latent-space query。DeepSeek-V2 的 Q 路径还使用单独的 query 降维、归一化和上投影。query latent 与缓存用的 KV latent 是不同对象，query 通常不需长期缓存。权重吸收只能合并相邻线性映射，不能随意跨过 norm 等非线性步骤。[MLA 的 Q/KV 压缩][mla-report]

### V 侧：先加权 latent，再做输出映射

对 head h，先计算 $z_h=\sum_jp_{h,j}c_j$，它是 `[1,r]` 的加权 latent。然后 $o_h=z_hW_h^{UV}$，恢复为该 head 的 value 输出维度。

将最终输出投影按 head 拆成 $W^O_h[d_v,D]$，有：

$$
y=\sum_h o_hW^O_h=\sum_hz_hW_h^{UV}W^O_h=\sum_hz_h\bar W^O_h,\qquad \bar W^O_h=W_h^{UV}W^O_h.
$$

V 侧的吸收是将上投影与输出投影合并。它不是先将 softmax 权重乘 V 投影，再与 latent 点乘。概率的轴是历史位置 T，上投影的轴是 latent/value 特征，两者不能直接按同一维度相乘。

```text
所有当前 query 按 head 排成 Qbar：[H,r]
所有历史 latent 按 token 排成 C：[T,r]

Scores = Qbar @ C.T → [H,T]
P = softmax(Scores, 沿 T) → [H,T]
Z = P @ C → [H,r]
各 head 的 Z 经输出映射并合并 → 当前 token 的 attention 输出 [1,D]
```

各 head 的 query 不同，所以概率分布和 Z 通常不同。加权后的 Z 是当前 query 的临时结果，不会替换历史 C。下一个 token 仍可用新的权重重新查阅每条历史 latent。

上述流程省略缩放、mask 和 RoPE，表示内容分支的代数关系。合并后的输出权重可能很大，实际 kernel 可以保留分解形式，或在部分阶段临时展开 K/V，以匹配高效 GEMM。数学上可吸收，不等于每个后端都物化同一张合并权重。将结果映射回 hidden 也不应一概称为“降维”，应以实际输入输出尺寸判断。

### RoPE 为什么需要单独缓存位置 key

若对展开 K 的全部内容做位置相关旋转，旋转矩阵会随历史位置 j 改变。它进入上投影与点积之间，妨碍用一次统一的 query 变换处理所有历史。

decoupled RoPE 将内容和位置分数分开。当前 token t、head h 对历史位置 j 的分数可写为：

$$
s_{t,j,h}=\frac{\bar q^C_{t,h}c_j^T+q^R_{t,h}(k^R_j)^T}{\sqrt{d_C+d_R}}.
$$

其中位置 query/key 已经过相应 RoPE，$\bar q^C$ 是吸收 K 上投影后的内容 query。随后应用 mask、沿历史位置做 softmax。真正缓存的是 $c_j$ 与 $k_j^R$；value 路径仍可使用前面的线性重排。[Decoupled RoPE][mla-report]

### 一个新 token 到来时，各层分别做什么

假设已经处理 token 1～3，现在计算 token 4。在第 ℓ 层，历史 cache 已有该层的 $c_1^{(\ell)},c_2^{(\ell)},c_3^{(\ell)}$ 和对应位置 keys。

1. 从当前 token 的该层输入生成 query、新 latent 与位置 key。实际投影路径含所需 norm。
2. 将 $c_4^{(\ell)}$ 与位置 key 追加到该层 cache。causal attention 允许访问当前位置自己。
3. 各 head 的当前 query 与四条历史/当前缓存计算内容和位置分数。
4. softmax 后，对四条 latent 加权求和，各 head 得到自己的 z。
5. 经 value/output 映射得到当前 token 的 attention 输出，再处理残差、norm 与 FFN。
6. 将当前 token 的新 hidden 交给第 ℓ+1 层，使用那一层自己的权重和 cache 重复。

历史 latent 不需要重新从旧 hidden 计算。所有层完成后，当前 token 的最后一层 hidden 经 LM head 产生 logits，再选出下一 token。每层的 latent 是独立缓存，不是全模型共用一份压缩状态。

### FFN 为什么不需要重新读取完整序列 hidden

FFN 需要一个 token 的完整特征向量，却不依赖其他 token 的 hidden 行。矩阵化执行只是在批量计算这些独立行：

$$
XW=\begin{bmatrix}x_1W\\x_2W\\x_3W\end{bmatrix},\qquad U[t,j]=\sum_dX[t,d]W[d,j].
$$

求和轴是 hidden 特征 d，token 位置 t 固定。SwiGLU 的 SiLU、逐元素乘法和 down projection 同样不跨 token 行。因此：

$$
\operatorname{FFN}\left(\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}\right)=\begin{bmatrix}\operatorname{FFN}(x_1)\\\operatorname{FFN}(x_2)\\\operatorname{FFN}(x_3)\end{bmatrix}.
$$

prefill 批量处理全序列，是因为这些位置都还需计算，而且 GEMM 能复用权重。普通单 token decode 只处理新位置的 `[1,D]`；旧位置的 FFN 已在之前完成。

历史信息通过 attention 进入当前 token 的表示。以常见 pre-norm block 为例，令当前 attention 分支输出为 a：

$$
u_t^{(\ell)}=x_t^{(\ell)}+a_t^{(\ell)},\qquad x_t^{(\ell+1)}=u_t^{(\ell)}+\operatorname{FFN}_{\ell}(\operatorname{Norm}(u_t^{(\ell)})).
$$

当前 hidden 还用于 query、latent/位置投影和残差，并非只在 FFN 中有用。旧 token 的 FFN 输出已经形成下一层 hidden，并进一步形成下一层的 cache。causal 推理中新 token 不改变过去位置的表示，所以不用重跑旧 token 的 FFN。这与双向 attention 或训练 backward 的状态需求不同。

### SGLang / vLLM 实际怎样保留 hidden

常规文本生成路径不会把“所有历史 token × 所有层”的完整 hidden 持续积累为 cache。hidden/residual 是当前 forward 的工作数据，最终 hidden 还用于 logits。长期供未来 attention 读取的是 KV，或 MLA 的 latent 与位置 key。

2026-10-29 对公开源码的静态核对：

| 实现 | 可核对的行为 |
| --- | --- |
| SGLang DeepSeek 模型 | 逐层更新 `hidden_states`；默认 `capture_aux_hidden_states=False`，捕获层列表为空 |
| SGLang MLA pool | 常规 `kv_cache_dim = kv_lora_rank + qk_rope_head_dim`，写入 latent/位置分支 |
| vLLM DeepSeek 模型 | 逐层更新 `hidden_states, residual`；默认辅助 hidden 层列表为空 |
| vLLM MLA wrapper | 从当前 hidden 生成 `kv_c_normed` 与 `k_pe`，交给 attention |

源码入口：[SGLang 模型][sg-model]、[SGLang MLA cache][sg-cache]、[vLLM 模型][vl-model]、[vLLM MLA wrapper][vl-mla]。这是指定版本的代码路径核对，不是对某个已启动服务的参数或显存作了运行时测量。

allocator 或 CUDA Graph 可能继续保留一块 buffer，之后复用或覆盖。旧数值尚未从物理显存消失，不表示程序把它们当作有效历史 hidden 保存。若启用 EAGLE/EAGLE3 辅助特征等功能，则可能显式保留指定层、指定 token 的 hidden。具体有效期按相应功能确定，不能将这类状态概括成默认全量历史缓存。

[mla-report]: https://arxiv.org/html/2405.04434v5#S2.SS1
[sg-model]: https://github.com/sgl-project/sglang/blob/a99f1052946150614000cf8727c4e1a325a00591/python/sglang/srt/models/deepseek_v2.py
[sg-cache]: https://github.com/sgl-project/sglang/blob/a99f1052946150614000cf8727c4e1a325a00591/python/sglang/srt/mem_cache/memory_pool.py
[vl-model]: https://github.com/vllm-project/vllm/blob/d6fbd947824959eabc6165751f8b9cd4aa83f546/vllm/model_executor/models/deepseek_v2.py
[vl-mla]: https://github.com/vllm-project/vllm/blob/d6fbd947824959eabc6165751f8b9cd4aa83f546/vllm/model_executor/layers/mla.py

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

**V4.1-Flash 把40层分成前20层 encoder 和后20层 decoder。前半段为历史 token 产生表示，后半段继续加工当前 token，并读取前半段建立的历史资料。** 两段都有 attention、FFN 和残差连接。下面的层号从1开始。

#### 从一个输入走到下一词预测

文本只在入口查一次 embedding。之后传递的一直是 hidden：

```text
token IDs → embedding
                ↓
       第1～20层 encoder
                ↓
       encoder 输出 hidden
                ↓
       第21～40层 decoder
                ↓
             LM head
                ↓
       下一个 token 的概率
```

两段都保持因果关系：位置 j 的计算不能读取未来位置。Decoder 的第一层接收 encoder 最后一层的输出，并不重新查一次 embedding。

这里有两种历史缓存，先把用途分开：

- **Global KV**：保存可供长程读取的历史资料。当前 query 通过 indexer 选择其中一部分。
- **SWA KV**：每层自己的近期记录，窗口最多128个 token。窗口内的位置直接参与该层的局部 attention。

**V4.1 的重要变化是：后20层共用一份由第20层输出生成的 global KV。各层的 query 和 SWA KV 仍从各自的 hidden 生成。**

例如第25层会这样计算：

```text
当前 token 进入第25层的 hidden
           ├→ 第25层自己的 query
           └→ 第25层自己的 SWA KV

历史 token 的第20层输出
           ↓ 指定投影
       共享 global KV
           ↓ 第25层按自己的query读取
       attention 输出，再经过FFN更新hidden
```

第26层可以读取同一份 global KV，但会生成不同的 query，因此仍然能得到不同结果。这就是 CED 在这份模型里的分工。[CED §2.2][ds41-report]

#### Encoder 也需要自己的历史缓存

Encoder 需要读取历史，不能只有 decoder 有缓存。它的实际安排是：

| 层范围 | Global KV | SWA KV |
| --- | --- | --- |
| 1～2 | 没有 global 分支 | 每层自己保存 |
| 3～8 | 第3层生成一份，供本组6层使用 | 每层自己保存 |
| 9～14 | 第9层生成另一份，供本组6层使用 | 每层自己保存 |
| 15～20 | 第15层生成另一份，供本组6层使用 | 每层自己保存 |

这三份 global cache 供 encoder 自己计算使用。它们把两个相邻 token 的信息合成一个512维条目。

Decoder 使用另外一份缓存。第21层的来源模块把第20层输出投影成512维条目，每个 token 一条。第21～40层共享这些条目。

```text
Encoder 自己用：3份压缩后的global cache + 每层自己的SWA cache
Decoder 自己用：1份从encoder输出生成的global cache + 每层自己的SWA cache
```

这些数量描述单个请求的逻辑缓存。跨GPU是否复制或分片，要另外看并行实现。SWA cache是当前推理所需的局部状态，不表示所有状态都永久存入SSD或prefix cache。

### CSA2：跨层共享 KV 和跨层复用索引是两件事

即使共享同一份 global KV，不同层仍然可以重新选择要读的位置。CSA2 把层分成三种角色：

| 角色 | 这层做什么 |
| --- | --- |
| **Full** | 负责生成本组global KV和indexer keys，并选出top-k位置 |
| **Reindex** | 使用已有global KV，用自己的indexer query重新选位置 |
| **Reuse** | 使用已有global KV，也沿用前面一层选出的位置 |

Full 指这些组件都执行，不是“计算全历史 full attention”。三种角色都要生成自己的主 query 和局部 SWA KV，并计算本层 attention 输出。

具体安排如下：

| 层范围 | 谁负责生成缓存和选择位置 |
| --- | --- |
| 3～8、9～14、15～20 | 每组第一层Full，后5层Reuse |
| 21～24 | 第21层Full，后3层Reuse |
| 25～28、29～32、33～36、37～40 | 每组第一层Reindex，后3层Reuse；都用第21层的global缓存 |

可以跟着当前 token 走一遍：第21层选出一组历史位置，22～24层用同一组位置，但各自重新算attention权重。第25层用自己的indexer重新选择，26～28层再使用这次选择。下一轮来了新token，负责索引的层会重新选，不会永远使用上一轮的位置。

#### 分级 indexer 怎样少做重复扫描

第21层先扫描所有可见global条目，选出自己的top-512。它还会建立一个更大的候选池。先把每8个位置划为一组，用组内最高分给组排序，再选最多2048组。展开这些组后，最多得到16384个候选位置。

第25、29、33、37层只需在候选池中重新选择512条。**16384是搜索范围，512是最终主attention读取的global条目数量。** 第21层的全历史扫描仍然存在。

CSA2 还简化了索引key的生成：先得到512维主KV条目，再把它投影成128维indexer key。Encoder里的两token压缩也不采用V4 CSA的相邻组重叠。[CSA2 §2.3][ds41-report]

### SWA Bounded Replay：只重算一小段，补回局部缓存

**它解决的是：global KV还能复用，但继续计算需要的局部SWA KV缺了。做法是只运行一小段末尾输入，重新算出这些层的局部KV，再接着处理。** V4.1 把这段长度限制为最多128个token，所以叫 bounded；重新执行这段输入的前向计算，就是 replay。

这里不是重新生成这几个token的文字，也不是只把token IDs复制进缓存。局部KV来自各层hidden，因此要让这些已知token重新经过所需的attention、FFN和残差计算，才能得到各层的局部KV。

#### 情景一：旧对话的global缓存还在，encoder局部缓存没了

假设已经处理过10000个token。过了一段时间，请求再次到来，服务系统还留着这段前缀的global KV，但encoder的近期SWA状态已经被回收。

当前持有的东西是：

| 对象 | 是否还在 | 能做什么 |
| --- | --- | --- |
| 已知前缀的token内容与位置 | 在 | 可以重新执行所需的一段输入 |
| 旧global KV | 在 | 长程attention继续读取旧历史 |
| Encoder各层SWA KV | 缺失 | 需要补回，不能直接复制global KV代替 |

要继续处理新输入，各层都需要自己的局部记录。Global KV来自不同来源层，有自己的投影和压缩，不能直接充当每层缺失的SWA KV。

Encoder Bounded Replay 选择旧前缀最后128个位置，即第9873～10000个token：

```text
旧前缀：  [1 …… 9872] [9873 …… 10000]
                         └─重算这段─┘
新输入：                              [10001 ……]
```

这段尾部重新从embedding进入encoder，使用原来的位置编号。重放时：

1. Global attention继续使用已保存的global KV，只读取因果可见的位置。
2. 局部SWA只读取重放段内已经算出的局部条目。
3. 把重算出的各层SWA KV留下，供后面的计算使用。
4. 旧global KV保持原样，不用重放结果覆盖。

接着，新输入正常经过模型，并生成新的global KV和SWA KV。它依赖刚恢复的局部状态，因此数值也可能与从头完整重算不同。

#### 为什么“窗口只有128”却不能保证精确恢复

关键在于：**深层KV来自深层hidden，而这个hidden已经受前面层的历史影响。** 只拿到最后128个token的原文，不等于拿到了它们原先进入每层时的hidden。

先把窗口缩小为4个token来观察。假设旧前缀有1000个token，只重放997～1000：

| 重放位置 | 正常SWA可以看哪些位置 | 截断后这次SWA只能看哪些位置 |
| --- | --- | --- |
| 997 | 994、995、996、997 | 997 |
| 998 | 995、996、997、998 | 997、998 |
| 999 | 996、997、998、999 | 997、998、999 |
| 1000 | 997、998、999、1000 | 997、998、999、1000 |

第一行已经发生了变化：997在局部路径上少看了三个位置。它的attention输出可能变化，这个输出进入更深层后，又会影响997在深层生成的KV。

到1000时，窗口里的四个位置虽然已经齐了，但其中997、998的深层表示可能已经变了。因此，**“最后窗口凑齐”不代表恢复了原先完整前向的数值。**

表中截断的是SWA这一条局部路径。有global分支的层仍可访问缓存中因果可见的长程信息，不能把它理解成整个模型只能看到这4个token。

如果要精确恢复，就需要往前追溯那些生成hidden所需的局部依赖。层数增加，所需前缀还会向前扩展。论文用层数乘窗口大小描述这种多层回溯范围。Bounded Replay则固定只取一个窗口，接受近似状态，以控制重算量。

#### 情景二：长prompt刚经过encoder，decoder局部缓存还没建

现在换成一次没有prefix缓存的新请求。10000个prompt token已完整经过前20层encoder，所以已经有：

```text
全部prompt的encoder输出 [10000,5120]
    ↓ 投影
供后20层共享的global KV
```

这时global历史资料准备好了，但decoder各层的SWA KV来自它们自己的hidden，仍然需要计算。

如果让10000个token都经过后20层，就失去了CED节省prefill工作的主要收益。Decoder Bounded Replay只拿最后128个位置的encoder输出，送过后20层：

```text
全部prompt → 前20层encoder → 全部位置的encoder输出
                                  ├→ 生成decoder global KV
                                  └→ 取最后128个位置的hidden
                                             ↓
                                      经过后20层decoder
                                             ↓
                                      生成各层SWA KV
```

对decoder来说，这里可以理解成**补跑尾部**：这些位置未必此前已执行过decoder。它们从encoder输出进入后半段，不重新查embedding，也不重新跑前20层。

补跑时仍读取可见的global缓存，局部SWA则限制在尾部这段内。因此它同样是近似的。论文报告质量影响很小，并在后训练中模拟这种过程，让模型适应；这并不构成数学等价保证。[报告 §3.2.2][ds41-report]

#### 两种 replay 分别什么时候发生

| 场景 | 从哪里开始重算 | 补回什么 | 主要节省什么 |
| --- | --- | --- | --- |
| Encoder replay：global prefix cache命中，但encoder SWA缺失 | 旧前缀末尾token的embedding | Encoder各层SWA状态 | 避免为局部状态回溯很长前缀，降低长期缓存负担 |
| Decoder replay：prefill需要建立decoder局部状态 | Prompt末尾位置的encoder输出 | Decoder各层SWA状态 | 避免全部prompt都完整经过后20层 |

完成 prefill 后，新 token 的普通 decode 仍经过全部40层，并更新现有 SWA 状态。生成过程无需每步都重放最近128个 token。

Decoder SWA状态用于后续decode，不存入长期prefix cache。Encoder SWA可以短时间保留，缺失时再恢复。Global KV则继续作为可长期复用的历史缓存。SWA Bounded Replay与ReplaySSM解决的是不同状态管理问题，不能把两者的“replay”混为一个算法。

### V4.1 的具体尺寸与其他模块

先看一层已经取得输入hidden、且历史足够长时的主attention：

```text
当前hidden [1,5120]
    ↓ Q下投影、RMSNorm
Q中间表示 [1,1280]
    ↓ 主Q上投影
64个query heads [64,512]

选中的global条目 [512,512] ─┐
                           ├→ 读取集合 [640,512]
本层最近的SWA条目 [128,512] ┘

Q @ 读取集合.T → 分数 [64,640]
统一归一化，包含sink规则 → 权重 [64,640]
权重 @ 读取集合 → 各head输出 [64,512]
```

所有query heads使用一组共享条目，每条512维向量兼作K和V。512维已包含位置处理使用的64维。输出每8个heads分成一组，每组从4096维投影到1024维；8组拼成8192维，再投影回5120维，交回主干。实际kernel可以直接按索引读缓存，以上形状不要求全部物化。

| 部件 | 具体配置与作用 |
| --- | --- |
| 语言主干 | 40层，hidden 5120；前20层encoder，后20层decoder |
| 主attention与indexer | 64个主Q heads，每头512维；32个indexer Q heads，每头128维；top-512 |
| MoE FFN | 384个routed experts，每token选6个，另有1个shared expert。输入输出均为5120维，单专家gate/up中间宽度2304 |
| FP4 main KV | 减少global主缓存的存储字节；逻辑宽度仍为512，量化scales另计 |
| Single-Pass mHC | 4条残差流；调整输入混合系数的依赖，使残差读写更容易融合 |
| Engram | 额外196B参数；在第2、15层使用2/3/4-gram查表并按上下文融合。V4.1省略原Engram的短因果卷积 |
| DSpark | 独立训练的draft模块，3个blocks，产生5个候选位置的草稿信息，再交给target验证 |
| DeepSeek-ViT | 图像特征经过projector成为5120维表示，与文本embedding一起进入语言主干 |

模型卡口径为552B主干，另有196B Engram；prefill激活8B，decode激活16B。报告中的“890 bytes/token”描述指定精度下的global缓存，不能当作整个请求的总显存。当前SWA缓存、临时激活、参数表和模型权重都要另外考虑。[固定配置][ds41]、[官方模型卡][ds41-card]

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

GR 的逐元素读门、Block AttnRes 的深度候选、Stable LatentMoE 的张量形状，以及 Engram/n-gram 查表的逐步解释，见[特殊结构详解](residual-experts-memory.md)。

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

[ds41-report]: https://arxiv.org/html/2609.19969v1
