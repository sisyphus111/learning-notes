# 特殊结构详解：GR、AttnRes、LatentMoE 与 Engram

## 1. 先区分这些模块在改什么

| 模块 | 处理对象 | 输出交给谁 |
| --- | --- | --- |
| Gated Residual（GR） | 同一 token 的多个残差分支 | 当前 attention/FFN 子层，以及下一子层的残差状态 |
| Block Attention Residuals（AttnRes） | 同一 token 在不同深度产生的表示 | 当前子层 |
| Stable LatentMoE | 当前 token 的特征通道与专家变换 | 当前层的 FFN 输出 |
| N-gram Embedding | 当前及紧邻之前的 token IDs | 当前 token 的 hidden/residual |
| Engram | n-gram 参数查表结果与当前上下文 hidden | 通过门控、卷积和残差融合进入主干 |

它们可以与 attention 共同存在。token attention 沿序列读取其他位置；GR/AttnRes 改变跨深度的信息传递，MoE 改变特征计算。条件记忆增加按 token 组合访问的模型参数。家族和版本对照见[模型结构](model-architectures.md)。

下文单 token 向量按列向量表示，D 为主干 hidden 宽度。公式解释数据流，手算数值是教学示例；未运行完整模型或 GPU benchmark。

## 2. Gated Residual：四份残差状态怎样读写

### 从普通残差开始

普通 pre-norm 子层可以写为：

$$
y=F(\operatorname{RMSNorm}(h)),\qquad h'=h+y.
$$

F 可以是 attention 或 FFN。残差加法让旧表示保留在输出里，但连续层写入的是同一份 D 维状态。GR 将它扩成 n 个分支，Qwen3.8-Flash-Next 使用 n=4：

$$
R=[R_1;R_2;R_3;R_4]\in\mathbb R^{4\times D}.
$$

这 4 份状态不是 4 个 token、4 个 experts，也不是把 attention 执行 4 次。GR 从四个分支读出一个 D 维向量，子层只处理这个输入，再将一个 D 维结果写回各分支。[Qwen GR 设计][q-report]

### 读：每个分支、每个特征都有一个 gate

先独立归一化各分支，得到 $\widehat R_i$，每个分支有自己的 RMSNorm gain。把所有分支拼成长度 nD 的向量，再通过低秩投影预测 gate：

$$
G=\operatorname{reshape}\left[\sigma\left(W_u\operatorname{SiLU}\left(\frac1nW_d\operatorname{vec}(\widehat R)\right)\right)\right]\in\mathbb R^{n\times D}.
$$

这里 $W_d[r,nD]$、$W_u[nD,r]$，报告采用瓶颈宽度 r=D/8。随后：

$$
x=\frac1n\sum_{i=1}^{n}G_i\odot\widehat R_i,\qquad y=F(x).
$$

G 由当前 token 的所有分支共同决定。某个分支第一个特征可以读得多、第二个特征读得少。sigmoid 将单个读门限制在 0～1，但这里没有沿分支做 softmax；各门之和不要求等于 1。

例如只用两个分支、两个特征说明读操作，假设归一化并乘 gain 后是：

```text
Rhat1 = [1, 2]    G1 = [0.8, 0.2]
Rhat2 = [3, 4]    G2 = [0.1, 0.9]

x = (G1 * Rhat1 + G2 * Rhat2) / 2
  = ([0.8, 0.4] + [0.3, 3.6]) / 2
  = [0.55, 2.0]
```

分支不是固定承担某个语义。训练可以学到不同的信息保留和读取方式，不能预先命名为“语法分支”“事实分支”。

### 写：每个分支一个标量，旧状态继续保留

Qwen 的写门为：

$$
s=2\sigma\left(\frac1nW_w\operatorname{vec}(\widehat R)\right)\in\mathbb R^n,\qquad R'_i=R_i+s_i y.
$$

$W_w[n,nD]$ 为每个分支生成一个标量。**读门是 `[n,D]`，写门是 `[n]`；写回加在原始 R 上，不是归一化后的 Rhat 上。** 写门范围为 0～2，也不是概率分布。

只看一次写回，假设原始状态为 `R1=[1,2]`、`R2=[3,4]`，子层输出 `y=[2,-1]`，写门为 `[0.5,1.5]`：

```text
R1' = [1,2] + 0.5*[2,-1] = [2,1.5]
R2' = [3,4] + 1.5*[2,-1] = [6,2.5]
```

两个分支得到同一子层输出的不同写入量，下一个子层再按自身 gate 读取它们。GR 没有保留 HC/mHC 的完整 n×n 残差混合算子；旧分支由恒等路径保留，新结果通过写门注入。报告中 attention 和 FFN 子层分别有自己的 GR。[GR 公式 30–34][q-report]

### 成本在哪里

子层输入仍为 D，因此 attention/FFN 主矩阵尺寸不因四个分支就变成 4D。增加的是分支状态读写、归一化和 gate 投影。

以 M=128、D=2560、BF16 为例，单份 `[M,D]` residual 是 0.625 MiB，四分支是 2.5 MiB。这只是该状态的容量，不是整个模型显存变成 4 倍。融合 read/norm/gate 和 write 可以减少中间写回；效果需按真实 kernel 测量。

## 3. Block AttnRes：attention 的候选轴改为深度

### 不是对历史 token 做第二遍 attention

固定一个 token t，普通 self-attention 从序列位置 j 读取 KV；AttnRes 从这个 token 在不同层或块产生的表示读取信息。

```text
序列 attention：token t ← token 0、1、2、…、t
深度 AttnRes：  当前子层 ← embedding、已完成深度块、当前块部分结果
```

更精确地说，候选是 embedding 与先前子层的输出。若把每个已包含全部残差的 hidden 再累加一次，就会重复计入旧信息。

### 权重怎样计算

令候选为 $v_0,\ldots,v_{J-1}\in\mathbb R^D$。当前层有学习得到的 pseudo-query $w_\ell[D]$：

$$
e_j=w_\ell^T\operatorname{RMSNorm}(v_j),\qquad \alpha_j=\frac{\exp e_j}{\sum_i\exp e_i},\qquad x_\ell=\sum_j\alpha_jv_j.
$$

层的 query 参数固定，但各 token 的候选表示不同，所以权重随 token 改变。softmax 沿候选深度轴计算，输出仍是 D 维。这里也不是常规 token attention 那种必需分别生成 Q/K/V 三套矩阵的接口。

假设候选向量为 `[1,0]`、`[0,2]`、`[2,1]`，得到权重 `[0.25,0.5,0.25]`，则子层输入为 `[0.75,1.25]`。这是候选的加权和，不是三个 token 的 attention 输出。

### 为什么分 block

Full AttnRes 需要保留每个候选子层的输出。Block AttnRes 将一个深度块内部的子层输出累加成一份表示：

$$
b_n=\sum_{j\in\mathcal B_n}F_j(x_j).
$$

当前块还未结束时，保留已完成子层的部分和。当前子层的候选为：

```text
embedding b0
已完成块 b1、b2、…
当前块的部分和（若已有子层输出）
```

它先对这些候选做加权读取，再运行自身 F，并将新输出加进当前块部分和。块结束后，这份累积结果成为后续层可访问的已完成块表示。

K3 报告采用每 12 层一块，93 层得到 8 个深度块，末块不满；embedding 另作一个来源。这里的 block 是深度分组，不是 attention 的 token tile，也不是 CUDA thread block。[K3 AttnRes][k-report]

### 与 GR 的区别和状态生命周期

| 项目 | GR | Block AttnRes |
| --- | --- | --- |
| 保存对象 | 固定数量的残差分支 | embedding、已完成深度块、当前块部分和 |
| 读取权重 | 每分支每通道的 sigmoid gate | 沿候选深度轴的 softmax |
| 写入方式 | 子层输出按标量写回多个分支 | 子层输出加入当前块部分和 |
| 主要代价 | 拓宽后的 residual 带宽与 gate 计算 | 保存、读取深度候选并做归一化/加权 |

对一次 forward 中正在处理的 M 个 token，保存 J 份 D 维候选需要约 `M×J×D` 个元素。分组减少 J，但不能把有同样 shape 的候选当作可以互相覆盖的同一个 buffer。

这些状态服务于当前 token 跨深度的计算，不因此成为永久保留所有历史 token 的 KV cache。prefill、训练 backward 和普通单 token decode 的生命周期还需分别处理。

## 4. Stable LatentMoE：专家仍有完整 FFN，只是工作宽度变窄

### 先看一条 routed 路径

K3 主干宽度 D=7168，routed latent 宽度 r=3584。用 M 个 token 表示：

```text
X [M,7168] ───────────────────→ 2 个 full-width shared experts ─┐
  ↓ 下投影 Wdown.T [7168,3584]                                  │
Z [M,3584]                                                    │
  ↓ router 选择、按专家分组                                    │
各专家处理自己的 [Me,3584]                                    │
  ↓ 按 token、路由权重合并                                    │
U [M,3584]                                                    │
  ↓ RMSNorm → Wup.T [3584,7168]                                 │
routed 输出 [M,7168] ──────────────────────────────────────────┴→ Y
```

图中批量 X 按行排列，右乘单 token 列向量公式中权重的转置。公式为：

$$
z=W_\downarrow x,\quad u=\sum_{e\in\mathrm{TopK}(router(x))}p_eE_e(z),\quad y=\sum_jE_j^{shared}(x)+W_\uparrow\operatorname{RMSNorm}(u).
$$

K3 有 896 个 routed experts，每 token 选 16 个，另有 2 个 shared experts。每个 routed expert 是 `r → intermediate → r` 的非线性 FFN，内部仍有 gate/up/down 等矩阵；外面的 `D → r → D` 只改变这些专家的工作宽度。[K3 Stable LatentMoE][k-report]、[配置][k-config]

所有选中专家的结果先在 r 维合并，再做一次 norm/up-projection。不能先给每个专家分别 RMSNorm 再加权，因为归一化不是线性变换，顺序会改变函数。

### 为什么叫 Stable

额外 down/up 投影与专家内部矩阵串联，会放大中间数值和训练稳定性问题。K3 报告采用三类处理：

- 在 routed 聚合结果与上投影之间加入 RMSNorm，控制聚合后的尺度。
- 使用 SiTU-GLU 对 gate/up 的激活范围做约束，抑制极端值。
- 使用 Quantile Balancing 调整专家负载均衡，解决大专家池的选择分布问题。

SiTU-GLU 的简化逐元素表达如下。a、b 分别是 gate/up 两个线性投影的结果：

$$
g(a)=\beta_1\tanh(a/\beta_1)\sigma(a),\quad u(b)=\beta_2\tanh(b/\beta_2),\quad E(z)=W_{out}[g(a)\odot u(b)].
$$

K3 的对应参数为 $\beta_1=4,\beta_2=25$。乘积激活绝对值不超过 100，但经过后续输出矩阵后，整个 expert 输出不受同一个 100 上限约束。不能把它等同于普通 SwiGLU，也不能将“稳定”理解为消除全部训练风险。

### 成本与信息取舍

固定专家中间宽度 I，普通三矩阵 GLU 的权重约为 `3DI`，latent 版本约为 `3rI`。另加每层共享的两张 `D↔r` 投影，合计 `2Dr`。r=D/2 时，每个 routed expert 的这部分参数约减半；总模型还包括 shared experts、router、attention 与额外投影。

较窄输入也减少专家处理/传输的特征量，但引入投影、norm 和容量约束。保留全宽 shared 路径可以补充通用变换，不能推断 latent 能无损保留任意 D 维信息。

| latent 名称 | 压缩对象 | 是否是逐历史 token 的 attention cache |
| --- | --- | --- |
| MLA latent | 历史 token 的 K/V 内容表示 | 是，通常还要考虑位置分支 |
| LatentMoE latent | 当前 routed FFN 的特征宽度 | 否，是当前计算的中间激活 |

## 5. N-gram embedding：从一个 token 查表，扩展到几个 token 查表

### n 是组合长度，不是向量维度

普通 embedding 用一个 token ID 查参数表 `E[id]`。n-gram embedding 用连续 n 个 token IDs 的组合确定条目：

```text
token IDs：[…,17,42,9]
当前位置 unigram：[9]
当前位置 bigram： [42,9]
当前位置 trigram：[17,42,9]
```

组合以当前位置结束，因此不读取未来 token。token 不一定等于自然语言的一个词；相同最后一个 token，在不同前缀下可以查到不同组合向量。一个位置也可以同时查询多种 n-gram order。

### 为什么需要 hash

若词表有 V 个 token，为所有 trigram 直接分配参数需考虑 $V^3$ 个组合。hash 将组合映射到有限槽位：

$$
i_{n,h}=hash_{n,h}(ids_{t-n+1:t})\bmod V_{n,h},\qquad e_{n,h}=E_{n,h}[i_{n,h}].
$$

h 表示 hash head，不是 attention head。不同 order/head 可以使用不同表和 hash。读出的向量拼接或按模型定义组合，形成 memory vector，再投影和融合。

一个可手算的教学 hash（不是官方算法）是：

```python
def toy_hash(token_ids, buckets):
    index = 0
    for token_id in token_ids:
        index = (31 * index + token_id) % buckets
    return index

assert toy_hash([42, 9], 101) == 99
assert toy_hash([17, 42, 9], 103) == 35
assert toy_hash([1, 68], 101) == 99  # 不同组合发生碰撞
```

假设 `E2[101,4]`、`E3[103,4]`，本位置读取 `E2[99]` 和 `E3[35]`，拼接成 8 维向量。读的是参数表中的行，不是把 ID 数字本身当成语义向量。

碰撞意味着不同组合共享条目，梯度可能互相干扰。多组 hash 可以降低两个组合在所有表中都碰撞的概率，但不保证彻底消除碰撞。

### 这些参数怎样训练，怎样使用

hash 和 token IDs 不需要可微。训练的梯度通过查表输出回到命中的 embedding 行，也会更新后面的投影和 gate。这与普通 `nn.Embedding` 的索引选择方式类似。重复访问同一条目的梯度按训练实现累积。

普通推理只读这些参数，不因用户刚输入了一句新话就修改表。表项也不必可解释为一条事实或原文句子。它是训练学到的局部模式表示，不是外部文本搜索结果。

## 6. Engram：查表之后怎样结合当前上下文

Engram 是 DeepSeek 提出的具体条件记忆模块，n-gram embedding 是它的重要组成。原始论文的流程包括 token ID 规范化、多个 hash 表查询、上下文门控、短卷积与残差注入。[Engram 论文][engram]、[官方示意代码][engram-code]

```text
原始 token IDs → 规范化后的 IDs → suffix n-grams → 多头 hash
                                                       ↓
当前层 hidden h ────────────────→ gate ← 查表拼接向量 e
                                    ↓
                              gated value
                                    ↓
                          短因果卷积与残差融合
                                    ↓
                              主干后续计算
```

### 规范化只服务于这条查表路径

原始 Engram 将某些文本规范化后等价的 tokenizer 条目映射到同一个 canonical ID，提高表的利用率。不能据此认为语言模型输入从此丢掉了大小写或空格：主 token embedding 路径仍保留原来的 token 身份。

### 静态条目与动态 gate

设拼接查表向量为 $e\in\mathbb R^{d_m}$，当前层 hidden 为 $h\in\mathbb R^D$：

$$
k=W_Ke,\quad v=W_Ve,\quad \alpha=\sigma\left(\frac{\operatorname{RMSNorm}(h)^T\operatorname{RMSNorm}(k)}{\sqrt D}\right),\quad \tilde v=\alpha v.
$$

Wk/Wv 将 memory vector 映射到与 hidden 对齐的 D 维。这里 k/v 是查表向量的投影，不是请求历史 KV cache。相同 n-gram 会读到相同 e，但 h 含当前上下文，门值因此可以不同。

例如相同局部片段出现在金融或地理语境中，早先层形成的 h 不同，门可以改变查表结果的影响。这是机制示意，不保证某个训练条目恰好学到了特定词义。

原论文随后使用按通道的短因果卷积、SiLU 和局部残差，最后将结果加进主干 hidden。卷积可以结合邻近位置的 gated values；因果实现只使用当前与过去位置。Engram 只放在指定层，不是给每个 Transformer 层都附加整张参数表。

### 不能与请求状态混淆

| 对象 | 内容来源 | 普通推理中的变化 |
| --- | --- | --- |
| Engram 参数表 | 训练 | 只读，跨请求共享 |
| lookup e / gated value | 当前 token 查表与融合 | 当前计算的激活，可按生命周期复用 |
| KV / MLA cache | 当前请求各历史位置 | 随历史追加 |
| GDN / KDA state | 当前请求的递推 | 每个 token 更新 |
| 短卷积状态 | 当前请求最近的 gated values | 滚动更新 |
| RAG 文档 | 外部知识库 | 由检索系统维护，结果通常作为模型输入 |

“Engram 表固定”不表示整个模块没有请求状态。流式生成仍需正确处理 n-gram 边界所需的最近 IDs，以及卷积历史。它也不替代 attention 对当前对话事实的读取。

### 为什么适合 host memory 预取

表索引由 IDs 决定，不依赖直到深层才生成的 router hidden。prefill 的输入 IDs 已知；decode 的当前新 token 确定后，也能在目标层执行前安排查询和传输。

```text
知道当前 IDs → 计算表地址 → host 读取所需行 → 传到 GPU
                    与前面几层计算重叠 ───────────────┘
```

不是把整张表每步搬到 GPU，也不是预知尚未生成的未来 token。需要搬的是当前查询的行，外加融合工作。地址已知不保证传输总能完全隐藏：batch、随机访问、带宽、预取距离都会影响等待。

固定 n-gram order、hash heads、embedding 宽度时，每个 token 的查表次数不随总表容量增长，因此常称 O(1) lookup。这不表示整段序列 O(1)，也不表示内存访问和后续投影没有成本。训练还需考虑表梯度和优化器状态，不能直接照搬推理 offload 的成本结论。

## 7. Qwen 的 N-gram Embedding 与 Engram 有何不同

Qwen3.8-Flash-Next 报告同样采用多头 hash 与上下文门控，将表表示注入 residual stream。但它是另一份模型设计，不能把 DeepSeek 的 tokenizer 规范化、卷积或所有超参数原样套过去。Qwen 报告还说明，尝试 token normalization 等策略并未在其训练方案中得到稳定收益。[Qwen n-gram 设计][q-report]

公开 Flash-Next 在第 2 层插入一个 n-gram embedding 模块，使用 bigram/trigram。放在较早但不是最开始的位置，可以用前一层计算为 host 预取留出时间。插入位置仍是模型结构选择，不保证所有机器上都是最优延迟。

官方参数口径为 **125B 主干、6B 激活，另有 51B n-gram embedding 和 4B MTP**。51B 描述整张表的参数容量，不表示每个 token 对 51B 参数执行 dense GEMM。若仅按 BF16 保存这 51B 参数，原始容量约 102 GB；实际格式、分片、优化器和缓存策略会改变部署需求。[Flash-Next 模型卡][q-card]

### MoE 与 MTP 在截图里分别意味着什么

MoE 的 router 依据 hidden 选择少数专家，执行相应 FFN；n-gram lookup 由 token 组合确定条目。两者都可以扩大总参数容量，但一个选择计算网络，一个选择参数向量，成本并不等价。

MTP 是 Multi-Token Prediction，训练增加预测后续多个位置的目标与辅助路径。推理可以将其用于草稿生成，再让 target 验证；接受的 token 才成为正式结果。它不保证一次主干前向无条件得到多个正确 token，实际收益还取决于接受率和验证开销。Flash-Next 将 MTP 单独计入参数，不应把它和 48 层主干或 n-gram 表混为一项。

## 8. 把几种结构放回一个 token 的前向过程

对 Flash-Next，可以分别追踪两条输入：token IDs 提供 n-gram 表地址，当前层 residual 提供上下文融合和子层输入。GR 读出 D 维输入，GDN/QSA 或 MoE 计算子层结果，GR 再写回拓宽的 residual。n-gram 模块只在指定位置注入额外信息。

对 K3，Block AttnRes 从深度候选中形成当前子层输入；KDA/MLA 完成时间方向的信息读取；Stable LatentMoE 完成特征变换。新的子层输出更新深度块状态，继续交给后续计算。它们不是同一个 attention kernel 的三个名称。

| 需要回答的问题 | 对应机制 |
| --- | --- |
| 当前 token 看哪些历史位置？ | token attention、MLA、QSA |
| 历史怎样累积成固定状态？ | GDN/KDA |
| 当前子层从哪些深度信息读取？ | AttnRes 或 GR |
| 当前 token 的 FFN 在多宽空间计算？ | LatentMoE |
| 当前局部 token 组合能读取哪些静态参数？ | n-gram embedding / Engram |

## 9. 参考来源与验证边界

核对日期：2026-10-29。公式按原论文机制展开；小数值、hash 和容量例子为自行构造的教学推导。未将论文跑分当作本机性能，也未运行完整 checkpoint。

- [Qwen3.8-Next 报告 §2.2–2.3][q-report]：GR 的 read/write gate、n-gram lookup 和插入位置。
- [Qwen3.8-Flash-Next 模型卡][q-card]：参数口径与主要配置。
- [Kimi K3 报告 §2.2–2.3][k-report]：Block AttnRes、Stable LatentMoE、SiTU-GLU。
- [K3 固定配置][k-config]：hidden、latent、专家数量与结构字段。
- [Engram 原论文][engram]与[官方 demo][engram-code]：规范化、hash、融合和预取。demo 自身明确不代表生产优化实现。

[q-report]: https://arxiv.org/html/2608.30320v1
[q-card]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[k-report]: https://arxiv.org/html/2607.24653v1
[k-config]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json
[engram]: https://arxiv.org/html/2601.07372v1
[engram-code]: https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py
