# Transformer 运行全过程：从文本输入到逐 token 生成

## 1. 一次生成请求经过哪些步骤

以现代 decoder-only 语言模型为主线，采用 Llama 风格的 pre-norm、RoPE、GQA 和 SwiGLU。原始 Transformer 还有 encoder 与 cross-attention，第 15 节单独说明。不是所有模型都采用相同的 norm、FFN 或位置编码。

```text
文本或聊天消息
    ↓ chat template、tokenizer
token IDs
    ↓ embedding 查表
每个位置的初始 hidden state
    ↓ Transformer block 0
    ↓ Transformer block 1
    ↓ ……共 L 层
最后一层 hidden state
    ↓ final norm、LM head
词表上的 logits
    ↓ 选择下一个 token
输出 token ID → 还原文本
    └→ 把新 token 作为下一次模型输入，继续生成
```

**一次模型 forward 接收已有 token，计算下一个 token 的分布；生成整段文本通常需要多次 forward。** prefill 一次处理整个 prompt，后续 decode 利用 KV cache 处理新 token。[Hugging Face 生成流程](https://huggingface.co/docs/transformers/main/llm_tutorial)

本文是计算与数据流说明，不安装或运行模型。Llama 风格模块的参考实现固定为 [Transformers v4.57.1 的 modeling_llama.py](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/llama/modeling_llama.py)，便于对应具体模块。一个版本的 dispatch 不能代表所有推理引擎。

## 2. 全文使用同一组形状

下面的参数是教学配置，不对应某个具体已发布模型：

| 符号 | 含义 | 本文示例 |
| --- | --- | --- |
| B | batch size，请求数量 | 1 |
| S | 本次送入模型的 token 数 | prefill 为 4，单 token decode 为 1 |
| T | 本层 attention 可访问的 KV 总长度 | 无历史时为 S，有历史时为 past + S |
| D | hidden size，每个 token 的表示长度 | 1024 |
| Hq | query head 数 | 16 |
| Hkv | key/value head 数 | 4 |
| d | 每个 head 的特征长度 | 64 |
| F | FFN 中间维度 | 2816 |
| L | Transformer block 数 | 24 |
| Vocab | 词表大小 | 32000 |

示例满足 `Hq × d = D`，`Hq / Hkv = 4`。在这个 GQA 配置中，每 4 个 query heads 共享一个 KV head。

**S 是“这一轮计算多少个新位置”，T 是“这些位置可以读取多少个 KV 位置”。** decode 时 S 可以是 1，而 T 已经达到几千。把两者都叫 seq_len，容易在矩阵乘法和 mask 上写错。

## 3. 文本怎样变成 token IDs

聊天系统先按照模型的 chat template，将 system/user/assistant 消息组织成 token 序列。角色边界、消息结束标记和 assistant 开始标记都可能进入输入。

tokenizer 再把内容映射为整数 ID。token 可能是词、子词、字符片段或字节片段，不能直接理解成“一个汉字一个 token”。同一句话使用不同 tokenizer，得到的序列也可能不同。[Chat templates](https://huggingface.co/docs/transformers/main/chat_templating)

后面只假设输入有 4 个 token：

```text
input_ids.shape = [1, 4]
位置：                0    1    2    3
ID：                 a    b    c    d
```

a/b/c/d 是四个合法词表 ID 的代称，不是实际 tokenizer 的编码结果。batch 中存在 padding 时，还需有效长度或 attention mask，避免把 padding 当成正文。

## 4. embedding：每个 ID 查出一个长度 D 的向量

embedding 参数形状为 `[Vocab,D]`。ID 为 a，就读取这张表的第 a 行：

```text
input_ids：[1, 4]
embedding weight：[32000, 1024]
    ↓ 查表
X⁰：[1, 4, 1024]
```

X 中每个位置的一行，称作该 token 的 hidden state。最初这行来自 embedding；经过后续层后，它会结合允许看到的上下文。token ID 是离散索引，hidden state 是模型计算中的浮点向量，两者不是同一个对象。

embedding 不是 one-hot 向量必须先显式生成，再做一次大矩阵乘法；实现通常直接查表。模型参数在请求之间共享，每个请求的 hidden state 和 KV cache 则是运行数据。

位置也要进入计算。原始 Transformer 向 embedding 加位置编码。本文的 RoPE 路线主要在每层 Q/K 上引入位置，不能将这两种做法重复套用。

## 5. 一个 Transformer block 的完整顺序

令 X 为进入当前层的 `[B,S,D]` hidden state。本文采用两个串联的残差分支：

$$
A=X+\operatorname{AttentionBranch}(\operatorname{RMSNorm}_1(X)),
$$

$$
Y=A+\operatorname{FFN}(\operatorname{RMSNorm}_2(A)).
$$

```text
X ────────────────┐
 ↓ RMSNorm₁       │
Q/K/V 投影        │
 ↓ RoPE、attention│
多头合并、输出投影│
 ↓                │
加上 X ←──────────┘
 ↓ A ─────────────┐
RMSNorm₂          │
 ↓ SwiGLU FFN     │
加上 A ←──────────┘
 ↓ Y，交给下一层
```

Norm₁、Norm₂ 有各自的参数。RMSNorm 对每个 token 的最后一个 D 维向量独立计算统计量；它不把 batch 和全部 token 混在一起归一化。残差保留主路径，让分支在已有表示上添加更新，而不是直接用 norm 后的数据替换主路径。

对一行向量 x，本文 RMSNorm 的形式为：

$$
\operatorname{RMSNorm}(x)_a=\gamma_a\frac{x_a}{\sqrt{\frac1D\sum_{r=0}^{D-1}x_r^2+\epsilon}}.
$$

$\gamma$ 是长度 D 的学习参数，$\epsilon$ 保证分母稳定。它不减去均值；不能把它误当成“所有特征最后必须围绕 0 分布”。

本层输入输出通常都保持 `[B,S,D]`，内部却会经过不同维度和不同语义的张量。各层有各自权重，重复相同结构不表示重复使用同一组参数。

## 6. Q、K、V 从哪里来，怎样拆成多个 head

### 三次投影接收同一个 norm 后的 hidden state

令 `Z = RMSNorm₁(X)`，shape 为 `[B,S,D]`。PyTorch Linear 保存 `[输出维,输入维]` 权重，忽略 bias：

$$
Q=ZW_q^T,\qquad K=ZW_k^T,\qquad V=ZW_v^T.
$$

| 权重或结果 | 本文形状 |
| --- | --- |
| $W_q$ | `[1024,1024]`，即 `[Hq*d,D]` |
| $W_k$、$W_v$ | 各为 `[256,1024]`，即 `[Hkv*d,D]` |
| Q 投影结果 | `[1,4,1024]` |
| K、V 投影结果 | 各为 `[1,4,256]` |

Q、K、V 使用不同权重，学习不同的表示。Q 用于本位置发起匹配，K 用于各位置被匹配，V 提供按权重汇集的内容。它们不是从外部知识库直接取来的三份数据。

### 拆 head 先 reshape，再调整轴顺序

```python
# 形状示意，q/k/v 已经完成投影。
q = q.reshape(B, S, Hq, d).transpose(1, 2)     # [B,Hq,S,d]
k = k.reshape(B, S, Hkv, d).transpose(1, 2)    # [B,Hkv,S,d]
v = v.reshape(B, S, Hkv, d).transpose(1, 2)    # [B,Hkv,S,d]
```

投影将最后一个特征轴组织成 `head × head_dim`。reshape 和 transpose 本身不增加一层学习计算，只改变张量的逻辑分组与访问方式。转轴后不能直接假设存储仍连续。

### GQA 的共享关系

本文 16 个 Q heads 对应 4 个 KV heads：

```text
Q heads  0、 1、 2、 3 → KV head 0
Q heads  4、 5、 6、 7 → KV head 1
Q heads  8、 9、10、11 → KV head 2
Q heads 12、13、14、15 → KV head 3
```

每个 Q head 都有自己的分数与输出，但组内使用相同 K/V。教学代码可以逻辑上 repeat K/V，优化 kernel 则可直接复用，避免实际复制成 16 份。MHA 是 `Hkv=Hq`，MQA 是 `Hkv=1`，GQA 位于两者之间。[GQA 原论文](https://arxiv.org/abs/2305.13245)

## 7. RoPE：让 Q/K 的点积感知位置

仅有 Q/K 的内容变换还不够，模型需要区分 token 的位置。RoPE 按位置旋转 Q/K 中成对的特征，不改变 shape。某一对特征可以写成：

$$
\begin{bmatrix}x'_0\\x'_1\end{bmatrix}=
\begin{bmatrix}\cos\theta_p&-\sin\theta_p\\\sin\theta_p&\cos\theta_p\end{bmatrix}
\begin{bmatrix}x_0\\x_1\end{bmatrix}.
$$

p 是 token 位置，不同特征对使用不同频率。Q 位于位置 i、K 位于位置 j 时，二者点积会包含与相对位置有关的旋转关系。实现可以采用不同特征配对布局，但需要与模型训练时的约定一致。

本文给 Q/K 应用 RoPE，V 不旋转。缓存中保存的是已按其位置处理好的 K，以及对应 V。decode 的新 token 必须使用正确的绝对位置，而不是每轮都从位置 0 开始。[RoPE 原论文](https://arxiv.org/abs/2104.09864)

## 8. attention：分数、mask、归一化和 V 加权求和

### 每个 head 的 QK 点积

完成 RoPE，并取得历史与当前 KV 后：

```text
Q：     [B,Hq,S,d]
K：     [B,Hkv,T,d]
V：     [B,Hkv,T,d]

对 Q head h，选择它对应的 KV head g(h)：
Q_h @ K_g(h).T：[S,d] @ [d,T] → [S,T]
```

定义 score：

$$
s_{b,h,i,j}=\frac{q_{b,h,i}\cdot k_{b,g(h),j}}{\sqrt d}.
$$

i 表示当前 query 位置，j 表示被访问的 key 位置。一个 score 是两条长度 d 向量的点积，不是沿 token 位置把 Q/K 的所有元素一起相乘。

### causal mask 按位置关系屏蔽未来

prefill 的四个位置使用：

```text
             key 0  key 1  key 2  key 3
query 0        ✓      ×      ×      ×
query 1        ✓      ✓      ×      ×
query 2        ✓      ✓      ✓      ×
query 3        ✓      ✓      ✓      ✓
```

允许位置加 0，屏蔽位置加 `-inf`，然后沿 key 轴做 softmax。不能把未来位置的 score 乘成 0：`exp(0)=1`，仍会给它分配概率。

若已有 past 个位置，本次 query 的局部索引 i 对应绝对位置 `past+i`。屏蔽条件应为 `key_position > past+i`，再结合 padding、window 或其他限制。decode 的 `[1,T]` 分数不能未经位置偏移就套一个左上角三角 mask。

### 每个 query/head 输出一条 value 加权向量

$$
p_{b,h,i,j}=\frac{e^{s_{b,h,i,j}+M_{b,i,j}}}{\sum_{j'}e^{s_{b,h,i,j'}+M_{b,i,j'}}},\qquad
o_{b,h,i}=\sum_j p_{b,h,i,j}v_{b,g(h),j}.
$$

softmax 沿最后的 T 维，表示同一个 query 在各 key 之间分配权重。没有 dropout 时，这些权重非负且和为 1，输出是 V 的加权平均。

例如权重 `[0.2,0.3,0.5]`，V 为 `[[1,0],[0,2],[2,2]]`，输出为 `[1.2,1.6]`。这个二维示例说明混合方式，不代表真实 head 只有两个特征。

```text
scores / probs：[B,Hq,S,T]
每个 head 的结果：[B,Hq,S,d]
```

这些是逻辑形状。FlashAttention 可以直接分块计算输出，不必在显存里生成完整 score/probability 矩阵，详见 [FlashAttention 1–4](../CUDA/flash-attention.md)。

## 9. 合并 heads、输出投影、第一次残差

各 head 已经得到上下文相关的 value 混合结果，先还原 token-first 布局，再拼接 head 特征：

```text
[B,Hq,S,d]
    ↓ transpose(1,2)
[B,S,Hq,d]
    ↓ reshape
[B,S,Hq*d] = [B,S,D]
    ↓ output Linear
[B,S,D]
    ↓ 加上进入分支前的 X
A：[B,S,D]
```

```python
# 逻辑示意；transpose 后 reshape 可能需要拷贝。
merged = head_output.transpose(1, 2).reshape(B, S, Hq * d)
attn_update = merged @ w_o.T       # w_o：[D,Hq*d]
a = x + attn_update
```

拼接只是把不同 head 的结果放到不同特征位置。输出投影 $W_o$ 再学习如何混合这些特征，给残差路径提供 `[B,S,D]` 更新量。因此，“V 的加权平均”是 attention 核心的结果，还要经过输出投影才能完成本层 attention 分支。

## 10. FFN：每个 token 独立变换特征

第二个分支先对 A 做 RMSNorm，再执行 SwiGLU：

$$
Z=\operatorname{RMSNorm}_2(A),\quad U=ZW_{up}^T,\quad G=ZW_{gate}^T,
$$

$$
H=\operatorname{SiLU}(G)\odot U,\qquad Y=A+HW_{down}^T.
$$

| 中间量 | 本文形状 |
| --- | --- |
| Z | `[1,4,1024]` |
| up、gate 投影 | 各 `[1,4,2816]` |
| gate 激活后与 up 逐元素相乘 | `[1,4,2816]` |
| down 投影 | `[1,4,1024]` |
| 加上残差 A 得到 Y | `[1,4,1024]` |

up/gate/down 都是学习到的线性变换，SiLU 提供非线性。中间结果随后直接经过 down 投影，已经构成完整的这类 FFN；不要求再插入额外 Linear。

这里 $\operatorname{SiLU}(x)=x\,\sigma(x)$，$\sigma(x)=1/(1+e^{-x})$，$\odot$ 表示对应元素相乘。gate 不只是一个 0/1 开关，它产生连续的逐特征调节值。

FFN 在每个位置独立执行，同一层的不同 token 共用 FFN 权重。它本身不直接读取其他 token 的行，但其输入已经由 attention 混入上下文。**attention 混合位置，FFN 变换每个位置的特征。** 具体公式和组件实现可参见 [常见计算与组件](../PyTorch/operations-and-components.md)。

## 11. 堆叠 L 层后，怎样得到下一个 token

Y 进入下一层，再执行 norm、attention、残差、norm、FFN、残差。本文共有 24 层，所有位置逐层推进。最后仍是 `[B,S,D]`，然后执行 final RMSNorm 与 LM head：

```text
最后一层输出：[1,4,1024]
    ↓ final norm
[1,4,1024]
    ↓ LM head，权重 [32000,1024]
logits：[1,4,32000]
```

attention 输出投影 $W_o$ 混合 head 特征；LM head 把最终 hidden state 映射到词表。两个投影位于不同层次。部分模型让 LM head 与 embedding 共享权重，但不能默认所有模型都共享。

对于 causal LM，位置 i 的 logits 预测位置 i 后面的 token。要续写四个 token 的 prompt，取最后一个有效位置的 logits，即本例 `logits[:,3,:]`，得到 `[1,32000]` 的下一 token 分数。有些推理实现只计算需要位置的 LM head，省去其他位置的词表投影。

logits 是未归一化分数。可以使用 argmax 做 greedy，也可以根据 temperature、top-k、top-p 处理后的分布采样。这里的 softmax 沿 **Vocab 轴**，与 attention 沿 **KV 位置轴**的 softmax 不同。选择策略在模型 forward 之外组织。[生成策略](https://huggingface.co/docs/transformers/main/generation_strategies)

选出的 token ID 再由 tokenizer 解码成文本。单个 token 不一定构成完整可显示字符，流式输出也需要按 tokenizer 的字节和边界规则处理。

## 12. prefill 与 decode：用四个 prompt token 走三轮

### prefill：计算 prompt，并选出第一个新 token

输入 `[a,b,c,d]`，位置为 0、1、2、3：

- 每层一次处理 4 行 hidden state，Q/K/V 都为这 4 个新位置生成。
- causal mask 保证每行只读取自身及之前的位置。
- 每层保存这四个位置的 K/V。
- 最后一层位置 3 的 logits 用来选出第一个新 token e。

**得到 e 时，缓存通常还只有 a/b/c/d，没有 e 的 K/V。** e 是这次 forward 之后选出的离散结果，还没作为输入经过各层计算。

### 第一次 decode：输入 e，预测 f

接下来只输入 e，位置为 4。每层接收 `[1,1,1024]`，为 e 生成当前层的新 Q/K/V，再把新 K/V 加入该层缓存：

```text
Q_new：   [1,16,1,64]
K_cache： [1, 4,5,64]   包含 a/b/c/d/e
V_cache： [1, 4,5,64]
scores：  [1,16,1,5]    按 GQA 关系访问 KV heads
```

e 依次经过全部 24 层，最后产生 `[1,1,32000]` logits，从中选择 f。它仍然经过完整的 block，不会因为只有一个新 token 就跳过 FFN 或输出投影。

### 第二次 decode：输入 f，预测 g

输入 f，位置为 5。每层缓存从 5 个位置增长为 6 个位置，最后选择 g：

| 阶段 | 本次输入 | 输入位置 | 这次 forward 后缓存位置数 | 用来选择的下一个 token |
| --- | --- | --- | --- | --- |
| prefill | a/b/c/d | 0～3 | 4 | e |
| decode 1 | e | 4 | 5 | f |
| decode 2 | f | 5 | 6 | g |

如果选出 g 后就停止，通常无须再为了缓存 g 而额外执行一次 forward。因此“已输出 token 数”和“已进入 KV cache 的生成 token 数”在停止边界处可能不同。

以上是普通单 token 自回归流程。chunked prefill 会把 prompt 分段；speculative decoding 可一次验证多个候选位置，不能将这些扩展与基础 S=1 decode 混为一谈。

## 13. KV cache 的所有权、有效期与显存

### 每一层都有自己的 K/V

第 l 层的 K/V 由第 l 层输入及其投影权重产生，不能拿第 0 层的 K/V 供全部层复用：

```text
layer 0：K₀、V₀ → layer 0 attention 使用
layer 1：K₁、V₁ → layer 1 attention 使用
……
layer 23：K₂₃、V₂₃ → layer 23 attention 使用
```

Q 通常只为当前 query 服务。未来 token 用自己的 Q 读取历史 K/V，因此不必把全部历史 Q 一直保存。

### 为什么旧 K/V 可以复用

对于本文的固定权重、固定位置编码规则的 causal 模型，旧位置不会看到新追加的 token。其逐层 hidden state 和 K/V 不因未来追加而改变，所以可以复用。

这依赖模型与缓存配置一致。换模型权重、修改已缓存前缀，或改变影响旧 K 的位置处理规则时，不能直接视作同一份有效缓存。双向 encoder 的旧表示会受到新位置影响，也不能照搬这个理由。

cache 省去历史位置的重复投影和逐层计算，但新 query 仍需读取它允许访问的历史 KV。它不是把长上下文读取成本消掉。[KV cache 原理](https://huggingface.co/docs/transformers/main/cache_explanation)

### 容量估算

若每层都有全长 KV，K/V head dimension 均为 d，每元素占 b 字节，则：

$$
\mathrm{KVBytes}=2\,L\,B\,T\,H_{kv}\,d\,b.
$$

前面的 2 表示 K 和 V。本文配置若 `T=4096`、BF16 为 2 字节，KV 大小为 `2×24×1×4096×4×64×2 = 96 MiB`。这是有效元素的理论容量，不含分页碎片、元数据、量化 scale 和其他运行数据。

GQA 按 Hkv 而非 Hq 保存缓存，因此可减少 KV 元素数量。实际推理引擎通常预分配或分页管理 KV。append 可以是在空闲槽位写入并更新长度，不必每步 `torch.cat` 复制整个历史。分页改变物理寻址，不改变每个 query 应访问哪些历史内容的语义。

## 14. 训练为什么能一次处理多个位置

训练时完整目标序列已经给定。对 `[a,b,c,d,e]`，可以组织：

```text
模型输入：a  b  c  d
监督目标：b  c  d  e
```

所有输入位置在每层都可以并行计算，但保留 causal mask。位置 0 不能读取输入中的 b/c/d，否则会提前看到要预测的答案。目标已知只意味着不必等待采样，不代表允许未来信息泄漏。

每个有效位置得到词表 logits，计算目标 token 的交叉熵：

$$
\mathcal L=-\frac{1}{N_{\mathrm{valid}}}\sum_{(b,i)\,\mathrm{valid}}\log p(x_{b,i+1}\mid x_{b,0:i}).
$$

训练随后反向传播，计算参数梯度，再由 optimizer 更新权重。padding 或不参与监督的位置需要在 loss 中忽略；指令微调也可能只监督 assistant 片段。模型返回的 loss 是否已经内部 shift，要按接口确认，避免调用方再 shift 一次。

训练通常不用推理式 KV cache 复用历史图。它需要反向所需状态，FlashAttention 或 activation checkpointing 可以通过重计算节省中间存储。标准训练 forward 接收真实前缀，称为 teacher forcing；推理则不断把刚选出的 token 加入前缀。两者共用核心网络，但输入组织和反向路径不同。[原始 Transformer 训练与 mask](https://arxiv.org/abs/1706.03762)

## 15. 原始 Transformer 与现代 decoder-only 的区别

原始 encoder-decoder Transformer 常用于翻译：先编码源句，再让 decoder 逐步生成目标句。decoder 每层除了 causal self-attention，还有读取 encoder 输出的 cross-attention。

| 结构 | Q 来自哪里 | K/V 来自哪里 | 可见范围 |
| --- | --- | --- | --- |
| encoder self-attention | encoder 当前层输入 | 同一输入序列 | 通常允许有效源位置互相访问 |
| decoder causal self-attention | decoder 当前层输入 | decoder 的已知前缀 | 自身及过去位置 |
| decoder cross-attention | decoder 当前子层输入 | encoder 最终表示的投影 | 目标位置读取有效源位置 |

```text
源句 → encoder → 源句表示 ────────────────┐
                                          ↓ K/V
目标前缀 → causal self-attention → cross-attention → FFN
                                          ↑ Q
```

现代 decoder-only LLM 通常把提示词、对话和待续写内容组织成同一条序列，反复执行本文主线。它叫 decoder-only，不表示每层还暗中存在一套 encoder 或必然有 cross-attention。

原始论文采用的位置编码、LayerNorm 放置和 ReLU FFN，也不同于本文的 pre-norm/RMSNorm/RoPE/SwiGLU 组合。学习“Transformer”时，应先确认讨论的是哪一种 block。[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 16. 各种 attention 应按什么关系继续学习

不同名字可能改变不同维度，不能仅按名字当成互斥的算法列表：

| 讨论维度 | 代表概念 | 主要改变 |
| --- | --- | --- |
| 信息来源 | self-attention、cross-attention | Q 与 KV 来自同一序列还是不同序列 |
| 允许的位置 | causal、bidirectional、sliding window、block sparse | 哪些 query-key 配对存在 |
| head 共享 | MHA、GQA、MQA | Q heads 如何映射到 KV heads |
| 数学结构 | softmax attention、linear attention 等 | 权重函数或状态更新公式 |
| 执行算法 | FlashAttention、split-KV | 怎样分块、重计算和并行完成目标计算 |
| KV 存储 | contiguous、paged KV | 历史 KV 怎样组织地址与容量 |

例如“causal GQA + paged KV + FlashAttention”可以同时成立：三个词分别描述可见性、head 关系与执行/存储方式。

落实到 GPU，一次 block 通常涉及 RMSNorm、QKV GEMM、RoPE/cache 写入、attention kernel、输出 GEMM、残差、FFN GEMM 和逐元素运算。实现可以融合相邻步骤，所以数学步骤数不等于 kernel 数。[GEMM 教程](../CUDA/gemm.md) 解释矩阵工作，[FlashAttention 1–4](../CUDA/flash-attention.md) 解释 attention 内部的数据流，[CUDA Graph](../GPU/graphs-and-compilation.md) 则解释如何复用提交安排。

## 17. 把一次生成的控制流连起来

下面是接口语义示意，`model` 代表完整 embedding、L 层 blocks、final norm 和 LM head，不是单个 attention 函数。代码省略 batch padding、采样策略细节和资源管理，不作为独立可运行实现。

```python
# prompt_ids：[B,S]，先完成整个 prompt 的 prefill。
logits, cache = model(prompt_ids, cache=None)
next_id = select_token(logits[:, -1, :])     # [B]

for step in range(max_new_tokens):
    emit(next_id)
    if should_stop(next_id) or step + 1 == max_new_tokens:
        break

    # 只输入刚选出的 token；model 根据 cache 长度安排其位置。
    logits, cache = model(next_id[:, None], cache=cache)
    next_id = select_token(logits[:, -1, :])
```

这里假设 batch 中所有请求长度一致且一起停止。实际服务会维护每个请求的长度与完成状态，动态组织 batch，并在请求结束后回收 KV 槽位。它不改变“已有前缀产生 logits，选择 token，再将新 token 送回完整模型”的基础依赖。

沿着一行 hidden state 解释每层发生的变化，再追踪一个 KV 位置在各层的存储与使用。这些对应关系能帮助理解各类 attention 优化在完整 Transformer 中的作用。本文的 shape 和容量示例均为推导，未运行模型或 CUDA 用例。
