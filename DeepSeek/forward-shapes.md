# DeepSeek V3、V3.2、V4-Pro：一次前向与张量形状

## 1. 固定输入、形状约定与模型版本

以单请求 decode 为主线：已有 P=131072 个历史 token，现在输入一个已知 token，预测它后面的 token。

```text
batch size B = 1
本次输入长度 S = 1
此前历史长度 P = 131072
写入本次 token 后 T = P + S = 131073
当前 token 的零基位置 t = P = 131072
```

以下都是未按 TP/EP 切分的**全局逻辑形状**，不表示模型能装进单卡。单 token 的 `[B,S,D]=[1,1,D]` 通常省略 batch 维写成 `[1,D]`；当前 query 的 `[1,1,H,d]` 简写为 `[H,d]`。缓存 `[1,T,d]` 简写为 `[T,d]`。

矩阵统一使用行向量约定：`y=x@W`，W 为 `[in,out]`。PyTorch `nn.Linear.weight` 的存储形状是 `[out,in]`，调用时等价于乘其转置。这里给出 FP8/FP4 解包后的逻辑元素形状，不包含量化 scales、padding 或分片。

本页讲普通主干自回归生成，不把额外 MTP 草稿预测拼入主干。V4 固定为 **Pro**；Flash 的 hidden、heads、层数与 top-k 不同。机制解释与 Flash 对照见[架构演进](evolution.md)。

配置核对于 2026-10-29，固定来源见本页末尾。没有运行模型或 GPU 测试；下面展示数学形状和参考代码路径，融合 kernel 未必显式分配所有中间 tensor。

## 2. 尺寸表与全模型路径

| 参数 | V3 | V3.2 | V4-Pro |
| --- | ---: | ---: | ---: |
| 主干层数 L | 61 | 61 | 61 |
| hidden D | 7168 | 7168 | 7168 |
| vocabulary V | 129280 | 129280 | 129280 |
| 主 Q heads H | 128 | 128 | 128 |
| Q latent | 1536 | 1536 | 1536 |
| 内容 QK / RoPE 每头宽度 | 128 / 64 | 128 / 64 | 448 / 64 |
| V 每头宽度 | 128 | 128 | 512，共享 K=V |
| MLA KV latent | 512 | 512 | 不沿用此结构 |
| Indexer heads / width / top-k | 无 | 64 / 128 / 2048 | CSA：64 / 128 / 1024 |
| Routed experts / 每 token 选中 | 256 / 8 | 256 / 8 | 384 / 6 |
| Shared experts | 1 | 1 | 1 |
| Expert 中间维度 | 2048 | 2048 | 3072 |
| 开头 dense FFN | 3 层，中间宽度 18432 | 同 V3 | 无；前 3 层为 hash-routed MoE |
| 层间 residual | `[1,7168]` | `[1,7168]` | `[1,4,7168]` |

配置：[V3][v3-config]、[V3.2][v32-config]、[V4-Pro][v4-config]。

```text
V3 / V3.2
ID → Embedding [7168] → 61 × {Attention + residual → FFN + residual}
   → Final RMSNorm [7168] → LM head [129280] → 采样得到下一 ID

V4-Pro
ID → Embedding [7168] → 扩成 [4,7168]
   → 61 × {mHC + Attention → mHC + MoE}
   → 合并为 [7168] → Final RMSNorm → LM head [129280] → 下一 ID
```

这条路径中的层数是顺序依赖：当前 token 先完成第 l 层，再把该层输出交给第 l+1 层。每层都有自己的模型权重和历史缓存，61 层不会共用同一份 latent。

## 3. V3：Embedding、归一化与 Q 投影

### 3.1 输入 token ID 取出一行 embedding

Embedding 表 E 的形状为 `[129280,7168]`。输入 ID 原本是 `[1,1]` 的整数索引，取出该行后得到 `[1,1,7168]`，下文简写为 x=`[1,7168]`。

E 是长期存在的模型权重；x 是本次输入产生的 activation。这里没有对完整 embedding 表做一次 GEMM。

### 3.2 普通 block 的两次 residual

$$
u=x+\operatorname{Attention}(\operatorname{RMSNorm}(x)),
\qquad
x_{\mathrm{next}}=u+\operatorname{FFN}(\operatorname{RMSNorm}(u)).
$$

RMSNorm 沿最后的 D=7168 维归一化，形状不变。Attention 和 FFN 分别有自己的 norm 权重 `[7168]`。先保存 x 作为 residual，令 attention 输入 a=`[1,7168]`。

### 3.3 Q 先降维，再展开到各 head

| 运算 | 输入 | 权重，按 `[in,out]` | 输出 |
| --- | --- | --- | --- |
| Q down | `[1,7168]` | `[7168,1536]` | `[1,1536]` |
| Q latent RMSNorm | `[1,1536]` | gain `[1536]` | `[1,1536]` |
| Q up | `[1,1536]` | `[1536,24576]` | `[1,24576]` |
| reshape | `[1,24576]` | 无 | `[128,192]` |
| split | `[128,192]` | 无 | 内容 `[128,128]`，位置 `[128,64]` |
| RoPE | 位置 `[128,64]` | 当前 token 的位置旋转 | `[128,64]` |

`24576=128×(128+64)`。Q 的总宽度不等于 7168；多头 attention 不要求 `head_dim=hidden_size/num_heads`。

`q_latent` 只服务当前 token 的 Q 生成，通常不作为历史缓存保留。

## 4. V3：生成当前层缓存，吸收 K，再对历史打分

### 4.1 本次新增的 cache 只有 512 + 64

```text
a：[1,7168]
   @ W_KV_down：[7168,576]
kv_raw：[1,576]
   ├─ 前 512 维 → RMSNorm → c：[1,512]
   └─ 后  64 维 → RoPE    → k_R：[1,64]
```

写入当前层缓存后：

```text
C_cache：[131073,512]
R_cache：[131073,64]
```

512 维 latent 联合生成内容 K/V。64 维 k_R 是跨 head 共享的位置 key，但各头的 Q_rope 不同。

### 4.2 K/V up 权重的逻辑形状

联合 up 权重为：

```text
W_KV_up：[512,32768]
32768 = 128 heads × (128 content-K + 128 V)
```

逐头拆开：

```text
U_K：[128,512,128]
U_V：[128,512,128]
```

这里的三个维度分别是 head、输入 latent、输出 feature。按 PyTorch 原权重 view 时，维度顺序可能是 `[128,256,512]`，与本文 `[in,out]` 约定不同。

### 4.3 先变换当前 Q，避免恢复全部历史 K

对每个 head h：

$$
q_{\mathrm{abs},h}=q_h^C U_{K,h}^{T}.
$$

```text
Q_content[h]：[1,128]
U_K[h].T：   [128,512]
结果：       [1,512]

128 个头合起来 Q_absorb：[128,512]
```

然后计算两个分数矩阵：

```text
内容分数： [128,512] @ [512,131073] → [128,131073]
位置分数： [128,64]  @ [64,131073]  → [128,131073]
```

$$
L=\alpha\left(Q_{\mathrm{abs}}C_{\mathrm{cache}}^T+Q_RR_{\mathrm{cache}}^T\right).
$$

L 的行表示 head，列表示历史 token 位置。基础 scale 来自原始 QK 的 192 维，还可能叠加 YaRN 修正。吸收改变计算方式，不应把它擅自改成 `1/sqrt(576)`。

当前 query 已经是序列最后一个位置，可以访问当前 token 与此前所有位置。若做多 token prefill，则需要对未来位置施加 causal mask。

## 5. V3：latent 加权、V 投影与输出投影

沿历史轴归一化：

```text
P = softmax(L, dim=-1)
P：[128,131073]
```

latent 加权：

```text
P @ C_cache
[128,131073] @ [131073,512] → Z：[128,512]
```

每个 head 再执行自己的 V up：

```text
Z[h]：[1,512] @ U_V[h]：[512,128] → O[h]：[1,128]
所有 heads → O：[128,128]
```

这里先对 latent 求加权和，再投影 V。没有逐位置展开全部历史 V，也不需要把 attention 权重先乘 U_V。

输出投影：

```text
O：[128,128]
   ↓ flatten
[1,16384]
   @ W_O：[16384,7168]
attention_output：[1,7168]
   + 原来的 residual x
u：[1,7168]
```

V3 decode 的这条吸收路径见[官方 inference/model.py][v3-code]。逻辑上的 scores 和 P 可以在融合 attention kernel 内分块计算，不要求先把完整矩阵写回 HBM。

## 6. V3：dense FFN、MoE 与层间输出

### 6.1 前 3 层 dense SwiGLU

f=`RMSNorm(u)`，形状 `[1,7168]`：

```text
gate = f @ W_gate：[7168,18432] → [1,18432]
up   = f @ W_up：  [7168,18432] → [1,18432]
z = silu(gate) * up            → [1,18432]
y = z @ W_down： [18432,7168]  → [1,7168]
x_next = u + y                → [1,7168]
```

中间维度 18432 是一层 dense FFN 的宽度，不能套到后续单个 MoE expert。

### 6.2 后 58 层的 router 与 expert

路由输入仍是 f=`[1,7168]`：

```text
f @ W_router：[7168,256] → logits：[1,256]
sigmoid                 → affinities：[1,256]
加 selection bias、分组筛选、选择专家
                        → expert IDs：[1,8]
取原 affinities 并归一化、缩放
                        → mixing weights：[1,8]
```

V3 配置把 256 个 experts 分为 8 组，保留 4 组后再选 8 个 experts；不是对原始 logits 无条件直接 top-k。选择偏置影响选谁，实际输出权重来自未加该偏置的 affinities。权重归一化后还乘 routed scaling factor 2.5。

每个选中的 expert e 独立计算：

```text
W_gate[e]：[7168,2048]
W_up[e]：  [7168,2048]
W_down[e]：[2048,7168]

[1,7168] → 两个 [1,2048] → silu × up → [1,2048] → [1,7168]
```

每个 expert 的三张大矩阵合计 `3×7168×2048=44040192` 个参数，约 44.0M；这是静态矩阵计数，不含路由器。一个 MoE 层同时存有 256 个 routed experts，但当前 token 只使用其中 8 个，再使用一个同宽度的 shared expert。

$$
y_{\mathrm{MoE}}=\sum_{e\in\mathrm{selected}}w_e\operatorname{Expert}_e(f)+\operatorname{SharedExpert}(f).
$$

所有 expert 输出都是 `[1,7168]`，加权求和后仍是 `[1,7168]`。最后加 u，交给下一层。

### 6.3 第 61 层之后

```text
主干输出       [1,7168]
Final RMSNorm [1,7168]
@ W_LM        [7168,129280]
logits        [1,129280]
```

Embedding 与 LM head 没有绑定同一份权重；`tie_word_embeddings=false`。采样器读取 logits，例如应用 temperature 和 top-p 设置来产生下一个 ID。这一步在模型主干之外。

## 7. V3.2：在同一 MLA 主干旁增加 indexer

### 7.1 三组额外投影

Indexer 复用主 attention 的归一化 Q latent，但使用自己的上投影：

| 对象 | 运算 | 输出 |
| --- | --- | --- |
| Indexer query | `[1,1536] @ [1536,8192]`，reshape | `[64,128]` |
| 当前 indexer key | `[1,7168] @ [7168,128]`，LayerNorm | `[1,128]` |
| Head combination weights | `[1,7168] @ [7168,64]` | `[1,64]` |

Indexer key 的 norm 在参考实现中是 LayerNorm，不是主 latent 的 RMSNorm。Indexer 还包含 RoPE、Hadamard 旋转、低精度量化和 scale；这些步骤保持逻辑维度。写入后额外 key cache 为 `[T,128]`，物理存储还有量化 scales。

### 7.2 扫描历史并归并 indexer heads

```text
[64,128] @ [128,131073] → per-head index scores：[64,131073]
ReLU、按 head 加权、沿 head 求和 → [131073]
top-2048                         → selected IDs：[2048]
```

选出的集合供主 attention 的 128 个 Q heads 使用，不是 64 个 indexer heads 各保留一套 2048。长序列中选择数为 2048；历史不足时取因果可见的有效位置。

### 7.3 主 attention 在所选集合上计算

沿用 V3 的 Q_absorb=`[128,512]`、Q_R=`[128,64]`：

```text
C_selected：[2048,512]
R_selected：[2048,64]

内容分数：[128,512] @ [512,2048] → [128,2048]
位置分数：[128,64]  @ [64,2048]  → [128,2048]
相加、缩放、softmax             → P：[128,2048]
P @ C_selected                  → [128,512]
逐头 V up                       → [128,128]
flatten、W_O                    → [1,7168]
```

这里是选中集合上的数学形状，不要求显式 gather 一份完整 `C_selected`。高性能 kernel 可通过位置索引直接读取原 cache。

[官方教学参考实现][v32-code]会先计算完整 `[128,T]` scores，再把未选中的位置 mask 成负无穷。其 prefill 分支还会暂时展开 K/V。不能把该表达方式当作生产稀疏 kernel 已减少计算量的证据。

Embedding、61 层、前三层 dense FFN、后续 MoE、普通 residual 和 LM head 的尺寸都沿用上一节。每层长期缓存为 `[T,512]`、`[T,64]`、额外 `[T,128]` indexer keys；未被选中的位置并未永久删除。

## 8. V4-Pro：Embedding 扩展为四条 residual streams

### 8.1 全模型入口

```text
input IDs：[1,1]
Embedding：[1,1,7168]
增加 stream 轴并复制 4 份：[1,1,4,7168]
```

4 份表示起初相同，之后在各层混合中形成不同状态。单 token 简写为 X=`[4,7168]`。

### 8.2 mHC 产生 attention 子层输入

控制分支：

```text
X flatten：[1,28672]
按 RMS 尺度归一化后投影：@ [28672,24] → [1,24]
加静态 bias、应用 scale 和约束
    ├─ pre： [4]
    ├─ post：[4]
    └─ comb：[4,4]
```

`28672=4×7168`，`24=4+4+4×4`。随后对原 residual streams 加权：

```text
pre：[1,4] @ X：[4,7168] → [1,7168]
Attention RMSNorm      → a：[1,7168]
```

控制分支使用展平状态的归一化尺度；子层输入是对 streams 的加权结果，随后再经过 attention norm。不要把这两处归一化当成同一操作。

## 9. V4-Pro：Q 和局部共享 K/V

### 9.1 Q 的实际宽度

```text
a：[1,7168]
@ W_Q_down：[7168,1536] → [1,1536]
Q latent RMSNorm       → [1,1536]
@ W_Q_up：[1536,65536] → [1,65536]
reshape                → [128,512]
逐头 RMS 归一化         → [128,512]
最后 64 维施加 RoPE      → [128,512]
```

`65536=128×512`。内容分量 448 维，RoPE 分量 64 维。这不是 V3 的“内容 Q 128 维吸收后变 512，再附加 64”；V4 直接以总宽度 512 做主 attention。

### 9.2 局部窗口条目

```text
a：[1,7168]
@ W_KV_window：[7168,512] → [1,512]
RMSNorm、最后 64 维 RoPE → kv_new：[1,512]
写入环形窗口             → window_cache：[128,512]
```

窗口包含当前 token 与最近 127 个 token。序列开头不足 128 时，只读有效条目。同一个条目同时作为 key 和 value，所有 Q heads 共用它。

实际参考实现会模拟非 RoPE 分量的低精度误差；最后 64 维的位置分量保留更高精度。这不改变上述逻辑宽度。

## 10. V4-Pro CSA：压缩器、indexer 与读取集合

### 10.1 两组候选与 pooling logits

CSA 压缩器有独立于局部窗口的权重：

```text
value 候选：[1,7168] @ [7168,1024] → [1,1024]
pooling logits：[1,7168] @ [7168,1024] → [1,1024]
1024 = 两组 × 每组 512 维
```

两组分别服务相邻块的重叠连接。一个完整摘要结合前块的一组和当前块的另一组：

```text
候选值 U：[8,512]
带位置偏置的 logits Z：[8,512]
alpha = softmax(Z, dim=0)：[8,512]
summary = sum(alpha * U, dim=0)：[512]
```

再做 512 维 RMSNorm 和最后 64 维的位置编码。第一个摘要缺失前块时，对缺失位置屏蔽。

摘要输出步长为 4，因此历史压缩 cache 在本例中为：

```text
C_comp：[32768,512]
```

当前 t=131072 是下一组第一个位置，只更新待完成组的状态，不增加摘要。Decode 在组边界增量生成摘要，不会每步从全部历史 hidden 重建。

### 10.2 Indexer 自己的压缩 keys

CSA indexer 使用相同的步长与重叠结构，但 key 宽度为 128：

```text
indexer 候选、logits：各 [1,256]，256=2×128
摘要前逻辑候选：        [8,128]
新 indexer key：       [128]
indexer compressed cache：[32768,128]
```

Indexer Q 来自同一个 1536 维 Q latent：

```text
[1,1536] @ [1536,8192] → reshape → [64,128]
```

然后计算：

```text
[64,128] @ [128,32768] → [64,32768]
ReLU、head 加权与求和   → [32768]
top-1024               → [1024] 个摘要编号
```

### 10.3 当前 query 的读取集合

```text
所选长程摘要：[1024,512]
局部窗口：  [128,512]
逻辑合并：  [1152,512]
```

窗口与摘要可以覆盖部分相同原始位置；它们是不同表示，不按原始 token ID 简单去重。未选中的摘要继续保留，供后续 query 选择。

## 11. V4-Pro HCA：强压缩后读取全部摘要

HCA 压缩器不使用 CSA 的相邻块重叠：

```text
每 token 候选：[1,7168] @ [7168,512] → [1,512]
每 token logits：[1,7168] @ [7168,512] → [1,512]
```

完整组中：

```text
U、Z：各 [128,512]
沿 128 个 token 做 softmax 和加权求和 → [512]
RMSNorm、位置编码                    → [512]
```

已有 131072 个历史 token，所以本例 cache 为 `[1024,512]`。当前 token 还没完成下一组，只更新压缩状态。

HCA 没有 indexer，读取全部可见摘要：

```text
长程摘要：[1024,512]
局部窗口：[128,512]
读取集合：[1152,512]
```

Pro 的主干配置是 31 层 HCA、30 层 CSA；前两层 HCA，随后交错。`compress_ratios` 的第 62 项服务额外 MTP，不应误算成第 62 个普通 block。[Pro 配置][v4-config]

## 12. V4-Pro：attention sink、加权与 grouped output projection

### 12.1 主 attention

令当前层读取集合为 C_read=`[1152,512]`：

```text
Q @ C_read.T
[128,512] @ [512,1152] → scores：[128,1152]
乘 1/sqrt(512)
```

每个 head 有一个 sink logit，参数形状 `[128]`：

$$
p_{h,j}=\frac{\exp(\ell_{h,j})}{\sum_r\exp(\ell_{h,r})+\exp(b_h)}.
$$

sink 只增加分母，真实条目权重和可以小于 1。稳定实现会用减最大值等方法计算，不直接依赖未经稳定处理的指数。

```text
P：[128,1152]
P @ C_read：[128,1152] @ [1152,512] → O：[128,512]
O 最后 64 维应用当前 query 位置的逆 RoPE → [128,512]
```

没有 V3 中每头 `[512,128]` 的 V 恢复步骤。原始窗口和长程摘要共同归一化；分开计算的 kernel 也必须正确合并统计量。

### 12.2 16 组输出投影

```text
O：[128,512]
reshape：[16,4096]，每组 8 个 heads
```

每组使用自己的 `[4096,1024]` 矩阵：

```text
逐组投影：[16,4096] → [16,1024]
flatten：            [1,16384]
最终投影：@ [16384,7168]
attention_output：  [1,7168]
```

分组权重的逻辑形状 `[16,4096,1024]`。参考代码可能先存成 `[16384,4096]` 再 view 为 `[16,1024,4096]`，这来自 PyTorch 的 `[out,in]` 布局，不改变分组语义。

这一设计降低直接对 65536 维拼接结果做大输出投影的参数与计算开销。它是模型学出的低秩参数化，不是对任意完整 W_O 做无损替换。

## 13. V4-Pro：两次 mHC 写回与 MoE

### 13.1 Attention 写回四条流

设 attention 输出 y=`[1,7168]`，旧状态 X=`[4,7168]`：

$$
X'=BX+Cy.
$$

```text
B：[4,4] @ X：[4,7168] → [4,7168]
C：[4,1] @ y：[1,7168] → [4,7168]
相加                   → [4,7168]
```

这里 B 采用本文“输出流 × 输入流”的矩阵约定。实现中的 `comb` buffer 可以存其转置顺序，需按实际 contraction 轴解释。

### 13.2 FFN 入口与路由

使用另一套 mHC 参数，从 X' 混合出 `[1,7168]`，再经过 FFN RMSNorm：

```text
f：[1,7168]
@ router：[7168,384] → [1,384]
sqrt(softplus(logits)) → affinities：[1,384]
```

普通路由层按带选择偏置的分数选 6 个 experts；权重由原 affinities 取出、归一化，再乘 2.5。前 3 层用 token ID 查固定表得到 6 个 expert IDs，仍用计算出的 affinity 形成权重。

```text
expert IDs：    [1,6]
expert weights：[1,6]
```

### 13.3 Expert 的 SwiGLU

每个 routed expert：

```text
W_gate：[7168,3072]
W_up：  [7168,3072]
W_down：[3072,7168]

f：[1,7168]
  ├─ gate：[1,3072]，上限 clamp 到 10
  └─ up：  [1,3072]，clamp 到 [-10,10]
z = silu(gate) * up：[1,3072]
y = z @ W_down：    [1,7168]
```

每个 expert 的三张逻辑权重共 `3×7168×3072=66060288` 个参数。六个 routed outputs 加权求和，再加一个 shared expert，得到 `[1,7168]`。参考实现可把路由权重乘在 down projection 前，利用线性性得到等价的输出加权。低精度舍入细节则取决于实现。

V4-Pro 的发布配置中 MoE expert 权重使用 FP4，其他部分混用 FP8/BF16。这不是说中间 gate/up、归一化或所有累加都用 FP4。[模型卡][v4-card]

FFN 对应的 mHC 再把 `[1,7168]` 写回四条流。下一个 block 接收 `[4,7168]`。以上对应[参考代码][v4-code]中的 `Block.hc_pre → Attention/MoE → Block.hc_post`。

## 14. V4-Pro：最终合并与 logits

61 层之后得到 `[4,7168]`。Final hyper-head 从当前四条流生成 4 个合并系数：

```text
flatten：[1,28672]
归一化尺度、投影 @ [28672,4]、偏置与 sigmoid → [1,4]
对四条原状态加权求和                          → [1,7168]
Final RMSNorm                                → [1,7168]
LM head @ [7168,129280]                       → [1,129280]
```

Final hyper-head 是独立的学习模块，不应直接假设取四条流的平均。采样器从 logits 选择下一 token，下一轮从 embedding 再开始。

## 15. Prefill：把一个新 token 换成 S 个

### 15.1 主要 activation 的维度

| 对象 | V3 / V3.2 | V4-Pro |
| --- | --- | --- |
| 输入 ID | `[B,S]` | `[B,S]` |
| Embedding | `[B,S,7168]` | `[B,S,7168]` |
| 层间 residual | `[B,S,7168]` | `[B,S,4,7168]` |
| 子层输入 | `[B,S,7168]` | `[B,S,7168]` |
| Q latent | `[B,S,1536]` | `[B,S,1536]` |
| 主 Q | `[B,S,128,192]` | `[B,S,128,512]` |
| Attention 输出 | `[B,S,7168]` | `[B,S,7168]` |
| FFN 输入/输出 | `[B,S,7168]` | `[B,S,7168]` |

没有已有前缀时，一种 V3 prefill 路径暂时展开 K/V：

```text
Q：[B,128,S,192]
K：[B,128,S,192]，其中位置分量由共同的 key 扩展
V：[B,128,S,128]

Q @ K.transpose(-1,-2) → [B,128,S,S]
加 causal mask、softmax
@ V → [B,128,S,128]
```

这是数学形状。FlashAttention 不必显式保存 `[S,S]` 分数矩阵。暂时展开 K/V 也不意味着必须把它们作为下一轮 decode 的持久缓存。Prefill 与 decode 可以选择不同计算路径，只要缓存契约与数学语义一致。

### 15.2 每个 query 有自己的可见集合

V3.2 在每个 query 位置选择自己的 top-k；开头的位置少于 2048。V4 的每个位置只能读取已完成、因果可见的压缩组，以及自己的局部窗口。不能先把整段 prompt 压成摘要，再让开头 query 无条件读取所有摘要。

以 T=S=131072、单请求为例：

| 对象 | 逻辑形状或上界 |
| --- | --- |
| V3.2 全历史 indexer 汇总分数 | `[B,S,S]`，可分块执行 |
| V3.2 选择编号 | `[B,S,2048]`，开头含无效槽位或使用有效长度 |
| V4 CSA 主摘要 / indexer 摘要 | `[B,32768,512]` / `[B,32768,128]` |
| V4 CSA indexer 汇总分数 | `[B,S,32768]`，带因果 mask，可分块执行 |
| V4 CSA 主读取集合上界 | 每个 query 128 个局部条目 + 1024 个摘要 |
| V4 HCA 主摘要 | `[B,1024,512]` |
| V4 HCA 主读取集合上界 | 每个 query 128 个局部条目 + 1024 个摘要，开头更少 |

表中的矩阵是计算域，不要求一次物化。CSA indexer 仍可能有随 S 二次增长的扫描工作；HCA 的摘要数也随 S 增长。

### 15.3 FFN 始终按 token 变换特征

线性层可先合并 batch 与 sequence：

```text
[B,S,7168] → [B*S,7168]
```

MoE 路由后，第 e 个 expert 收到 N_e 个 token：

```text
V3 / V3.2：[N_e,7168] → [N_e,2048] → [N_e,7168]
V4-Pro：  [N_e,7168] → [N_e,3072] → [N_e,7168]
```

N_e 随路由变化。多个 token 被放在同一个 GEMM 中，不表示 FFN 在这些 token 之间做 attention。每一行依旧独立变换，完成后按原 token 位置组合输出。

普通生成的 prefill 通常只需最后位置 logits=`[B,129280]`。训练或返回所有位置分数时，逻辑 logits 才是 `[B,S,129280]`。训练还需要 loss、backward 和 optimizer；本文没有把它们当成推理 forward 的步骤。

## 16. 下一轮究竟保留什么

| 每层、单请求 | 随历史保留的状态 | 本次临时状态 |
| --- | --- | --- |
| V3 | `[T,512]` latent、`[T,64]` RoPE key | 当前 Q、latent 加权结果、hidden、FFN 中间结果 |
| V3.2 | V3 cache + `[T,128]` indexer keys 与 scales | indexer Q/分数/选择结果，以及主干临时 activation |
| V4 CSA | 约 `[T/4,512]` 主摘要、`[T/4,128]` indexer 摘要、局部窗口、压缩状态 | 当前 Q、输出、四条 residual streams |
| V4 HCA | 约 `[T/128,512]` 主摘要、局部窗口、压缩状态 | 当前 Q、输出、四条 residual streams |

缓存的 T/r 要按实际完成组数取整。V4 的未完成组与 CSA 前邻重叠组需要有限状态，不能只数主摘要就声称是完整 cache 大小。参考代码的主压缩状态 buffers 在 CSA 为 `[B,8,1024]` 的候选和同形状 logits，HCA 为 `[B,128,512]`；CSA indexer 另外有 `[B,8,256]` 状态。它们是参考实现的布局，不是压缩算法的唯一最小存储方式。

**历史 hidden 不需要为了下一个 token 的 FFN 而永久保留。** 第 j 个历史 token 已经完成各层 FFN；当前 token 经过某层时，读取该层 cache，再只计算自己的 FFN。因果模型里追加未来 token 不会改变历史位置已经算出的输出。

CUDA Graph 保留某块 activation buffer 的地址，不代表里面永久保存全序列历史 hidden。训练反向、返回 hidden states、EAGLE 类草稿模型或特定服务接口可能额外保留 activation，那是另一份状态契约。[MLA 与 hidden 生命周期](../Attention/model-architectures.md#4-mla压缩每个-token-的表示不压缩-token-数)

## 17. 权重、cache、activation 与执行路径对照

| 容易混淆的对象 | 实际角色 |
| --- | --- |
| `W_Q_down`、`W_KV_up`、expert 权重 | 训练学得，所有请求复用；普通推理期间不更新 |
| `q_latent`、`Q_absorb` | 当前输入的临时计算结果 |
| `C_cache`、`R_cache` | 某个请求在某一层保留的历史信息 |
| V4 压缩器状态 | 等待凑齐组或服务重叠压缩的有限状态 |
| Indexer top-k | 当前 query 的读取位置选择；不是删除其余缓存 |
| mHC 的 4 | 当前 token 的 residual stream 轴；不是 token 或 head 轴 |
| Muon | backward 后的权重更新算法；不进入普通推理 forward |
| MTP | 额外预测模块；不应计作普通主干第 62 层 |

这些尺寸说明数据怎样流动，不直接给出 kernel 数、launch 次数或实测延迟。服务系统可能融合 norm/projection、选择不同 prefill/decode backend，并把权重和缓存分片。继续分析性能时，应先确认这些逻辑张量在各 rank 上的归属，再看实际 kernel 和通信路径。

## 18. 官方来源与配置快照

- [完整配置快照](../assets/deepseek-configs.json)：V2、V3、V3.2、V4-Flash、V4-Pro 的固定 revision 与原始配置。
- [V3 配置][v3-config]、[V3 参考代码][v3-code]、[V3 技术报告](https://arxiv.org/html/2412.19437v1)。
- [V3.2 配置][v32-config]、[V3.2 参考代码][v32-code]、[DSA 技术报告](https://arxiv.org/html/2512.02556v1)。
- [V4-Pro 配置][v4-config]、[V4-Pro 参考代码][v4-code]、[V4 技术报告](https://arxiv.org/html/2606.19348v1)。
- [架构演进与机制推导](evolution.md)：说明这些尺寸背后的设计变化。

[v3-config]: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json
[v3-code]: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/inference/model.py
[v32-config]: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/config.json
[v32-code]: https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/a7e62ac04ecb2c0a54d736dc46601c5606cf10a6/inference/model.py
[v4-config]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/config.json
[v4-code]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py
[v4-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/README.md
