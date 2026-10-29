# 特殊结构详解：哪些模型在用，数据怎样流动

## 1. 先把模块放回各自的模型

| 机制 | 本文对应的模型或工作 | 改变 Transformer 的哪个部分 |
| --- | --- | --- |
| Gated Residual（GR） | **Qwen3.8-Flash-Next** | 每个 attention/FFN 子层怎样读取和更新残差状态 |
| Block Attention Residuals（Block AttnRes） | **Kimi K3** | 当前子层怎样读取此前不同深度的表示 |
| Stable LatentMoE | **Kimi K3** | routed experts 在多宽的特征空间里计算 |
| N-gram Embedding | **Qwen3.8-Flash-Next** 的条件记忆模块；也是一类通用方法 | 用相邻 token 的组合查可学习参数表 |
| Engram | **DeepSeek 与北京大学的 Engram 论文**提出的条件记忆模块；**DeepSeek-V4.1-Flash** 也采用 Engram | 在 n-gram 查表之后，根据上下文融合记忆 |
| MoE、MTP | 截图中指 **Qwen3.8-Flash-Next** 的专家与多 token 预测模块；这些名称本身并非它专有 | 分别改变 FFN 计算和预测/草稿生成路径 |

Qwen3.8-Flash-Next 的 GR、n-gram 模块是一组设计；Kimi K3 的 Block AttnRes、Stable LatentMoE 是另一组设计。**这些机制没有全部装在同一个模型里，也不能按厂商名字推广到该厂商的全部模型。** 例如，Flash-Next 的 GR 不能直接套到其他 Qwen3.8 配置；K3 的结构也不能直接套到 K2。完整版本对照见[模型结构](model-architectures.md)。[Qwen 报告][q-report]、[K3 报告][k-report]、[DeepSeek-V4.1 模型卡][ds41-card]

可以先把它们放进熟悉的 Transformer：attention 让不同位置的 token 交换信息，FFN 对每个 token 的特征做变换，residual 将表示交给后续层。GR 和 AttnRes 改 residual 的组织方式；LatentMoE 改 FFN；n-gram embedding 和 Engram 增加一条查参数表的路径。

下文用 **D 表示一个 token 的 hidden 宽度，M 表示这次一起处理的 token 数**。因此 `[M,D]` 是一批 token 的表示。其余维度在出现时解释，形状只用于看清哪个轴发生了变化。

## 2. Qwen3.8-Flash-Next 的 GR：多份状态，分别控制读取和写入

### 为什么要改普通 residual

普通残差可以理解为：attention 或 FFN 先根据当前 hidden 算出一个增量，再把增量加回旧 hidden。这个 token 继续向深层流动时，前面各层的信息已经混在同一份 D 维向量里。

这条路径让模型容易保留旧信息，但也限制了信息的组织方式：每一层的新结果都加入同一个状态。GR 的想法是给 residual 更多保存空间，再让当前子层决定从哪里读、往哪里写。

Flash-Next 为一个 token 保留 **4 个 D 维残差分支**。四个分支都是同一个 token 的状态，会在训练中学出不同内容；它们没有预先指定的“语法”“知识”分工。[Qwen GR 设计][q-report]

```text
同一个 token 的 4 份状态 [4,D]
           ↓ 按特征门控读取
一个子层输入 [D]
           ↓ attention 或 FFN
一个子层输出 [D]
           ↓ 各分支控制写入量
更新后的 4 份状态 [4,D]
```

这里增加的轴是“残差分支”。序列长度没有增加，也没有新增四个专家；attention 或 FFN 仍只接收一个 D 维输入。

### 读门决定每份状态里的哪些特征进入当前计算

读取前，GR 分别对四个分支做 RMSNorm。归一化提供适合后续处理的数值尺度；原始状态仍然保留，后面写回时还要使用。

一个小网络观察四个归一化分支，生成读门。**读门形状是 `[4,D]`：每个分支的每个特征都有自己的系数。** gate 的作用是逐元素缩放。例如，当前子层可以多读取第一分支的一组特征，同时少读取该分支的另一组特征。

四份缩放后的向量沿“分支轴”相加，再按分支数缩放，形成 `[D]` 输入。D 个特征位置仍然保留，消掉的是那个长度为 4 的分支轴。gate 用 sigmoid 生成，各个系数处于 0～1；这里没有对四个分支做 softmax，因此无需把它理解成四选一的概率分布。

读门随当前状态变化，不是一组所有 token 共用的固定混合比例。生成 gate 的网络参数由训练得到，执行时由这些参数和当前状态共同算出 gate。

### 写门决定同一个新结果对各分支影响多大

子层输出一个 `[D]` 向量后，GR 为每个分支生成一个写门。**写门形状是 `[4]`，每个分支一个标量**，与逐特征的读门不同。

更新规则只需记住这一行：

```text
某分支的新状态 = 该分支的旧状态 + 该分支的写门 × 子层输出
```

同一个输出会写入四个分支，但写入量可以不同。写门接近零时，该分支主要保留旧信息；写门较大时，本次子层结果影响更强。Flash-Next 的写门范围是 0～2，它表示更新幅度，不是概率。写回使用原始 residual，而不是刚才用于读取的归一化副本。

下一子层会看到更新后的四份状态，并通过自己的读门重新组合。attention 和 FFN 子层分别有自己的 GR，因此“读一次、算一个子层、写一次”的过程会沿网络深度重复。

### 扩宽的是残差状态，子层计算宽度仍然是 D

批量情况下，residual 从 `[M,D]` 变成 `[M,4,D]`，子层输入和输出仍是 `[M,D]`。这解释了为什么 GR 可以增加状态容量，而无需把所有 attention/FFN 矩阵一并扩成四倍宽。

代价是要读取和更新更多状态，还要计算归一化与 gate。它主要增加 residual 路径的存储和访存，不能由此断言整个模型显存或运行时间变成四倍。具体开销取决于这些步骤能否与相邻 kernel 融合。

## 3. Kimi K3 的 Block AttnRes：让后面的层选择读取前面的成果

### 这里的 attention 沿网络深度选择信息

GR 维护固定的四份状态。K3 的 Block AttnRes 采取另一种做法：把此前深度块的结果分别留下，让当前子层按需要读取。

先固定一个 token 的位置。它经过浅层、中层和深层时，会形成不同表示。普通 residual 将各次更新不断相加；AttnRes 让后面的子层有机会单独访问这些来源，再决定各自的影响大小。[K3 AttnRes 设计][k-report]

```text
普通 token attention：
当前 token ← 序列中的其他 token

K3 的 Block AttnRes：
当前子层 ← 这个 token 的 embedding、此前深度块、当前块的部分结果
```

例如，序列中的第 20 个 token 正在经过某个深层模块。AttnRes 读取的候选仍然属于第 20 个 token，只是它们在不同深度产生。候选本身可能已通过前面的 attention 融入其他位置的信息，但 **AttnRes 这次加权的轴仍然是深度来源**。

### 一次读取怎样发生

假设当前可读 J 份候选，每份宽度都是 D。它们在逻辑上可以排成 `[J,D]`。

当前子层有一个训练得到的 `[D]` 向量，论文称为 pseudo-query。它与每份归一化候选做点积，得到 J 个分数；softmax 把分数变成 J 个权重，再对候选原始向量做加权求和。输出仍是一份 `[D]` 向量，交给当前 attention 或 FFN。

```text
J 份候选 [J,D]
     ↓ 每份候选得到一个分数
分数 [J] → 沿 J 做 softmax → 权重 [J]
     ↓ 对候选加权求和
当前子层输入 [D]
```

pseudo-query 是当前子层的参数，在普通推理时固定；候选内容随 token 变化，因此同一个子层对不同 token 可以算出不同权重。归一化用于打分，避免仅因某份表示数值尺度大就占优势；被加权的 value 仍是候选本身。

这里保留了 attention 的“打分、softmax、加权和”结构，但不需要照搬序列 attention 的完整 Q/K/V 投影接口。

### 为什么用 Block，而不是保留每个子层输出

如果将每个子层的结果都单独留下，后面的层就有更多来源可选，但前面的结果也必须一直存活到被读取。深度增加后，保存、读取和跨流水线阶段传输这些结果都会增加成本。

Block AttnRes 把连续的网络层分成组，每组内部的子层输出累加成一份块表示。它保留组与组之间的区别，省去逐子层保留全部结果的开销。

跟着一个 token 走到第三个深度块中途，可以看到这些候选：

```text
embedding：最初的输入表示
第一块：第一块内各子层输出的和
第二块：第二块内各子层输出的和
第三块部分结果：本块已执行子层输出的和
```

当前子层从这些候选读出输入，完成自身计算，再把新输出加到第三块的部分结果中。第三块结束后，这份结果就成为后续块可读的完整候选。刚进入一个新块、尚无子层结果时，不需要加入空的部分结果。

**块表示累加的是该块产生的子层输出。** 不能把每层已经包含全部旧 residual 的 hidden 再全部相加，否则旧信息会被重复计入。K3 报告以 12 层为一组组织 93 层主干，得到 8 个深度块，embedding 另外保留。这里的 block 是网络深度分组，与 CUDA thread block、序列 token 分块无关。

### 与 GR 相比，它保留了什么

| 观察点 | Qwen3.8-Flash-Next：GR | Kimi K3：Block AttnRes |
| --- | --- | --- |
| 当前保存的状态 | 固定 4 个持续更新的分支 | embedding、已完成深度块、当前块部分结果 |
| 来源是否对应一段明确深度 | 分支持续接收各层更新，不对应固定层段 | 已完成块对应明确的网络层段 |
| 怎样读取 | 每分支、每特征的 gate | 每个深度候选一个 softmax 权重 |
| 怎样写入 | 同一个输出按不同幅度加到各分支 | 新输出加到当前深度块 |

分块也有取舍：一个块内部的各次输出已相加，后续层无法再分别选择它们。块越细，可单独访问的来源越多，但保存和读取的负担也越大。

这些候选用于当前 token 沿深度的计算，不是为以后生成的所有 token 保存的 KV cache。训练还可能为 backward 保留激活，但那是另一种生命周期。

## 4. Kimi K3 的 Stable LatentMoE：在较窄空间里运行 routed experts

### 从普通 MoE 的一次 FFN 开始

普通 MoE 先用 router 为当前 token 选专家，再把 token 表示送给这些专家。每个专家都是一个 FFN，输出按路由权重合并，形成这个 token 的 FFN 结果。

如果每个专家都直接读写主干宽度 D，那么模型扩大主干宽度时，专家的输入和输出矩阵也会变宽。专家池很大时，这会增加整套专家参数和数据搬运的负担。

K3 用 Stable LatentMoE 将两种宽度分开：**主干继续使用宽表示，routed experts 在较窄的 latent 表示上工作。** 此处 latent 指经过可学习投影形成的中间特征，不是截掉 hidden 的后半段。[K3 Stable LatentMoE][k-report]

### 沿一个 token 跟完 routed 路径

K3 主干 hidden 宽度是 **7168**，routed latent 宽度是 **3584**。它有 896 个 routed experts，每个 token 选 16 个，另有 2 个 shared experts。[K3 配置][k-config]

```text
主干输入 [7168]
     ↓ 一个共享的下投影
latent 输入 [3584]
     ↓ 送给 router 选中的 16 个专家
每个专家输出 [3584]
     ↓ 按路由权重合并
合并结果 [3584]
     ↓ RMSNorm，再上投影
routed 路径输出 [7168]
     ↓ 与 full-width shared 路径相加
FFN 输出 [7168]
```

“共享的下投影”表示这一步由 routed 路径共同使用，当前 token 不必为每个被选中的专家各做一套独立压缩。router 决定这个 token 交给哪些专家；latent 向量是专家实际执行 FFN 时读到的输入。

每个专家内部仍然有完整的门控 FFN：先从 latent 宽度投影到更宽的 intermediate，经过门控非线性，再投影回 latent 宽度。因此存在两组不同的“缩放宽度”：

- **专家外部**：主干宽度 → latent 宽度 → 主干宽度，规定专家工作在哪个空间。
- **专家内部**：latent 宽度 → intermediate 宽度 → latent 宽度，完成 FFN 的特征变换。

外部的下投影和上投影没有替代专家计算。它们为专家提供窄接口，专家依然在这个接口之间完成非线性变换。

### 为什么先合并，再恢复主干宽度

选中的专家都返回同样宽度的 latent 向量，所以可以先在这个空间中加权合并。合并之后只需对结果执行一次 RMSNorm 和上投影，就能回到主干宽度。

顺序很重要：这里是“专家输出合并 → RMSNorm → 上投影”。如果改成每个专家各自做 RMSNorm，再合并结果，就改变了函数，因为归一化依赖被归一化向量的整体数值。

shared experts 则走另一条路径，直接使用主干宽度的输入和输出，最后与 routed 结果相加。shared 指每个 token 都经过它们；routed 指每个 token 只选择专家池中的一部分。全宽 shared 路径保留通用的特征变换能力，但不能由此认定 latent 压缩没有信息损失。

### Stable 指哪些稳定性处理

窄空间并不自动意味着容易训练。新增投影与专家内部矩阵串联后，中间数值可能放大；专家选择也可能不均衡。K3 将几项处理组合在 Stable LatentMoE 中：

| 处理 | 作用位置 | 为什么要做 |
| --- | --- | --- |
| RMSNorm | 专家输出合并后、上投影前 | 控制合并结果的数值尺度 |
| SiTU-GLU | 专家内部的门控激活 | 对两个投影分支的激活做平滑限制，抑制极端值 |
| Quantile Balancing | 专家路由的负载调节 | 减轻某些专家过忙、另一些专家很少被选中的不均衡 |

SiTU-GLU 可以接在已学过的 SwiGLU 后理解：仍然是两路投影、非线性门控、逐元素相乘，但加入基于 tanh 的平滑限制。它约束中间激活，不意味着后续线性层输出也有相同的硬上限。这里先记住它解决数值稳定性，具体公式留在[原报告 §2.3][k-report]按需查阅。

### 它省什么，又增加什么

专家读写的特征变窄后，在其他尺寸相同的前提下，专家输入和输出矩阵可以更小。若分布式实现也按 latent 宽度分发和合并 token，专家间通信需要搬运的特征也更少。

代价是新增下投影、上投影和归一化，并且 routed experts 只能利用压缩后保留下来的信息。是否更快仍取决于实际专家尺寸、批量、通信和 kernel 实现，不能把宽度减半直接等同于总延迟减半。

**LatentMoE 和 MLA 的 latent 指不同对象。** MLA 压缩 attention 的历史 K/V 内容；LatentMoE 缩小当前 FFN 的工作宽度。前者影响请求历史缓存，后者产生当前计算的中间激活。

## 5. 通用基础：n-gram embedding 是怎样查表的

### 先区分 token embedding 与组合 embedding

n-gram embedding 不是某一个模型的专有名称。下面先讲 Qwen 的模块和 DeepSeek Engram 都会用到的基本思路，再分别说明各自怎样融合结果。

普通 token embedding 根据一个 token ID 查表。假设参数表是 `[词表大小,D]`，选中其中一行，就得到这个 token 的 `[D]` 向量。相同 token ID 在这一阶段得到同一行，之后才通过网络获得上下文信息。

n-gram embedding 把“查表的键”从一个 ID 扩展为连续多个 ID：

```text
已读到的 token IDs：[…,17,42,9]

当前 token：9
bigram：   [42,9]
trigram：  [17,42,9]
```

n 表示组合包含几个 token，**不表示 embedding 向量有几维**。bigram 向量可以很宽，trigram 向量也可以与它一样宽。token 来自 tokenizer，不一定恰好对应一个汉字或一个完整单词。

组合以当前 token 结束，因此生成时只用已知内容。同样以 token 9 结尾，前面跟着不同 token，就能用不同的组合键查询参数。这样，模型可以直接取得一份与局部搭配有关的表示，而无需每次都仅靠多层计算重新构造这份局部信息。

### 查到的是向量，不是短语原文或下一词答案

可以把一张组合表理解成：

```text
局部 token 组合 → 表地址 → 一行训练得到的向量
```

参数表中的一行没有固定的人类可读含义。它可能帮助模型识别局部搭配或模式，但并不是一条写好的事实，也不是“看到这个短语就直接输出某个词”的规则。

n-gram embedding 与传统 n-gram 语言模型的输出不同。传统 n-gram 语言模型可以直接用短上下文估计下一个 token 的概率。这里查到的是特征，还要交给神经网络继续处理。

### 为什么先 hash，再查表

直接为所有可能的 token 组合各分配一行，会使表迅速变大。实际设计使用 hash：根据 token 组合算出一个有界整数，拿它作为有限参数表的行号。

hash 在这里是地址计算，不是神经网络判断语义相似度。不同组合可能映射到同一个槽位，这称为碰撞。发生碰撞时，它们会共享那一行参数，训练时也可能互相影响。

多头 hash 给同一个组合使用多组地址计算和参数表。某一张表发生碰撞时，其他表仍可能提供不同的向量，从而减少所有查询结果都无法区分的机会。它降低碰撞影响，但不保证完全没有碰撞。此处的 head 指 hash 查询分支，与 attention head 无关。

只看形状即可理解一次 lookup：

```text
一张参数表 [槽位数,每行宽度]
       ↑ 使用一个 hash 行号索引
取出一行 [每行宽度]

多个 order / hash head 的查询结果
       ↓ 按该模型定义拼接或组合
memory vector [dm]
```

dm 是组合后的记忆向量宽度，可以与主干 D 不同。后续投影负责把它变成适合主干融合的形状。

### 表里的内容什么时候改变

训练时，查表输出参与 loss 计算，梯度会回到被选中的参数行。索引和 hash 本身不需要可微，原理与普通 `nn.Embedding` 类似。

普通推理时，模型只读参数表。新请求不会自动改写表里的内容，模型也不会因为用户输入了一句新事实，就把这句话存进某一行。因此这里的“记忆”主要指训练保存在模型参数中的信息。

## 6. DeepSeek 的 Engram：查到局部记忆之后，决定此刻怎样使用

### 它属于谁，和 n-gram embedding 是什么关系

**Engram 是 DeepSeek 与北京大学提出的一套具体条件记忆设计。** 原论文的实验模型包括 Engram-27B 和 Engram-40B。后来的 DeepSeek-V4.1-Flash 模型卡也明确列出 Engram。不能据此把所有 DeepSeek 模型都标成使用 Engram。[Engram 原论文][engram]、[V4.1 模型卡][ds41-card]

n-gram embedding 解决“按局部 token 组合读哪份参数”。Engram 继续解决“这些静态参数在当前上下文里该起多大作用，以及怎样加入主干”。所以，n-gram lookup 是它的一部分，不是整个模块。

以下解释按 **Engram 原论文** 的设计展开，用来理解机制；不把原论文每个超参数都当成 V4.1 的逐项部署配置。Qwen 的实现则单独放在下一节。

### 第一步：从 token IDs 得到静态记忆向量

原论文先为查表路径建立规范化后的 token ID 映射，让规范化后等价的 tokenizer 条目能够复用参数。接下来构造当前位置结尾的 n-grams，通过多头 hash 读取向量，组合成 memory vector。

规范化只用于这条查表路径。主 token embedding 路径仍保留原 token 身份，因此不能理解成语言模型从此丢掉了大小写或空格差异。

给定相同的组合与固定参数，这一步查到的 memory vector 相同。它认识的是局部片段，尚未单独考虑当前整段上下文。

### 第二步：让当前 hidden 决定这份记忆有多适用

同一个局部片段可能出现在不同语境。查表得到相同向量，并不意味着每次都应该用相同幅度加入主干。Engram 因此引入上下文 gate。

模块把 memory vector 投影成 key 和 value：key 用于与当前 hidden 匹配，value 是准备注入的内容。这里的 k/v 是查表结果的投影，不是从历史 KV cache 读出的 k/v。

```text
查表向量 e [dm]
    ├→ key 投影 → k [D] ─┐
    └→ value 投影 → v [D] │
                          ↓
当前层 hidden h [D] → 匹配打分 → sigmoid gate
                                      ↓
                               缩放 value [D]
```

具体打分会先归一化 h 和 k，再做点积、尺度调整和 sigmoid。对这条融合路径，结果是一个 gate，控制整份 value 的注入幅度。gate 大表示这份局部记忆更适合当前表示，gate 小则减弱其影响。gate 并不保证具有可解释的“事实正确率”含义。

当前 h 已经过前面网络的处理，可以包含局部片段之外的上下文。因此，相同 n-gram 可以查到相同 e，却在两个上下文里得到不同 gate。这是静态参数表能够参与上下文相关计算的关键。

### 第三步：短卷积让相邻位置的记忆结果协作

原论文在门控之后使用短因果卷积、SiLU 和局部残差，再把结果加入主干。短卷积读取最近几个位置的 gated values，让相邻位置查到的模式能够进一步组合。

这里是按通道的卷积：可以先理解为“每个特征通道沿附近 token 位置做短距离处理”。它只读取当前和过去的位置，不需要对全部历史做一遍 attention。

最终加入主干的结果与 hidden 具有相同宽度 D，后续层继续使用融合后的表示。Engram 放在指定层；不用把它想成每个 Transformer 层都附带一张完整大表。[原论文融合路径][engram]、[官方示意代码][engram-code]

### 为什么它可以用很大的表，但每步只读很小一部分

参数表大小决定模型能存放多少可学习条目。一个 token 实际访问多少行，则由使用几种 n-gram、几个 hash head 决定。增加表的槽位数，并不要求每个 token 把整张表都读一遍。

这让“参数存储容量”和“单 token 参与计算的参数量”可以分开扩大。代价仍然存在：参数要放在某处，所需行要被读出、搬运，融合也需要计算。

另一个特点是地址由 token IDs 决定。当前 token 一旦已知，就能提前算表地址，不必等到深层 hidden 才知道访问哪一行。如果表放在 host memory，就可以尝试把读取和传输与前面几层计算重叠：

```text
当前 token 已知 → 算出地址 → 从 host 读取所需行 → 传到 GPU
                    前面几层继续计算 ──────────────┘
                                                   ↓
                                      到注入层时使用记忆向量
```

这是预取所需行，不是每步搬整张表，也不是提前知道未来 token。若数据没有及时到达，目标层仍会等待；是否能隐藏传输延迟取决于带宽、批量和可重叠的计算时间。

### 参数记忆与请求缓存分别保存什么

| 对象 | 内容从哪里来 | 普通生成时怎样变化 |
| --- | --- | --- |
| Engram 参数表 | 训练得到的向量 | 只读，多个请求共用 |
| 当前 lookup 与 gate 结果 | 本次查表、当前 hidden | 临时激活，用完后可释放或复用存储 |
| 最近 token IDs | 当前请求已生成的 token | 滚动保留，用于组成下一位置的 n-gram |
| 短卷积状态 | 当前请求最近的 gated values | 滚动更新 |
| attention KV cache | 当前请求各历史位置产生的 K/V | 按 attention 方案追加、压缩或淘汰 |
| RAG 文档 | 外部知识库 | 由检索系统管理，通常转成模型输入 |

所以，“表是静态的”和“整个模块没有请求状态”是两回事。Engram 仍需保存短历史来支持流式生成；它也不能替代 attention 对本轮对话内容的使用。

## 7. Qwen3.8-Flash-Next 的 n-gram 模块、MoE 与 MTP

### Qwen 的查表路径：与 Engram 相近，但不能逐项照搬

Flash-Next 同样使用多头 hash 和上下文门控，向 residual stream 注入组合 embedding。公开设计使用 bigram、trigram，并将 n-gram 模块放在第 2 层。[Qwen n-gram 设计][q-report]

这个插入位置有两层含义。它放得较早，后面的网络有机会反复利用查表信息。它又没有放在一切计算之前，因此前面的计算能为表数据预取提供重叠时间。这个位置是模型设计选择，不代表在所有硬件上都保证传输完全隐藏。

它与 Engram 原论文共享“局部组合查表，再按上下文融合”的思路，但不能把原论文的规范化、卷积或全部超参数直接套到 Qwen。Qwen 报告还说明，在其训练方案中尝试 token normalization 并未得到稳定收益。

### 截图中的 125B、51B 和 4B 分别是什么

| 参数项 | Qwen3.8-Flash-Next 的官方口径 | 应怎样理解 |
| --- | --- | --- |
| 主干 | 125B，总体激活口径 6B | 包括 attention 和 FFN；MoE 每个 token 只用部分专家 |
| N-gram embedding | 另有 51B | 全部查表参数的容量；一次只访问命中的行 |
| MTP | 另有 4B | 多 token 预测辅助模块的参数 |

这些是不同组成部分。不能把主干的 125B 当成全部部署参数，也不能看到 51B 表就认为每个 token 都要对它做一次完整 dense GEMM。[Flash-Next 模型卡][q-card]

### MoE 与查表，都选择部分参数，但执行的工作不同

MoE 的 router 读取当前 hidden，决定运行哪些专家网络。被选中的专家会对输入执行矩阵投影和非线性变换。

n-gram 模块由 token 组合算地址，取出已经训练好的向量，再做融合。它没有为每一行都运行一个专家 FFN。因此，“都是稀疏访问大参数池”只能说明相似点，不能据此把两者的计算成本看成相同。

### MTP 怎样参与训练与生成

MTP 是 Multi-Token Prediction。训练时，模型除了学习紧接着的 token，还通过辅助路径学习后续多个位置的预测，让表示承担更远的预测任务。

生成时，MTP 可以用于提出一段候选 token，再由 target 模型验证。通过验证的候选才成为正式结果；遇到不被接受的位置，需要按验证结果继续生成。

因此 MTP 提供的是可用于草稿生成的能力，不是保证主干每次前向都直接产出多个正确 token。加速效果取决于草稿质量、接受率和验证开销。它与 n-gram lookup 也没有同样的时间方向：lookup 使用已经知道的 token，MTP 尝试预测尚未确定的后续 token。

## 8. 按模型回看一次前向过程

### Qwen3.8-Flash-Next

token IDs 除了提供普通 embedding，还能在指定位置触发 n-gram 查表。残差状态以四个分支向深层传递。每个子层先由 GR 读出一份输入，交给 GDN/QSA 或 MoE 计算，再由 GR 将结果写回各分支。n-gram 模块在指定层注入记忆，MTP 则承担辅助预测及相应草稿路径。

这里，**GR 管残差读写，GDN/QSA 管序列信息，MoE 管特征变换，n-gram 管参数记忆，MTP 管额外预测**。它们处于不同位置，不能理解成几种相互替代的 attention。

### Kimi K3

当前子层通过 Block AttnRes，从 embedding、此前块和当前块部分结果中读出输入。KDA/MLA 处理序列方向的信息，Stable LatentMoE 处理 FFN 特征变换。每次子层产生新输出，都更新当前深度块，供后续模块读取。

这里，**Block AttnRes 选择深度来源，KDA/MLA 处理序列信息，Stable LatentMoE 规定专家的工作空间**。AttnRes 保存的深度候选、MLA 的历史缓存、LatentMoE 的中间激活虽然都是 tensor，但服务对象和生命周期不同。

### DeepSeek Engram 与 V4.1 的关系

Engram 论文给出条件记忆模块的完整机制，DeepSeek-V4.1-Flash 是公开采用 Engram 的具体模型之一。理解查表、门控和预取时可以从原论文入手；要复现 V4.1，则还需以它自己的配置和实现确定具体组织。

Engram 这一条参数记忆路径，与 V4.1 的 attention 和残差设计共同工作。不能因为它使用 Engram，就认为它也采用 Qwen 的 GR 或 Kimi 的 Block AttnRes。

## 9. 参考来源与阅读边界

本页按 2026-10-29 可核对的公开报告与配置解释结构。形状用于说明数据流，没有将结构推导当成 GPU 实测；未运行完整 checkpoint。

- [Qwen3.8-Next 报告 §2.2–2.3][q-report]：Flash-Next 的 GR 与 n-gram embedding。
- [Qwen3.8-Flash-Next 模型卡][q-card]：主干、n-gram、MTP 参数口径。
- [Kimi K3 报告 §2.2–2.3][k-report]：Block AttnRes 与 Stable LatentMoE。
- [K3 固定配置][k-config]：主干与 latent 宽度、专家配置。
- [Engram 原论文][engram]与[官方 demo][engram-code]：查表、上下文融合和预取机制。demo 不代表生产优化实现。
- [DeepSeek-V4.1-Flash 模型卡][ds41-card]：该模型采用 Engram；其余架构见[模型结构对照](model-architectures.md)。

[q-report]: https://arxiv.org/html/2608.30320v1
[q-card]: https://huggingface.co/Qwen/Qwen3.8-Flash-Next
[k-report]: https://arxiv.org/html/2607.24653v1
[k-config]: https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json
[engram]: https://arxiv.org/html/2601.07372v1
[engram-code]: https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py
[ds41-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash
