# 常见计算与组件的 Torch 实现

本页用 Tensor 运算展开常见组件的计算过程。函数、层对象、模型与参数管理先看 [神经网络基础](neural-networks.md)；shape、广播和存储布局查 [Tensor 与 GPU 内存](tensors.md)。

## nn.Embedding：按整数 ID 查表

`nn.Embedding(num_embeddings=V, embedding_dim=D)` 持有一张 `[V,D]` 的可学习表。输入里的 ID 指定行号，每个位置读取该行的 D 个浮点值。用于 token embedding 时，V 是词表大小，D 通常是模型的 hidden width。

| 对象 | shape / dtype | 含义 |
| --- | --- | --- |
| `embedding.weight` | `[V,D]`，浮点类型 | V 行向量，每行 D 个特征 |
| `token_ids` | `[B,S]`，整数类型 | B 条序列，每条 S 个 ID |
| `embedding(token_ids)` | `[B,S,D]`，与 weight 同 dtype | 为每个 ID 查出一条向量 |

索引可用 int32 或 int64，下面统一用 `torch.long`（int64）。ID 必须在 `[0,V)`，负数也不是合法 ID。这里让索引和权重位于同一 GPU；输出跟随权重的 dtype。把权重转成 BF16 时，索引仍保持整数类型。

对任意输入 shape，输出都在末尾增加 D：单个标量 ID 得到 `[D]`，`[S]` 得到 `[S,D]`，`[B,S]` 得到 `[B,S,D]`。固定权重时，同一个 ID 在不同位置读取同一行；后续位置处理和 attention 再让表示带上位置与上下文。[Embedding 接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)

### 查表原理与可手算的例子

令输入 ID 为 I、权重表为 W，则：

$$
Y_{b,s,d}=W_{I_{b,s},d}.
$$

数学上可以把它写成 `one_hot(I, V) @ W`。实际使用 Embedding 时按索引读取权重行，无需创建 `[B,S,V]` 的稠密 one-hot 张量。

默认参数下，新建模块会随机初始化权重，训练再用梯度调整。下面用固定数值替换初值，以便手算查表结果。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def embedding_lookup(token_ids, weight):
    """基础查表：ID 为合法非负整数；不扩展 padding 等参数的反向语义。"""
    return weight[token_ids]

weight = torch.tensor([
    [0.0, 0.0, 0.0],       # ID 0
    [1.0, 2.0, 3.0],       # ID 1
    [10.0, 20.0, 30.0],    # ID 2
    [-1.0, -2.0, -3.0],    # ID 3
    [4.0, 5.0, 6.0],       # ID 4
], device='cuda')
token_ids = torch.tensor(
    [[2, 1, 2], [4, 0, 3]], dtype=torch.long, device='cuda'
)
embedding = nn.Embedding(5, 3).to(device='cuda')
with torch.no_grad():
    embedding.weight.copy_(weight)   # 用固定数值演示查表

y = embedding(token_ids)
manual = embedding_lookup(token_ids, embedding.weight)
reference = F.embedding(token_ids, embedding.weight)
print(tuple(y.shape))                # 预期：(2,3,3)
# 第一条序列：[[10,20,30], [1,2,3], [10,20,30]]
# 第二条序列：[[4,5,6], [0,0,0], [-1,-2,-3]]
```

上面的三种写法在这组默认参数下表达同一次查表。`nn.Embedding` 注册并持有 `weight` 参数，没有 bias；`F.embedding` 使用调用方传入的权重。手写函数只展示基础索引，不实现 `padding_idx`、`max_norm` 或梯度频次缩放的全部行为。本例未设置 `padding_idx`，第 0 行虽然手动填了零向量，仍可学习。

查表结果有自己的数值存储，修改输出不会直接修改权重表。训练通过梯度更新参数，存储与梯度的区别见 [存储独立与梯度关系](tensors.md#存储独立与梯度关系是两件事)。[functional.embedding 接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.embedding.html)

### 梯度回到权重表，重复 ID 的贡献相加

整数 ID 不参与求导，梯度流向选中的权重行。若同一个 ID 出现两次，两处输出对这一行的梯度会累加。以下使用默认的 `scale_grad_by_freq=False`、`sparse=False`：

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(5, 3).to(device='cuda')
token_ids = torch.tensor([2, 1, 2], dtype=torch.long, device='cuda')
loss = embedding(token_ids).sum()
loss.backward()
# 预期：embedding.weight.grad 的 shape 为 [5,3]。
# ID 1 出现一次：第 1 行梯度 [1,1,1]。
# ID 2 出现两次：第 2 行梯度 [2,2,2]。
# 其余行未被本次查表使用，梯度为 [0,0,0]。
```

这里的 loss 对每个输出元素的导数为 1，所以梯度数值只取决于 ID 出现次数。`backward()` 写入 `.grad`，优化器的 `step()` 才更新参数；训练流程见 [一次训练更新](neural-networks.md#7-一次训练更新-谁计算-谁改参数)。

### padding、范数限制与稀疏梯度

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `num_embeddings` | 必填 | 表的行数 V；不是序列长度 |
| `embedding_dim` | 必填 | 每行的特征数 D |
| `padding_idx` | `None` | 该 ID 的查表结果不向对应权重行贡献梯度 |
| `max_norm` | `None` | 前向时，对本次用到且范数超限的权重行重归一化 |
| `norm_type` | `2.0` | `max_norm` 使用的 p 范数；默认是 L2 |
| `scale_grad_by_freq` | `False` | 为 True 时，把对应行的梯度除以该 ID 在本次输入中的出现次数 |
| `sparse` | `False` | 为 True 时生成稀疏的权重梯度；权重表本身仍是稠密 Tensor |

新建 `nn.Embedding(V,D,padding_idx=0)` 时，第 0 行初始化为全零。前向仍然读取这一行；如果手动加载了非零 padding 向量，查表会返回该向量。`F.embedding` 接收现成权重，不负责把 padding 行初始化或清零。

```python
import torch
import torch.nn as nn

embedding = nn.Embedding(5, 3, padding_idx=0).to(device='cuda')
token_ids = torch.tensor([[0, 2, 0]], dtype=torch.long, device='cuda')
y = embedding(token_ids)             # [1,3,3]
# 新建模块的第 0 行为零，因此第 0、2 个位置输出 [0,0,0]。
```

`padding_idx` 限制的是本次查表的梯度。它不会自动建立 attention mask 或排除 padding 位置的 loss；这两处仍需由模型和损失函数处理。其他使用共享权重的计算也可能影响这一行。

设置 `max_norm` 后，forward 会原地修改相应权重行。若同一计算图还直接使用 `embedding.weight`，需要按官方说明处理克隆与调用顺序。`sparse=True` 则要求优化器支持稀疏梯度，例如 `torch.optim.SparseAdam`。[Embedding 参数与限制](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)

### 加载已有向量表，以及与 LM head 的关系

已有 `vectors: [V,D]` 时，可以使用 `nn.Embedding.from_pretrained(vectors, freeze=False)`。它从表的 shape 推断 V 和 D；默认 `freeze=True` 会关闭权重求导，继续训练时显式指定 `freeze=False`。输入表应是已准备好的浮点向量表。

部分语言模型把 embedding 与 LM head 绑定为同一份参数。输入端用 ID 查行，输出端用 `hidden @ embedding.weight.T` 计算整个词表的 logits，得到 `[B,S,V]`。两者共享权重时，来自输入与输出两条路径的梯度都会汇总到这份参数。运行位置可对照 [Transformer 运行全过程](../Attention/transformer-runtime.md)。

## 1. Linear：线性层

`x: [..., in_features]`、`weight: [out_features, in_features]`、`bias: [out_features]`，返回 `[..., out_features]`。权重按 PyTorch Linear 的方向存放：`y = x @ weight.T + bias`。

```python
import torch

def linear(x, weight, bias=None):
    """对最后一个特征轴做线性变换，保留前面的 batch/token 轴。"""
    y = x @ weight.transpose(-2, -1)
    if bias is not None:
        y = y + bias                 # 偏置沿所有前导轴广播
    return y

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
weight = torch.tensor([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0],
                       [1.0, 1.0, 1.0]], device='cuda')
bias = torch.tensor([10.0, 20.0, 30.0, 40.0], device='cuda')
y = linear(x, weight, bias)
print(tuple(y.shape))                # 预期：(2, 4)
print(y.tolist())                    # 预期：[[11.0, 22.0, 33.0, 46.0], [14.0, 25.0, 36.0, 55.0]]
reference = torch.nn.functional.linear(x, weight, bias)
```

`x` 可以从 `[B, D]` 扩展成 `[B, S, D]`，不需要循环逐 token 计算。输入、权重与偏置使用相同的 GPU 和 dtype；这里是前向函数，层对象如何持有参数见 [F.linear 与 nn.Linear](neural-networks.md#2-linear-同一个计算的两种用法)。

官方接口：[torch.nn.Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html)。

## 2. Stable softmax：归一化为权重

沿 `dim` 做 `exp(x - max(x)) / sum(exp(x - max(x)))`。输入为有限 FP16 / BF16 / FP32 数值，输出 shape 与输入相同；先转 FP32 计算，再返回原 dtype。

```python
import torch

def stable_softmax(x, dim=-1):
    """沿指定轴做 softmax；减最大值避免大正数的指数溢出。"""
    values = x.float()
    shifted = values - values.amax(dim=dim, keepdim=True)
    exps = shifted.exp()
    probs = exps / exps.sum(dim=dim, keepdim=True)
    return probs.to(dtype=x.dtype)

x = torch.tensor([[1000.0, 1001.0, 1002.0],
                  [1.0, 1.0, 1.0]], dtype=torch.float32, device='cuda')
probs = stable_softmax(x)
print(tuple(probs.shape))            # 预期：(2, 3)
# 第一行约 [0.0900, 0.2447, 0.6652]；第二行各约 1/3
reference = torch.softmax(x, dim=-1)
print(torch.allclose(probs, reference, rtol=1e-5, atol=1e-6))  # 预期：True
```

两个归约都保留长度为 1 的轴，以便广播回输入。低精度输出有舍入误差，不能要求逐元素完全相等。输入包含 NaN、正无穷或整行负无穷时，需要另外定义数值处理策略；这里不处理这些输入。

官方接口：[torch.softmax](https://docs.pytorch.org/docs/2.14/generated/torch.softmax.html)。

## Norm、normalize 和归一化层分别做什么

Transformer 中常见的归一化层有 LayerNorm 和 RMSNorm。数学里的 norm（范数）用一个数描述向量的大小；`normalize` 则用这个大小缩放向量。先区分它们的输入、输出和统计范围：

| 操作 | 对一条特征向量做什么 | `x: [B,S,D]` 沿末轴计算后的 shape |
| --- | --- | --- |
| L1 / L2 norm | 计算一个表示大小的数 | `[B,S]`；`keepdim=True` 时为 `[B,S,1]` |
| L2 normalize | 每个元素除以同一条向量的 L2 长度 | `[B,S,D]` |
| LayerNorm | 减均值、除标准差，再逐特征缩放和偏移 | `[B,S,D]` |
| RMSNorm | 除均方根，再逐特征缩放 | `[B,S,D]` |

### 普通向量 Norm：L1 与 L2

对实数向量 $x=(x_1,\ldots,x_D)$：

$$
\lVert x\rVert_1=\sum_{i=1}^{D}|x_i|,\qquad
\lVert x\rVert_2=\sqrt{\sum_{i=1}^{D}x_i^2}.
$$

例如 `[3,4]` 的 L1 norm 是 7，L2 norm 是 5。求范数只得到长度，不会自动把原向量改成单位长度。

```python
import torch

def l2_norm(x, dim=-1, keepdim=False):
    """沿 dim 求 L2 范数；使用 FP32 计算并返回 FP32 结果。"""
    values = x.float()
    return values.square().sum(dim=dim, keepdim=keepdim).sqrt()

x = torch.tensor([[3.0, 4.0], [0.0, 0.0]], device='cuda')
lengths = l2_norm(x)                              # 预期：[5, 0]，shape [2]
lengths_kept = l2_norm(x, keepdim=True)            # 预期：[[5], [0]]，shape [2,1]
l1 = torch.linalg.vector_norm(x, ord=1, dim=-1)    # 预期：[7, 0]
reference = torch.linalg.vector_norm(x.float(), ord=2, dim=-1)
```

对 `[B,S,D]` 显式指定 `dim=-1`，就是分别计算每个 token 的特征向量长度。`torch.linalg.vector_norm` 的默认 `dim=None` 会先把全部元素展平；它与逐 token 求长度的统计范围不同。[vector_norm 接口](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.vector_norm.html)

### L2 normalize：按长度缩放向量

L2 normalize 的计算为：

$$
y_i=\frac{x_i}{\max(\lVert x\rVert_2,\epsilon)}.
$$

当输入长度不小于 eps 时，输出的 L2 长度为 1。零向量仍输出零向量。分母使用 `max(length, eps)`，与后面 Norm 层在平方根内加 eps 的写法不同。

```python
import torch
import torch.nn.functional as F

def l2_normalize(x, dim=-1, eps=1e-12):
    """沿 dim 缩放向量；分母保留归约轴，以便广播。"""
    values = x.float()
    length = values.square().sum(dim=dim, keepdim=True).sqrt()
    return (values / length.clamp_min(eps)).to(dtype=x.dtype)

x = torch.tensor([[3.0, 4.0], [0.0, 0.0]], device='cuda')
y = l2_normalize(x)                               # 预期：[[0.6,0.8], [0,0]]
reference = F.normalize(x.float(), p=2, dim=-1, eps=1e-12).to(x.dtype)
```

`F.normalize` 默认沿 `dim=1` 计算。对 `[B,S,D]` 的特征向量归一化时，显式写 `dim=-1`。本小节的实现处理 FP16 / BF16 / FP32 实数输入，在 FP32 中计算统计量。[normalize 接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.normalize.html)

## 3. LayerNorm：沿特征轴减均值、除标准差

`x: [..., D]`、`weight: [D]`、`bias: [D]`，输出 `[..., D]`。本实现只归一化最后一个轴；均值与方差在 FP32 中计算，方差除以 D，不用无偏估计的 D−1。

### 数学公式：每个 token 单独统计

固定 batch 和 token 位置，把它的 D 个特征记作 $x_1,\ldots,x_D$：

$$
\mu=\frac{1}{D}\sum_{i=1}^{D}x_i,\qquad
\sigma^2=\frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2,
$$

$$
\widehat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}},\qquad
y_i=\widehat{x}_i\gamma_i+\beta_i.
$$

`weight` 对应 $\gamma$，`bias` 对应 $\beta$。它们各有 D 个可学习参数，同一层的所有 token 共用。末尾的乘法与加法都是逐元素操作，不使用 `[D,D]` 矩阵混合特征。

以 `[3,4]` 为例，均值为 3.5，中心化结果为 `[-0.5,0.5]`，方差为 0.25。忽略 eps、令 weight 为全 1、bias 为全 0，输出就是 `[-1,1]`。常量向量减均值后全为零，最终输出等于 bias。

`eps>0` 时，$\widehat{x}$ 的方差为 $\sigma^2/(\sigma^2+\epsilon)$，不严格等于 1。乘 weight、加 bias 后，输出的均值和方差也不再固定。[LayerNorm 的统计与参数约定](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)

### Torch 实现：保留统计量的特征轴

| 步骤 | `x: [B,S,D]` 时的 shape | 含义 |
| --- | --- | --- |
| `mean` | `[B,S,1]` | 每个 token 自己的特征均值 |
| `centered` | `[B,S,D]` | 每个特征减去对应 token 的均值 |
| `variance` | `[B,S,1]` | 沿 D 求平均的平方偏差 |
| `normalized`、`y` | `[B,S,D]` | 统计量广播回 D 个特征，随后应用参数 |

```python
import torch

def layer_norm(x, weight, bias, eps=1e-5):
    """每个样本/token 独立归一化最后一个特征轴。"""
    values = x.float()
    mean = values.mean(dim=-1, keepdim=True)       # [..., 1]
    centered = values - mean                      # [..., D]
    variance = centered.square().mean(dim=-1, keepdim=True)
    normalized = centered * torch.rsqrt(variance + eps)
    y = normalized * weight.float() + bias.float()
    return y.to(dtype=x.dtype)

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
weight = torch.ones(3, device='cuda')
bias = torch.zeros(3, device='cuda')
y = layer_norm(x, weight, bias)
print(tuple(y.shape))                         # 预期：(2, 3)
# 两行分别约为 [-1.2247, 0.0, 1.2247]
reference = torch.nn.functional.layer_norm(x, (3,), weight, bias, eps=1e-5)
```

`eps` 放在开方前，避免方差为零时除零。与 BatchNorm 不同，这里不沿 batch 统计，也不维护 running mean / variance。FP16 / BF16 / FP32 输入和参数在同一 GPU 上；多轴 normalized_shape 和其他 dtype 不在这个函数的约定中。

### 层对象与 functional 接口

```python
import torch
import torch.nn.functional as F

x = torch.randn(2, 4, 8, device='cuda')    # [B,S,D]
norm = torch.nn.LayerNorm(8, eps=1e-5).to(device='cuda')
y = norm(x)                             # [2,4,8]
# norm.weight：[8]，初始全 1；norm.bias：[8]，初始全 0。
reference = F.layer_norm(x, (8,), norm.weight, norm.bias, eps=1e-5)
```

`nn.LayerNorm(8)` 持有参数，`F.layer_norm` 接收调用方传入的参数。`normalized_shape=8` 对应最后一个轴；若写成 `(4,8)`，则同时统计最后两个轴，参数 shape 也变为 `[4,8]`。Transformer 中按 token 归一化特征时，这里使用 `8`。

官方接口：[LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)、[functional.layer_norm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.layer_norm.html)。

## 4. RMSNorm：按均方根缩放

`x: [..., D]`、`weight: [D]`，输出 `[..., D]`。计算 `x / sqrt(mean(x²) + eps) * weight`，不减均值，不加 bias。输入使用 FP16 / BF16 / FP32，在 FP32 中计算统计量与缩放。

### 数学公式：控制均方根，保留均值信息

RMS 是 root mean square（均方根）。对一个 token 的 D 个特征：

$$
r^2=\frac{1}{D}\sum_{i=1}^{D}x_i^2,\qquad
y_i=\frac{x_i}{\sqrt{r^2+\epsilon}}\gamma_i.
$$

它先求平方、沿特征轴取平均、加 eps、开平方，再用这个分母缩放所有特征。每个 token 有自己的分母；每个特征有自己的 weight。在乘 weight 之前，正值向量归一化后仍然为正。

以 `[3,4]` 为例，均方为 12.5，均方根为 $\sqrt{12.5}\approx3.5355$。忽略 eps、令 weight 全 1，输出约为 `[0.8485,1.1314]`。归一化后的 RMS 为 1，但每个元素的绝对值不必为 1。

RMS 与 L2 范数存在关系：

$$
\operatorname{RMS}(x)=\frac{\lVert x\rVert_2}{\sqrt D}.
$$

因此，对非零向量忽略 eps 且令 weight 全 1 时，RMSNorm 输出的 L2 长度为 $\sqrt D$；L2 normalize 输出的长度为 1。两者使用不同的尺度。

### Torch 实现：平方均值仍要 keepdim

对 `[B,S,D]`，`mean_square` 的 shape 为 `[B,S,1]`，广播缩放后恢复 `[B,S,D]`。如果不保留末轴，统计量通常无法按预期广播回特征轴。

```python
import torch

def rms_norm(x, weight, eps=1e-6):
    """沿最后一个轴计算均方根，按各特征的权重缩放。"""
    values = x.float()
    mean_square = values.square().mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(mean_square + eps)
    return (normalized * weight.float()).to(dtype=x.dtype)

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
weight = torch.ones(3, device='cuda')
y = rms_norm(x, weight)
print(tuple(y.shape))                         # 预期：(2, 3)
reference = torch.nn.functional.rms_norm(x, (3,), weight, eps=1e-6)
```

对照官方结果时显式指定相同的 eps，不能把不同默认 eps 的结果直接比较。本实现的低精度中间计算与转换顺序是明确的教学选择，不保证与所有后端逐位一致。

### 层对象：只有逐特征 weight

```python
import torch
import torch.nn.functional as F

x = torch.randn(2, 4, 8, device='cuda')
norm = torch.nn.RMSNorm(8, eps=1e-6).to(device='cuda')
y = norm(x)                             # [2,4,8]
# norm.weight：[8]，初始全 1；这个模块没有 bias 参数。
reference = F.rms_norm(x, (8,), norm.weight, eps=1e-6)
```

官方 LayerNorm 和 RMSNorm 都可用 `elementwise_affine=False` 关闭可学习的逐元素变换。两者在训练和推理时都使用当前输入的统计量，不维护 BatchNorm 式的 running statistics。实际模型中可以放在 attention / FFN 分支之前，位置关系见 [Transformer 运行全过程](../Attention/transformer-runtime.md#5-一个-transformer-block-的完整顺序)。

### 用同一个向量比较三种归一化

以下均忽略 eps；Norm 层的 weight 全为 1，LayerNorm 的 bias 全为 0：

| 输入 `[3,4]` | 分母 | 输出 | 输出的 L2 长度 |
| --- | --- | --- | --- |
| L2 normalize | $\sqrt{3^2+4^2}=5$ | `[0.6,0.8]` | 1 |
| LayerNorm | 中心化后的标准差 0.5 | `[-1,1]` | $\sqrt2$ |
| RMSNorm | $\sqrt{(3^2+4^2)/2}=\sqrt{12.5}$ | 约 `[0.8485,1.1314]` | $\sqrt2$ |

官方接口：[RMSNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.RMSNorm.html)。

## 5. SwiGLU FFN：门控前馈组件

ReLU、SiLU、GELU 的现成接口，以及 SwiGLU 与普通 GLU 的区别，见 [激活函数与门控](neural-networks.md#激活函数与门控-relu-silu-gelu-与-swiglu)。

`x: [..., D]`，`w_gate / w_up: [F, D]`，`w_down: [D, F]`，返回 `[..., D]`。F 是中间特征数；gate 与 up 都从同一个 x 产生。

```python
import torch
import torch.nn.functional as F

def swiglu_ffn(x, w_gate, w_up, w_down):
    """无 bias 的门控前馈：两路投影、门控相乘、投影回原特征数。"""
    gate = x @ w_gate.transpose(-2, -1)       # [..., F]
    up = x @ w_up.transpose(-2, -1)           # [..., F]
    hidden = F.silu(gate) * up                # SiLU(gate) = gate * sigmoid(gate)
    return hidden @ w_down.transpose(-2, -1)  # [..., D]

x = torch.randn(2, 3, 4, device='cuda')       # [batch, tokens, D]
w_gate = torch.randn(8, 4, device='cuda')
w_up = torch.randn(8, 4, device='cuda')
w_down = torch.randn(4, 8, device='cuda')
y = swiglu_ffn(x, w_gate, w_up, w_down)
print(tuple(y.shape))                        # 预期：(2, 3, 4)
```

SiLU 是逐元素激活，gate 与 up 的乘法也是逐元素乘法；三个投影使用矩阵乘法。这里只实现 FFN 本身，residual connection 和 norm 属于外层 block。输入和权重需位于同一 GPU、使用相同浮点 dtype。

官方激活接口：[functional.silu](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.silu.html)。门控 FFN 的计算结构见 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。

## 6. Scaled dot-product attention：分数、mask 与加权求和

| 输入/中间结果 | shape | 含义 |
| --- | --- | --- |
| q | `[B, H, L, D]` | L 个 query，每个 D 个特征 |
| k | `[B, H, S, D]` | S 个 key |
| v | `[B, H, S, Dv]` | 每个 key 对应 Dv 个 value 特征 |
| scores / probs | `[B, H, L, S]` | 每个 query 对每个 key 的分数/权重 |
| output | `[B, H, L, Dv]` | 按权重组合 value |

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, causal=False):
    """普通多头 attention；causal=True 仅支持 L=S 的完整自注意力。"""
    scores = q.float() @ k.float().transpose(-2, -1)  # [B, H, L, S]
    scores = scores / (q.size(-1) ** 0.5)
    if causal:
        length = q.size(-2)
        assert length == k.size(-2), '此函数的 causal mask 要求 L=S'
        blocked = torch.ones(length, length, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(blocked, float('-inf'))
    probs = torch.softmax(scores, dim=-1)           # 沿 key 轴归一化
    return (probs @ v.float()).to(dtype=q.dtype)     # [B, H, L, Dv]

q = torch.randn(2, 4, 8, 16, device='cuda')
k = torch.randn(2, 4, 8, 16, device='cuda')
v = torch.randn(2, 4, 8, 16, device='cuda')
y = scaled_dot_product_attention(q, k, v, causal=True)
print(tuple(y.shape))                           # 预期：(2, 4, 8, 16)
reference = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
```

缩放因子是 head 的特征维度 `sqrt(D)`，softmax 沿 key 轴 S 做；最后 `[L, S] @ [S, Dv] → [L, Dv]`。本函数的 `blocked=True` 表示禁止关注，填成负无穷后权重为零。官方 SDPA 的 bool `attn_mask` 则以 True 表示允许关注，使用不同接口时要核对约定。

causal mask 只保留对角线及其下方，第 i 个 token 可以关注第 0 到 i 个 token。本实现没有 dropout、padding mask、GQA 或 KV cache；decode 的 query/key 长度与绝对位置关系需要单独定义，不能直接套用这里的方形 causal mask。输入为同一 GPU 上同 dtype 的 FP16 / BF16 / FP32 Tensor，特征轴长度非零。

这里显式保存了 `[B, H, L, S]` 分数矩阵，用于展示计算关系；它不是 FlashAttention 的实现或性能基线。与官方融合后端比较数值时需要合适容差，也不能假设低精度中间计算顺序完全一致。

官方接口：[functional.scaled_dot_product_attention](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)。

## 运行约定与验证范围

上述代码块各自包含导入，GPU 调用使用 `device='cuda'`；需要支持 CUDA 的 PyTorch 和可用的 NVIDIA GPU。函数自身跟随输入的 device；没有 CPU fallback 或自动设备迁移。

标出的数值和 shape 是推导预期，`reference` 用来说明对应的官方接口，不表示已执行对照。只做 Python 语法、网站构建和链接检查，未执行 GPU 实机验证，也没有性能测量结果。回查基础写法见 [Tensor 与 GPU 内存](tensors.md)。
