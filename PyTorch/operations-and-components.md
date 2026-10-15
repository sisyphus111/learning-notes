# 常见计算与组件的 Torch 实现

本页用 Tensor 运算展开常见组件的计算过程。函数、层对象、模型与参数管理先看 [神经网络基础](neural-networks.md)；shape、广播和存储布局查 [Tensor 与 GPU 内存](tensors.md)。

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

## 3. LayerNorm：沿特征轴减均值、除标准差

`x: [..., D]`、`weight: [D]`、`bias: [D]`，输出 `[..., D]`。本实现只归一化最后一个轴；均值与方差在 FP32 中计算，方差除以 D，不用无偏估计的 D−1。

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

官方接口：[LayerNorm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LayerNorm.html)、[functional.layer_norm](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.layer_norm.html)。

## 4. RMSNorm：按均方根缩放

`x: [..., D]`、`weight: [D]`，输出 `[..., D]`。计算 `x / sqrt(mean(x²) + eps) * weight`，不减均值，不加 bias。输入使用 FP16 / BF16 / FP32，在 FP32 中计算统计量与缩放。

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
