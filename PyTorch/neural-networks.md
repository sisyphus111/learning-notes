# 神经网络基础：函数、层、模型与参数

神经网络用 Tensor 保存输入、权重和中间结果。`torch.nn` 把计算封装成层，再把层组织成模型；自动求导计算梯度，优化器根据梯度更新参数。

前置写法见 [Tensor 与 GPU 内存](tensors.md)：shape、广播、矩阵乘法、device、dtype 与存储。本页从计算函数走到模型的使用；各组件的公式与手写实现见 [常见计算与组件](operations-and-components.md)。

## 1. torch、nn 与 F 的关系

```python
import torch                    # Tensor 创建、运算与自动求导
from torch import nn            # 层、模型和参数管理
import torch.nn.functional as F # 神经网络计算函数，F 是常用简称
```

| 写法 | 负责什么 | 参数由谁保存 |
| --- | --- | --- |
| `x @ weight.T + bias` | 用 Tensor 运算表达线性计算 | 调用方提供 weight、bias |
| `F.linear(x, weight, bias)` | 调用现成的线性计算函数 | 调用方提供 weight、bias |
| `layer = nn.Linear(3, 2)`，再 `layer(x)` | 创建并使用一个线性层对象 | layer 保存自己的 weight、bias |
| 自定义 `nn.Module` | 把多个层组织成模型，定义计算顺序 | 管理注册的子层、参数和 buffer |

`nn` 是 neural network 的缩写。`F.linear` 与 `nn.Linear` 都能参与自动求导；是否记录梯度取决于 Tensor 和梯度模式，不能按“函数还是层对象”判断。`nn.Module` 是层和模型共同使用的基类，单个 `nn.Linear` 也是一个 Module。

## 2. Linear：同一个计算的两种用法

Linear（线性层，也常称全连接层）对每个输入向量做 `y = x @ weight.T + bias`。`in_features` 是输入特征数，`out_features` 是输出特征数；前面的 batch、token 轴保留。

| 对象 | shape |
| --- | --- |
| x | `[..., in_features]` |
| weight | `[out_features, in_features]` |
| bias | `[out_features]` |
| y | `[..., out_features]` |

### F.linear：自己提供权重和偏置

```python
import torch
import torch.nn.functional as F

x = torch.tensor([[1., 2., 3.]], device='cuda')
weight = torch.tensor([[1., 0., 0.], [1., 1., 1.]], device='cuda')
bias = torch.tensor([10., 20.], device='cuda')
y = F.linear(x, weight, bias)    # 3 个输入特征变为 2 个输出特征
print(y.tolist())               # 预期：[[11.0, 26.0]]
print(torch.allclose(y, x @ weight.T + bias))  # 预期：True
```

第一项是 `1×1 + 2×0 + 3×0 + 10 = 11`，第二项是 `1+2+3+20 = 26`。F.linear 使用传入的 Tensor，不会替你创建或保存一套模型权重。

### nn.Linear 的构造参数与常用写法

构造接口是 `nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)`。它创建一个层对象及其参数；随后用 `layer(x)` 传入 Tensor 做计算。

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `in_features` | 输入 Tensor 最后一轴的长度 | 必填整数 |
| `out_features` | 输出 Tensor 最后一轴的长度 | 必填整数 |
| `bias` | 是否创建偏置参数 | `True`；设为 False 时 `layer.bias is None` |
| `device` | 创建 weight、bias 的设备 | `None`：使用默认设备；未改配置时通常是 CPU |
| `dtype` | weight、bias 的数据类型 | `None`：使用默认浮点类型；未改配置时是 float32 |

下面都是同一套构造接口的常用写法，均把最后一轴从 3 个特征变成 2 个：

| 构造写法 | 含义 |
| --- | --- |
| `nn.Linear(3, 2)` | 按位置传入输入、输出特征数，包含 bias |
| `nn.Linear(in_features=3, out_features=2)` | 用参数名写清输入、输出特征数，与上一行配置相同 |
| `nn.Linear(3, 2, bias=False)` | 不创建 bias，只计算 `x @ weight.T` |
| `nn.Linear(3, 2, False)` | 第三个位置参数也是 bias；通常用 `bias=False` 更易读 |
| `nn.Linear(3, 2, device='cuda')` | 直接在当前 CUDA 设备上创建参数 |
| `nn.Linear(3, 2, dtype=torch.float64)` | 用 float64 创建参数，设备仍由默认配置决定 |
| `nn.Linear(3, 2, device='cuda:0', dtype=torch.float16)` | 同时指定 GPU 和参数精度 |
| `nn.Linear(3, 2).to(device='cuda', dtype=torch.float16)` | 先按默认配置构造，再迁移、转换已有参数 |

最后一行的 `.to(...)` 修改刚创建的层并返回自身；直接传 device/dtype 则在目标配置下创建参数。每次调用 `nn.Linear(...)` 都创建并初始化一套独立参数；“构造写法配置相同”不代表两层的随机权重数值相同。默认 weight 和启用的 bias 都会随机初始化，不能当作零使用；需要指定数值时，见下一小节的 copy_ 示例。

```python
import torch
from torch import nn

basic = nn.Linear(3, 2)         # 默认包含偏置，3 个输入特征变为 2 个
print(basic)                   # 预期：Linear(in_features=3, out_features=2, bias=True)
no_bias = nn.Linear(in_features=3, out_features=2, bias=False)
print(tuple(no_bias.weight.shape))  # 预期：(2, 3)，先输出特征数，再输入特征数
print(no_bias.bias is None)     # 预期：True，没有偏置参数

gpu = nn.Linear(3, 2, device='cuda', dtype=torch.float16)
print(gpu.weight.device.type, gpu.weight.dtype)  # 预期：cuda torch.float16
x = torch.ones(4, 3, device=gpu.weight.device, dtype=gpu.weight.dtype)
print(tuple(gpu(x).shape))      # 预期：(4, 2)，batch 数 4 由输入决定
# nn.Linear((4, 3), 2)          # 报错：in_features 要整数，不是输入的完整 shape
```

`nn.Linear(3, 2)` 的两个数字是特征数；输入可以是 `[3]`、`[B,3]`、`[B,S,3]`，输出分别是 `[2]`、`[B,2]`、`[B,S,2]`。构造时不用传入 batch/token 数，也不传入输入 Tensor 或现成权重。参数与初始化规则见 [Linear 官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html)，迁移语义见 [Module.to](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.to)。

### nn.Linear：由层对象持有参数

```python
import torch
from torch import nn
import torch.nn.functional as F

layer = nn.Linear(3, 2, device='cuda')  # 创建层；默认包含 bias，参数会被初始化
print(tuple(layer.weight.shape))      # 预期：(2, 3)
print(tuple(layer.bias.shape))        # 预期：(2,)

with torch.no_grad():           # 设置示例权重时不记录梯度，后文介绍此上下文
    layer.weight.copy_(torch.tensor([[1., 0., 0.], [1., 1., 1.]], device='cuda'))
    layer.bias.copy_(torch.tensor([10., 20.], device='cuda'))

x = torch.tensor([[1., 2., 3.]], device='cuda')
y = layer(x)                    # 使用层中保存的 weight、bias 计算
print(y.tolist())               # 预期：[[11.0, 26.0]]
print(torch.allclose(y, F.linear(x, layer.weight, layer.bias)))  # 预期：True
tokens = torch.ones(2, 4, 3, device='cuda')  # [B,S,in_features]
print(tuple(layer(tokens).shape))          # 预期：(2, 4, 2)
```

同一个 layer 每次调用都复用它持有的参数。改变 batch 或 token 数，不需要重新创建层；最后一轴的特征数必须匹配。接口见 [Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html) 与 [F.linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.linear.html)。

## 3. nn.Module：把层组织成模型

在 `__init__` 中创建需要复用的层，在 `forward` 中写 Tensor 怎样依次经过这些层。下面用一个 MLP（多层感知机）示例：把全连接层和非线性激活串起来，最后一轴从 3 变成 4，经过激活后再变成 2。

```python
import torch
from torch import nn

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()      # 先初始化 Module 的子层与参数管理
        self.fc1 = nn.Linear(3, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        hidden = self.relu(self.fc1(x))  # [...,3] → [...,4]，负数置零
        return self.fc2(hidden)          # [...,4] → [...,2]

model = TinyMLP().to('cuda')
x = torch.ones(2, 5, 3, device='cuda')
print(tuple(model(x).shape))     # 预期：(2, 5, 2)
print([name for name, _ in model.named_parameters()])
# 预期：['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
```

赋给 `self.fc1`、`self.fc2` 的层会注册为子模块，父模型能统一找到它们的参数。ReLU 也是 Module，但没有可学习参数；它为网络加入非线性，否则多个线性层串联仍只能表达线性加偏置的变换。

通常调用 `model(x)`，让 Module 执行 forward 及相关调用逻辑；直接调用 `model.forward(x)` 会绕过 hooks 等机制。层放在 `__init__` 中创建，避免每次 forward 都创建一套新参数。

纯顺序结构也可以用 `nn.Sequential`：

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).to('cuda')
print(tuple(model(torch.ones(2, 3, device='cuda')).shape))  # 预期：(2, 2)
```

如果只想注册一组层、自己在 forward 中决定调用顺序，用 `nn.ModuleList`。它负责管理子层，不像 Sequential 那样自动依次执行。接口见 [Module](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)、[Sequential](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Sequential.html) 与 [ModuleList](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ModuleList.html)。

## 4. Parameter、buffer 与模型的 device / dtype

### 为什么要把 Tensor 保存到模型里

一个线性层计算 `y = x @ weight.T + bias`。每次输入 x 可以不同，但同一个层需要反复使用同一套 weight、bias：训练时逐步调整它们，推理时复用训练好的数值。因此，层把参数保存在对象里，每次调用 `layer(x)` 都读取当前的参数。

模型还可能保存固定数据、运行统计量和临时结果，它们也可以用 Tensor 表示。**把数据注册为参数或 buffer 后，PyTorch 才能按相应规则统一管理它们。** 对包含很多子层的模型，参数查找、迁移和保存可以递归完成，不必逐层手动枚举。

| 要做的事 | 使用的接口 | 处理哪些数据 |
| --- | --- | --- |
| 把权重交给优化器更新 | `model.parameters()` | 注册参数 |
| 把整个模型搬到 GPU 或转换精度 | `model.to(...)` | 注册参数和 buffer |
| 取得用于保存的模型状态 | `model.state_dict()` | 注册参数和持久 buffer，按名称组织 |

计算本身也可以写成 `F.linear(x, weight, bias)`；Module 将需要复用的数据和使用这些数据的计算组织在一起。普通 Tensor 属性也能保存一份引用，但不会自动进入上面的管理流程。

### Parameter、buffer 与普通属性分别是什么

假设一个模型计算 `y = weight * (x - offset)`：weight 是准备训练调整的缩放系数，offset 是本例预先给定的固定校正量，cache 用来演示普通临时数据。

- **Parameter 是一种特殊 Tensor。** 将 `nn.Parameter(t)` 赋给 Module 的属性，会自动登记进参数列表。`nn.Linear` 已经替你创建并注册了 weight、bias，通常不用再次手动包装。
- **buffer 也是 Tensor。** `register_buffer('offset', t)` 把它登记为模型管理的状态，但不放进参数列表。用 `model.parameters()` 创建的优化器通常不会更新它；forward 或你写的代码仍可以修改 buffer，例如更新统计量。
- **普通 Tensor 属性没有这两种注册身份。** `self.cache = t` 只保存引用，不会自动成为参数或 buffer。

下面把计算和数据登记放在同一个模型中。为演示迁移，先显式在 CPU 创建三份数据：

```python
import torch
from torch import nn

class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3, device='cpu'))  # 注册参数
        self.register_buffer('offset', torch.tensor([1., 2., 3.], device='cpu'))
        self.cache = torch.zeros(3, device='cpu')  # 普通属性，演示不会自动迁移

    def forward(self, x):
        return self.weight * (x - self.offset)    # 使用模型保存的数据计算

model = Scale()
returned = model.to(device='cuda', dtype=torch.float16)
print(returned is model)        # 预期：True，修改同一个 Module 并返回自身
print(model.weight.device.type, model.offset.device.type, model.cache.device.type)
# 预期：cuda cuda cpu
x = torch.tensor([3., 5., 7.], device='cuda', dtype=torch.float16)
print(model(x).tolist())        # 预期：[2.0, 3.0, 4.0]
print([name for name, _ in model.named_parameters()])  # 预期：['weight']
print(list(model.state_dict()))                       # 预期：['weight', 'offset']
```

`model(x)` 读取 weight 和 offset，进行 Tensor 运算。训练更新 weight 后，下次调用就会使用更新后的值；offset 在本例中按固定值使用。cache 没有参与这个 forward，也没有注册，所以迁移模型时不会处理它。

### device / dtype 怎样作用于这些数据

device 表示数值放在哪里，例如 CPU、`cuda:0`；dtype 表示数值用什么类型保存，例如 float32、float16。上例的 `model.to(device='cuda', dtype=torch.float16)` 得到：

| 数据 | 管理身份 | 迁移后的 device / dtype |
| --- | --- | --- |
| `model.weight` | 参数 | CUDA / float16 |
| `model.offset` | buffer | CUDA / float16 |
| `model.cache` | 普通属性 | 仍是 CPU / float32 |

浮点、复数参数和 buffer 会转换为指定 dtype；整型状态只迁移设备，保留其整数 dtype。模型的输入 x 是调用时从外面传入的，不属于注册状态，仍需单独迁移，例如 `x = x.to(device='cuda', dtype=torch.float16)`。

再看三种保存方式对应的管理规则：

| 保存方式 | 在 `parameters()` 中 | 随 `model.to(...)` 迁移 | 在默认 `state_dict()` 中 |
| --- | --- | --- | --- |
| `self.weight = nn.Parameter(t)` | 是 | 是 | 是 |
| `self.register_buffer('offset', t)` | 否 | 是 | 是；`persistent=False` 可排除 |
| 普通 Tensor 属性 `self.cache = t` | 否 | 否 | 否 |

**注册身份与 requires_grad 是两个概念。** requires_grad 控制是否求梯度，Parameter / buffer 的身份控制模型怎样管理 Tensor。Parameter 默认 `requires_grad=True`，也可以冻结为 False；普通 Tensor 即使要求梯度，也不会自动出现在 `model.parameters()` 中。

注意调用对象的区别：`Tensor.to(...)` 返回转换结果，需要接收；目标 device/dtype 已匹配时可能返回原 Tensor。`Module.to(...)` 修改模型并返回自身。先把模型放到目标设备，再创建持有它参数的优化器。规则见 [Parameter](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parameter.Parameter.html)、[Tensor.to](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.to.html) 与 [Module.to](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.to)。

## 5. 常见层与 functional 写法

| 层对象 | 计算函数 | 作用 |
| --- | --- | --- |
| `nn.ReLU()` | `F.relu(x)` | 负数置零，shape 不变 |
| `nn.SiLU()` | `F.silu(x)` | `x * sigmoid(x)`，shape 不变 |
| `nn.GELU()` | `F.gelu(x)` | 平滑的逐元素激活，shape 不变 |
| `nn.Softmax(dim=-1)` | `F.softmax(x, dim=-1)` | 沿指定轴变成和为 1 的权重 |
| `nn.LayerNorm(D)` | `F.layer_norm(x, (D,), weight, bias)` | 对末轴 D 做归一化，层可持有缩放与偏移参数 |
| `nn.Dropout(p=0.5)` | `F.dropout(x, p=0.5, training=...)` | 训练时随机置零并缩放保留值；评估时直接返回输入 |

这些 Module 可以插入 Sequential 或自定义模型；functional 写法适合在 forward 中显式表达计算。使用 `F.dropout` 时需要传入 `training=self.training`，它默认 `training=True`，不会自动读取所属模型的模式。接口见 [functional](https://docs.pytorch.org/docs/2.14/nn.functional.html)；softmax、归一化与 SiLU 的具体计算见 [常见计算与组件](operations-and-components.md)。

## 激活函数与门控：ReLU、SiLU、GELU 与 SwiGLU

激活函数通常放在 Linear 等层之后，对中间激活值做非线性变换。只堆叠 Linear 仍等价于一次线性加偏置的变换；加入非线性，模型才能表达更复杂的关系。**PyTorch 已提供 ReLU、SiLU、GELU 的现成接口，不需要自己写公式才能使用。**

### 常用激活接口速查

| 名称 | 层对象写法 | 函数写法 | 每个元素做什么 |
| --- | --- | --- | --- |
| ReLU | `nn.ReLU()` | `F.relu(x)` | `max(0, x)`：负数变成 0，正数保留 |
| SiLU | `nn.SiLU()` | `F.silu(x)` | `x * sigmoid(x)`；也称 Swish（β=1） |
| GELU | `nn.GELU()` | `F.gelu(x)` | `x * Φ(x)`，Φ 是标准正态分布的累积分布函数；按输入值平滑缩放 |
| Sigmoid | `nn.Sigmoid()` | `torch.sigmoid(x)` | `1 / (1 + exp(-x))`，把实数映射到 (0, 1) |
| Tanh | `nn.Tanh()` | `torch.tanh(x)` | 双曲正切，把实数映射到 (-1, 1) |

这些接口都返回 Tensor，**支持任意维数，逐元素计算，不需要 dim，输出 shape 与输入相同**。例如输入 `[B, S, D]`，每个 `x[b, s, d]` 分别经过激活，输出仍是 `[B, S, D]`。表中层对象没有可学习参数；输入在 GPU 上，就在 GPU 上计算，不需要给 `nn.ReLU()` 等构造器传 device。

```python
import torch
from torch import nn
import torch.nn.functional as F

x = torch.tensor([[-1., 0., 1.]], device='cuda')
print(F.relu(x).tolist())             # 负数置零，预期：[[0.0, 0.0, 1.0]]
print(F.silu(x).shape)                # 逐元素 SiLU，预期：torch.Size([1, 3])
print(F.gelu(x).shape)                # 逐元素 GELU，预期：torch.Size([1, 3])
assert torch.allclose(nn.SiLU()(x), F.silu(x))  # 层对象与函数的计算一致
assert torch.allclose(F.silu(x), x * torch.sigmoid(x))

model = nn.Sequential(
    nn.Linear(4, 8),                  # 最后一轴：4 → 8
    nn.GELU(),                        # 加入非线性，保留所有轴的长度
    nn.Linear(8, 4),                  # 最后一轴：8 → 4
).to('cuda')
y = model(torch.randn(2, 3, 4, device='cuda'))
print(tuple(y.shape))                 # 预期：(2, 3, 4)
```

ReLU 会直接截掉负数；SiLU、GELU 对负数也可能输出负值，不是“负数全部清零”。它们的计算本身不随 train/eval 切换，也不会像 Dropout 一样随机丢弃激活。这里默认返回计算结果，不修改输入；ReLU、SiLU 可显式设置 `inplace=True` 做原地计算，但仍须遵守自动求导的原地修改限制。

GELU 默认 `approximate='none'`，也提供 tanh 近似；复现模型时应与原模型设置保持一致：

```python
import torch
from torch import nn
import torch.nn.functional as F

x = torch.randn(2, 3, 4, device='cuda')
y = F.gelu(x, approximate='none')      # 默认公式
y_approx = F.gelu(x, approximate='tanh')  # tanh 近似，数值可能有差别
layer = nn.GELU(approximate='tanh')     # 同一个选项也能传给层对象
assert torch.allclose(layer(x), y_approx)
```

接口与公式见 [ReLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ReLU.html)、[SiLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SiLU.html)、[GELU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html)；其他激活接口见 [nn 非线性激活函数](https://docs.pytorch.org/docs/2.14/nn.html#non-linear-activations-weighted-sum-nonlinearity)。

### SwiGLU：SiLU 激活一路，再与另一路逐元素相乘

SiLU 接收一个 Tensor；SwiGLU 的门控核心接收两路同 shape 的激活值：`hidden = F.silu(gate) * up`。gate（门控分支）经过 SiLU 后，逐位置缩放 up（另一条特征分支）。这里的“门”不是随机开关，也不保证缩放系数在 0 到 1 之间。

```python
import torch
import torch.nn.functional as F

gate = torch.tensor([[0., 1.]], device='cuda')
up = torch.tensor([[10., 20.]], device='cuda')
hidden = F.silu(gate) * up             # 对应位置相乘，不是矩阵乘法
print(tuple(hidden.shape))            # 预期：(1, 2)
# 第一个位置：SiLU(0) * 10 = 0
# 第二个位置：SiLU(1) * 20 ≈ 14.6212

packed = torch.cat([gate, up], dim=-1) # 两路拼接，shape：(1, 4)
gate2, up2 = packed.chunk(2, dim=-1)   # 沿末轴拆成等宽两路
assert torch.allclose(F.silu(gate2) * up2, hidden)
```

在常见的 SwiGLU FFN 中，gate 和 up 来自同一个输入的两次独立 Linear 投影；相乘后，再用第三个 Linear 投影回模型维度：

```text
x [..., D] → gate_proj → gate [..., H] → SiLU ─┐
           → up_proj   → up   [..., H] ────────┴→ 逐元素乘 → [..., H] → down_proj → [..., D]
```

H 是中间特征数。这三个 Linear 持有可学习权重；SiLU 和逐元素乘法本身没有可学习参数。完整实现见 [SwiGLU FFN](operations-and-components.md#5-swiglu-ffn-门控前馈组件)。

在本文核对的 PyTorch 2.14 标准 `torch.nn` / `F` 激活接口中，没有直接对应的 `nn.SwiGLU()` / `F.swiglu()`；用现成的 Linear、`F.silu` 和 `*` 就能组合，自动求导会处理这些计算。`nn.GLU(dim=-1)` / `F.glu(x, dim=-1)` 虽然是现成接口，但它把输入拆成 a、b，计算 `a * sigmoid(b)`，**不等于 SwiGLU**；输入末轴为 `2H` 时，它的输出末轴为 H。

GLU 的契约见 [nn.GLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GLU.html)，标准激活实现见 [PyTorch activation.py](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/nn/modules/activation.py)；SwiGLU 的门控结构见 [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。

## 6. 自动求导：loss、backward 与 grad

### 从损失值到梯度

**先计算一个希望降低的损失值，再计算它对各个数的变化有多敏感，把结果存到 `.grad` 中。** 在神经网络中，loss 通常衡量预测与目标的差距，自动求导负责计算 loss 相对于模型参数的梯度。

| 名字 | 作用 |
| --- | --- |
| `loss` | 希望降低的目标值，本例是一个标量 Tensor |
| `loss.backward()` | 从 loss 反向计算梯度 |
| `x.grad` | 保存计算出来的 loss 对 x 的梯度 |

先把 x 当作两个需要求梯度的数，用 `loss = x0² + x1²` 演示求导。这个目标可以理解为让它们接近 0，不需要先引入整个模型。

### 前向计算：算出 loss，并记录求导关系

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True, device='cuda')
squared = x.square()           # 前向计算：[4, 9]
loss = squared.sum()           # 前向计算：4 + 9 = 13
print(loss.item())              # 预期：13.0
print(x.grad)                   # 预期：None，算出 loss 还没有求梯度

loss.backward()                # 反向求梯度，累加到 x.grad
print(x.grad.tolist())          # 预期：[4.0, 6.0]
print(x.tolist())               # 预期：[2.0, 3.0]，backward 没有更新数值
```

requires_grad=True 表示需要计算涉及 x 的梯度。前向运算既算出数值，也记录相关运算的求导关系；只运行 square、sum 时，x.grad 仍是 None。上例把结果 sum 成一个标量，因此可以直接调用无参数 backward；非标量输出需要传入上游梯度或先做归约。

### 梯度 [4, 6] 表示什么

梯度表示：保持其他数不变，某个数小幅变化时，loss 大约会变化多少。平方函数的导数是 `2x`，所以 loss 对 x0 的导数为 `2×2=4`，对 x1 的导数为 `2×3=6`。

| 当前的数 | 对应梯度 | 小幅变化时的含义 |
| --- | --- | --- |
| `x0=2` | `4` | x0 增加 0.01，loss 大约增加 0.04 |
| `x1=3` | `6` | x1 增加 0.01，loss 大约增加 0.06 |

x 的数值 `[2,3]`、loss 的数值 `13`、x.grad 的数值 `[4,6]` 描述不同事情。梯度是局部变化率，不是新的 x，也不是 loss 的各个组成项。

### backward：沿计算关系反向应用求导规则

```text
前向计算：x=[2,3] → square → [4,9] → sum → loss=13
反向求导：从 loss 开始 → 穿过 sum → 穿过 square → x.grad=[4,6]
```

每种运算有自己的求导规则。调用 backward 时，PyTorch 沿前向记录的关系反向组合这些规则，得到 loss 对 x 的梯度。这里 sum 对每个输入项的导数为 1，square 对输入的导数为 2x，组合后就是 `[4,6]`。

**backward 只计算并累加梯度，不更新原来的数值。** 如果用学习率 0.1 做最简单的梯度下降，更新规则是 `新数值 = 原数值 - 学习率×梯度`：

```text
x0：2 - 0.1×4 = 1.6
x1：3 - 0.1×6 = 2.4
新的平方和：1.6² + 2.4² = 8.32，低于原来的 13
```

上面是手算更新，不是 backward 自动执行的操作。在正常模型训练中，优化器的 step 根据参数的 .grad 完成更新，见[一次训练更新](#7-一次训练更新-谁计算-谁改参数)。

### 梯度保存在哪里，为什么要清理

本例直接创建且要求梯度的 x 是叶子 Tensor，squared、loss 是带梯度计算产生的中间结果。默认把梯度累加到要求梯度的叶子 Tensor 的 .grad；中间结果仍参与反向传播，只是通常不保留自己的 .grad。

梯度会累加。下面每次都重新做前向计算，产生新的求导关系；没有清理旧梯度时，两次相同计算得到的梯度会相加：

```python
import torch

x = torch.tensor([2., 3.], requires_grad=True, device='cuda')
x.square().sum().backward()     # 第一次前向与反向
print(x.grad.tolist())          # 预期：[4.0, 6.0]
x.square().sum().backward()     # 重新前向，再反向；累加到已有梯度
print(x.grad.tolist())          # 预期：[8.0, 12.0]
x.grad = None                  # 清理本例普通 Tensor 的旧梯度
print(x.grad)                   # 预期：None
```

真实训练中，主要需要梯度的是模型的 weight、bias 等参数；输入数据一般不必设置 requires_grad=True。模型参数的梯度由 optimizer.zero_grad 清理，再由 loss.backward 计算并累加，最后由 optimizer.step 更新参数。

自动求导与叶子规则见 [Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)，累加行为见 [Tensor.backward](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.backward.html)。

## 局部输入求导：detach、requires_grad_ 与 autograd.grad

有时需要的是“loss 对输入的梯度”，并把梯度作为函数结果返回；同时不改变调用方输入的求导标记，也不往原输入或权重的 `.grad` 中累加。可以用 **`detach()` 分离历史 → `requires_grad_(True)` 开始局部跟踪 → `autograd.grad()` 取回梯度**。

### detach()：分离求导历史，不复制数值

`local = x.detach()` 返回新的 Tensor 对象，数值、shape、dtype、device 与 x 相同，但不再连接 x 原来的计算图，且 `local.requires_grad` 为 False。它不会修改原 x 的 requires_grad 或已有 `.grad`。

**detach 后仍与原 x 共享底层 storage。** 对 local 原地修改数值，也可能修改原 x；分离求导关系不等于复制数据。只读取 local 并做非原地计算时，无需额外复制；需要独立修改数值时用 `x.detach().clone()`。存储对照见 [Tensor 的复制与梯度关系](tensors.md#存储独立与梯度关系是两件事)。

### requires_grad_(True)：从局部 Tensor 开始记录后续计算

`requires_grad` 是 bool 属性；`requires_grad_(True)` 是方法，修改调用对象的标记并返回该对象自身。下划线表示这里原地修改的是标记，不是 Tensor 的数值。

```python
import torch

x = torch.tensor([2., 3.], device='cuda')
x_local = x.detach().requires_grad_(True)  # 分离历史，再开启局部梯度跟踪
print(x.requires_grad)                    # 预期：False，原对象的标记不变
print(x_local.requires_grad)              # 预期：True
print(x_local.is_leaf)                    # 预期：True，局部计算图的叶子输入
print(x_local.grad)                       # 预期：None，尚未计算梯度
```

这个标记让正常梯度记录模式下的后续可求导运算建立求导关系，**不会立刻计算梯度，也不会恢复 detach 前的历史**。直接调用 `x.requires_grad_(True)` 会修改原对象；若要求保留原 x 的标记，应像上面一样先 detach。它也不会覆盖外层 `torch.no_grad()` 的设置。

### autograd.grad 与 backward：返回梯度，还是累加到 .grad

| 写法 | 返回什么 | 梯度放在哪里 |
| --- | --- | --- |
| `loss.backward()` | `None` | 默认累加到参与求导且要求梯度的叶子 Tensor 的 `.grad` |
| `torch.autograd.grad(loss, x_local)` | tuple，包含所请求的梯度 | 直接返回，不累加到输入的 `.grad` |

`grad_x = torch.autograd.grad(loss, x_local)[0]` 表示求 loss 对 x_local 的梯度。返回 tuple 是因为可以请求多个输入的梯度，例如 `grad_a, grad_b = torch.autograd.grad(loss, (a, b))`；只请求一个时，`[0]` 取出对应的梯度 Tensor，shape 与该输入相同。

这两个接口都会在调用时执行反向求导；读取 `.grad` 只是读取已有字段，不会触发懒计算。这里 loss 是实数标量，可以省略 `grad_outputs`；多元素输出通常需要显式提供它，指定各输出在反向求导中的权重。

### 完整例子：返回输入梯度，保留调用方的状态

```python
import torch

def input_gradient(x, weight):
    """正常梯度记录模式下求输入梯度，不修改原输入、权重或它们的 .grad。"""
    x_local = x.detach().requires_grad_(True)  # [B, Din]，新的局部求导起点
    weight_local = weight.detach()            # [Dout, Din]，本次把权重当常量
    y = x_local @ weight_local.T              # [B, Dout]，前向必须使用 x_local
    loss = y.square().mean()                  # []，对所有输出元素求平方均值
    grad_x = torch.autograd.grad(loss, x_local)[0]  # [B, Din]，直接取回梯度
    return loss, grad_x

x = torch.tensor([[2.]], requires_grad=True, device='cuda')
weight = torch.tensor([[3.]], requires_grad=True, device='cuda')
x.grad = torch.full_like(x, 7.)               # 模拟调用方已有的梯度
weight.grad = torch.full_like(weight, 11.)
loss, grad_x = input_gradient(x, weight)
print(loss.item(), grad_x.item())             # 手算预期：36.0 36.0
assert torch.equal(x, torch.full_like(x, 2.))  # 原输入数值不变
assert x.requires_grad and weight.requires_grad
assert torch.equal(x.grad, torch.full_like(x, 7.))
assert torch.equal(weight.grad, torch.full_like(weight, 11.))
```

手算：`y = 2×3 = 6`，`loss = (3x)² = 9x²`，所以在 x=2 处输入梯度为 `18×2 = 36`。权重 detach 后只是本次不对它求导，它的数值仍参与输入梯度的计算。即使不 detach 权重，仅请求输入梯度的 `autograd.grad` 也不会给 weight 累加 `.grad`。

**常见错误：创建了 x_local，却仍写 `y = x @ weight.T`。** 这样 loss 没有依赖 x_local 的计算路径，对 x_local 求导默认会报错；两个 Tensor 数值相同、共享存储，并不会自动建立求导关系。正确路径是 `x_local → y → loss`。

默认求导后，计算图为反向保存的中间数据会释放；不要直接对同一个 loss 再调用一次 backward 或 grad，通常应重新前向计算。这里返回的 grad_x 默认也不建立供高阶求导使用的计算图；需要继续对梯度求导时才考虑 `create_graph=True`。

接口见 [Tensor.detach](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html)、[Tensor.requires_grad_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.requires_grad_.html)、[autograd.grad](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html)。

## 7. 一次训练更新：谁计算，谁改参数

常见训练顺序是：清理旧梯度 → 前向预测 → 计算 loss → backward 求梯度 → optimizer.step 更新参数。优化器通过 `model.parameters()` 获得要更新的参数。

```python
import torch
from torch import nn
import torch.nn.functional as F

model = nn.Linear(1, 1, bias=False, device='cuda')
with torch.no_grad():
    model.weight.fill_(1)      # 设置初始权重 w=1，方便手算
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.tensor([[2.]], device='cuda')
target = torch.tensor([[0.]], device='cuda')

model.train()                  # 设为训练模式
optimizer.zero_grad(set_to_none=True)  # 清理上一轮的梯度
prediction = model(x)          # 前向：w*x = 2
loss = F.mse_loss(prediction, target)  # 均方误差，本例为 (2-0)^2 = 4
loss.backward()                # 求 d(loss)/dw = 8，此时还没更新 w
print(loss.item(), model.weight.grad.item())  # 预期：4.0 8.0
print(model.weight.item())      # 预期：1.0
optimizer.step()               # SGD 更新：w = 1 - 0.1*8 = 0.2
print(torch.allclose(model.weight, torch.full_like(model.weight, 0.2)))  # 预期：True
```

`zero_grad` 清梯度，`backward` 算梯度，`step` 改参数；step 不会自动清理梯度。`set_to_none=True` 让已有梯度字段变为 None，后续 backward 再产生梯度。本例的 SGD 没有设置动量或权重衰减，因此可按 `参数 -= 学习率 × 梯度` 手算。流程见 [优化训练教程](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。

## 8. train / eval 与 no_grad：模式和梯度分开控制

| 操作 | 控制什么 |
| --- | --- |
| `model.train()` | 递归设为训练模式，影响 Dropout、BatchNorm 等层 |
| `model.eval()` | 递归设为评估模式，改变这些层的行为；不自动关闭梯度 |
| `with torch.no_grad():` | 在上下文中关闭常规梯度记录；不切换模型模式 |
| `x.detach()` | 得到脱离当前梯度关系的 Tensor，仍共享原数值存储 |

推理时通常同时使用 eval 和 no_grad：

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(3, 2), nn.Dropout(p=0.5)).to('cuda')
x = torch.ones(4, 3, device='cuda')
model.eval()                   # Dropout 不再随机丢弃元素
print(model(x).requires_grad)   # 预期：True，模型参数仍要求梯度
with torch.no_grad():
    prediction = model(x)      # 计算预测，不记录这次前向的梯度关系
print(prediction.requires_grad) # 预期：False
print(tuple(prediction.shape))  # 预期：(4, 2)
```

仅含 Linear、ReLU 等层的模型，train/eval 往往不改变前向数值，但模式仍需要按用途设置。`no_grad()` 也不会把参数本身的 requires_grad 永久改成 False。detach/clone 的存储区别见 [Tensor 的赋值与复制](tensors.md#11-赋值-修改与数据复制)，模式规则见 [Evaluation mode](https://docs.pytorch.org/docs/2.14/notes/autograd.html#evaluation-mode-nn-module-eval)。

## 9. state_dict：保存和恢复模型状态

`model.state_dict()` 返回按名称组织的参数和持久 buffer；它不包含模型的 forward 代码，也不包含优化器状态。恢复时先创建结构匹配的模型，再调用 `load_state_dict(...)`。

```python
import torch
from torch import nn

model = nn.Linear(2, 1, device='cuda')
snapshot = {name: value.clone() for name, value in model.state_dict().items()}
# 本例的状态值都是 Tensor；逐个 clone，保存独立的数值快照
with torch.no_grad():
    model.weight.zero_()       # 修改当前模型参数
model.load_state_dict(snapshot)  # 把快照中的数值加载回匹配的模型
print(list(model.state_dict()))  # 预期：['weight', 'bias']
print(torch.equal(model.weight, snapshot['weight']))  # 预期：True
```

直接保存 `saved = model.state_dict()` 时，里面的 Tensor 仍与模型共享数据，后续更新参数可能改变你看到的 saved 值；独立快照需复制数值，或当时就用 `torch.save(model.state_dict(), path)` 写入文件。加载文件可用 `torch.load(path, map_location='cuda', weights_only=True)` 再传给 load_state_dict。保存/恢复见 [官方教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended)。

## 常见错误速查

| 写法或判断 | 问题 | 对应写法 |
| --- | --- | --- |
| 把 `nn.Linear(3, 2)` 当作计算结果 | 它创建层对象 | 先创建 layer，再 `layer(x)` |
| 在每次 forward 中创建 Linear | 每次产生新的参数 | 在 `__init__` 中创建并复用 |
| 用普通 list 保存需要管理的子层 | 不会注册其中的层 | 用 ModuleList 或 Sequential |
| 把普通 Tensor 属性当作模型参数 | 不会出现在 `parameters()` 中 | 显式使用 nn.Parameter |
| 模型在 GPU，输入仍在 CPU | 设备不匹配 | 分别迁移模型和输入 |
| 认为 backward 会更新权重 | 它只计算并累加梯度 | 再由优化器 step 更新 |
| 认为 eval 会关闭梯度 | 模式与梯度记录独立 | 推理时结合 eval 与 no_grad |
| 把 state_dict 当作自动冻结的快照 | 其中数值仍可与模型共享 | 保存到文件或复制 Tensor 数值 |

## 接下来读什么

[Tensor 与 GPU 内存](tensors.md)解释每一步计算怎样处理 shape、dtype、device 和 storage；[常见计算与组件](operations-and-components.md)展开 Linear、softmax、LayerNorm、RMSNorm、SwiGLU 与 attention 的 Tensor 实现。

本文核对官方接口，示例做静态语法检查，网页做构建与链接检查。注释数值为推导预期；**未执行 CUDA 示例，也未提供训练性能结论。**
