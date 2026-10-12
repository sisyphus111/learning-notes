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

模型的状态也由 Tensor 保存，但注册方式决定了框架怎样管理它。

| 保存方式 | 在 `parameters()` 中 | 随 `model.to(...)` 迁移 | 在默认 `state_dict()` 中 |
| --- | --- | --- | --- |
| `self.weight = nn.Parameter(t)` | 是 | 是 | 是 |
| `self.register_buffer('offset', t)` | 否 | 是 | 是；`persistent=False` 可排除 |
| 普通 Tensor 属性 `self.cache = t` | 否 | 否 | 否 |

Parameter 常用于可学习权重，默认 `requires_grad=True`。buffer 常用于需要保存、随模型迁移但不交给优化器更新的状态，例如 BatchNorm 的运行统计量。普通 Tensor 即使设置了 requires_grad，也不会因赋给一个属性而自动成为注册参数。

```python
import torch
from torch import nn

model = nn.Module()             # 这个例子只演示状态注册和迁移
model.weight = nn.Parameter(torch.ones(3))
model.register_buffer('offset', torch.zeros(3))
model.cache = torch.zeros(3)    # 普通属性，初始也在 CPU

returned = model.to(device='cuda', dtype=torch.float32)
print(returned is model)        # 预期：True，修改同一个 Module 并返回自身
print(model.weight.device.type, model.offset.device.type, model.cache.device.type)
# 预期：cuda cuda cpu
print([name for name, _ in model.named_parameters()])  # 预期：['weight']
print(list(model.state_dict()))                       # 预期：['weight', 'offset']
```

这里为演示迁移而先创建 CPU 状态；实际可像 Linear 示例一样直接在 GPU 创建参数。`model.to(...)` 递归处理注册参数和 buffer；指定 dtype 时转换浮点、复数状态，整型 buffer 保留其 dtype。输入 x 仍需单独放到合适设备，例如 `x = x.to('cuda')`。浮点输入与层参数也需要匹配的 dtype，本页不展开混合精度。

注意调用对象的区别：`Tensor.to(...)` 返回转换结果，需要接收；目标 device/dtype 已匹配时可能返回原 Tensor。`Module.to(...)` 修改模型并返回自身。先把模型放到目标设备，再创建持有它参数的优化器。规则见 [Parameter](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parameter.Parameter.html)、[Tensor.to](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.to.html) 与 [Module.to](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html#torch.nn.Module.to)。

## 5. 常见层与 functional 写法

| 层对象 | 计算函数 | 作用 |
| --- | --- | --- |
| `nn.ReLU()` | `F.relu(x)` | 负数置零，shape 不变 |
| `nn.SiLU()` | `F.silu(x)` | `x * sigmoid(x)`，shape 不变 |
| `nn.Softmax(dim=-1)` | `F.softmax(x, dim=-1)` | 沿指定轴变成和为 1 的权重 |
| `nn.LayerNorm(D)` | `F.layer_norm(x, (D,), weight, bias)` | 对末轴 D 做归一化，层可持有缩放与偏移参数 |
| `nn.Dropout(p=0.5)` | `F.dropout(x, p=0.5, training=...)` | 训练时随机置零并缩放保留值；评估时直接返回输入 |

这些 Module 可以插入 Sequential 或自定义模型；functional 写法适合在 forward 中显式表达计算。使用 `F.dropout` 时需要传入 `training=self.training`，它默认 `training=True`，不会自动读取所属模型的模式。接口见 [functional](https://docs.pytorch.org/docs/2.14/nn.functional.html)；softmax、归一化与 SiLU 的具体计算见 [常见计算与组件](operations-and-components.md)。

## 6. 自动求导：loss、backward 与 grad

自动求导本身作用于 Tensor 运算；在神经网络中，它负责求出损失相对于模型参数的梯度。loss 是衡量预测误差的量，梯度描述参数变化对 loss 的局部影响。

先用一个可手算的例子：`loss = x0² + x1²`，对 x 的梯度为 `[2×x0, 2×x1]`。

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True, device='cuda')
loss = x.square().sum()         # 计算一个标量 loss，并记录求导所需关系
loss.backward()                # 求梯度，累加到叶子 Tensor 的 .grad
print(loss.item())              # 预期：13.0
print(x.grad.tolist())          # 预期：[4.0, 6.0]
print(x.tolist())               # 预期：[2.0, 3.0]，backward 没有更新数值
```

直接创建且要求梯度的 x 是叶子 Tensor；由带梯度运算产生的中间结果通常不是叶子，默认不保留自己的 `.grad`。模型参数一般是要求梯度的叶子 Tensor，backward 会为参与计算的参数累加梯度。上例是标量 loss，所以直接调用 backward；非标量输出需要明确传入上游梯度或先归约。

重复计算并 backward，梯度会继续累加，下一步训练通常需要先清理。自动求导与叶子规则见 [Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)和 [Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html)。

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
