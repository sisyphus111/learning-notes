# PyTorch Tensor：从形状到 GPU 内存

## 1. 创建 Tensor

### 从具体数据创建

```python
import torch

a = torch.tensor([1, 2, 3], device='cuda')
b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device='cuda')
x = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
print(a.dtype)                    # 预期：torch.int64
print(b.dtype)                    # 预期：torch.float32
print(tuple(x.shape))             # 预期：(2, 3)
print(x.tolist())                 # 预期：[[1, 2, 3], [4, 5, 6]]
```

Python 整数字面量通常推断成 int64。本文没有修改默认浮点类型；需要确定类型时显式写 `dtype=`，不要依赖程序其他位置的默认设置。

### 按 shape 创建

```python
import torch

zeros = torch.zeros(2, 3, device='cuda')
ones = torch.ones((2, 3), device='cuda')
full = torch.full((2, 3), 7, device='cuda')
seq = torch.arange(0, 6, device='cuda')
grid = seq.reshape(2, 3)
eye = torch.eye(2, device='cuda')
print(zeros.tolist())             # 预期：[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
print(full.tolist())              # 预期：[[7, 7, 7], [7, 7, 7]]
print(grid.tolist())              # 预期：[[0, 1, 2], [3, 4, 5]]
print(eye.tolist())               # 预期：[[1.0, 0.0], [0.0, 1.0]]
```

`torch.zeros(2, 3)` 和 `torch.zeros((2, 3))` 都表示两行三列。`arange(start, end, step)` 不包含 `end`；整数范围优先用整数步长，避免浮点累积使边界难以判断。

```python
import torch

x = torch.randn(2, 3, dtype=torch.float32, device='cuda')  # 标准正态随机数
u = torch.rand(2, 3, device='cuda')                       # [0, 1) 均匀随机数
ids = torch.randint(0, 10, (2, 3), device='cuda')         # 整数，0 到 9
blank = torch.empty(2, 3, device='cuda')                  # 分配空间，不承诺初始数值
same = torch.zeros_like(x)                              # 沿用 x 的 shape/dtype/device
half = torch.ones_like(x, dtype=torch.float16)           # 指定不同 dtype
print(tuple(same.shape), same.is_cuda)   # 预期：(2, 3) True
```

`empty` 里的数不能当作零或随机初始化使用；读取前先写满。`randn` 与 `rand` 的分布不同；固定随机 seed 方便复现，但不保证跨版本、跨设备得到完全相同的随机序列。

## 2. shape、维度与元素数量

### 维度是轴的数量，不是元素数量

```python
import torch

x = torch.tensor([[10, 20, 30], [40, 50, 60]], device='cuda')
print(tuple(x.shape))             # 预期：(2, 3)
print(x.ndim, x.dim())            # 预期：2 2
print(x.size(0), x.size(1))        # 预期：2 3
print(x.size(-1))                 # 预期：3
print(x.numel())                  # 预期：6
print(len(x))                     # 预期：2
```

`dim=0` 是第一个轴，`dim=1` 是第二个轴；`dim=-1` 是最后一个轴。二维表可以叫“行轴、列轴”，更高维时最好直接给轴命名。

例如 `X.shape = [B, S, D]`：`B` 表示 batch 数，`S` 表示每个样本的 token 数，`D` 表示每个 token 的特征数。`X[b, s, d]` 取第 `b` 个样本、第 `s` 个 token 的第 `d` 个特征。字母的含义来自你对数据的约定，Tensor 本身不会自动知道哪个轴是 batch。

### 标量 Tensor 与长度为 1 的向量

```python
import torch

scalar = torch.tensor(7, device='cuda')
vector = torch.tensor([7], device='cuda')
print(tuple(scalar.shape), scalar.ndim)  # 预期：() 0
print(tuple(vector.shape), vector.ndim)  # 预期：(1,) 1
print(scalar.numel(), vector.numel())    # 预期：1 1
print(scalar.item())                     # 预期：7
# len(scalar)                            # 报错：0 维 Tensor 没有第一个轴
# vector.reshape(2)                      # 报错：不能把 1 个元素变成 2 个
```

`item()` 要求总元素数为 1，不要求一定是 0 维。一个 GPU 标量 Tensor 仍在 GPU 上；调用 `item()` 后才得到 Python 数值。

## 3. dtype：类型会影响运算和存储

| 常见 dtype | 单元素字节数 | 常见用途 |
| --- | --- | --- |
| `torch.float32` | 4 | 浮点计算、参考结果 |
| `torch.float16` | 2 | 低精度模型计算，范围较小 |
| `torch.bfloat16` | 2 | 低精度模型计算，指数范围较大 |
| `torch.int64` / `torch.long` | 8 | 索引、token ID |
| `torch.int32` | 4 | 整数数据，部分算子有类型要求 |
| `torch.bool` | 1 | mask、比较结果 |

FP16 与 BF16 虽然都占 2 字节，精度和数值范围不同。用哪种类型、计算时怎样累加，取决于算子、硬件和数值要求；不能从输入 dtype 推断所有中间计算都用同一精度。

```python
import torch

x = torch.tensor([1, 2, 3], device='cuda')
f = x.float()                    # 转成 float32；原 x 不变
h = f.to(dtype=torch.float16)     # 转成 float16，仍在原 GPU
print(x.dtype, f.dtype, h.dtype)  # 预期：torch.int64 torch.float32 torch.float16
print(h.element_size())          # 预期：2
print(h.numel() * h.element_size())  # 预期：6
print(f.mean().item())           # 预期：2.0
# x.mean()                       # 报错：普通整数 Tensor 不能直接求 mean
```

`numel() × element_size()` 是**逻辑元素总字节数**。对于 view，它不等于独立分配的显存大小；广播 view 可能有很多逻辑元素，却共享很小的 storage。它也不是整个进程的显存占用，后者还包括 allocator 缓存和其他资源。

<span id="4-索引-切片与-mask"></span>

## 4. 索引：切片、高级索引、index_select 与 gather

### 按位置读取：是否保留轴

```python
import torch

x = torch.tensor([[10, 20, 30], [40, 50, 60]], device='cuda')
print(x[0, 1].item())            # 预期：20
print(x[0].tolist())             # 预期：[10, 20, 30]
print(x[:, 1].tolist())          # 预期：[20, 50]
print(x[:, 1:2].tolist())        # 预期：[[20], [50]]
print(tuple(x[:, 1].shape))      # 预期：(2,)
print(tuple(x[:, 1:2].shape))    # 预期：(2, 1)
print(x[-1, -1].item())          # 预期：60
```

整数索引 `x[:, 1]` 选中一个位置，去掉对应轴；切片 `x[:, 1:2]` 选中一个区间，保留该轴。这个区别会直接影响后续广播。

```python
import torch

x = torch.arange(24, device='cuda').reshape(2, 3, 4)
print(tuple(x[0].shape))         # 预期：(3, 4)
print(tuple(x[..., 0].shape))    # 预期：(2, 3)
print(tuple(x[:, None, :, :].shape))  # 预期：(2, 1, 3, 4)
```

`...` 代表“补齐中间剩余的轴”，`None` 插入一个长度为 1 的轴。PyTorch 切片不支持负步长；逆序可用 `torch.flip(x, dims=[某个轴])`。

### 布尔条件逐元素筛选

```python
import torch

x = torch.tensor([[1, -2, 3], [-4, 5, -6]], device='cuda')
mask = x > 0
print(mask.tolist())             # 预期：[[True, False, True], [False, True, False]]
print(x[mask].tolist())          # 预期：[1, 3, 5]
y = x.masked_fill(x < 0, 0)      # 新结果；原 x 不变
z = torch.where(x > 0, x, 0)     # 每个位置按条件选 x 或 0
print(y.tolist())                # 预期：[[1, 0, 3], [0, 5, 0]]
print(z.tolist())                # 预期：[[1, 0, 3], [0, 5, 0]]
```

`x[mask]` 的结果是一维选中元素，不保留原矩阵结构；选中数量由数据决定。`masked_fill` / `where` 适合保留网格形状。注意动态形状的筛选与固定形状逐元素选择有不同的执行代价。

<span id="用索引-tensor-选择元素"></span>

### 三种按下标选取方式速查

下面都以二维 `x.shape = [B, D]` 为例。列下标从 0 开始；整数索引 Tensor 使用 `torch.long`，示例把它和 x 放在同一 GPU 上。

| 写法 | 下标 shape | 选取规则 | 输出 shape |
| --- | --- | --- | --- |
| `x[:, columns]` | `[K]` | 每行按同一份 columns 选列，可调换顺序、重复 | `[B, K]` |
| `x.index_select(1, columns)` | `[K]` | 沿列轴按同一份 columns 选列 | `[B, K]` |
| `x.gather(1, index)` | `[B, K]` | 每行按自己的 index 选列 | `[B, K]` |

这三种**读取**都会得到独立存储的结果，允许重复下标。`index_select` 把“沿哪一个轴”写得明确；`gather` 的索引表可以为每个输出位置指定来源。高级索引还支持同时指定多个轴，见下面的配对示例。

### 高级索引：每行按同一份下标选列

```python
import torch

x = torch.tensor([[10, 20, 30], [40, 50, 60]], device='cuda')
columns = torch.tensor([2, 0, 2], dtype=torch.long, device='cuda')
out = x[:, columns]             # 保留所有行，按 2、0、2 的顺序选列
print(out.tolist())             # 预期：[[30, 10, 30], [60, 40, 60]]
print(tuple(out.shape))         # 预期：(2, 3)
print(x[:, [2, 0, 2]].tolist())  # 预期：[[30, 10, 30], [60, 40, 60]]
```

对应关系是 `out[i, j] = x[i, columns[j]]`：`:` 保留所有行，columns 的第 j 个元素决定输出第 j 列从哪里取。除了整数 Tensor，也可以像上例一样用整数列表作下标。

比较三种“只取下标 1 的列”的写法，数值来源相同，shape 和存储关系不同：

| 写法 | 输出 shape | 与 x 共享存储吗 |
| --- | --- | --- |
| `x[:, 1]` | `[B]`，去掉列轴 | 是，基础索引读取 |
| `x[:, 1:2]` | `[B, 1]`，保留列轴 | 是，基础切片读取 |
| `x[:, [1]]` | `[B, 1]`，保留列轴 | 否，高级索引读取 |

### index_select：明确沿哪个轴选取

```python
import torch

x = torch.tensor([[10, 20, 30], [40, 50, 60]], device='cuda')
columns = torch.tensor([2, 0, 2], dtype=torch.long, device='cuda')
out = x.index_select(dim=1, index=columns)  # 沿列轴选择，所有行共用下标
print(out.tolist())             # 预期：[[30, 10, 30], [60, 40, 60]]
rows = torch.tensor([1, 0], dtype=torch.long, device='cuda')
print(x.index_select(0, rows).tolist())  # 预期：[[40, 50, 60], [10, 20, 30]]
```

index 是一维整数 Tensor；输出维数与 x 相同，只有 dim 轴长度变为 index 的长度，其他轴保持不变。上例 `x.index_select(1, columns)` 与 `x[:, columns]` 选出相同数值。

### gather：每行可以按不同下标选列

```python
import torch

x = torch.tensor([[10, 20, 30], [40, 50, 60]], device='cuda')
index = torch.tensor([[2, 0], [1, 1]], dtype=torch.long, device='cuda')
out = x.gather(dim=1, index=index)  # 第一行取 2、0 列；第二行取两次 1 列
print(out.tolist())             # 预期：[[30, 10], [50, 50]]
print(tuple(out.shape))         # 预期：(2, 2)

columns = torch.tensor([2, 0, 2], dtype=torch.long, device='cuda')
same_index = columns.unsqueeze(0).expand(x.size(0), -1)  # 扩成每行相同的索引表
print(x.gather(1, same_index).tolist())  # 预期：[[30, 10, 30], [60, 40, 60]]
```

沿列轴时，关系是 `out[i, j] = x[i, index[i, j]]`，输出 shape 与 index 相同。这里 `[2, 0]` 和 `[1, 1]` 是两行各自的选列规则，不是矩阵中的数值。

gather 要求 input 与 index 维数相同；除 dim 外，`index.size(d) <= input.size(d)`。它们**不会自动广播**，不能直接把一维 columns 传给二维 x 的 gather。需要相同规则时可以像上例显式扩展索引表，也可以直接用 index_select。

### 同时给行列下标：配对选点，不是选子矩阵

```python
import torch

x = torch.tensor([[10, 20, 30], [40, 50, 60]], device='cuda')
rows = torch.tensor([0, 1], dtype=torch.long, device='cuda')
columns = torch.tensor([2, 0], dtype=torch.long, device='cuda')
print(x[rows, columns].tolist())          # 预期：[30, 40]，取 (0, 2)、(1, 0)
print(x[rows[:, None], columns].tolist()) # 预期：[[30, 10], [60, 40]]，取行列组合
```

多个整数索引 Tensor 先按广播规则对齐，再逐位置配对：两个 `[2]` 下标选出两个点；把 rows 变成 `[2, 1]`，与 columns 的 `[2]` 广播为 `[2, 2]`，才得到所有行列组合。**这个广播规则属于高级索引，不能套到 gather 上。**

index_select / gather 的下标应满足 `0 <= index < 对应轴长度`；高级索引也支持有效范围内的负数下标。读取结果与直接索引赋值的区别见[赋值与修改板块](#11-赋值-修改与数据复制)。接口契约见 [index_select](https://docs.pytorch.org/docs/2.14/generated/torch.index_select.html)、[gather](https://docs.pytorch.org/docs/2.14/generated/torch.gather.html)，高级索引规则见 [NumPy 索引说明](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)；PyTorch 的索引读取与存储关系见 [Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html)。

## 5. 逐元素计算与比较

### 加减乘除与比较

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
y = torch.tensor([10.0, 20.0, 30.0], device='cuda')
print((x + y).tolist())          # 预期：[11.0, 22.0, 33.0]
print((x * y).tolist())          # 预期：[10.0, 40.0, 90.0]
print((x / 2).tolist())          # 预期：[0.5, 1.0, 1.5]
print(x.square().tolist())       # 预期：[1.0, 4.0, 9.0]
print((x > 1).tolist())          # 预期：[False, True, True]
print(((x > 1) & (x < 3)).tolist())  # 预期：[False, True, False]
```

`+ - * /` 是逐元素运算，可以使用广播。`x == y` 返回逐元素的 bool Tensor；Python 的 `and` / `or` 不能用于多元素 Tensor 的逐元素逻辑，用 `&` / `|` 并给比较表达式加括号。

### 常用逐元素数学运算

这些运算对每个元素分别计算，返回 Tensor。下面以实数浮点 Tensor 为例：单输入运算保持 shape 和 device；二元运算按广播规则确定输出 shape。这里的写法都不原地修改 x。

| 写法 | 对每个元素做什么 |
| --- | --- |
| `x.square()` | 平方：等价于 `x * x`、`x.pow(2)` 或 `x ** 2` |
| `x.pow(p)` | p 次幂；p 可以是数值或可广播的 Tensor |
| `x.sqrt()` | 平方根：√x |
| `x.rsqrt()` | 平方根的倒数：1 / √x，归一化中常用 |
| `x.reciprocal()` | 倒数：1 / x |
| `x.abs()` | 绝对值：\|x\| |
| `x.neg()` | 取相反数：−x |
| `x.exp()` | 以 e 为底的指数：e 的 x 次方 |
| `x.expm1()` | e 的 x 次方减 1；x 接近 0 时比 `x.exp() - 1` 更准确 |
| `x.log()` | 自然对数：ln(x)，底数为 e |
| `x.log2()` / `x.log10()` | 底数分别为 2 / 10 的对数 |
| `x.log1p()` | ln(1 + x)；x 接近 0 时比 `(1 + x).log()` 更准确 |
| `x.sin()` / `x.cos()` / `x.tan()` | 正弦 / 余弦 / 正切；输入按弧度计算 |
| `x.tanh()` | 双曲正切，常用作激活函数 |
| `x.floor()` | 向下取整，朝负无穷方向 |
| `x.ceil()` | 向上取整，朝正无穷方向 |
| `x.trunc()` | 截去小数部分，朝 0 方向 |
| `x.round()` | 取最近整数；恰好在中间时取偶数 |
| `x.clamp(min=a, max=b)` | 截断到区间 [a, b]；也可只指定一个边界 |

`x.square()` 与 `torch.square(x)` 的数学含义相同；其他表中运算也有对应的 `torch.xxx(x, ...)` 写法。函数入口见 [逐元素运算参考](https://docs.pytorch.org/docs/2.14/torch.html#pointwise-ops)。

### 平方、幂、开方与倒数

```python
import torch

x = torch.tensor([-2.0, 0.0, 3.0], device='cuda')
print(x.square().tolist())       # 预期：[4.0, 0.0, 9.0]，逐元素平方
print(x.pow(3).tolist())         # 预期：[-8.0, 0.0, 27.0]，逐元素立方
print(x.abs().tolist())          # 预期：[2.0, 0.0, 3.0]，取绝对值
print(x.clamp(0, 1).tolist())    # 预期：[0.0, 0.0, 1.0]，截断到 [0, 1]
print(x.tolist())               # 预期：[-2.0, 0.0, 3.0]，原 x 不变

positive = torch.tensor([1.0, 4.0, 16.0], device='cuda')
print(positive.sqrt().tolist())       # 预期：[1.0, 2.0, 4.0]，开平方
print(positive.rsqrt().tolist())      # 预期：[1.0, 0.5, 0.25]，平方根的倒数
print(positive.reciprocal().tolist()) # 预期：[1.0, 0.25, 0.0625]，元素本身的倒数

base = torch.tensor([1.0, 2.0, 3.0], device='cuda')
power = torch.tensor([1.0, 2.0, 3.0], device='cuda')
print(base.pow(power).tolist())  # 预期：[1.0, 4.0, 27.0]，每个位置用自己的指数
```

`.square()` 对整个 Tensor 的每个位置平方，shape 不变；接 `.mean()` 才会把结果归约。例如 `torch.tensor([1., 2., 3., 4.], device='cuda').square().mean()` 得到标量 Tensor，值为 7.5。接口见 [square](https://docs.pytorch.org/docs/2.14/generated/torch.square.html)、[pow](https://docs.pytorch.org/docs/2.14/generated/torch.pow.html)、[rsqrt](https://docs.pytorch.org/docs/2.14/generated/torch.rsqrt.html)。

### 指数与对数

```python
import torch

x = torch.tensor([0.0, 1.0, 2.0], device='cuda')
y = x.exp()                    # 逐元素得到约 [1, 2.71828, 7.38906]
print(torch.allclose(y.log(), x))  # 预期：True，对正数取自然对数还原
print(torch.tensor([1.0, 2.0, 4.0], device='cuda').log2().tolist())  # 预期：[0.0, 1.0, 2.0]
print(torch.zeros(3, device='cuda').exp().tolist())    # 预期：[1.0, 1.0, 1.0]
print(torch.zeros(3, device='cuda').log1p().tolist())  # 预期：[0.0, 0.0, 0.0]
```

实数浮点 Tensor 的 `sqrt()` 对负数产生 nan；`log()` 对负数产生 nan，对 0 产生 −inf。`reciprocal()` / `rsqrt()` 在正零处产生 inf。对数和开方前应明确输入范围；并非所有运算都会对不合法的数值直接抛异常。整数 Tensor 的部分数学运算会返回浮点结果，原地版本则还要满足原 dtype 能容纳结果。

### 取整：floor、ceil、trunc 与 round

```python
import torch

x = torch.tensor([-1.5, -2.5, 1.5, 2.5], device='cuda')
print(x.floor().tolist())        # 预期：[-2.0, -3.0, 1.0, 2.0]，向下取整
print(x.ceil().tolist())         # 预期：[-1.0, -2.0, 2.0, 3.0]，向上取整
print(x.trunc().tolist())        # 预期：[-1.0, -2.0, 1.0, 2.0]，朝 0 截断
print(x.round().tolist())        # 预期：[-2.0, -2.0, 2.0, 2.0]，中间值取偶数
print(x.round().dtype)           # 预期：torch.float32，取整不等于转换为整数 dtype
```

`round()` 不是简单的“逢 5 就增大”：1.5 → 2，2.5 → 2，−1.5 → −2。`x.round(decimals=2)` 可保留两位小数，但浮点存储不保证十进制精确，低精度时还可能溢出。规则见 [round](https://docs.pytorch.org/docs/2.14/generated/torch.round.html)。

`x.square_()` 等带下划线的版本会修改原 Tensor；使用前看[赋值、修改与数据复制](#11-赋值-修改与数据复制)中的共享存储和 autograd 约束。

### 比较整个 Tensor 的内容

```python
import torch

x = torch.tensor([1.0, 2.0], device='cuda')
y = x.clone()
print(torch.equal(x, y))         # 预期：True
print(torch.allclose(x, y))      # 预期：True
print((x > 0).all().item())      # 预期：True
print((x < 0).any().item())      # 预期：False
# if x > 0:                     # 报错：多个元素不能直接当作一个布尔值
#     pass
```

`equal` 检查 shape 和元素是否完全相同，不用于检查 dtype 是否相同；`allclose` 用容差比较数值，适合浮点结果。它的基本条件是 `abs(a-b) <= atol + rtol * abs(b)`；实际容差应按 dtype 和计算需求选择，默认值不是所有算子的正确性标准。

## 6. reduction：沿一个轴把多个数归约成一个数

`sum`、`mean`、`max` 等把多个元素合成较少的结果。对于二维矩阵，`sum(dim=0)` 消去行轴，把不同行的同一列相加；`sum(dim=1)` 消去列轴，把每行内部相加。

```python
import torch

x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
print(x.sum().item())            # 预期：21.0，所有元素合成一个标量
print(x.sum(dim=0).tolist())     # 预期：[5.0, 7.0, 9.0]
print(x.sum(dim=1).tolist())     # 预期：[6.0, 15.0]
print(x.mean(dim=-1).tolist())   # 预期：[2.0, 5.0]
print(x.sum(dim=1, keepdim=True).tolist())  # 预期：[[6.0], [15.0]]
```

| 表达式 | 输出 shape | 含义 |
| --- | --- | --- |
| `x.sum()` | `[]`，0 维 | 所有元素求和 |
| `x.sum(dim=0)` | `[3]` | 每列的和 |
| `x.sum(dim=1)` | `[2]` | 每行的和 |
| `x.sum(dim=1, keepdim=True)` | `[2, 1]` | 每行的和，保留长度为 1 的归约轴 |

`keepdim=True` 主要用于让结果能沿原轴广播回去。例如 `[2, 3] / [2, 1]` 表示每行除以自己的一个数。

```python
import torch

x = torch.tensor([[1, 7, 3], [9, 5, 6]], device='cuda')
values, indices = x.max(dim=1)
print(values.tolist())           # 预期：[7, 9]
print(indices.tolist())          # 预期：[1, 0]
print(x.amax(dim=1).tolist())    # 预期：[7, 9]
print(x.argmax(dim=1).tolist())  # 预期：[1, 0]
```

`max(dim=...)` 同时返回最大值及其轴内下标，`amax` 只返回值，`argmax` 只返回下标。不要把 `x.max(dim=1)` 的整个返回对象当成一张数值 Tensor。

## 7. broadcasting：对齐轴，复用较小的数据

广播让不同 shape 的 Tensor 做逐元素运算。**从右往左对齐轴**：长度相同可以配对；其中一个长度为 1 可以扩展；缺少的前导轴按长度 1 理解。其他组合报错。

先看“给每列加一个偏置”：

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
bias = torch.tensor([10, 20, 30], device='cuda')
print((x + bias).tolist())       # 预期：[[11, 22, 33], [14, 25, 36]]
```

```text
x       [2, 3]
bias       [3]  → 对齐成 [1, 3]
result  [2, 3]
```

每一行使用同样的三个偏置。无需为了加法先用 `repeat` 复制两行；但新结果本身仍需要存储，广播不等于整次计算不分配内存。

再看“给每行加一个数”：

```python
import torch

x = torch.tensor([[1, 2, 3], [4, 5, 6]], device='cuda')
row_bias = torch.tensor([10, 20], device='cuda')
print((x + row_bias[:, None]).tolist())  # 预期：[[11, 12, 13], [24, 25, 26]]
# x + row_bias                  # 报错：[2, 3] 与 [2] 的末轴 3 和 2 不兼容
```

| 两个输入 shape | 能否广播 | 输出 shape |
| --- | --- | --- |
| `[2, 3]` 与 `[3]` | 能 | `[2, 3]` |
| `[2, 3]` 与 `[2, 1]` | 能 | `[2, 3]` |
| `[2, 3]` 与 `[2]` | 不能 | 末轴不兼容 |
| `[3, 1]` 与 `[1, 4]` | 能 | `[3, 4]` |
| `[2, 4, 8, 16]` 与 `[16]` | 能 | `[2, 4, 8, 16]` |

注意两个一维长度相同的数组默认按位置计算，广播不会理解你的业务意图。例如 `[2, 2] + [2]` 会给每列加数；如果你想给每行加数，仍应先变成 `[2, 1]`。

规则见 [Broadcasting semantics](https://docs.pytorch.org/docs/2.14/notes/broadcasting.html)。

## 8. 矩阵乘法：收缩内轴，保留外轴

### `*` 与 `@` 的关系

```python
import torch

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device='cuda')
b = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device='cuda')
print((a * b).tolist())          # 预期：[[10.0, 40.0], [90.0, 160.0]]
print((a @ b).tolist())          # 预期：[[70.0, 100.0], [150.0, 220.0]]
print(torch.matmul(a, b).tolist())  # 预期：[[70.0, 100.0], [150.0, 220.0]]
```

`a * b` 把对应位置相乘。`a @ b` 的一个输出元素，是 a 的一行与 b 的一列做点积，例如 `out[0, 0] = 1×10 + 2×30 = 70`。

```text
A: [M, K]  @  B: [K, N]  →  C: [M, N]
       内轴 K 必须相等；每个输出元素沿 K 累加
```

### 向量与批量矩阵

```python
import torch

v = torch.tensor([1.0, 2.0, 3.0], device='cuda')
print((v @ v).item())            # 预期：14.0，向量点积返回 0 维 Tensor
a = torch.ones(2, 3, 4, device='cuda')
b = torch.ones(2, 4, 5, device='cuda')
print(tuple((a @ b).shape))      # 预期：(2, 3, 5)
print(tuple(torch.bmm(a, b).shape))  # 预期：(2, 3, 5)
w = torch.ones(4, 5, device='cuda')
print(tuple((a @ w).shape))      # 预期：(2, 3, 5)，各 batch 复用同一个 w
```

`matmul` 把最后两个轴当矩阵轴，前面的轴按 batch 广播；一维输入有向量的特殊规则。`bmm` 专门处理两个三维输入，不做 batch 广播；`mm` 专门处理两个二维输入。

为 attention 提前认识这组 shape：

```python
import torch

q = torch.ones(2, 4, 8, 16, device='cuda')  # [batch, heads, query_tokens, head_dim]
k = torch.ones(2, 4, 10, 16, device='cuda') # [batch, heads, key_tokens, head_dim]
scores = q @ k.transpose(-2, -1)
print(tuple(scores.shape))        # 预期：(2, 4, 8, 10)
```

对每个 batch、每个 head，做 `[8, 16] @ [16, 10] → [8, 10]`。query token 数和 key token 数可以不同；匹配的是 head 内的特征轴 16。

## 9. 逻辑形状：reshape、permute 与增加/删除轴

### 新建 Tensor 对象，不等于新建数值存储

本节讨论普通稠密 strided Tensor。**permute、transpose 返回共享原 storage 的 view；reshape 不保证共享，布局不兼容时会复制数值。** 这不是由“默认参数”决定的，也不能把三个操作都理解成“不分配新显存”。

Tensor 对象保存 shape、stride 等元数据，storage 保存数值。创建一个新的 view 对象，只需让新元数据指向原 storage；它不会另分配一份数值 buffer。**“不复制数值”不代表连 Tensor 对象和元数据都不创建。**

| 写法 | 返回结果 | 是否新建数值 storage |
| --- | --- | --- |
| `y = x` | 同一个 Tensor 对象 | 否 |
| `y = x.permute(...)` | 新的 view 对象，重排所有轴 | 否，共享 x 的 storage |
| `y = x.transpose(i, j)` | 新的 view 对象，交换两个轴 | 否，共享 x 的 storage |
| `y = x.reshape(...)` | 布局兼容时是 view，否则复制 | 不保证；取决于当前 stride 与目标 shape |

共享 storage 时，修改 view 中的数值会影响原 Tensor 对应位置；改变 view 的 shape/stride 不会改变原 Tensor 的 shape/stride。要保证换 shape 时不复制，用 `view(shape)`，不兼容时它会报错；要保证数值独立，用 `clone()`。**非连续不意味着 reshape 一定复制，关键是目标 shape 能否由现有布局表达。** 详细例子见[共享与复制](#reshape-兼容时共享-不兼容时允许复制)和[非连续也能 view 的反例](#反例-非连续-tensor-也可能成功-view)。

接口契约见 [reshape](https://docs.pytorch.org/docs/2.14/generated/torch.reshape.html)、[permute](https://docs.pytorch.org/docs/2.14/generated/torch.permute.html) 与 [transpose](https://docs.pytorch.org/docs/2.14/generated/torch.transpose.html)。

### shape 描述逻辑轴，不决定底层元素怎样排列

先用同一组数区分三个概念：

| 概念 | 描述什么 | 本节中的值 |
| --- | --- | --- |
| storage | 底层 buffer 保存的数值 | 六个数 0、1、2、3、4、5 |
| shape | 逻辑上有几个轴，每个轴多长 | 两行三列 `[2, 3]` |
| stride | 沿某个轴走一步，在 storage 中跳过几个元素 | `(3, 1)` |

这里说的“物理排布”是 buffer 中元素的线性存储次序，不是 GPU 显存颗粒、bank 或页表等更底层结构。普通 strided Tensor 用 shape、stride 和起始偏移解释这份存储；多张不同 shape 的 Tensor 可以共享同一份数据。

```python
import torch

x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
print(tuple(x.shape), x.stride())    # 预期：(2, 3) (3, 1)
print(x.tolist())                   # 预期：[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
```

```text
底层 storage：
元素位置   0   1   2   3   4   5
保存数值  [0,  1,  2,  3,  4,  5]

用 shape=[2,3]、stride=(3,1) 解释：
[[0, 1, 2],
 [3, 4, 5]]
```

两个轴分别对应行、列。行轴 stride=3，表示换到下一行要跳 3 个元素；列轴 stride=1，表示下一列紧接着当前元素。读取 `x[i,j]` 的位置为 `i×3+j×1`，例如 `x[1,2]` 对应元素位置 5。

**逻辑上相邻，不保证存储位置也相邻。** 改变 shape/stride 可以改变看到的矩阵，却完全不移动原来的数。下一节会把索引与地址的关系展开。

### reshape：保持逻辑遍历顺序，重新分组

`reshape` 保持元素总数，将当前 Tensor 按各轴顺序、最后一轴变化最快的逻辑次序读出，再按新的 shape 分组。对这里连续的 x：

```text
原来的逻辑顺序：0, 1, 2, 3, 4, 5

reshape(3,2)：每两个元素分为一行
[[0, 1],
 [2, 3],
 [4, 5]]
```

```python
import torch

x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
r = x.reshape(3, 2)
print(r.tolist())                   # 预期：[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
print(tuple(r.shape), r.stride())    # 预期：(3, 2) (2, 1)
print(r.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # 预期：True
print(tuple(x.shape))               # 预期：(2, 3)，原 x 的 shape 没有改变
print(tuple(x.reshape(2, -1).shape)) # 预期：(2, 3)，推断唯一的 -1
# x.reshape(4, 2)                   # 报错：6 个元素不能变成 8 个
# x.reshape(-1, -1)                 # 报错：不能同时推断两个轴
```

这个例子只需把 stride 从 `(3,1)` 改为 `(2,1)`，就能在同一份 storage 上得到新矩阵。**这是本例的行为，不是 reshape 永远不复制的保证。** 当原布局无法表达目标分组时，reshape 允许复制；不能仅根据函数名判断是否发生显存分配。

`-1` 用元素总数与其他轴长度推断剩下的一个轴，不代表“自动交换轴”或“自动理解 batch”。

### permute / transpose：交换轴及其 stride

`permute(1,0)` 表示新第 0 轴来自原第 1 轴，新第 1 轴来自原第 0 轴。shape 和 stride 一起重排：

```text
原 x：shape=[2,3]，stride=(3,1)
新 p：shape=[3,2]，stride=(1,3)

原 x：               p=x.permute(1,0)：
[[0, 1, 2],          [[0, 3],
 [3, 4, 5]]           [1, 4],
                      [2, 5]]

两者仍共享 storage：[0,1,2,3,4,5]
```

```python
import torch

x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
p = x.permute(1, 0)
t = x.transpose(0, 1)
print(p.tolist())                   # 预期：[[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
print(tuple(p.shape), p.stride())    # 预期：(3, 2) (1, 3)
print(torch.equal(p, t))            # 预期：True
print(p.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # 预期：True
```

为什么 p 的第一行是 `[0,3]`？因为 `p[0,0]` 的位置是 `0×1+0×3=0`，`p[0,1]` 的位置是 `0×1+1×3=3`。这一行的两个相邻元素在底层存储中间隔 3 个位置。

普通 strided Tensor 的 `permute` 与 `transpose` 返回 view：改变轴与索引到地址的映射，不搬动原 buffer 中的数值。它们也不会修改原 x 的元数据。

| 操作 | 用法 | 轴的变化 |
| --- | --- | --- |
| `transpose(i,j)` | 交换两个轴 | 只交换 i、j |
| `permute(...)` | 给出原轴的新排列顺序 | 所有轴各出现一次 |
| `reshape(...)` | 给出目标各轴长度 | 重分组，不等于交换轴 |

三维时，`x.permute(1,0,2)` 就是把原轴 1 放到第一位、原轴 0 放到第二位、原轴 2 留在第三位。例如 `[B,S,D]` 变成 `[S,B,D]`，数值关系为 `out[s,b,d]=x[b,s,d]`。

### shape 相同，也可能对应完全不同的矩阵

```python
import torch

x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
regrouped = x.reshape(3, 2)
transposed = x.permute(1, 0)
print(tuple(regrouped.shape), tuple(transposed.shape))  # 预期：(3, 2) (3, 2)
print(regrouped.tolist())           # 预期：[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
print(transposed.tolist())          # 预期：[[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
print(torch.equal(regrouped, transposed))  # 预期：False
```

reshape 把 `0,1,2,3,4,5` 重新分组；permute 保留原元素的轴对应关系，通过交换轴让 `out[j,i]=x[i,j]`。**不要因为目标 shape 相同，就把 reshape 当成 transpose。**

### None / unsqueeze / squeeze：只增删长度为 1 的轴

```python
import torch

x = torch.tensor([10, 20, 30], device='cuda')
print(tuple(x[None, :].shape))      # 预期：(1, 3)
print(tuple(x[:, None].shape))      # 预期：(3, 1)
print(tuple(x.unsqueeze(0).shape))  # 预期：(1, 3)
print(tuple(x.unsqueeze(1).shape))  # 预期：(3, 1)
print(x[:, None].tolist())          # 预期：[[10], [20], [30]]
y = torch.zeros(1, 2, 1, device='cuda')
print(tuple(y.squeeze(0).shape))    # 预期：(2, 1)
print(tuple(y.squeeze().shape))     # 预期：(2,)
print(tuple(y.squeeze(1).shape))    # 预期：(1, 2, 1)，该轴长度不为 1
```

`None` 在索引里插入一个新轴，不向数值里添加 None。`unsqueeze` 也插入长度为 1 的轴；`squeeze(dim)` 只在指定轴长度为 1 时删除它。这些操作共享数据，不增加或减少元素总数。长度为 1 的轴只有一个有效下标，其 stride 不决定新的独立数值位置。

对 `[B,D]`，`x[:,None,:]` 得到 `[B,1,D]`；这里两个 `:` 使用原来的两个轴，None 不消耗原来的轴。直接 `squeeze()` 会删除所有长度为 1 的轴，batch=1 时也可能删掉 batch 轴，所以需要明确保留哪些轴。

## 10. 存储布局：stride、view 与 contiguous

### 用 stride 与 storage_offset 找到元素地址

对本节的普通 strided Tensor：

```text
storage 内的元素位置 = storage_offset + Σ(下标 × 对应 stride)

字节地址 = storage 起始地址 + 元素位置 × element_size()
```

stride 和 storage_offset 的单位都是**元素**。对 float32，一个元素 4 字节；对 int64，一个元素 8 字节。storage_offset 表示当前 Tensor 的第一个逻辑元素从共享 storage 的哪个位置开始。

```python
import torch

x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
part = x[1, 1:]
print(part.tolist())                # 预期：[4.0, 5.0]
print(tuple(part.shape), part.stride(), part.storage_offset())  # 预期：(2,) (1,) 4
print(part.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # 预期：True
print(part.data_ptr() == x.data_ptr())  # 预期：False，两个 Tensor 的起始元素不同
```

part 的第 0 个元素对应 storage 位置 `4+0×1=4`，第 1 个对应 `4+1×1=5`。因此，Tensor 的 `data_ptr()` 不同不能直接证明 storage 独立；本例仍共享同一份底层数据。

### contiguous 描述当前逻辑遍历与存储次序的关系

默认标准连续布局是最后一个轴变化最快，相邻逻辑元素在 storage 中依次相邻。对非空二维矩阵 `[rows,columns]`，通常的 stride 是 `(columns,1)`；长度为 1 的轴有特殊余地，不能只机械比较 stride 数字。

```python
import torch

x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
p = x.permute(1, 0)
print(x.is_contiguous(), p.is_contiguous())  # 预期：True False
print(x.stride(), p.stride())        # 预期：(3, 1) (1, 3)
```

```text
x 按当前 shape 逐行读：
数值/原位置  0 → 1 → 2 → 3 → 4 → 5
相邻元素都紧接着：连续

p 按当前 shape 逐行读：
数值/原位置  0 → 3 → 1 → 4 → 2 → 5
位置跳来跳去：非连续

两者都来自同一份 storage：
[0,1,2,3,4,5]
```

**非连续不表示这份 GPU allocation 碎成了多块。** 它描述逻辑索引在共享 buffer 中的排列方式。`is_contiguous()` 默认检查标准 contiguous memory format；channels-last 等格式有不同规则，本节不展开。

### contiguous()：保持逻辑矩阵，必要时复制成连续布局

```python
import torch

x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
p = x.permute(1, 0)
c = p.contiguous()
print(c.tolist())                   # 预期：[[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
print(tuple(c.shape), c.stride())    # 预期：(3, 2) (2, 1)
print(c.is_contiguous())            # 预期：True
print(c.untyped_storage().data_ptr() == p.untyped_storage().data_ptr())  # 预期：False
print(x.contiguous() is x)          # 预期：True，x 已经符合标准连续布局
```

```text
p：
逻辑矩阵     [[0,3],[1,4],[2,5]]
共享 storage [0,1,2,3,4,5]
stride       (1,3)

c=p.contiguous()：
逻辑矩阵     [[0,3],[1,4],[2,5]]    ← 保持不变
新 storage   [0,3,1,4,2,5]         ← 按这个逻辑顺序重新存放
stride       (2,1)
```

这里连续化复制出了新 buffer，没有改变原 storage。`contiguous()` 不是“恢复转置前的矩阵”；它保留 p 当前的元素对应关系，使这些元素连续存放。如果输入已经符合目标 memory format，它返回自身，所以也不能用它替代“保证独立副本”的 clone。

### view：只允许不用复制就能表达的目标 shape

这里的 `view(shape)` 表示换 shape，不讨论另一个重解释 dtype 的重载。它必须共享输入数据；如果目标 shape 不能由原 shape/stride 表达，就报错。

```python
import torch

x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
v = x.view(3, 2)
print(v.tolist())                   # 预期：[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
print(v.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # 预期：True
p = x.permute(1, 0)
# p.view(6)                        # 报错：不能只用一个一维 stride 表达其逻辑顺序
print(tuple(p.view(3, 2).shape))    # 预期：(3, 2)，保留原 shape 本身仍可 view
```

p 的逻辑展平顺序要求依次访问原位置：

```text
0 → 3 → 1 → 4 → 2 → 5
  +3  -2  +3  -2  +3
```

一维 Tensor 只有一个 stride，表示每走一步固定跳过几个元素。这里无法用一个固定 stride 表达这组访问位置，所以 `p.view(6)` 不能成功。

合并相邻轴时，一个关键兼容条件是：

```text
stride[i] = stride[i+1] × size[i+1]
```

原 x 的两个轴满足 `3=1×3`，能合并；p 的两个轴不满足 `1=3×2`，不能直接合并。完整条件允许拆轴、保留轴等情况，所以不要把它简化成“所有非连续输入都不能 view”。具体契约见 [Tensor.view](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html)。

### reshape：兼容时共享，不兼容时允许复制

```python
import torch

x = torch.arange(6, dtype=torch.float32, device='cuda').reshape(2, 3)
shared = x.reshape(6)              # 原 x 连续，本例展平可以共享 storage
t = x.transpose(0, 1)             # 交换行列轴，仍共享 x 的 storage
r = t.reshape(6)                  # 转置后的逻辑顺序不能用一个 stride 展平，本例复制
explicit = t.contiguous().view(6)
print(shared.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # 预期：True
print(t is x)                      # 预期：False，不同 Tensor 对象
print(t.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # 预期：True，共享数值存储
print(r.tolist())                   # 预期：[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
print(explicit.tolist())            # 预期：[0.0, 3.0, 1.0, 4.0, 2.0, 5.0]
print(r.untyped_storage().data_ptr() == t.untyped_storage().data_ptr())  # 预期：False，本例需要复制
print(tuple(t.shape), t.stride())    # 预期：(3, 2) (1, 3)，原 t 未被改动
```

reshape 保持的是**当前 Tensor 的逻辑遍历顺序**。转置后的 t 按顺序访问原 storage 的位置 `0,3,1,4,2,5`，相邻位置的间隔为 `+3,-2,+3,-2,+3`，无法用一个固定的一维 stride 表达。因此这次 reshape 必须复制，得到按 `0,3,1,4,2,5` 连续存放的新 storage；直接沿原 storage 读 `0,1,2,3,4,5` 会改变元素对应关系。

不要依赖 reshape 必然共享或必然复制；这里的具体例子需要复制，其他兼容布局则可以返回 view。性能关键路径要结合实际 stride 和下一步算子的要求判断。[reshape 官方说明](https://docs.pytorch.org/docs/2.14/generated/torch.reshape.html)

### 反例：非连续 Tensor 也可能成功 view

```python
import torch

a = torch.arange(12, dtype=torch.float32, device='cuda').reshape(3, 4)
s = a[:, ::2]
flat = s.view(6)
print(s.tolist())                   # 预期：[[0.0, 2.0], [4.0, 6.0], [8.0, 10.0]]
print(tuple(s.shape), s.stride(), s.is_contiguous())  # 预期：(3, 2) (4, 2) False
print(flat.tolist())                # 预期：[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
print(flat.stride())                # 预期：(2,)
print(flat.untyped_storage().data_ptr() == a.untyped_storage().data_ptr())  # 预期：True
```

s 的逻辑位置依次为 `0,2,4,6,8,10`，每一步都固定跳 2，可以用一维 stride=2 表达。它也满足两个轴的合并条件 `4=2×2`。**view 是否可行取决于目标 shape 的布局兼容性，而不只取决于 is_contiguous()。**

### attention head：拆分特征、重排轴，再合并

设 `B=1`、`S=2`、`H=2`、每个 head 的 `D=2`。输入为 `[B,S,H×D]=[1,2,4]`，每个 token 的四个特征分成两个 head：

```python
import torch

x = torch.arange(8, dtype=torch.float32, device='cuda').reshape(1, 2, 4)
split = x.reshape(1, 2, 2, 2)      # [B,S,H,D]：拆开特征轴
heads = split.permute(0, 2, 1, 3)  # [B,H,S,D]：把 head 放到 token 前面
print(split.stride(), heads.stride())  # 预期：(8, 4, 2, 1) (8, 2, 4, 1)
print(heads[0, 0].tolist())         # 预期：[[0.0, 1.0], [4.0, 5.0]]，head 0 的两个 token
print(heads[0, 1].tolist())         # 预期：[[2.0, 3.0], [6.0, 7.0]]，head 1 的两个 token
print(heads.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # 预期：True
```

因为这里 S 与 H 都等于 2，split 与 heads 的 shape 数字看起来完全相同，但 stride 和轴的含义已经不同。仅把 `[B,S,H×D]` reshape 成一个叫 `[B,H,S,D]` 的 shape，不能代替这次 permute。

合并时，先让同一个 token 的各个 head 相邻，再合并 H 与 D：

```python
import torch

x = torch.arange(8, dtype=torch.float32, device='cuda').reshape(1, 2, 4)
heads = x.reshape(1, 2, 2, 2).permute(0, 2, 1, 3)
back = heads.permute(0, 2, 1, 3).reshape(1, 2, 4)
print(torch.equal(back, x))         # 预期：True
print(back.untyped_storage().data_ptr() == x.untyped_storage().data_ptr())  # 预期：True
head_output = heads.contiguous()    # 模拟一份连续 [B,H,S,D] 结果
merged = head_output.permute(0, 2, 1, 3).reshape(1, 2, 4)
print(torch.equal(merged, x))       # 预期：True
print(merged.untyped_storage().data_ptr() == head_output.untyped_storage().data_ptr())  # 预期：False
```

第一条路径撤销原来的轴重排，恢复连续布局后可以共享原 storage。第二条路径从连续的 `[B,H,S,D]` 出发，permute 后 H、D 的布局不兼容直接合并，本例 reshape 需要复制。**同一组 shape 变换，是否复制还取决于实际输入布局。**

### 操作对逻辑与存储的影响速查

下表的“共享 / 复制”指底层数值存储，不是 Tensor 对象是否相同；按普通稠密 strided Tensor 的行为总结。

| 操作 | 对逻辑 shape / 元素关系的影响 | 对底层存储的影响 |
| --- | --- | --- |
| `view(shape)` | 按当前逻辑次序重新分组 | 保证共享，不兼容则报错 |
| `reshape(shape)` | 按当前逻辑次序重新分组 | 布局兼容时共享，否则复制；不保证不分配 |
| `permute(...)` | 重排所有轴及其对应 stride | 保证共享原 storage，只创建 view，不复制数值 |
| `transpose(i,j)` | 交换两个轴及其 stride | 保证共享原 storage，只创建 view，不复制数值 |
| `None/unsqueeze/squeeze` | 增删长度为 1 的轴 | 共享原数据，元素总数不变 |
| `contiguous()` | 保留当前 shape 与对应数值 | 不符合目标布局时复制；已符合时返回自身 |
| `clone()` | 保留 shape 与数值 | 得到独立存储；不等于指定某种 memory format |

view/permute 等操作只改元数据，通常不用复制 GPU 数值；连续化或不兼容的 reshape 需要读旧数据、写新 buffer。即使 permute 本身不搬数据，后续算子也会按不同 stride 访问，可能影响访问效率或触发内部复制。因此不应每次转置后都机械地 contiguous，也不能只看 shape 就判断布局与性能。

共享、连续布局与 view 的一般规则见 [Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html)，轴重排接口见 [permute](https://docs.pytorch.org/docs/2.14/generated/torch.permute.html)。本节代码与其他 GPU 示例一样，以注释给出推导预期，不把它们当作实测性能。


<span id="11-赋值-view-clone-与原地修改"></span>
<span id="11-赋值-修改-浅拷贝与深拷贝"></span>

## 11. 赋值、修改与数据复制

先区分三层：**变量名引用 Tensor 对象，Tensor 对象描述 shape/stride 等信息，storage 保存数值。** 两个名字可以引用同一个 Tensor 对象；两个不同 Tensor 对象也可以共享 storage。判断赋值或复制的影响，要分别看对象身份、数值存储与梯度关系。

### 赋值改名字绑定，原地修改改已有数据

```python
import torch

x = torch.tensor([1, 2], device='cuda')
alias = x                       # 不复制；两个名字引用同一个 Tensor
x = x + 10                     # 计算新 Tensor，再把 x 改绑到新对象
print(x.tolist())               # 预期：[11, 12]
print(alias.tolist())           # 预期：[1, 2]，alias 仍引用旧对象
current = x                     # 引用改绑后的新 Tensor
x += 1                         # 对普通 Tensor 原地加；不是 x = x + 1
print(x.tolist())               # 预期：[12, 13]
print(current.tolist())         # 预期：[12, 13]，原地修改影响同一对象的其他名字
print(alias.tolist())           # 预期：[1, 2]
alias[0] = 99                  # 修改旧 Tensor 的数据
print(alias.tolist())           # 预期：[99, 2]
```

`x = ...` 本身只改名字绑定；右侧操作决定是否分配新数据。`x[...] = ...`、`x.add_(...)` 修改已有数据，共享存储的 Tensor 也会看到变化。对普通 Tensor，`x += value` 是原地相加，受 dtype、shape 和 autograd 约束。

### 不同对象也可以共享一份数据

```python
import torch

x = torch.tensor([[1, 2], [3, 4]], device='cuda')
alias = x                       # 同一个 Tensor 对象
row = x[0]                      # 基础索引：不同 Tensor 对象，共享存储
copied = x.clone()              # 复制数值，独立存储
row[0] = 99
print(alias is x, row is x)      # 预期：True False
print(x.tolist())                # 预期：[[99, 2], [3, 4]]
print(copied.tolist())           # 预期：[[1, 2], [3, 4]]
```

`is` 只能判断是否是同一个 Tensor 对象，不能证明存储是否共享。view 有自己的 shape/stride 等信息，但仍读写原 storage；`clone()` 复制数值，得到独立存储。

### 存储独立与梯度关系是两件事

以下按普通 Tensor 的常用行为总结；梯度列假定 x 要求梯度，并启用梯度记录。

| 写法 | 与 x 的对象 / 存储关系 | 梯度能回传到 x 吗 |
| --- | --- | --- |
| `y = x` | 同一对象、同一存储 | 是 |
| `y = x[:, 1:2]` | 不同对象、共享存储 | 是 |
| `y = x[:, [1]]` | 不同对象、独立存储 | 是 |
| `y = x.clone()` | 不同对象、独立存储 | 是 |
| `y = x.detach()` | 不同对象、共享存储 | 否 |
| `y = x.detach().clone()` | 不同对象、独立存储 | 否 |

要隔离数值又保留梯度关系，用 `clone()`；要保存不参与梯度传播的独立数值快照，用 `detach().clone()`。`detach()` 单独使用仍共享数据；`contiguous()`、`reshape()`、`.to(...)` 也不能作为“保证复制”的通用写法。存储规则见 [Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html)，梯度规则见 [clone](https://docs.pytorch.org/docs/2.14/generated/torch.clone.html) 与 [detach](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html)。

### 高级索引读取会复制，直接索引赋值会写回

```python
import torch

x = torch.tensor([10, 20, 30], device='cuda')
selected = x[[0, 2]]            # 高级索引读取：复制选中元素
selected[0] = 99
print(x.tolist())                # 预期：[10, 20, 30]
x[[0, 2]][0] = 88              # 先生成临时副本，再修改副本；不会写回 x
print(x.tolist())                # 预期：[10, 20, 30]
x[[0, 2]] = -1                 # 直接索引赋值：写回原 x
print(x.tolist())                # 预期：[-1, 20, -1]
```

同一套索引写法放在读取右侧和赋值左侧，行为不同：`out = x[:, columns]` 复制选中数值；`x[:, columns] = values` 把值写回 x。避免链式高级索引赋值；先明确要修改原 Tensor 还是读取出的副本。写入时有重复目标下标，也不能套用读取时“重复取值”的规则，普通赋值不要依赖某次写入最后获胜。重复下标的写入限制见 [index_put_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.index_put_.html)。

### 下划线方法常表示原地修改

```python
import torch

x = torch.tensor([1.0, 2.0], device='cuda')
alias = x
y = x + 10                     # 新结果；x 不变
x.add_(3)                      # 原地加 3；alias 同样看到变化
print(y.tolist())                # 预期：[11.0, 12.0]
print(alias.tolist())            # 预期：[4.0, 5.0]
src = torch.tensor([8.0, 9.0], device='cuda')
x.copy_(src)                   # 把数值写进已有 x；不同于 alias = src
print(alias.tolist())            # 预期：[8.0, 9.0]
```

`add_`、`zero_`、`fill_`、`copy_` 是常见原地操作。`x.copy_(src)` 将数值写入已有 x，不会把 x 改绑为 src；返回 x 自身，见 [copy_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.copy_.html)。原地写入也影响共享存储的 view，并受 autograd 限制，例如要求梯度的叶子 Tensor 通常不能在梯度记录模式下直接原地修改。这些 Tensor 原地方法返回自身。

## 12. expand 与 repeat：复用存储还是复制元素

```python
import torch

x = torch.tensor([[10, 20, 30]], device='cuda')
expanded = x.expand(2, 3)        # 将长度为 1 的轴扩展，得到共享存储的 view
repeated = x.repeat(2, 1)        # 把数据复制两份
print(expanded.tolist())         # 预期：[[10, 20, 30], [10, 20, 30]]
print(expanded.stride())         # 预期：(0, 1)
x[0, 0] = 99
print(expanded.tolist())         # 预期：[[99, 20, 30], [99, 20, 30]]
print(repeated.tolist())         # 预期：[[10, 20, 30], [10, 20, 30]]
```

expand 的行 stride 为 0，表示“换一行仍访问同样的地址”；这不是为第二行另分配数值。不要把 expanded 的逻辑位置当作互相独立的位置来批量原地写；需要独立可写数据时先 `clone()`。

`repeat(2, 1)` 的数字是各轴的重复次数，不是最终 shape。对原 shape `[1, 3]`，结果是 `[1×2, 3×1] = [2, 3]`。一般只为参与逐元素广播，不必先 repeat。

## 13. cat、stack 与拆分

```python
import torch

a = torch.tensor([1, 2], device='cuda')
b = torch.tensor([3, 4], device='cuda')
print(torch.cat([a, b], dim=0).tolist())    # 预期：[1, 2, 3, 4]
print(torch.stack([a, b], dim=0).tolist())  # 预期：[[1, 2], [3, 4]]
print(torch.stack([a, b], dim=1).tolist())  # 预期：[[1, 3], [2, 4]]
```

`cat` 沿现有轴拼接，维数不变；除拼接轴外的 shape 要一致。`stack` 增加新轴，输入 shape 必须相同。它们通常会为结果复制数据；循环里反复把旧结果 cat 到越来越长的新结果中，会重复搬运旧数据。

```python
import torch

x = torch.arange(7, device='cuda')
parts = x.split(3)              # 每段最多 3 个元素；返回 tuple，段是 view
print([p.tolist() for p in parts])  # 预期：[[0, 1, 2], [3, 4, 5], [6]]
chunks = x.chunk(3)             # 请求拆成 3 段
print([p.tolist() for p in chunks]) # 预期：[[0, 1, 2], [3, 4, 5], [6]]
```

`split(3)` 的 3 是每段长度，`chunk(3)` 的 3 是期望段数；chunk 在某些大小下会返回少于请求数量的段。固定段数的需求可以查看 `tensor_split`。

## 14. 排序与 top-k

```python
import torch

x = torch.tensor([[3.0, 1.0, 5.0], [2.0, 9.0, 4.0]], device='cuda')
values, indices = x.sort(dim=-1)
print(values.tolist())           # 预期：[[1.0, 3.0, 5.0], [2.0, 4.0, 9.0]]
print(indices.tolist())          # 预期：[[1, 0, 2], [0, 2, 1]]
top_values, top_indices = x.topk(2, dim=-1)
print(top_values.tolist())       # 预期：[[5.0, 3.0], [9.0, 4.0]]
print(top_indices.tolist())      # 预期：[[2, 0], [1, 2]]
```

下标指向原输入在该轴上的位置。`sort` 默认升序；`topk` 默认选最大的 k 个并排序。值相同时不要假设 topk 的下标顺序稳定；算法若要求稳定排序，需要明确排序契约。

## 15. CPU / GPU 传输与异步执行

### `.to(...)` 不会自动改变原变量

```python
import torch

cpu = torch.tensor([1.0, 2.0])   # 这个例子专门演示传输，所以先创建 CPU 数据
gpu = cpu.to('cuda')            # 复制到当前 GPU，返回新的 Tensor
back = gpu.cpu()                # 复制回 CPU
print(cpu.is_cuda, gpu.is_cuda, back.is_cuda)  # 预期：False True False
same = gpu.to(device=gpu.device, dtype=gpu.dtype)
print(same is gpu)               # 预期：True，已经符合设备和类型要求，默认不强制复制
```

如果想让名字继续引用 GPU 结果，写 `x = x.to('cuda')`。尽量让参与同一次运算的 Tensor 在同一设备上；有多个 GPU 时，还要区分 `cuda:0` 与 `cuda:1`。部分算子支持 CPU 标量等例外，不能据此认为一般 CPU/GPU Tensor 可以混合计算。

需要在 GPU 上产生零张量时，直接用 `torch.zeros(..., device='cuda')`，或用 `zeros_like(x)` 继承设备，避免先在 CPU 创建再无意中传输。

### Python 返回，不等于 GPU 已经算完

```python
import torch

x = torch.ones(1024, device='cuda')
y = x * 2                      # GPU 工作通常异步提交，Python 可以继续执行
z = y + 1                      # 同一 stream 内按序执行，不必手动在两句之间同步
print(z[0].item())              # 预期：3.0；CPU 获取结果时需要等待相关 GPU 工作
```

stream 是 GPU 工作的有序队列。同一 stream 中后面的计算可以使用前面的结果；切换到多个 stream 后，需要额外处理依赖和存储生命周期。初学先使用默认执行路径。

不要用“Python 包住一行 GPU 运算花了多久”直接代表 GPU 计算时间。下面只是计时写法示例，没有提供实测数值：

```python
import torch

x = torch.ones(1024, device='cuda')
for _ in range(10):
    y = x + 1                  # warmup，不把首次初始化混入目标区间
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(100):
    y = x + 1
end.record()
end.synchronize()              # 等待结束 event，之后才能读取这段 GPU 时间
print(start.elapsed_time(end) / 100)  # 单位 ms，没有固定的预期值
```

这个例子包含分配和多次提交的影响，不是纯 kernel 的性能结论。完整 benchmark 还要定义 shape、dtype、硬件、重复次数、计时范围与统计方式。`torch.cuda.synchronize()` 等待指定设备上各 stream 的工作，不能当作每个算子后的惯例。

异步、stream 与显存管理语义见 [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)。

## 16. autograd：Tensor 的梯度信息

`requires_grad=True` 让浮点等支持求导的 Tensor 参与梯度记录。先用一个能手算的例子：`loss = x0² + x1²`，对 x 的梯度是 `[2×x0, 2×x1]`。

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True, device='cuda')
loss = x.square().sum()
loss.backward()
print(loss.item())               # 预期：13.0
print(x.grad.tolist())           # 预期：[4.0, 6.0]
d = x.detach()                  # 停止沿这个结果记录梯度，仍共享 x 的存储
c = x.detach().clone()          # 同时脱离梯度关系并复制数据
print(d.requires_grad, c.requires_grad)  # 预期：False False
with torch.no_grad():
    y = x * 2
print(y.requires_grad)           # 预期：False
```

`detach()` 不等于复制；`clone()` 本身也不等于停止梯度传播。`no_grad()` 是梯度记录上下文，不是设备切换或同步操作。梯度会累加进 `.grad`，训练循环需要按目标清理；这里仅执行一次 backward，不涉及完整训练循环。

这部分先做到读懂输入、计算、loss、backward 和梯度的关系。更完整的图与梯度说明见 [Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。

## 17. 常见错误速查

| 写法或判断 | 问题 | 处理方式 |
| --- | --- | --- |
| `x.size()` 当作元素总数 | 返回 shape，不是元素个数 | `x.numel()` |
| `len(x)` 当作元素总数 | 只取首轴长度，0 维会报错 | `x.numel()` |
| `a * b` 当作矩阵乘法 | 逐元素乘法 | 按 shape 使用 `a @ b` |
| `x.reshape(...)` 当作转置 | 重新分组没有交换元素对应关系 | `transpose` / `permute` |
| `x[:, 0]` 后仍假设二维 | 整数索引去掉一个轴 | 保留轴可用 `x[:, 0:1]` |
| 给 `[B, D]` 加 `[B]` 当作每行偏置 | 广播从末轴对齐，可能报错或沿错轴加 | 先变成 `[B, 1]` |
| `x.mean()`，x 是整数 | 算子不接受普通整数求均值 | 先明确所需浮点 dtype |
| `if x == y:`，结果有多个元素 | 多个布尔值不能直接进入 if | `equal` / `allclose` 或明确归约条件 |
| `x.to('cuda')` 后假设 x 已迁移 | 返回值没有重新绑定到 x | `x = x.to('cuda')` |
| 认为 `detach()` 会复制数据 | 它仍与原 Tensor 共享 storage | 隔离数据用 `detach().clone()` |
| 认为 `contiguous()` 总会复制 | 符合目标布局时返回自身 | 需要独立数据时用 `clone()` |
| 认为 reshape 总不分配显存 | 不兼容布局可能复制 | 看 stride 与操作契约 |
| 反复 `item()` / `tolist()` / 打印 GPU 数值 | CPU 取数，可能产生同步和传输 | 检查时少量使用，关键路径避免反复调用 |
| 用 Python 提交时间代表 GPU 时间 | GPU 通常异步执行 | 明确计时边界，使用 event 或同步计时 |

## 常见计算与组件的实现

[常见计算与组件的 Torch 实现](operations-and-components.md)：Linear、stable softmax、LayerNorm、RMSNorm、SwiGLU 与 scaled dot-product attention。

## 官方参考与验证范围

- [Tensors 教程](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)：创建、属性、运算与设备。
- [torch API](https://docs.pytorch.org/docs/2.14/torch.html)：各算子的参数、返回值与约束。
- [Broadcasting semantics](https://docs.pytorch.org/docs/2.14/notes/broadcasting.html)：对齐规则及原地操作的约束。
- [Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html)：共享存储、索引与连续布局。
- [Tensor.view](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html)、[reshape](https://docs.pytorch.org/docs/2.14/generated/torch.reshape.html)：换 shape 的契约。
- [CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html)：设备、异步、stream 与显存管理。
- [Autograd 教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)：梯度记录与 backward。

本文核对了官方 API 说明，示例做静态 Python 语法检查，网站做构建与链接检查。**未执行 CUDA 示例，也没有提供实测性能结论。**
