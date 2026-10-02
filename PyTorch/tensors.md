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

## 4. 索引、切片与 mask

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

`...` 代表“补齐中间剩余的轴”，`None` 插入一个长度为 1 的轴。PyTorch 切片不支持 Python list 的负步长逆序写法；逆序可用 `torch.flip(x, dims=[某个轴])`。

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

### 用索引 Tensor 选择元素

```python
import torch

x = torch.tensor([[10, 20, 30], [40, 50, 60]], device='cuda')
cols = torch.tensor([2, 0], dtype=torch.long, device='cuda')
print(x.index_select(1, cols).tolist())  # 预期：[[30, 10], [60, 40]]
index = torch.tensor([[2, 0], [1, 1]], dtype=torch.long, device='cuda')
print(x.gather(1, index).tolist())      # 预期：[[30, 10], [50, 50]]
```

`index_select` 沿选定轴使用同一份一维索引；`gather` 允许每行选不同位置。这里 `gather(1, index)` 的关系是 `out[i, j] = x[i, index[i, j]]`，输出 shape 与 `index` 相同。索引必须在有效范围内；`gather` 的 input 与 index 维数相同，不自动按广播规则扩展它们。

## 5. 逐元素计算与比较

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

## 9. reshape、transpose 与增加/删除轴

### reshape：重新分组元素

```python
import torch

x = torch.arange(6, device='cuda')
print(x.reshape(2, 3).tolist())   # 预期：[[0, 1, 2], [3, 4, 5]]
print(x.reshape(3, 2).tolist())   # 预期：[[0, 1], [2, 3], [4, 5]]
print(tuple(x.reshape(2, -1).shape))  # 预期：(2, 3)，自动推断唯一的 -1
# x.reshape(4, 2)                # 报错：元素总数不匹配
# x.reshape(-1, -1)              # 报错：不能同时推断两个轴
```

reshape 保持元素数量，按当前逻辑遍历顺序重新分组。它不会把轴的含义自动交换；是否复制数据要看布局，下一节会具体说明。

### transpose / permute：交换轴的解释

```python
import torch

x = torch.arange(6, device='cuda').reshape(2, 3)
print(x.transpose(0, 1).tolist()) # 预期：[[0, 3], [1, 4], [2, 5]]
print(x.reshape(3, 2).tolist())   # 预期：[[0, 1], [2, 3], [4, 5]]
t = torch.zeros(2, 3, 4, device='cuda')
print(tuple(t.permute(1, 0, 2).shape))  # 预期：(3, 2, 4)
```

两个结果都可以是 `[3, 2]`，但对应元素完全不同。`transpose(i, j)` 交换两个轴；`permute(...)` 给出**原来各轴的新排列顺序**，每个轴出现一次。普通 strided Tensor 的这些操作返回 view，不直接搬动所有元素。

### unsqueeze / squeeze：处理长度为 1 的轴

```python
import torch

x = torch.tensor([10, 20, 30], device='cuda')
print(tuple(x.unsqueeze(0).shape))     # 预期：(1, 3)
print(tuple(x.unsqueeze(1).shape))     # 预期：(3, 1)
y = torch.zeros(1, 2, 1, device='cuda')
print(tuple(y.squeeze(0).shape))       # 预期：(2, 1)
print(tuple(y.squeeze().shape))        # 预期：(2,)
print(tuple(y.squeeze(1).shape))       # 预期：(1, 2, 1)，该轴长度不为 1
```

`unsqueeze` 插入长度为 1 的轴；`squeeze(dim)` 只在指定轴长度为 1 时删除它。直接 `squeeze()` 会删掉所有长度为 1 的轴，batch=1 时也可能把 batch 轴删掉；需要保留哪些轴，应显式指定。

## 10. stride 与 contiguous：shape 之外的内存布局

### shape 相同或元素相同，不代表布局相同

对普通 strided Tensor，可用下面的式子找到元素在 storage 中的位置：

```text
元素偏移 = storage_offset + i0 * stride[0] + i1 * stride[1] + ...
```

这里的 stride 和偏移以**元素**为单位；转换成字节地址，还要乘以 `element_size()`。

```python
import torch

x = torch.arange(6, device='cuda').reshape(2, 3)
t = x.transpose(0, 1)
print(tuple(x.shape), x.stride())      # 预期：(2, 3) (3, 1)
print(tuple(t.shape), t.stride())      # 预期：(3, 2) (1, 3)
print(x.is_contiguous(), t.is_contiguous())  # 预期：True False
print(x[1, 2].item(), t[2, 1].item()) # 预期：5 5
```

原 storage 的顺序是 `0, 1, 2, 3, 4, 5`。x 中沿行轴走一步跳 3 个元素，沿列轴走一步跳 1 个元素。t 把这两个 stride 交换：`t[2, 1]` 的偏移为 `2×1 + 1×3 = 5`，仍读同一个数。**转置先改变“怎么找数据”，不等于已经重排设备内存。**

`is_contiguous()` 默认检查标准 contiguous 布局；不是单纯检查底层是否来自一块显存。其他 memory format 另有布局要求。

### view、reshape、contiguous 各自保证什么

| 操作 | 保证 | 是否可能复制数值 |
| --- | --- | --- |
| `view(shape)` | 布局兼容时换 shape，共享原存储；不兼容则报错 | 不复制 |
| `reshape(shape)` | 得到指定 shape，元素数量不变 | 可能，别依赖其是否复制 |
| `contiguous()` | 得到指定 memory format 的连续布局 | 已符合时返回自身，否则复制 |
| `clone()` | 复制数据，得到独立存储 | 是 |

```python
import torch

x = torch.arange(6, device='cuda').reshape(2, 3)
t = x.transpose(0, 1)
# t.view(6)                          # 报错：这份 stride 不能直接合并成一个轴
r = t.reshape(6)                     # 此例需要复制，按 t 的逻辑顺序取数
c = t.contiguous()                   # 重排成连续的 [3, 2] 数据
v = c.view(6)                        # 此时可以仅换解释方式
print(r.tolist())                    # 预期：[0, 3, 1, 4, 2, 5]
print(c.stride(), c.is_contiguous())  # 预期：(2, 1) True
print(v.tolist())                    # 预期：[0, 3, 1, 4, 2, 5]
print(x.contiguous() is x)            # 预期：True
```

并非“所有非连续 Tensor 都不能 view”；真正条件是新 shape 与原 shape/stride 兼容。`reshape` 接口允许在需要时复制，不能在性能分析中一律当成免费的元数据操作。`contiguous()` 也不该当作每个算子前的固定步骤；先看下一步是否要求连续布局。

共享、索引与连续布局规则见 [Tensor Views](https://docs.pytorch.org/docs/2.14/tensor_view.html)，换 shape 的契约见 [view](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html) 与 [reshape](https://docs.pytorch.org/docs/2.14/generated/torch.reshape.html)。

## 11. 赋值、view、clone 与原地修改

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

`is` 只能判断 Python 对象身份，不能证明存储是否共享。基础索引读取通常返回 view；高级索引读取返回副本，但直接索引赋值仍修改原 Tensor：

```python
import torch

x = torch.tensor([10, 20, 30], device='cuda')
selected = x[[0, 2]]            # 高级索引读取：复制选中元素
selected[0] = 99
print(x.tolist())                # 预期：[10, 20, 30]
x[[0, 2]] = -1                 # 直接索引赋值：写回原 x
print(x.tolist())                # 预期：[-1, 20, -1]
```

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

`add_`、`zero_`、`fill_`、`copy_` 是常见原地操作。原地写入节省某些新分配，也影响共享存储的 view，并受 autograd 限制；不能只按“少分配一次”选择写法。与 Python list 不同，很多 Tensor 原地方法会返回自身，不是返回 `None`。

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
