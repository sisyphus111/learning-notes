# 数学运算速查

普通四则运算直接写运算符；`abs`、`round`、`pow` 等内置函数直接调用；开方、取整、最大公约数等用标准库 `math`，先写 `import math`。

示例使用 Python 3.10+，注释标明结果与用途。参考 [运算符文档](https://docs.python.org/3.12/reference/expressions.html#binary-arithmetic-operations)、[内置函数文档](https://docs.python.org/3.12/library/functions.html)与 [math 文档](https://docs.python.org/3.12/library/math.html)。

## 基础运算：加减乘除、取余、乘方

```python
7 + 2                           # 9：加法
7 - 2                           # 5：减法
7 * 2                           # 14：乘法
7 / 2                           # 3.5：普通除法
8 / 2                           # 4.0：整数相除也返回 float
7 // 2                          # 3：向下取整除法
7 % 2                           # 1：取余
2 ** 3                          # 8：乘方，2 的 3 次方
2 ** -2                         # 0.25：负指数，等于 1 / 2**2
7.0 // 2                        # 3.0：有 float 参与，结果仍是 float
# 1 / 0                         # ZeroDivisionError：除数不能为零
```

`**` 是乘方，`^` 是按位异或。乘除先于加减；不确定优先级时加括号。

```python
2 + 3 * 4                       # 14：先乘后加
(2 + 3) * 4                     # 20：括号优先
-2 ** 2                         # -4：等于 -(2 ** 2)
(-2) ** 2                       # 4：对负数整体求平方
2 ** 3 ** 2                     # 512：等于 2 ** (3 ** 2)
2 ^ 3                           # 1：按位异或，不是乘方
```

## 整除与取余：负数也向下取整

`//` 向负无穷取整，不是直接去掉小数。对整数 `a`、非零整数 `b`，有 `a == (a // b) * b + a % b`；非零余数与除数 `b` 同号。

```python
7 // 3                          # 2：向下取整
7 % 3                           # 1：7 = 2 * 3 + 1
-7 // 3                         # -3：-2.333... 向下取整
-7 % 3                          # 2：-7 = (-3) * 3 + 2
7 // -3                         # -3：仍向下取整
7 % -3                          # -2：余数与除数同号
divmod(7, 3)                    # (2, 1)：同时得到商与余数
divmod(-7, 3)                   # (-3, 2)：遵循同样的规则

q, r = divmod(17, 5)            # 解包：商 q 为 3，余数 r 为 2
print(q, r)                     # 输出：3 2
```

和 C++ 对比：整数除法 `-7 / 3` 在 C++ 中朝零截断得到 `-2`，Python 的 `-7 // 3` 得到 `-3`。

### 常用整数写法

```python
n, k = 10, 3
(n + k - 1) // k                # 4：向上取整，适用于 n >= 0、k > 0
-(-n // k)                      # 4：整数向上取整，适用于整数 n、k > 0
n % 2 == 0                      # True：判断偶数
n % 2 != 0                      # False：判断奇数，也适用于负整数
123 % 10                        # 3：取非负整数的末位
123 // 10                       # 12：去掉非负整数的末位
```

大整数的整除直接用 `//`，不要先 `/` 再 `int()`，中间转成 float 可能损失精度。

## abs、pow：绝对值与幂

```python
abs(-7)                         # 7：求绝对值
abs(3 - 10)                     # 7：计算两数距离
pow(2, 10)                      # 1024：等价于 2 ** 10
pow(2, 10, 1000)                # 24：整数模幂，计算 2**10 % 1000
```

整数模幂用三参数 `pow(a, b, mod)`，可避免先生成完整的巨大幂。本页的模幂示例使用非负整数指数、正整数模数。

`math.pow()` 会转成浮点运算；整数乘方通常使用 `**` 或内置 `pow()`。

## 取整与 round：规则不同

| 写法 | 含义 | `2.7` 的结果 | `-2.7` 的结果 |
| --- | --- | --- | --- |
| `int(x)` | 朝零截断 | `2` | `-2` |
| `math.floor(x)` | 向下取整 | `2` | `-3` |
| `math.ceil(x)` | 向上取整 | `3` | `-2` |
| `round(x)` | 舍入到最近整数 | `3` | `-3` |

```python
import math

int(-2.7)                       # -2：去掉小数部分
math.floor(-2.7)                 # -3：向负无穷取整
math.ceil(-2.7)                  # -2：向正无穷取整
round(2.5)                      # 2：恰好在中间，取偶数
round(3.5)                      # 4：恰好在中间，取偶数
round(-2.5)                     # -2：负数同样取偶数
round(3.14159, 2)                # 3.14：保留到小数点后两位
round(125, -1)                  # 120：舍入到十位，中间取偶数
round(2.675, 2)                 # 2.67：受 float 的二进制表示误差影响
```

`round` 不是简单的“四舍五入”；恰好位于两候选值中间时取偶数。内置 float 不传 `ndigits` 时返回 int，传入时返回 float；`round(3.0, 2)` 仍是 `3.0`。

只想显示两位小数时用格式化，得到的是字符串：

```python
value = 3.5
text = f'{value:.2f}'            # '3.50'：格式化显示，不改变 value
print(text, value)              # 输出：3.50 3.5
```

## math：开方、对数与常量

```python
import math

math.sqrt(9)                    # 3.0：平方根，返回 float
math.isqrt(10)                  # 3：整数平方根，满足 3**2 <= 10 < 4**2
math.log(1)                     # 0.0：自然对数，底数为 e
math.log2(8)                    # 3.0：以 2 为底的对数
math.log10(100)                 # 2.0：以 10 为底的对数
math.log(8, 2)                  # 3.0：指定底数
math.exp(0)                     # 1.0：计算 e**x
round(math.pi, 4)               # 3.1416：圆周率
round(math.e, 4)                # 2.7183：自然常数
# math.sqrt(-1)                 # ValueError：math.sqrt 不支持负数
# math.log(0)                   # ValueError：实数对数的输入必须大于零
```

判断整数是否为完全平方数，用 `r = math.isqrt(n)` 再比较 `r * r == n`，避免浮点开方误差；`n` 必须是非负整数。

## math：最大公约数、最小公倍数、排列组合

```python
import math

math.gcd(12, 18)                # 6：最大公约数
math.gcd(12, 18, 24)            # 6：也支持多个整数
math.gcd(-12, 18)               # 6：结果非负
math.gcd(0, 0)                  # 0：两个输入均为零
math.lcm(4, 6)                  # 12：最小公倍数
math.lcm(0, 6)                  # 0：任一输入为零，结果为零
math.factorial(5)               # 120：阶乘 5!
math.factorial(0)               # 1：0! 等于 1
math.comb(5, 2)                 # 10：选 2 个，不考虑顺序
math.perm(5, 2)                 # 20：选 2 个，考虑顺序
math.comb(3, 5)                 # 0：选取数量超过总数
# math.factorial(-1)            # ValueError：阶乘要求非负整数
```

`factorial(n)`、`comb(n, k)`、`perm(n, k)` 使用非负整数参数，结果都是整数。

## 求和、连乘与最值

```python
import math

nums = [2, 3, 4]
sum(nums)                       # 9：求和
sum(x * x for x in nums)         # 29：平方和
math.prod(nums)                 # 24：连乘
sum([])                         # 0：空输入的和
math.prod([])                   # 1：空输入的积
min(nums)                       # 2：最小值
max(nums)                       # 4：最大值
min([], default=0)              # 0：指定空输入的默认值
# max([])                       # ValueError：空输入且没有 default
```

`sum`、`min`、`max` 的生成器写法与 `key=` 用法见[可迭代对象与常用函数](python-iterables-and-functions.html#sum-求和与计数)。

## float：近似比较与无穷大

许多十进制小数不能用二进制 float 精确表示，误差会参与后续运算。浮点误差的背景见[官方说明](https://docs.python.org/3.12/tutorial/floatingpoint.html)。

```python
import math

0.1 + 0.2                       # 0.30000000000000004：浮点误差
0.1 + 0.2 == 0.3                # False：直接比较不相等
math.isclose(0.1 + 0.2, 0.3)     # True：按容差比较
math.isclose(1e-12, 0.0, abs_tol=1e-9)  # True：接近零时设置绝对容差

inf = float('inf')              # 创建正无穷，也可写 math.inf
10 < inf                        # True：常用于最小值的初始上界
-inf < -10                      # True：负无穷可作最大值的初始下界
math.isinf(inf)                 # True：判断无穷大
math.isfinite(3.5)              # True：判断是否为有限数值
math.isnan(float('nan'))         # True：判断 NaN
```

`isclose` 的 `rel_tol` 按数值大小限定相对误差，`abs_tol` 限定绝对误差；容差应按题意设置。`inf` 是 float，不是一个很大的整数；NaN 用 `math.isnan()` 判断，不要用 `==`。

Python 的 int 不受固定 32/64 位宽度限制，但受可用内存限制；大整数运算也会随位数增加而变慢。涉及精确十进制或分数运算时，可以进一步使用 `decimal.Decimal` 或 `fractions.Fraction`。
