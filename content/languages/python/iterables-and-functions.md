---
title: 可迭代对象与常用函数
category: python
tags: [生成器, all, any, sum, map, filter]
description: 生成器表达式 · all / any · sum · min / max · map / filter
updated: 2026-11-04
---
# 可迭代对象与常用函数

`all`、`any`、`sum` 等是 **Python 内置函数**；`表达式 for 变量 in 对象` 则属于 **生成器表达式语法**。把它们组合起来，可以把“逐个检查、求和、转换”的循环写得简洁，同时明确要得到的结果。

本页使用 Python 3.10+，从字符异位词的判断出发，介绍生成器表达式与常用函数。相关 API 见 [官方内置函数文档](https://docs.python.org/3.12/library/functions.html)。

## 先拆开这句 all

```python
def is_anagram(s, t):
    diff_counts = {}
    for s_ch in s:
        diff_counts[s_ch] = diff_counts.get(s_ch, 0) + 1
    for t_ch in t:
        diff_counts[t_ch] = diff_counts.get(t_ch, 0) - 1
    return all(count == 0 for count in diff_counts.values())

print(is_anagram('aab', 'aba'))    # 输出：True
print(is_anagram('aab', 'abb'))    # 输出：False
print(is_anagram('', ''))         # 输出：True
```

这段写法是正确的：`diff_counts` 记录每个字符在两个字符串中的出现次数之差，所有差值都为零，两个字符串才是字符异位词。重复字符也会按次数比较。

最后一句可以分成三个部分：

| 部分 | 做什么 |
| --- | --- |
| `diff_counts.values()` | 提供各个字符的次数差，例如 `[0, 0]` 或 `[1, -1]` |
| `count == 0 for count in ...` | 每次取一个差值，产生一个 `True` / `False` 判断结果 |
| `all(...)` | 所有结果都为真才返回 `True`；遇到第一个假值就返回 `False` |

`values()` 本身是字典视图，上表中的列表只用于展示取到的数值。整个写法不会先创建一张完整的布尔结果列表。

如果把最后一句展开，它等价于下面的循环：

```python
def all_counts_zero(diff_counts):
    for count in diff_counts.values():
        if count != 0:
            return False
    return True

print(all_counts_zero({'a': 0, 'b': 0}))  # 输出：True
print(all_counts_zero({'a': 1, 'b': -1})) # 输出：False
print(all_counts_zero({}))               # 输出：True
```

注意不能直接写 `all(diff_counts.values())`：它判断的是“每个数值是否为真”，而整数 `0` 的真假值恰好是 `False`。

```python
counts = {'a': 0, 'b': 0}
print(all(counts.values()))       # 输出：False
print(all(count == 0 for count in counts.values()))  # 输出：True
```

## 可迭代对象与迭代器

**可迭代对象（iterable）**就是可以放在 `for x in ...` 后面、逐个取出元素的对象。字符串、列表、元组、字典、集合、字典视图，以及生成器都可以用于迭代。

**迭代器（iterator）**保存当前遍历的位置，用 `next()` 取下一个元素；取完后会抛出 `StopIteration`。生成器也是迭代器，通常只能消耗一次。

```python
nums = [3, 7]
it = iter(nums)
print(next(it))                  # 输出：3
print(next(it))                  # 输出：7
print(next(it, '已结束'))         # 输出：已结束
```

列表本身可以反复开始新的遍历；上面的 `it` 已经消耗完，再遍历它不会自动回到开头。

## 列表推导式与生成器表达式

### 方括号：立即创建列表

```python
nums = [1, 2, 3, 4]
squares = [x * x for x in nums if x % 2 == 0]
print(squares)                   # 输出：[4, 16]
print(sum(squares), sum(squares)) # 输出：20 20
```

通用形式是 `[结果表达式 for 变量 in 可迭代对象 if 筛选条件]`。`if` 筛选条件可以省略；结果会完整保存到新列表中，后续可以反复使用。

同一套推导式写法也能构造集合与字典：集合自动去重，字典需要同时写出键和值的表达式。

```python
nums = [1, 2, 2, 3]
odd_numbers = {x for x in nums if x % 2 == 1}
squares = {x: x * x for x in nums}
print(sorted(odd_numbers))        # 输出：[1, 3]
print(squares)                   # 输出：{1: 1, 2: 4, 3: 9}
```

### 圆括号：按需产生元素

```python
nums = [1, 2, 3, 4]
squares = (x * x for x in nums if x % 2 == 0)
print(next(squares))             # 输出：4
print(list(squares))             # 输出：[16]
print(list(squares))             # 输出：[]
```

生成器表达式先创建一个生成器，结果表达式在取元素时才执行。它不会保存完整结果；需要多次复用结果时，可以明确保存成列表。相关语言规则见 [官方生成器表达式说明](https://docs.python.org/3.12/reference/expressions.html#generator-expressions)。

生成器表达式是函数的唯一参数时，可以省略它外面单独那一层括号：

```python
nums = [1, 2, 3]
print(sum(x * x for x in nums))   # 输出：14
print(sum((x * x for x in nums), 10))  # 输出：24
```

第二句还有一个起始值参数 `10`，所以给生成器保留了单独的括号。

## 常用函数速查

| 目标 | 写法 | 结果 |
| --- | --- | --- |
| 是否全部满足条件 | `all(P(x) for x in xs)` | `bool`，遇到假值停止 |
| 是否至少一个满足条件 | `any(P(x) for x in xs)` | `bool`，遇到真值停止 |
| 求和 | `sum(f(x) for x in xs)` | 累加后的数值 |
| 求最小 / 最大元素 | `min(xs)` / `max(xs)` | 一个元素，可指定 `key` |
| 生成排序结果 | `sorted(xs, key=...)` | 一个新列表 |
| 同时取下标和值 | `enumerate(xs)` | 依次产生 `(下标, 值)` |
| 将多个输入并排遍历 | `zip(xs, ys)` | 依次产生元素元组 |
| 逐个应用函数 | `map(f, xs)` | 迭代器，按需产生 `f(x)` |
| 保留满足条件的元素 | `filter(P, xs)` | 迭代器，按需产生筛选结果 |

表中的 `P` 表示返回判断结果的函数，`f` 表示一个转换或计算函数；它们不是需要导入的特殊名字。上述函数都是内置的，直接使用即可。

### 函数作为参数与 lambda

`map`、`filter` 和 `key=` 等参数接收的是函数对象。例如 `map(double, nums)` 会对每个元素调用 `double(x)`，不是先写 `double()` 调用一次，再把返回值传进去。

`lambda 参数: 表达式` 创建一个匿名函数，调用时返回该表达式的结果。它适合在参数位置写简单的转换或条件；需要多条语句时用 `def`。

```python
double = lambda x: x * 2
print(double(3))                 # 输出：6
print(list(map(double, [1, 2, 3])))  # 输出：[2, 4, 6]
```

## all 与 any：全部 / 至少一个

`all`、`any` 检查元素的真假值，不要求元素必须已经是布尔类型。非零数、非空字符串等也为真。空输入时，`all` 返回 `True`，`any` 返回 `False`。[官方 all / any 说明](https://docs.python.org/3.12/library/functions.html#all)

```python
nums = [2, 4, 5]
print(all(x % 2 == 0 for x in nums))  # 输出：False
print(any(x % 2 == 0 for x in nums))  # 输出：True
print(all([]), any([]))           # 输出：True False
print(all([1, 2]), any([0, '']))   # 输出：True False
```

“全部满足”也就是“不存在不满足的元素”：

```python
counts = {'a': 0, 'b': 0}
print(not any(count != 0 for count in counts.values()))  # 输出：True
```

### 短路检查

生成器按需产生结果，因此 `all` 遇到第一个 `False` 后，后面的检查可以不执行：

```python
def checks():
    print('检查 1')               # 输出：检查 1
    yield True
    print('检查 2')               # 输出：检查 2
    yield False
    print('检查 3')               # 这一行不会执行
    yield True

print(all(checks()))              # 输出：False
```

`any` 类似，只是遇到第一个 `True` 就停止。若写成 `all([P(x) for x in xs])`，列表推导式会先执行完所有 `P(x)`；`all` 对列表的短路遍历不会撤销前面已经做过的计算。

## sum：求和与计数

`sum(xs)` 默认从 `0` 开始累加；可以用第二个参数指定起始值。空输入返回起始值。

```python
nums = [3, 5, 8]
print(sum(nums))                  # 输出：16
print(sum(x * x for x in nums))   # 输出：98
print(sum([], 10))                # 输出：10
```

布尔值参与数值运算时，`True` 相当于 `1`，`False` 相当于 `0`，所以可以统计满足条件的元素数目：

```python
nums = [2, 4, 5, 7]
print(sum(x % 2 == 0 for x in nums))  # 输出：2
```

这与 `all(...)`、`any(...)` 不同：求准确数量通常需要检查所有元素。字符串拼接用 `''.join(parts)`，不要用 `sum`。

## min / max：取元素与指定比较依据

```python
nums = [6, 2, 9]
print(min(nums), max(nums))       # 输出：2 9
print(min(6, 2, 9))              # 输出：2
print(max([], default=-1))       # 输出：-1
```

`min(xs)`、`max(xs)` 对空输入默认抛出 `ValueError`；可迭代对象形式可以用 `default=` 指定空输入的结果。

`key=` 接收一个函数：Python 对每个候选元素调用它，用函数返回的值比较大小，而 `min` / `max` 的返回值仍然是原来的元素。例如找最高分的名字：

```python
scores = {'Alice': 90, 'Bob': 80}
print(max(scores, key=scores.get))  # 输出：Alice
print(max(scores.values()))      # 输出：90
```

直接迭代字典得到键，所以第一句比较每个键对应的分数，返回名字；第二句只取分数，返回数值。

## sorted：得到新的有序列表

`sorted` 接受可迭代对象并返回新列表，原输入不因排序而修改。列表的 `sort()` 则原地修改并返回 `None`。

```python
words = ['pear', 'fig', 'apple']
print(sorted(words, key=len))     # 输出：['fig', 'pear', 'apple']
print(words)                     # 输出：['pear', 'fig', 'apple']
print(sorted(words, reverse=True))  # 输出：['pear', 'fig', 'apple']
```

多条件排序，可以让 `key` 返回元组；元组从第一个字段开始比较：

```python
scores = {'Alice': 90, 'Bob': 80, 'Carol': 90}
print(sorted(scores.items(), key=lambda item: (-item[1], item[0])))
# 输出：[('Alice', 90), ('Carol', 90), ('Bob', 80)]
```

上面先按分数降序，再按名字升序。

## enumerate / zip：组织循环中拿到的值

```python
words = ['a', 'b']
print(list(enumerate(words, start=1)))  # 输出：[(1, 'a'), (2, 'b')]
print(list(zip([1, 2, 3], ['a', 'b'])))  # 输出：[(1, 'a'), (2, 'b')]
```

`enumerate` 的下标默认从 `0` 开始。`zip` 默认在最短输入结束时停止；如果要求输入长度相等，Python 3.10+ 可用 `zip(..., strict=True)`，遇到长度不一致时抛出 `ValueError`。

```python
try:
    list(zip([1, 2, 3], ['a', 'b'], strict=True))
except ValueError:
    print('输入长度不同')         # 输出：输入长度不同
```

它们都按迭代方式产生结果，普通 `for` 循环直接使用即可，不必先转成列表。

## map / filter：转换与筛选

`map(f, xs)` 对每个元素应用函数，`filter(P, xs)` 保留判断结果为真的元素。两者都返回迭代器，不会立即创建完整结果列表。

```python
nums = map(int, '3 1 2'.split())
print(list(nums))                 # 输出：[3, 1, 2]
print(list(nums))                 # 输出：[]
```

刷题时，`map(int, input().split())` 常用于读取一行整数；若后面需要按下标访问或多次遍历，再写 `list(map(...))`。

```python
nums = [1, 2, 3, 4]
print(list(filter(lambda x: x % 2 == 0, nums)))  # 输出：[2, 4]
print([x for x in nums if x % 2 == 0])          # 输出：[2, 4]
print(list(map(lambda x: x * x, nums)))        # 输出：[1, 4, 9, 16]
print([x * x for x in nums])                  # 输出：[1, 4, 9, 16]
```

筛选与转换也能直接用推导式表达。需要保存结果时选列表推导式；只想让后续函数逐个消费时，可以用生成器表达式、`map` 或 `filter`。

## 使用前检查这几件事

- `all` / `any` 的条件是否写对？`all(counts.values())` 判断数值是否全为真，`all(x == 0 for x in counts.values())` 判断是否全为零，不能互换；它们也不是互为取反。
- 输入为空时，期望的结果是什么？`all` 为真、`any` 为假，`min` / `max` 默认报错。
- 拿到的是列表还是迭代器？迭代器消耗后不会自动重置，也不能直接用下标访问。
- `key=` 只控制比较依据；`min` / `max` 仍返回原元素，`sorted` 返回原元素组成的新列表。
- 字典直接迭代的是键；只处理值用 `values()`，键值一起处理用 `items()`。

## 继续查阅

- [基础数据结构](data-structures.md)：字典视图、字符串遍历、容器修改与复制。
- [官方内置函数](https://docs.python.org/3.12/library/functions.html)：`all`、`any`、`sum`、`min/max`、`sorted`、`enumerate`、`zip`、`map/filter`。
- [官方生成器表达式](https://docs.python.org/3.12/reference/expressions.html#generator-expressions)：语法与按需求值规则。
