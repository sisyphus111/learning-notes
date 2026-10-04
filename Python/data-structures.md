# Python 基础数据结构用法速查

重点是**怎么写、返回什么、是否修改原对象、哪里容易出错**。示例适用于 Python 3.10+，每个代码块可以独立运行。

行尾注释标出表达式的值或操作后容器的内容；方法返回值会单独注明。运行 `.py` 文件时，单独写表达式不会打印，需要使用 `print(...)`。报错示例已注释，可取消注释观察。

## 先选对容器

| 类型 | 示例 | 主要用途 | 能否修改自身 |
| --- | --- | --- | --- |
| `list` | `[10, 20]` | 按位置保存元素，增删改查、栈 | 能 |
| `tuple` | `(10, 20)` | 坐标、固定字段、多返回值 | 不能替换元素 |
| `dict` | `{'a': 1}` | key 到 value 的映射 | 能 |
| `set` | `{10, 20}` | 去重、成员查询、集合运算 | 能 |
| `frozenset` | `frozenset({1, 2})` | 不可变集合、可哈希状态 | 不能 |
| `str` | `'hello'` | 文本字符串 | 不能 |
| `range` | `range(5)` | 描述整数序列，用于循环 | 不能 |
| `bytes` | `b'hello'` | 二进制数据 | 不能 |
| `bytearray` | `bytearray(b'hello')` | 可修改的二进制数据 | 能 |

“有序”是有位置或插入顺序，不代表自动排序。`dict` 保留插入顺序；`set` 不承诺遍历顺序。`int`、`float`、`bool`、`None` 是常用值，不是多元素容器。

## 通用操作与对象引用

### 长度、成员检查与真假判断

```python
a = [10, 20, 30]
len(a)                           # 3：元素数量
20 in a                          # True：检查是否包含 20
99 not in a                      # True
isinstance(a, list)              # True：类型检查
bool([])                         # False：空容器
bool([0])                        # True：容器非空
bool('False')                    # True：非空字符串
```

| 对象 | `len(x)` 统计什么 |
| --- | --- |
| list、tuple、deque、range | 元素个数 |
| dict、defaultdict、字典视图 | 键值对 / 条目个数 |
| set、frozenset | 去重后的元素个数 |
| str | Unicode 码点个数，不是编码后的字节数 |
| Counter | 保存的键数，不是计数之和 |

生成器、map、filter 等迭代器通常不能调用 `len()`。空容器为假；判断 `None` 用 `x is None`，不要与 `0`、`False` 或空字符串混为一谈。

### 值、对象引用与变量名

`=` 绑定名字，不复制对象；`==` 比较内容；`is` 比较对象身份。数字、字符串按内容比较。

```python
a = [1, 2]
b = a                            # 两个名字引用同一个 list
c = a.copy()                     # 新建外层 list
a == c                           # True：内容相等
a is c                           # False：对象不同
a is b                           # True：同一个对象
b.append(3)
print(a)                         # 输出：[1, 2, 3]
print(c)                         # 输出：[1, 2]

nums = [10, 20]
x = nums[0]                     # x 绑定到该元素对象
x = 99                           # x 改绑，不修改 nums[0]
print(x, nums)                   # 输出：99 [10, 20]
```

整数、字符串不可变；列表、字典可变。`x = 99` 改名字的绑定，`nums[0] = 99` 改列表位置，`nums[0].append(99)` 则修改该位置引用的内层列表。

### 赋值、浅拷贝与二维列表

```python
from copy import deepcopy

a = [[1], [2]]
shallow = a.copy()               # 新外层 list，内层仍共享
deep = deepcopy(a)               # 这个例子里的内层 list 也独立
shallow[0].append(9)
print(a)                         # 输出：[[1, 9], [2]]
print(deep)                      # 输出：[[1], [2]]

bad = [[0, 0]] * 2               # 两个位置引用同一行
bad[0][0] = 7
print(bad)                       # 输出：[[7, 0], [7, 0]]
good = [[0, 0] for _ in range(2)] # 每行独立创建
good[0][0] = 7
print(good)                      # 输出：[[7, 0], [0, 0]]
```

`a.copy()`、`a[:]`、`list(a)` 都可浅拷贝列表。深拷贝用于确实需要隔离嵌套可变对象的情况，不是每次赋值或交换都要做的步骤。

### 传进函数后：改参数名，还是改列表？

```python
def swap_values(a, b):
    a, b = b, a                  # 只改变函数内的参数名绑定

def swap_positions(nums, left, right):
    nums[left], nums[right] = nums[right], nums[left]

nums = [10, 20]
swap_values(nums[0], nums[1])
print(nums)                      # 输出：[10, 20]
swap_positions(nums, 0, 1)
print(nums)                      # 输出：[20, 10]
```

函数参数引用调用方提供的对象：改绑参数名不改变调用方变量，修改共享的可变对象会被调用方看到。

## list：列表

### 创建、索引与切片

```python
a = []                           # 空 list
b = list('cat')                  # ['c', 'a', 't']
c = list(range(3))               # [0, 1, 2]
zeros = [0] * 4                  # [0, 0, 0, 0]

a = [10, 20, 30, 40, 50]
a[0]                             # 10
a[-1]                            # 50：最后一个元素
a[-2]                            # 40：倒数第二个
a[1:4]                           # [20, 30, 40]：不包含 stop
a[:3]                            # [10, 20, 30]
a[2:]                            # [30, 40, 50]
a[:]                             # [10, 20, 30, 40, 50]：浅拷贝
a[::2]                           # [10, 30, 50]
a[::-1]                          # [50, 40, 30, 20, 10]：逆序副本
a[100:]                          # []：切片越过边界允许
# a[100]                         # IndexError：单个下标越界
# a[::0]                         # ValueError：步长不能为 0
```

切片一般创建新列表，不是免费视图；长度为 `k` 的切片要处理 `k` 个元素。

### 添加、删除与查找

```python
a = [1, 2]
a[1] = 9                         # 修改下标 1；a 为 [1, 9]
a.append(3)                      # 末尾添加；a 为 [1, 9, 3]；返回 None
a.extend([4, 5])                 # 逐个添加；a 为 [1, 9, 3, 4, 5]；返回 None
a.insert(1, 8)                   # 在下标 1 前插入；a 为 [1, 8, 9, 3, 4, 5]；返回 None
last = a.pop()                   # 删除末尾并返回；last 为 5，a 为 [1, 8, 9, 3, 4]
first = a.pop(0)                 # 删除头部并返回；first 为 1，a 为 [8, 9, 3, 4]
a.remove(9)                      # 删除首个值 9；a 为 [8, 3, 4]；返回 None
del a[1:]                        # 删除下标 1 起的区间；a 为 [8]
a.clear()                        # 清空；a 为 []；返回 None

b = [1, 2]
b.append([3, 4])                 # 末尾添加整个 list；b 为 [1, 2, [3, 4]]
c = [1, 2]
c.extend([3, 4])                 # 末尾逐个添加；c 为 [1, 2, 3, 4]
```

`pop(i)` 按下标删除并返回元素；`remove(x)` 按值删除第一个匹配项，不返回元素。`del` 是语句，不是方法。

```python
a = [10, 20, 20, 30]
a.count(20)                      # 2：出现次数
a.count(99)                      # 0：统计 99 的出现次数
a.index(20)                      # 1：第一个匹配的下标
a.index(20, 2)                   # 2：从下标 2 开始找
20 in a                          # True：检查是否包含 20
# a.index(99)                    # ValueError：不是返回 -1
# a.remove(99)                   # ValueError
# a.pop(100)                     # IndexError
```

尾部 `append`、`pop()` 通常为摊销 $O(1)$；中间插入、删除需要移动后续元素，最坏为 $O(n)$。反复查成员时考虑 `set` 或 `dict`。

### 排序、反转与栈

```python
a = [3, 1, 2]
b = sorted(a)                    # 升序排序为新 list；b 为 [1, 2, 3]，a 不变
result = a.sort()                # 原地升序排序；a 为 [1, 2, 3]，result 为 None
a.reverse()                      # 原地逆序；a 为 [3, 2, 1]；返回 None
print(b, a, result)              # 输出：[1, 2, 3] [3, 2, 1] None

words = ['pear', 'fig', 'banana']
sorted(words, key=len)           # ['fig', 'pear', 'banana']
records = [('Lin', 80), ('Mei', 95), ('An', 80)]
sorted(records, key=lambda row: (-row[1], row[0]))
# [('Mei', 95), ('An', 80), ('Lin', 80)]：分数降序，同分按名字升序
```

`sort()` 原地排序，`sorted()` 返回新列表；不要写 `a = a.sort()`。`key` 是提取比较依据的函数，同 key 元素保持原来的相对顺序。

```python
stack = []
stack.append(6)                  # 入栈
stack.append(9)
stack[-1]                        # 9：查看栈顶
stack.pop()                      # 9：出栈，stack 剩 [6]
# 访问栈顶或出栈前，先确认 stack 非空。
```

### 交换两个位置：先求右边，再写左边

```python
nums = [10, 20]
left, right = 0, 1
nums[left], nums[right] = nums[right], nums[left]
print(nums)                      # 输出：[20, 10]
```

先完整求出右边两个结果，再从左到右写入目标位置；原来的元素不会被第一次写入覆盖掉。普通列表的固定下标交换为 $O(1)$，不复制元素对象。

```python
nums = [10, 20]
nums[0] = nums[1]
nums[1] = nums[0]                # 此时才读取 nums[0]，已经是 20
print(nums)                      # 输出：[20, 20]
# 两行直接赋值不是交换。
```

### 交换嵌套列表：对象没有被复制

```python
a, b = [10], [20]
nums = [a, b]
saved = nums[0]                  # saved 引用 a
nums[0], nums[1] = nums[1], nums[0]
nums[0] is b                     # True
saved is nums[1]                 # True：saved 仍引用 a，不跟着下标改绑
saved.append(99)
print(nums)                      # 输出：[[20], [10, 99]]
```

### += 也要看对象类型

```python
x = 10
y = x
x += 1                           # 整数不可变，x 改绑到结果
print(x, y)                      # 输出：11 10

a = [1]
alias = a
a += [2]                        # 对 list 原地扩展
print(a, alias, a is alias)      # 输出：[1, 2] [1, 2] True
a = a + [3]                     # 新建列表，再让 a 改绑
print(a, alias, a is alias)      # 输出：[1, 2, 3] [1, 2] False
```

### 切片赋值与拼接

```python
a = [10, 20, 30, 40]
a[1:3] = [7, 8, 9]              # a 为 [10, 7, 8, 9, 40]：允许改变长度
a[1:1] = [5]                    # a 为 [10, 5, 7, 8, 9, 40]：插入
a[1:3] = []                     # a 为 [10, 8, 9, 40]：删除
b = a + [50]                    # 新列表；a 不变
print(a, b)                      # 输出：[10, 8, 9, 40] [10, 8, 9, 40, 50]
```

## tuple：元组

用于坐标、固定字段、多返回值。不能替换元素，但元素引用的可变对象仍可修改。

### 直接构造：空元组、单元素、多元素

```python
empty = ()                      # 构造空 tuple
empty = tuple()                 # 同样得到空 tuple：()
single = (10,)                   # 单元素 tuple，逗号不能省
single = 10,                     # 省略括号，仍是 (10,)
not_tuple = (10)                 # 整数 10，括号只做分组
point = (3, 5)                   # 构造两个元素的 tuple
point = 3, 5                     # 省略括号，仍是 (3, 5)
mixed = (1, 'cat', True)          # 元素可以是不同类型
```

非空元组由**逗号**构造，括号主要用于分组；空元组写成 `()`。在函数调用里要保留括号：`f((3, 5))` 传入一个元组，`f(3, 5)` 传入两个参数。

### 从其他可迭代对象转换

`tuple(iterable)` 按迭代顺序收集元素。可传入 list、字符串、range、生成器等。

```python
tuple([1, 2])                    # (1, 2)：从 list 转换
tuple('cat')                     # ('c', 'a', 't')：逐字符转换
tuple(range(3))                  # (0, 1, 2)：从 range 转换
tuple(x * x for x in range(3))   # (0, 1, 4)：收集生成器的结果
tuple({'a': 1, 'b': 2})          # ('a', 'b')：迭代 dict 的 key

g = (x * x for x in range(3))    # 创建生成器，不是 tuple
tuple(g)                         # (0, 1, 4)：消费生成器，构造 tuple
tuple(g)                         # ()：生成器已耗尽
# tuple(10)                      # TypeError：整数不可迭代
# tuple(1, 2)                    # TypeError：不能传入两个元素参数
```

没有“元组推导式”：`(x for x in ...)` 是生成器表达式，要得到元组需写 `tuple(x for x in ...)`。转换保留元素引用，不会深拷贝内部对象。

### 读取、解包与不可变性

```python
point = (3, 5)
x, y = point                    # 解包：x 为 3，y 为 5
point[0]                         # 3：读取第一个元素
point[-1]                        # 5：读取最后一个元素
# point[0] = 9                   # TypeError

t = ([1], 'label')
t[0].append(2)                   # 修改内部 list，没有替换 tuple 元素
print(t)                         # 输出：([1, 2], 'label')
```

只含可哈希元素的 tuple 可以作字典 key；`(1, 2)` 可以，`([1, 2],)` 不可以。

## dict：字典

### 创建、读取与修改

```python
d = {}                           # 空 dict，不是空 set
d = {'name': 'Lin', 'age': 20}
d['name']                        # 'Lin'：按 key 读取 value
'name' in d                      # True：检查 key
'Lin' in d                       # False：不检查 value
'Lin' in d.values()              # True：检查是否存在这个 value
d['age'] = 21                   # 修改已有 key
d['city'] = 'Shanghai'           # 新增 key
d.update({'age': 22})            # 更新或添加；返回 None
positions = {(1, 2): 'A'}         # tuple 可作 key
# positions[[1, 2]] = 'B'         # TypeError：list 不可哈希
```

key 必须可哈希，常用数字、字符串、可哈希 tuple、frozenset 可以；list、dict、set 不可以。value 不受此限制。

### 键不存在时：返回默认值，还是插入默认值？

| 操作 | 缺失时结果 | 插入 key 吗 |
| --- | --- | --- |
| `d[key]`、`d[key] += 1` | `KeyError` | 否 |
| `d.get(key)` | `None` | 否 |
| `d.get(key, 0)` | `0` | 否 |
| `d.setdefault(key, 0)` | `0` | 是 |
| `d.pop(key, 0)` | `0` | 否 |
| `defaultdict(int)[key]` | `0` | 是 |
| `Counter()[key]` | `0` | 否 |

```python
d = {}
d.get('x', 0)                    # 0：读取缺失 key 的默认值，不插入，d 仍为空
d.setdefault('x', 0)             # 0：读取或插入默认值，此处新增 'x': 0
d.setdefault('x', 99)            # 0：读取已有值，不覆盖它
d['x'] += 1
print(d)                         # 输出：{'x': 1}

counts = {}
for ch in 'aba':
    counts[ch] = counts.get(ch, 0) + 1
print(counts)                    # 输出：{'a': 2, 'b': 1}
```

`get` 不插入默认值；`setdefault` 缺失时插入，不传默认值则插入 `None`。`defaultdict` 的默认工厂为 `None` 时仍会抛出 `KeyError`。

### 删除、复制与合并

```python
d = {'a': 1, 'b': 2, 'c': 3}
value = d.pop('b')               # 按 key 删除并返回；value 为 2，删除 'b'
pair = d.popitem()               # 删除并返回最后插入的条目；pair 为 ('c', 3)
del d['a']                       # 按 key 删除；d 为 {}
d.pop('missing', 0)              # 0：缺失时返回默认值
# d.pop('missing')               # KeyError

d = {'a': 1}
copied = d.copy()                # 浅拷贝为新字典，内部对象仍可共享
merged = d | {'a': 9, 'b': 2}    # 合并为新字典；merged 为 {'a': 9, 'b': 2}；d 不变
d |= {'b': 2}                    # 原地合并
d.clear()                        # 原地清空；返回 None
```

### 遍历、计数与分组

```python
d = {'a': 1, 'b': 2}
for key in d:                    # 与 for key in d.keys() 等价
    print(key)
# 输出：a
# 输出：b
for value in d.values():         # 遍历所有 value
    print(value)
# 输出：1
# 输出：2
for key, value in d.items():     # 遍历键值对，每轮解包 (key, value)
    print(key, value)
# 输出：a 1
# 输出：b 2
list(d.keys())                   # ['a', 'b']：取 key 快照
list(d.values())                 # [1, 2]：取 value 快照
list(d.items())                  # [('a', 1), ('b', 2)]：取键值对快照
```

字典保留插入顺序，不自动排序。`keys/values/items` 返回动态视图，不是列表快照；视图不能用 `[0]` 访问。`values()` 保留重复值。

```python
d = {'a': 1, 'b': 2}
view = d.values()
snapshot = list(view)
d['c'] = 3
print(list(view), snapshot)      # 输出：[1, 2, 3] [1, 2]

groups = {}
groups.setdefault('fruit', []).append('apple')
print(groups)                    # 输出：{'fruit': ['apple']}
```

遍历时不要增删字典 key。要删除，可遍历 `list(d)` 这个 key 快照。

```python
bad = dict.fromkeys(['a', 'b'], [])  # 两个 key 共享同一个 list
bad['a'].append(1)
print(bad)                       # 输出：{'a': [1], 'b': [1]}
good = {key: [] for key in ['a', 'b']}  # 每次新建 list
```

## set 与 frozenset：集合

### 创建、查询与修改

```python
s = set()                        # 空集合；{} 是 dict
s = {1, 2, 2, 3}                 # 去重，元素为 1、2、3
2 in s                           # True
len(s)                           # 3
s.add(4)                         # 原地加入元素 4；返回 None
s.update([4, 5])                 # 原地逐个加入，重复值不再添加；返回 None
s.remove(2)                      # 删除元素 2；缺失会抛出 KeyError
s.discard(99)                    # 尝试删除元素 99；缺失不报错，返回 None
removed = s.pop()                # 任意删除并返回，不保证最小值
copied = s.copy()                # 浅拷贝
s.clear()                        # 清空集合；返回 None
# s[0]                           # TypeError：集合没有位置下标
```

### 集合运算

```python
a, b = {1, 2, 3}, {3, 4}
a | b                            # {1, 2, 3, 4}：并集
a & b                            # {3}：交集
a - b                            # {1, 2}：差集
a ^ b                            # {1, 2, 4}：对称差
{1, 2} <= a                      # True：子集，允许相等
{1, 2} < a                       # True：真子集，不允许相等
a.isdisjoint({8, 9})             # True：没有共同元素

f = frozenset([1, 2])            # 不可修改，可作字典 key
cache = {f: 'visited'}
cache[frozenset([2, 1])]         # 'visited'
```

集合不承诺遍历顺序；稳定输出用 `sorted(s)`。保留首次出现顺序去重可用 `list(dict.fromkeys(values))`。集合运算方法如 `a.union([4, 5])` 可接收一般可迭代对象。

## str：字符串

字符串不可变，索引返回的单个字符也仍是 `str`；Python 没有单独的 char 类型。

### 非负下标与负数下标

```python
s = 'abcba'
s[0]                             # 'a'：第一个字符
s[-1]                            # 'a'：最后一个字符
s[-2]                            # 'b'：倒数第二个
s[-len(s)]                       # 'a'：等同 s[0]
s[-0]                            # 'a'：-0 仍是 0
s[1:4]                           # 'bcb'：不包含 stop
s[::-1]                          # 'abcba'：逆序结果
# s[-6]                          # IndexError
# s[0] = 'x'                     # TypeError：字符串不能原地修改
```

长度为 `n` 时，合法负下标 `j` 对应非负下标 `n + j`；`s[-1 - i]` 就是 `s[n - 1 - i]`。

### 逐字符遍历

```python
s = 'cat'
for ch in s:
    print(ch)
# 输出：c
# 输出：a
# 输出：t
```

### 同时获取下标和字符

```python
s = 'cat'
for i, ch in enumerate(s):
    print(i, ch)
# 输出：0 c
# 输出：1 a
# 输出：2 t
# 需要手动控制下标时：for i in range(len(s)): ... s[i] ...
```

### 倒序遍历

```python
s = 'cat'
list(reversed(s))                # ['t', 'a', 'c']：也可直接 for ch in reversed(s)
for i in range(len(s) - 1, -1, -1):
    print(i, s[i])
# 输出：2 t
# 输出：1 a
# 输出：0 c
```

### 回文判断：// 与左右配对

```python
5 / 2                            # 2.5：普通除法
5 // 2                           # 2：向下取整除法
-5 // 2                          # -3：向下取整，不是朝零截断

def is_palindrome(s):
    for i in range(len(s) // 2):
        if s[i] != s[-1 - i]:
            return False
    return True

is_palindrome('abcba')            # True
is_palindrome('abba')             # True
is_palindrome('abca')             # False
is_palindrome('')                 # True
```

`'abcba'` 长度为 5，`range(5 // 2)` 产生 `0、1`，比较位置 `(0, 4)`、`(1, 3)`；中间字符不必与自己比较。最坏时间 $O(n)$，额外空间 $O(1)$。

### 转成 list：修改后用 join 拼回字符串

```python
s = 'hello'
original = s
chars = list(s)                  # ['h', 'e', 'l', 'l', 'o']：新 list
chars[0] = 'H'                   # 改列表位置，不修改 s
result = ''.join(chars)          # 'Hello'：返回字符串结果
print(result, s)                 # 输出：Hello hello
s = result                      # s 改绑，original 仍是原字符串
print(s, original)               # 输出：Hello hello

parts = ['ab', 'cd']
''.join(parts)                   # 'abcd'：'' 是分隔符
'-'.join(parts)                  # 'ab-cd'
str(['a', 'b'])                  # "['a', 'b']"：列表的展示文本，不等于 join
# ''.join(['a', 1])               # TypeError：元素必须为字符串
''.join(map(str, [1, 2, 3]))      # '123'
```

`join` 不会把列表原地变成字符串；列表元素可以是多字符片段，结果长度不一定等于列表长度。反复修改时，先转换一次、最后拼接一次；总时间与额外空间通常为 $O(n)$。

### 索引、切片与常用方法

```python
s = 'banana'
s.find('na')                     # 2：查找首次出现的下标
s.rfind('na')                    # 4：从右侧查找，返回最后一次出现的下标
s.find('x')                      # -1：查找 x，没找到
s.index('na')                    # 2：查找首次出现的下标，缺失则报错
# s.index('x')                    # ValueError
s.count('an')                    # 2：非重叠出现次数
s.startswith('ban')              # True：检查前缀
s.endswith('na')                 # True：检查后缀

'  a   b  '.split()              # ['a', 'b']：按连续空白切分
'a,,b'.split(',')                 # ['a', '', 'b']：显式分隔符保留空项
'  hello\n'.strip()              # 'hello'：去两端空白
'xxhellox'.strip('x')            # 'hello'：删除两端字符集合，不是完整前后缀
'prefix_data'.removeprefix('prefix_')  # 'data'：移除完整前缀
'banana'.replace('na', 'X', 1)    # 'baXna'：最多替换一次
'Hello'.lower()                  # 'hello'
'hello'.upper()                  # 'HELLO'
```

`find` 结果不能直接当真假值：找到下标 `0` 为假，没找到的 `-1` 为真。查存在性用 `x in s`。

### 格式化与编码

```python
name, score = 'Lin', 3.14159
f'Hello, {name}'                 # 'Hello, Lin'
f'{score:.2f}'                    # '3.14'：两位小数
f'{7:03d}'                       # '007'：宽度 3，补零
ord('A')                         # 65：Unicode 码点
chr(65)                          # 'A'
encoded = '你好'.encode('utf-8')
encoded.decode('utf-8')          # '你好'
len('你好')                       # 2：码点数
len(encoded)                     # 6：UTF-8 字节数
```

## range：整数序列描述

```python
list(range(4))                   # [0, 1, 2, 3]
list(range(2, 5))                # [2, 3, 4]：stop 不包含
list(range(1, 7, 2))             # [1, 3, 5]
list(range(5, 0, -1))            # [5, 4, 3, 2, 1]
r = range(2, 10, 2)
len(r)                           # 4
r[-1]                            # 8
6 in r                           # True
# range(0, 5, 0)                  # ValueError
```

`range` 保存起点、终点与步长，不预先生成全部整数；可以反复遍历，不是一次性迭代器。

## bytes、bytearray 与 memoryview

文本与字节不同：`'A'` 是字符串，`b'A'` 是字节序列。

```python
data = b'ABC'
data[0]                          # 65：单个字节的下标返回整数
data[:1]                         # b'A'
data.hex()                       # '414243'
data.decode('ascii')             # 'ABC'
# data[0] = 90                    # TypeError

buf = bytearray(data)
buf[0] = 90                     # buf 为 bytearray(b'ZBC')
view = memoryview(buf)           # 不复制整段缓冲区
view[1] = 88                     # buf 为 bytearray(b'ZXC')
view.tobytes()                    # b'ZXC'：生成 bytes 副本
view.release()                   # 释放视图
```

持有活动的 memoryview 时，不能随意改变底层 bytearray 的长度，否则可能报 `BufferError`。与其他语言交换缓冲区，见 [互操作参考](native-interop.md)。

## collections 和 heapq 常用工具

### Counter：计数器

```python
from collections import Counter

c = Counter('aab')
c['a']                           # 2
c['missing']                     # 0：不自动插入 key
len(c)                           # 2：键数
c.total()                        # 3：计数总和，Python 3.10+
c.most_common(1)                 # [('a', 2)]：取频次最高的一项
c['a'] -= 2                     # 零计数的 key 不自动删除
'a' in c                         # True
dict(+c)                         # {'b': 1}：过滤零和负计数
Counter(a=1) == Counter(a=1, b=0) # True：缺失计数当作零
Counter('aabb') >= Counter('ab') # True：逐元素计数包含，不是字典序
```

### defaultdict：默认值字典

```python
from collections import defaultdict

groups = defaultdict(list)      # 每次缺失调用 list()，创建新 list
groups.get('fruit')              # None：不调用工厂，不插入 key
groups['fruit'].append('apple')
groups['fruit'].append('pear')
print(dict(groups))              # 输出：{'fruit': ['apple', 'pear']}
counts = defaultdict(int)       # int() 返回 0
counts['a'] += 1                 # 缺失读取时自动插入
print(dict(counts))              # 输出：{'a': 1}
```

### deque：双端队列

```python
from collections import deque

q = deque([1, 2])
q.append(3)                      # 原地从右端加入；返回 None
q.appendleft(0)                  # 原地从左端加入；返回 None
q.pop()                          # 3：从右端删除并返回
q.popleft()                      # 0：从左端删除并返回
q.extendleft([8, 9])             # 逐个从左端加入；q 为 deque([9, 8, 1, 2])
q.rotate(1)                      # 向右轮转 1 位；q 为 deque([2, 9, 8, 1])
print(list(q))                   # 输出：[2, 9, 8, 1]
```

队列用 `append` 入队、`popleft` 出队，两端操作大致为 $O(1)$。中间下标访问用 list；不要用反复 `list.pop(0)` 来实现大队列。

### heapq：堆操作

```python
from heapq import heapify, heappush, heappop, nlargest

h = [7, 2, 5]
heapify(h)                       # 原地建小顶堆；返回 None
h[0]                             # 2：堆顶最小，其余位置不保证排序
heappush(h, 1)                   # 加入元素 1 并维持堆结构；返回 None
heappop(h)                       # 1：删除并返回堆顶最小值
nlargest(2, [3, 7, 1, 5])        # [7, 5]：取最大的两项

big = [-x for x in [7, 2, 5]]    # 跨版本大顶堆写法：取负
heapify(big)
-heappop(big)                     # 7
```

堆中的元素要能比较。若用 `(优先级, 对象)` 且优先级可能相同，可在对象之前加唯一编号，避免继续比较不可比较的对象。

## 遍历、内置函数与推导式

```python
nums = [1, 2, 3]
list(enumerate(nums))             # [(0, 1), (1, 2), (2, 3)]
list(zip(nums, ['a', 'b']))       # [(1, 'a'), (2, 'b')]：按最短输入结束
[x * x for x in nums]            # [1, 4, 9]：列表推导式
{x: x * x for x in nums}         # {1: 1, 2: 4, 3: 9}：字典推导式
all(x > 0 for x in nums)         # True：全部满足
any(x % 2 == 0 for x in nums)    # True：至少一个满足
sum(x * x for x in nums)         # 14
```

`all/any`、生成器表达式、`map/filter`、`lambda`、`key=` 和迭代器消费规则，见 [可迭代对象与常用函数](iterables-and-functions.md)。模块、包与 import 见 [模块参考](modules-and-packages.md)。

## 常见错误速查

| 容易写错 | 实际问题 | 写法 |
| --- | --- | --- |
| `a = a.append(3)` / `a = a.sort()` | a 变成 None | 原地操作不要赋回；或用 `sorted(a)` |
| `(3)` 当单元素 tuple | 实际是整数 | `(3,)` |
| `{}` 当空 set | 实际是字典 | `set()` |
| 缺失 key 时 `d[key] += 1` | KeyError | `d[key] = d.get(key, 0) + 1` |
| `s.strip('abc')` 当删除完整前缀 | 删除两端任意 a/b/c 字符 | `s.removeprefix('abc')` |
| `if s.find('x'):` | 0 为假，-1 为真 | `if 'x' in s:` |
| `rows = [[]] * 3` | 内部列表共享 | `[[] for _ in range(3)]` |
| `b = a` 当复制 | 对象共享 | 按需浅拷贝 / 深拷贝 |
| 用 `is` 比较数字或字符串的值 | 比较身份 | `==` |
| `str(chars)` 当字符拼接 | 得到列表展示文本 | `''.join(chars)` |

可变默认参数在函数定义时创建，不是每次调用时创建：

```python
def collect(x, out=None):
    if out is None:
        out = []
    out.append(x)
    return out

collect(1)                       # [1]
collect(2)                       # [2]：两次调用各自创建 list
```

## 查题解时对应到哪里

- [数组](../数组/fuck数组.md)：dict 查位置、set 去重、Counter 做滑窗计数。
- [图](../图/fuck图.md)：deque 做 BFS，heapq 做最短路。
- [回溯](../回溯/fuck回溯.md)：用 list 维护路径，保存答案时 `path.copy()`。
- [动态规划](../动态规划/fuck动态规划.md)：独立创建二维列表，明确初始化与遍历顺序。

## 官方资料

- [内置类型与方法](https://docs.python.org/3.12/library/stdtypes.html)
- [数据结构教程](https://docs.python.org/3.12/tutorial/datastructures.html)
- [内置函数](https://docs.python.org/3.12/library/functions.html)
- [赋值规则与对象模型](https://docs.python.org/3.12/reference/simple_stmts.html#assignment-statements)
- [浅拷贝与深拷贝](https://docs.python.org/3.12/library/copy.html)
- [collections](https://docs.python.org/3.12/library/collections.html) / [heapq](https://docs.python.org/3.12/library/heapq.html)
