# Python 基础数据结构

刷题时先选容器：要按位置访问，用 `list`；要记录一个固定组合，用 `tuple`；要按键查值，用 `dict`；只关心是否出现，用 `set`；处理文本，用 `str`。队列、计数和堆则有对应的标准库工具。

这一页面向 Python 3.10+。示例中的 `# 输出：` 标出实际打印结果，代码块可单独运行。API 行为以 [Python 官方内置类型文档](https://docs.python.org/3.12/library/stdtypes.html)为依据。

## 先选对容器

| 要做什么 | 常用结构 | 关键区别 |
| --- | --- | --- |
| 保存序列、按下标读写、模拟栈 | `list` | 有序、可变，保存对元素对象的引用 |
| 保存坐标、状态、固定组合 | `tuple` | 序列本身不可变；元素都可哈希时，可作字典键 |
| 记录元素的位置、次数或关联信息 | `dict` | 键到值的映射，按插入顺序遍历 |
| 判断是否出现、去重、集合运算 | `set` | 元素唯一，没有序列下标，不保证遍历顺序 |
| 保存和处理文本 | `str` | 不可变，切片、替换等操作返回字符串结果 |
| 从两端入队、出队 | `collections.deque` | 避免 `list.pop(0)` 的整体搬移 |
| 统计出现次数 | `collections.Counter` | 字典式计数，缺失元素的计数为 `0` |
| 分组、邻接表、自动创建默认值 | `collections.defaultdict` | 用 `d[key]` 读缺失键时会创建默认值 |
| 反复取当前最小元素 | `heapq` + `list` | 列表保存堆，只有堆顶保证是最小值 |

如果从 C++ 切过来，先记住：**Python 的赋值通常只是让另一个名字指向同一个对象；不会像复制一个 `vector` 那样自动得到独立容器。** 切片、`copy()` 与 `list(...)` 才会新建外层列表，但通常仍共享其中的对象。

## list：列表

### 创建、索引与切片

下标从 `0` 开始，负下标从末尾倒数。切片 `a[start:stop:step]` 的结束位置不包含在结果中。

```python
nums = [8, 3, 5, 1]
print(nums[0], nums[-1])            # 输出：8 1
print(nums[1:3])                   # 输出：[3, 5]
print(nums[:2], nums[2:])          # 输出：[8, 3] [5, 1]
print(nums[::-1])                  # 输出：[1, 5, 3, 8]
print([0] * 4)                    # 输出：[0, 0, 0, 0]
```

索引越界会抛出 `IndexError`；普通切片越过两端会截到有效范围，不会因端点过大报错。列表切片是浅拷贝，长为 `k` 的切片需要处理 `k` 个元素，不能把它当作一个免费视图。

### 添加、删除与查找

`append(x)` 添加一个对象，`extend(xs)` 逐个添加可迭代对象中的元素。`pop(i)` 按位置删除并返回元素，`remove(x)` 删除第一个等于 `x` 的元素。

```python
nums = [4, 7]
nums.append([2, 3])
print(nums)                       # 输出：[4, 7, [2, 3]]
nums.pop()
nums.extend([2, 3])
print(nums)                       # 输出：[4, 7, 2, 3]
print(nums.pop())                  # 输出：3
nums.remove(7)
print(nums, nums.index(2))         # 输出：[4, 2] 1
```

`index(x)`、`remove(x)` 找不到时抛出 `ValueError`。只想判断是否存在，用 `x in nums`；反复查找时，应考虑 `set` 或 `dict`。删除一段用 `del nums[l:r]`，清空用 `nums.clear()`。

### 排序、反转与栈

`sort()`、`reverse()` 修改原列表，返回 `None`；`sorted(...)` 返回一个新列表。不要把 `nums = nums.sort()` 写成获取排序结果的方式。[官方列表方法说明](https://docs.python.org/3.12/tutorial/datastructures.html#more-on-lists)

```python
nums = [5, 2, 4]
ordered = sorted(nums)
print(nums, ordered)              # 输出：[5, 2, 4] [2, 4, 5]
print(nums.sort())                # 输出：None
nums.reverse()
print(nums)                       # 输出：[5, 4, 2]

pairs = [(2, 8), (1, 9), (2, 3)]
print(sorted(pairs, key=lambda p: (p[0], -p[1])))  # 输出：[(1, 9), (2, 8), (2, 3)]
```

模拟栈时，用尾部 `append()`、`pop()` 和 `stack[-1]`。后两者要求栈非空，通常先写 `if stack:`。

```python
stack = []
stack.append(6)
stack.append(9)
print(stack[-1])                  # 输出：9
print(stack.pop(), stack)         # 输出：9 [6]
```

### 赋值、浅拷贝与二维列表

```python
nums = [1, 2]
alias = nums
copy = nums.copy()
alias.append(3)
print(nums, copy)                 # 输出：[1, 2, 3] [1, 2]
print(alias is nums, copy is nums) # 输出：True False
```

`is` 判断是不是同一个对象，`==` 判断值是否相等。对数字、字符串、列表等做值比较，用 `==`；判断空指针式的哨兵值，用 `x is None`。

```python
rows = [[1], [2]]
copy = rows.copy()
copy[0].append(9)
print(rows, copy)                 # 输出：[[1, 9], [2]] [[1, 9], [2]]
```

浅拷贝只复制外层：上面的两个列表仍引用同一个第一行。刷题的二维 DP 数组，应该让每行独立创建。[官方 FAQ：二维列表](https://docs.python.org/3.12/faq/programming.html#how-do-i-create-a-multidimensional-list)

```python
bad = [[0] * 3] * 2               # 两个位置引用同一行
bad[0][1] = 7
print(bad)                        # 输出：[[0, 7, 0], [0, 7, 0]]

good = [[0] * 3 for _ in range(2)]
good[0][1] = 7
print(good)                       # 输出：[[0, 7, 0], [0, 0, 0]]
```

遍历时不要一边按原下标向前走、一边从同一列表删除元素：后面的元素会前移，容易漏处理。筛选通常直接新建列表：`kept = [x for x in nums if x != 0]`；如果题目要求原地操作，再使用双指针。

## tuple：元组

元组适合坐标、边、返回多个结果等固定组合。逗号决定单元素元组；仅写 `(7)` 仍然是整数。

```python
point = (3, 5)
x, y = point
print(x, y)                       # 输出：3 5
print(type((7)).__name__, type((7,)).__name__)  # 输出：int tuple

distance = {(0, 0): 0, (0, 1): 1}
print(distance[(0, 1)])            # 输出：1
```

元组不可变，指的是不能替换它的元素位置；如果元素本身是可变对象，该对象仍可修改。

```python
state = ([1, 2], 3)
state[0].append(4)
print(state)                      # 输出：([1, 2, 4], 3)
```

`tuple` 只有在所有元素都可哈希时才能作为 `dict` 的键或 `set` 的元素。`(1, 2)` 可以，`([1, 2],)` 不可以，因为里面的列表不可哈希。

## dict：字典

### 按键查询、插入与删除

字典适合 `元素 → 次数`、`值 → 下标`、`状态 → 结果` 等映射。`key in d` 查的是键，不是值。

```python
positions = {'red': 2, 'blue': 5}
print(positions['red'])           # 输出：2
print(positions.get('green', -1)) # 输出：-1
print('green' in positions)       # 输出：False
positions['green'] = 8
print(positions.pop('blue'))      # 输出：5
print(list(positions.items()))    # 输出：[('red', 2), ('green', 8)]
```

`d[key]` 在键不存在时抛出 `KeyError`，`d.get(key, default)` 返回默认值且不插入键；`d.pop(key, default)` 删除并返回值，缺失时返回默认值。`del d[key]` 删除但不返回值，键不存在时会报错。

### 遍历、计数与分组

`for key in d` 遍历键；键值一起用 `d.items()`，只看值用 `d.values()`。字典按插入顺序遍历，**不是按键大小排序**。

```python
counts = {}
for word in ['pear', 'apple', 'pear']:
    counts[word] = counts.get(word, 0) + 1
print(counts)                     # 输出：{'pear': 2, 'apple': 1}
print(sorted(counts))             # 输出：['apple', 'pear']
```

`setdefault(key, default)` 适合分组：键存在就返回原来的值，不存在就插入默认值并返回它。这里返回的是字典内那个列表本身。

```python
groups = {}
for word in ['aa', 'b', 'cc']:
    groups.setdefault(len(word), []).append(word)
print(groups)                     # 输出：{2: ['aa', 'cc'], 1: ['b']}
```

遍历字典时不要增删键。需要按条件删除，可以先收集要删除的键，再逐个删除；或新建字典。键和值的浅拷贝与列表类似，`d.copy()` 不会递归复制里面的可变对象。

## set 与 frozenset：集合

集合适合去重、判断是否出现、维护 `visited`。空集合写 `set()`，`{}` 是空字典。`set` 没有下标，也不保证遍历顺序；需要稳定输出时用 `sorted(...)`。

```python
seen = set()
seen.add(4)
seen.add(4)
seen.update([2, 7])
print(sorted(seen))               # 输出：[2, 4, 7]
seen.discard(9)                   # 缺失也不报错
seen.remove(2)                    # 缺失会抛出 KeyError
print(4 in seen, len(seen))        # 输出：True 2
```

`set.pop()` 删除并返回任意一个元素，不是最小值，也不是最后插入的值。

```python
a, b = {1, 2, 3}, {2, 3, 4}
print(sorted(a & b))              # 输出：[2, 3]
print(sorted(a | b))              # 输出：[1, 2, 3, 4]
print(sorted(a - b))              # 输出：[1]
print(sorted(a ^ b))              # 输出：[1, 4]
print({2, 3} <= a)                # 输出：True
```

`&` 是交集，`|` 是并集，`-` 是差集，`^` 是对称差，`<=` 判断是否为子集。`frozenset` 是不可变集合，可以用来表示一个与顺序无关的可哈希状态。

```python
key = frozenset([2, 1, 2])
cache = {key: 'visited'}
print(cache[frozenset([1, 2])])    # 输出：visited
```

## str：字符串

字符串是不可变序列：可以索引、切片、遍历，但不能写 `s[i] = 'x'`。要反复修改字符，可以先转成列表，最后 `''.join(chars)`。

```python
s = 'stone'
chars = list(s)
chars[0] = 'a'
print(''.join(chars), s)          # 输出：atone stone
print(s[1:4], s[::-1])            # 输出：ton enots
print(s.find('on'), s.find('x'))   # 输出：2 -1
```

`find()` 找不到返回 `-1`，`index()` 找不到抛出 `ValueError`。`x in s` 判断子串存在。`strip()` 去掉两端指定字符集合中的字符；不会删除中间字符，也不是删除一个指定的完整前后缀。

```python
text = '  a  b   c  '
print(text.strip().split())       # 输出：['a', 'b', 'c']
print('a,,b'.split(','))           # 输出：['a', '', 'b']
print('-'.join(['a', 'b', 'c']))   # 输出：a-b-c
print('banana'.replace('na', 'X', 1))  # 输出：baXna
```

`split()` 不传分隔符时按连续空白切分，且省略两端空白造成的空项；显式分隔符会保留空项。构造大量文本时，先收集片段再 `join`，可以避免依赖循环拼接的性能表现。

## deque：双端队列

BFS 队列通常用 `deque`：右端 `append()` 入队、左端 `popleft()` 出队。`appendleft()`、`pop()` 操作另一端。两端插入、删除大致是 $O(1)$，而 `list.pop(0)` 需要移动后面的元素。[官方 deque 文档](https://docs.python.org/3.12/library/collections.html#collections.deque)

```python
from collections import deque

queue = deque([3, 6])
queue.append(9)
queue.appendleft(1)
print(queue.popleft(), queue.pop())  # 输出：1 9
print(list(queue))                # 输出：[3, 6]
```

要频繁访问中间下标，用 `list`，不要因为 `deque` 支持 `queue[i]` 就把它当作动态数组。

## Counter：计数器

`Counter(iterable)` 直接统计元素出现次数。它是字典的子类，但查询缺失元素返回 `0`。修改到 `0` 的键不会自动删除；`+counter` 会过滤掉零和负计数。[官方 Counter 文档](https://docs.python.org/3.12/library/collections.html#collections.Counter)

```python
from collections import Counter

counts = Counter('aab')
print(counts['a'], counts['z'])    # 输出：2 0
print(counts.most_common(1))       # 输出：[('a', 2)]
counts['a'] -= 2
print('a' in counts)              # 输出：True
print(dict(+counts))             # 输出：{'b': 1}
```

Python 3.10+ 的 `Counter` 比较把缺失计数当作零，`>=` 表示逐元素计数包含关系。这与普通列表、元组的字典序比较不同。

```python
from collections import Counter

have = Counter('aabbc')
need = Counter('abc')
print(have >= need)               # 输出：True
print(Counter(a=1) == Counter(a=1, b=0))  # 输出：True
```

## defaultdict：默认值字典

`defaultdict(int)` 为缺失键提供 `0`，`defaultdict(list)` 为缺失键创建新列表。只有 `d[key]` 这种缺失键访问触发默认工厂，`.get()` 和 `in` 不会。

```python
from collections import defaultdict

groups = defaultdict(list)
print(groups.get('new'), 'new' in groups)  # 输出：None False
groups['new'].append(4)
print(dict(groups))               # 输出：{'new': [4]}
```

`defaultdict` 适合邻接表、分组、计数，但在只想查询时要留意隐式插入。[官方 defaultdict 文档](https://docs.python.org/3.12/library/collections.html#collections.defaultdict)

## heapq：用列表维护小顶堆

`heapq` 不是容器类，而是修改普通列表的一组堆操作。`heapify()` 原地建堆；`heappush()` 入堆；`heappop()` 返回并删除堆顶；`heap[0]` 查看堆顶。堆中其余位置不是排好序的。

```python
from heapq import heapify, heappush, heappop

heap = [7, 2, 5]
heapify(heap)
heappush(heap, 1)
print(heap[0])                    # 输出：1
print(heappop(heap), heappop(heap))  # 输出：1 2
```

Python 3.10+ 的通用大顶堆写法是把数值取负后放入小顶堆：

```python
from heapq import heapify, heappop

heap = [-x for x in [7, 2, 5]]
heapify(heap)
print(-heappop(heap))             # 输出：7
```

元组按字段从左到右比较，适合 `(距离, 节点编号)`。如果最后一个字段是无法比较的对象，给它之前加一个唯一编号，避免优先级相同时继续比较对象。[官方 heapq 文档](https://docs.python.org/3.12/library/heapq.html)

## 遍历与真假判断

```python
nums = [4, 7, 9]
print(list(enumerate(nums)))       # 输出：[(0, 4), (1, 7), (2, 9)]
print(list(zip([1, 2, 3], ['a', 'b'])))  # 输出：[(1, 'a'), (2, 'b')]
print([x * x for x in nums if x > 4])  # 输出：[49, 81]
print(bool([]), bool([0]), bool(''))  # 输出：False True False
```

`enumerate()` 同时给出下标和值；`zip()` 默认在最短输入耗尽时结束；推导式适合筛选或构造容器。空容器为假，**含有一个零的列表 `[0]` 仍为真**。

列表、元组、字符串的大小比较是字典序；集合的大小比较是包含关系；`Counter` 的大小比较是逐元素计数包含关系。不要把这几种 `>=` 混在一起使用。

## 查题解时对应到哪里

- [数组](../数组/fuck数组.md)：`dict` 查位置、`set` 去重、`Counter` 做滑窗计数。
- [图](../图/fuck图.md)：`deque` 做 BFS，`defaultdict(list)` 建邻接表，`heapq` 做 Dijkstra。
- [回溯](../回溯/fuck回溯.md)：用 `list` 维护路径，收集答案时用 `path.copy()` 保存当前快照。
- [动态规划](../动态规划/fuck动态规划.md)：独立创建二维列表，明确状态、初始化和遍历顺序。

## 官方资料

- [Python 教程：数据结构](https://docs.python.org/3.12/tutorial/datastructures.html)
- [内置类型：序列、集合与字典](https://docs.python.org/3.12/library/stdtypes.html)
- [collections：deque、Counter、defaultdict](https://docs.python.org/3.12/library/collections.html)
- [heapq：堆操作](https://docs.python.org/3.12/library/heapq.html)
