---
title: 类与对象
category: python
tags: [class, self, __init__, 继承, 特殊方法]
description: class · self · __init__ · 属性与方法 · 继承 · 特殊方法
updated: 2026-11-04
---
# 类与对象

**类把相关的数据和操作放在一起；实例保存某个具体对象的状态。** 例如 `Student` 定义学生有哪些属性和方法，`Student('Lin', 80)` 则创建一个具体学生。

本页使用 Python 3.10+。每个 Python 代码块都可以独立运行，`assert` 检查实际结果；基础规则见 [Python 官方类教程](https://docs.python.org/3.12/tutorial/classes.html)。

## class、实例、属性与方法

| 写法 | 含义 | 返回值 / 得到什么 |
| --- | --- | --- |
| `class Student: ...` | 定义一个类，名字 `Student` 指向类对象 | 定义语句，不返回值 |
| `s = Student('Lin', 80)` | 创建并初始化一个实例 | `Student(...)` 返回实例 |
| `s.name` | 读取该对象的属性 | 属性值 |
| `s.score = 90` | 修改该对象的属性 | 赋值语句，不返回值 |
| `s.improve(5)` | 调用该对象的方法 | 由方法的 `return` 决定 |
| `isinstance(s, Student)` | 判断是否是该类或其子类的实例 | `bool` |

```python
class Student:
    def __init__(self, name, score=0):
        self.name = name
        self.score = score

    def improve(self, delta):
        self.score += delta
        return self.score


lin = Student('Lin', 80)
mei = Student('Mei')
assert lin.name == 'Lin' and lin.score == 80
assert mei.score == 0
assert isinstance(lin, Student)

assert lin.improve(5) == 85
assert lin.score == 85 and mei.score == 0  # 两个实例各有自己的分数

alias = lin                              # 赋值不复制实例
alias.score = 90
assert alias is lin and lin.score == 90

print('类与实例练习通过')                 # 输出：类与实例练习通过
```

`Student` 是类对象，`lin` 是它的实例；二者都是 Python 对象。类里通常用 `def` 定义方法，用 `self.xxx` 访问当前实例的数据。

## self：方法自动收到当前实例

`self` 是实例方法的第一个参数，代表当前调用它的对象。它是约定名称，不是关键字；调用 `lin.improve(5)` 时，Python 自动把 `lin` 传给它。

| 写法 | 含义 | 返回值 / 得到什么 |
| --- | --- | --- |
| `lin.improve(5)` | 绑定当前实例后调用 | 本例返回更新后的分数 |
| `Student.improve(lin, 5)` | 显式传入实例；对本例普通方法与上一行等价 | 更新后的分数 |
| `lin.improve` | 取得绑定方法，还没有调用 | 方法对象 |
| `self.score` | 访问当前实例的属性 | 属性值 |
| `score` | 查找普通变量名，不自动等同于 `self.score` | 变量值；未定义时可能报 `NameError` |

```python
class Student:
    def __init__(self, score):
        self.score = score

    def improve(self, delta):
        self.score += delta
        return self.score


s = Student(80)
assert s.improve(5) == 85
assert Student.improve(s, 5) == 90

method = s.improve                        # 保存方法，稍后再调用
assert method.__self__ is s               # 这个方法已经绑定到 s
assert method(5) == 95
```

方法中读写实例属性要写 `self.score`；单写 `score = 90` 只创建或修改当前函数的局部变量。

## `__init__`：初始化实例，不是包文件

对于本页这样的普通类，调用顺序可以理解为：

```text
Student('Lin', 80)
  → 创建 Student 实例（__new__ 的职责）
  → 调用 __init__(实例, 'Lin', 80)
  → self.name = 'Lin'，self.score = 80
  → Student(...) 返回这个实例
```

| 名字 | 作用 | 使用时注意 |
| --- | --- | --- |
| `__init__(self, ...)` | 初始化已经创建的实例 | 必须返回 `None`；通常不写 `return` |
| `__new__(cls, ...)` | 创建并返回实例 | 入门时通常不需要重写 |
| `__init__.py` | 普通包的初始化文件，导入包时执行 | 与类的初始化方法是两件事 |

不必手动调用 `s.__init__(...)` 来创建对象；直接调用类 `Student(...)`。`self.name = name` 中，左边是实例属性，右边是传进来的参数，名字相同不代表它们是同一个变量。

详见 [对象初始化规则](https://docs.python.org/3.12/reference/datamodel.html#object.__init__)与本站 [模块、包与运行方式](modules-and-packages.md)。

## 实例属性与类属性

| 写法 | 保存在哪里 | 常见用途 |
| --- | --- | --- |
| `self.name = name` | 当前实例 | 每个学生自己的名字 |
| 类体里的 `school = 'North'` | 类上 | 实例可共用的学校名称 |
| `Student.school = 'South'` | 修改类属性 | 没有同名实例属性的对象读到新值 |
| `s.school = 'Local'` | 对本例创建同名实例属性 | 遮蔽这个实例对类属性的读取，不修改类属性 |

以下规则针对普通属性；后面介绍的 `property` 等机制可以改变属性访问行为。

```python
class Student:
    school = 'North'

    def __init__(self, name):
        self.name = name


a = Student('Lin')
b = Student('Mei')
assert a.school == b.school == Student.school == 'North'

a.school = 'Local'
assert a.school == 'Local'
assert b.school == Student.school == 'North'

Student.school = 'South'
assert b.school == 'South'
assert a.school == 'Local'                # 实例属性仍遮蔽类属性

del a.school
assert a.school == 'South'                # 删除实例属性后，再从类上读取
```

### 每个实例的列表，放在初始化方法里创建

```python
class SharedScores:
    scores = []                          # 类属性：所有实例读到同一个列表


a = SharedScores()
b = SharedScores()
a.scores.append(80)
assert a.scores is b.scores
assert b.scores == [80]


class Student:
    def __init__(self, name, scores=None):
        self.name = name
        self.scores = [] if scores is None else list(scores)


lin = Student('Lin')
mei = Student('Mei')
lin.scores.append(80)
assert lin.scores == [80] and mei.scores == []
assert lin.scores is not mei.scores

original = [70]
s = Student('An', original)
s.scores.append(90)
assert original == [70]                   # 本例主动复制输入列表
```

也避免写 `def __init__(self, scores=[]): ...`：默认参数只在函数定义时求值，会被后续调用复用。本例用 `None` 作默认值；提供输入时用 `list(scores)` 浅拷贝，若输入包含嵌套可变对象，内部对象仍可能共享。

## 实例方法、类方法与静态方法

写在 `def` 前的 `@classmethod`、`@staticmethod` 是装饰器，用于改变函数在类中被访问和调用的方式。

| 定义方式 | 自动接收什么 | 常见用途 |
| --- | --- | --- |
| `def improve(self, delta)` | 当前实例 `self` | 读取或修改实例状态 |
| `@classmethod` + `def from_text(cls, text)` | 当前类 `cls` | 从另一种输入格式创建实例 |
| `@staticmethod` + `def passing(score)` | 不自动接收实例或类 | 与该类相关的独立工具函数 |

```python
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    @classmethod
    def from_text(cls, text):
        name, score = text.split(',')
        return cls(name, int(score))

    @staticmethod
    def passing(score):
        return score >= 60


s = Student.from_text('Lin,80')
assert isinstance(s, Student)
assert s.name == 'Lin' and s.score == 80
assert Student.passing(s.score)
assert not Student.passing(59)
```

`cls` 和 `self` 一样是约定名称。通过子类调用类方法时，`cls` 是那个子类；静态方法不自动收到 `self` 或 `cls`。说明见 [classmethod](https://docs.python.org/3.12/library/functions.html#classmethod)与 [staticmethod](https://docs.python.org/3.12/library/functions.html#staticmethod)。

## 继承、方法重写与 super()

| 写法 | 含义 | 返回值 / 得到什么 |
| --- | --- | --- |
| `class Graduate(Student): ...` | 定义 `Student` 的子类 | 定义语句，不返回值 |
| 子类定义同名方法 | 重写继承的方法 | 由子类方法决定 |
| `super().__init__(...)` | 本例单继承中调用父类初始化方法 | `None` |
| `super().describe()` | 本例中调用父类方法，当前实例仍是子类实例 | 父类方法的结果 |
| `isinstance(g, Student)` | 子类实例也属于父类类型 | `bool` |

```python
class Student:
    def __init__(self, name):
        self.name = name

    def describe(self):
        return self.name


class Graduate(Student):
    def __init__(self, name, topic):
        super().__init__(name)           # 初始化同一个实例的 name 属性
        self.topic = topic

    def describe(self):
        return f'{super().describe()} / {self.topic}'


g = Graduate('Lin', 'MLSys')
assert g.name == 'Lin' and g.topic == 'MLSys'
assert g.describe() == 'Lin / MLSys'
assert isinstance(g, Graduate) and isinstance(g, Student)
assert issubclass(Graduate, Student)
```

子类没有定义 `__init__` 时会继承父类的实现；定义了自己的 `__init__` 后，父类初始化不会因此自动执行。`super()` 不会新建一个父类实例；在多继承中它按 MRO（方法解析顺序）继续查找，不能一概理解为固定调用某个直接父类。[官方 super 说明](https://docs.python.org/3.12/library/functions.html#super)

## property：像属性一样读取计算结果

`@property` 把一个方法暴露为属性访问：写 `s.passed`，不用写 `s.passed()`。它可以在读取时计算结果，不要求预先保存一个同名值。

```python
class Student:
    def __init__(self, score):
        self.score = score

    @property
    def passed(self):
        return self.score >= 60


s = Student(59)
assert not s.passed
s.score = 60
assert s.passed

# s.passed = True                         # AttributeError：没有定义 setter
```

需要支持赋值时，可以定义 `@passed.setter`；本例只提供读取。见 [官方 property 文档](https://docs.python.org/3.12/library/functions.html#property)。

## 特殊方法：接入 print、len 和下标操作

前后都有双下划线的方法常被称作 **dunder / 特殊方法**；它们让自定义对象参与 Python 已有的操作。

| 普通写法 | 对应特殊方法 | 返回值 |
| --- | --- | --- |
| `repr(obj)` | `__repr__(self)` | 调试用的 `str` |
| `str(obj)`、`print(obj)` 的字符串转换 | `__str__(self)`；未定义时通常使用 `__repr__` | 面向阅读的 `str` |
| `len(obj)` | `__len__(self)` | 非负整数 |
| `obj[i]` | `__getitem__(self, i)` | 对应元素 |

```python
class ScoreBook:
    def __init__(self, scores):
        self.scores = list(scores)

    def __repr__(self):
        return f'ScoreBook({self.scores!r})'

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, index):
        return self.scores[index]


book = ScoreBook([80, 90])
assert len(book) == 2
assert book[0] == 80 and book[-1] == 90
assert book[:] == [80, 90]                 # 本例把切片转交给内部列表
assert repr(book) == 'ScoreBook([80, 90])'
print(book)                              # 输出：ScoreBook([80, 90])
```

`!r` 表示在 f-string 中使用 `repr` 格式化。日常调用优先写 `len(book)`、`book[0]`，由 Python 按协议调用相应方法。以单下划线开头的 `_score` 通常只是内部使用约定；并不是这里的特殊方法，也不强制禁止外部访问。详见 [特殊方法规则](https://docs.python.org/3.12/reference/datamodel.html#special-method-names)。

## dataclass：主要保存数据时减少重复代码

`@dataclass` 根据字段定义生成 `__init__`、`__repr__`、`__eq__` 等常用方法。字段用类型注解声明；**类型注解本身不强制检查运行时输入类型**。

```python
from dataclasses import dataclass, field


@dataclass
class Student:
    name: str
    scores: list[int] = field(default_factory=list)


lin = Student('Lin')
mei = Student('Mei')
lin.scores.append(80)
assert mei.scores == []                   # default_factory 每次创建新列表
assert lin == Student('Lin', [80])        # 自动生成的相等比较按字段判断
assert lin is not Student('Lin', [80])    # 内容相等不代表是同一个对象
assert repr(lin) == "Student(name='Lin', scores=[80])"
```

可变默认字段用 `field(default_factory=list)`，不要直接写 `scores: list[int] = []`。见 [dataclasses 官方文档](https://docs.python.org/3.12/library/dataclasses.html)。

## 常见错误速查

| 现象 | 检查什么 |
| --- | --- |
| 调用方法时报参数数量错误 | 普通实例方法是否漏写 `self`，调用时是否又多传了实例 |
| 方法里改了 `score`，对象没变 | 应该改局部变量，还是 `self.score` |
| 多个实例的列表一起变化 | 列表是否是类属性、可变默认参数，或来自同一个输入对象 |
| `__init__() should return None` | 不要在初始化方法中返回实例或其他非 `None` 值 |
| 子类没有父类初始化的属性 | 子类的 `__init__` 是否调用了 `super().__init__(...)` |
| `AttributeError` | 属性是否实际创建；拼写是否正确；property 是否支持赋值 |
| 定义了类，却没有运行方法 | 创建实例并调用方法；类定义不会自动执行方法体 |

理解类和实例后，可以继续读 [模块、包与运行方式](modules-and-packages.md)了解类怎样跨文件使用，或读 [PyTorch 神经网络基础](../PyTorch/neural-networks.md)理解 `nn.Module` 子类。
