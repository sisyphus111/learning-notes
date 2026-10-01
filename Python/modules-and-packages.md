# 模块、包与运行方式

当程序从一个文件变成多个文件，你就会遇到三件事：**怎样组织代码，怎样找到代码，怎样启动程序。** `module` / `package` 管组织，`import` 管加载，`python -m` 管启动；`pip` 和虚拟环境则决定第三方代码安装到哪里。`__pycache__` 是解释器产生的缓存，不是你要维护的源码。

本页面向刚接触 Python 项目的读者，使用 Python 3.10+ 的常见写法，运行机制以 CPython 3.12 为例。先读本页，再读 [Python 与其他语言互操作](native-interop.md)。文件示例需要按标注保存到同一个项目中；它们不是可以逐块独立执行的片段。

## 先分清这几个名字

| 名字 | 解决什么问题 | 本页的例子 |
| --- | --- | --- |
| module，模块 | 给一组代码一个可导入的名字 | `stats.py` 对应 `studytools.stats` |
| import package，导入包 | 按层级组织模块；包本身也是模块 | `studytools/`，含 `__init__.py` |
| distribution package，发行包 | 把代码、元数据和依赖打包，交给安装工具 | 发行名 `studytools-demo` |
| library，库 | 描述“提供给别的代码使用”的用途 | 一个库可以包含多个包和本地扩展 |
| script / CLI | 描述“作为程序启动”的用途 | `cli.py` / 命令 `study-squares` |
| environment，环境 | 决定解释器、已安装依赖和搜索路径 | 项目里的 `.venv/` |

“模块就是一个 `.py` 文件”是入门时好用的近似：模块还可以由 C/C++ 扩展实现，也可以内置在解释器中。普通文件包通常有 `__init__.py`；**namespace package** 可以没有它，用于把同一个包的内容分布在不同位置，不需要在第一个项目里引入这种结构。

发行名和导入名属于两个命名空间，并不要求一致。例如安装 `Pillow` 后使用 `from PIL import Image`。本页也刻意用不同的名字：安装 `studytools-demo`，导入 `studytools`。依据见 [PyPA 对两种 package 的区分](https://packaging.python.org/en/latest/discussions/distribution-package-vs-import-package/)与 [Python import 机制](https://docs.python.org/3.12/reference/import.html#packages)。

## 用一个小项目串起来

我们写一个“求若干数的平方和”的工具：计算函数能被其他程序复用，也能从命令行调用。先创建下面的文件结构，`pyproject.toml` 在后面的安装小节补上。

```text
square-demo/                 ← 项目根目录，不是导入名
├── studytools/              ← 导入包
│   ├── __init__.py          ← 包初始化与对外接口
│   ├── stats.py             ← 计算函数
│   ├── cli.py               ← 命令行入口
│   └── __main__.py          ← python -m studytools 的入口
└── pyproject.toml           ← 构建与发行信息
```

`stats.py` 放计算逻辑，函数内部才会做计算；单纯导入不会自动调用这个函数。

```python
# 文件：studytools/stats.py
def sum_squares(nums):
    """计算平方和，不修改输入。"""
    return sum(x * x for x in nums)
```

`__init__.py` 可以是空文件。这里主动导出一个函数，让调用者可以直接写 `studytools.sum_squares(...)`，而不必了解内部文件怎么划分。

```python
# 文件：studytools/__init__.py
from .stats import sum_squares
```

`cli.py` 解析命令行数字，再调用计算函数。`argparse` 是标准库，不需要用 pip 安装。

```python
# 文件：studytools/cli.py
import argparse
from .stats import sum_squares


def main():
    parser = argparse.ArgumentParser(description="计算整数的平方和")
    parser.add_argument("nums", type=int, nargs="+")
    args = parser.parse_args()
    print(sum_squares(args.nums))


if __name__ == "__main__":
    main()
```

`__main__.py` 把“运行整个包”转交给同一个入口函数。

```python
# 文件：studytools/__main__.py
from .cli import main

if __name__ == "__main__":
    main()
```

在 `square-demo/` 根目录运行以下命令。本页用 `python3` 表示你选择的 Python 3.10+ 解释器；稍后会换成明确的虚拟环境路径。

```sh
python3 -c 'from studytools import sum_squares; print(sum_squares([2, 3]))'
# 输出：13
python3 -m studytools.cli 2 3
# 输出：13
python3 -m studytools 2 3
# 输出：13
```

三条命令分别展示了：复用函数、运行包中的模块、运行整个包。它们共享计算逻辑。

## import 实际做了什么

`import studytools.stats` 会先加载父包 `studytools`，再取得子模块 `stats`。首次加载一个普通 Python 模块时，解释器会执行它的顶层语句：赋值、`def`、`class`、其他 `import` 都会执行；`def` 创建函数对象，函数体要等调用才执行。

对本页项目，首次 `import studytools` 会执行 `__init__.py`，它又导入 `stats`。因此不宜在模块顶层直接启动服务器、跑训练或解析命令行，否则别人只是想导入函数，也会触发这些动作。

几种写法的区别在于当前文件里绑定什么名字：

| 写法 | 后续调用 |
| --- | --- |
| `import studytools.stats` | `studytools.stats.sum_squares([2, 3])` |
| `from studytools import stats` | `stats.sum_squares([2, 3])` |
| `from studytools.stats import sum_squares` | `sum_squares([2, 3])` |
| `import studytools.stats as st` | `st.sum_squares([2, 3])` |

`from ... import ...` 不会跳过模块加载，也不会复制一份函数实现。`as` 只改变当前文件里使用的名字。先避免 `import *`，这样读代码时能知道每个名字从哪里来。

**同一进程中，已经导入的模块通常会按模块名保存在 `sys.modules` 中，再次导入会复用模块对象。** 它不等于每次都重新运行源文件。观察这个缓存：

```python
import importlib
import sys

first = importlib.import_module("collections")
second = importlib.import_module("collections")
print(first is second)                       # 输出：True
print(sys.modules["collections"] is first)   # 输出：True
```

注意缓存是按名字查找的：把同一个文件作为入口运行，又按普通模块名导入，可能形成两个模块实例。因此入口应尽量薄，公共状态与计算逻辑放在正常导入的模块中。细节见 [模块缓存](https://docs.python.org/3.12/reference/import.html#the-module-cache)。

## 绝对导入、相对导入与搜索路径

`from studytools.stats import sum_squares` 是**绝对导入**，从顶层导入名出发；`from .stats import sum_squares` 是**相对导入**，从当前包出发。`.` 表示当前包，`..` 表示上一级包，含义取决于包层级，**不是 shell 当前目录，也不是任意文件路径的父目录**。

对于常规文件模块，Python 从 `sys.path` 提供的位置查找顶层模块。找到包之后，查找子模块会使用包的 `__path__`。`sys.path` 的初始化通常包含入口位置、`PYTHONPATH`、标准库、环境的 `site-packages` 等；内置模块和自定义导入器还有其他查找路径，不是每次 import 都扫描这些目录。

先用只读命令观察你真正使用的解释器，而不是猜它来自系统、Conda 还是虚拟环境：

```sh
python3 -c 'import sys; print(sys.executable); print(sys.version); print(*sys.path, sep="\n")'
python3 -m pip --version
python3 -c 'import studytools; print(studytools.__file__)'
```

最后一条需要在项目根目录或安装后运行，输出应指向你预期的 `__init__.py`。对于普通文件模块，`__file__` 是很直接的来源线索；有些内置模块没有这个属性。

`PYTHONPATH` 是额外的导入搜索路径，不是 shell 用来找命令的 `PATH`。临时加路径可以做诊断，但长期依赖全局 `PYTHONPATH` 或在代码里到处 `sys.path.append(...)`，容易让同名模块来自意外位置。先检查来源，再修正项目的安装或启动方式。依据见 [sys.path 初始化](https://docs.python.org/3.12/library/sys_path_init.html)。

## python file.py 与 python -m 的区别

**按文件运行时，你给的是文件路径；用 `-m` 时，你给的是可导入的模块名，Python 负责找到它，再作为程序运行。** `-m` 后面不用 `.py`，也不用斜杠。

| 命令 | 默认入口搜索位置 | 包上下文 | 本页结果 |
| --- | --- | --- | --- |
| `python3 studytools/cli.py 2 3` | 把脚本所在的 `studytools/` 放到 `sys.path` 开头 | 没有可供相对导入使用的包上下文 | `.stats` 导入失败 |
| `python3 -m studytools.cli 2 3` | 把当前工作目录放到 `sys.path` 开头 | 知道入口属于 `studytools` | 输出 `13` |
| `python3 -m studytools 2 3` | 同上 | 查找 `studytools.__main__` 作为入口 | 输出 `13` |

这个表描述默认启动行为，未启用 `-I`、`-P` 等改变搜索路径的选项。`-m` 不会把进程的工作目录切换到包目录，也不会让 `open("data.txt")` 自动相对源码寻找文件。

直接运行 `cli.py` 时，常见报错是 `ImportError: attempted relative import with no known parent package`。文件明明存在，问题在于它没有以包成员的身份启动。回到项目根目录，用 `python3 -m studytools.cli ...`；或者先安装项目，再从其他目录启动。

`python3 -m studytools` 还要求包提供 `__main__.py`。并不是每个可导入模块都适合 `-m`：纯库可能没有命令行入口，原生扩展通常也不能直接用它运行。依据见 [官方 -m 说明](https://docs.python.org/3.12/using/cmdline.html#cmdoption-m)。

### `__name__`、`__main__` 与入口保护

`__name__` 是模块运行时的名字。普通导入 `studytools.cli` 时，它是 `"studytools.cli"`；把该模块作为入口运行时，它是 `"__main__"`。`-m` 同时保留了包上下文，所以相对导入仍可工作。

```text
导入 studytools.cli
  → 定义 main
  → __name__ 不是 "__main__"
  → 不调用 main，不读取调用者的命令行参数

python3 -m studytools.cli 2 3
  → 定义 main
  → __name__ 是 "__main__"
  → 调用 main，解析 2 和 3，输出 13
```

`if __name__ == "__main__":` 是普通条件判断，不是特殊语法指令。它只保护缩进在其中的语句；放在前面的顶层代码仍会在导入时执行。`__main__.py` 是入口文件名，`__name__` 是模块属性，二者有关联但不是同一个东西。详见 [官方入口教程](https://docs.python.org/3.12/library/__main__.html)。

### 为什么常写 python -m pip

`python3 -m pip install ...` 表示“让这个 `python3` 解释器运行它找到的 pip”。直接执行 `pip` 时，shell 会从 `PATH` 找可执行文件，它可能绑定另一套 Python。`-m` 能明确解释器，但如果 `python3` 本身选错了，仍然会装错环境。

其他常见例子：

| 命令 | 用途 |
| --- | --- |
| `python3 -m venv .venv` | 用所选解释器创建项目环境 |
| `python3 -m http.server 8765 --bind 127.0.0.1` | 把当前目录作为本地静态文件目录，适合预览本站 |
| `python3 -m unittest` | 运行标准库测试工具 |
| `python3 -m py_compile studytools/stats.py` | 编译源码以检查语法，写出字节码；不调用计算函数 |

`-m` 的统一含义始终是“按模块名启动”，不是“安装模块”。后面的模块决定具体做什么。

## 把项目安装到环境里

在根目录里能导入 `studytools`，不代表它已经安装：项目根目录可能刚好在搜索路径中。安装后，才可以从别的工作目录通过该环境找到它。

给 `square-demo/` 添加以下配置。TOML 是配置格式；`[project]` 存发行信息，`[build-system]` 指定怎样构建，setuptools 是这里选择的构建后端。

```toml
# 文件：pyproject.toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "studytools-demo"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = []

[project.scripts]
study-squares = "studytools.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["studytools*"]
```

`dependencies` 表示运行这个项目所需的第三方发行包；标准库 `argparse` 不写进去。`build-system.requires` 则是构建阶段需要的工具，两者用途不同。`studytools.cli:main` 表示安装工具要生成一个命令，调用该模块中的 `main` 函数。详见 [pyproject.toml 官方指南](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)。

先用前面的只读命令确认解释器，再在这个独立 demo 项目的根目录创建环境。以下是 macOS / Linux 命令；Windows 通常使用 `.venv\Scripts\python.exe`。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m studytools 2 3
.venv/bin/study-squares 2 3
# 两种启动方式都输出：13
```

用 `.venv/bin/python` 明确指向环境，不必激活。`source .venv/bin/activate` 主要是在当前 shell 中调整 `PATH`，方便把这个解释器写成 `python`。venv 默认隔离第三方包，但会使用创建它的基础 Python；它不是容器，也不会隔离操作系统与所有本地动态库。见 [venv 官方说明](https://docs.python.org/3.12/library/venv.html)。

### 普通安装、editable 安装、wheel 与源码包

| 做法或文件 | 你会看到的行为 |
| --- | --- |
| `python -m pip install .` | 从当前项目构建并安装；修改项目源码通常不会自动更新已安装内容 |
| `python -m pip install -e .` | editable 安装，让导入指向开发源码；纯 Python 修改通常在新进程中生效 |
| `.whl`，wheel | 可安装的发行产物，可能含 Python 源码，也可能含已编译扩展 |
| `.tar.gz` 等 sdist | 源码发行包，安装时通常需要调用构建后端生成 wheel |
| `requirements.txt` | pip 可读取的安装项列表，可含版本约束；不会自动定义自己项目的发行元数据 |

editable 安装不等于热更新：已经导入模块的进程一般仍使用缓存，需要重启；修改入口命令、依赖或构建配置，可能需要重新安装；修改 C++ 扩展通常还需要重新编译。具体机制由后端决定，并不总是简单的符号链接。见 [pip 本地项目安装](https://pip.pypa.io/en/stable/topics/local-project-installs/)。

`pyproject.toml` 不一定代表一个可安装项目，有些仓库只用它配置格式化或检查工具。也不要把一份宽泛的依赖列表当成完整可复现环境：解释器版本、平台、具体依赖版本及本地库都可能影响结果。

## `__pycache__`、pyc 与内存缓存

常规 CPython 源码模块的加载过程，可以这样理解：

```text
查找到 stats.py 的加载方式
  → 创建模块对象，先登记到 sys.modules
  → 检查是否有与当前解释器和源码匹配的 .pyc
  → 可用：读取字节码；不可用：编译源码得到字节码
  → 执行模块顶层代码，填充模块中的函数、变量等
  → 后续按相同名字导入时，复用 sys.modules 中的对象
```

先登记模块对象，可以避免相互导入时无休止地重新创建模块；但另一方此时也可能拿到一个尚未完成初始化的模块，这就是循环导入报错的一种来源。

`.pyc` 保存的是 Python 字节码，可由解释器执行；它不同于 C++ 编译出的 CPU 机器码。导入本页的 `stats.py` 后，常见文件名类似 `studytools/__pycache__/stats.cpython-312.pyc`，其中 `cpython-312` 标识解释器实现与版本。

| 缓存 | 在哪里 | 保存什么 | 解决什么问题 |
| --- | --- | --- | --- |
| `__pycache__/*.pyc` | 磁盘 | 编译后的字节码与校验信息 | 后续进程可省去部分源码编译工作 |
| `sys.modules` | 当前进程内存 | 按模块名保存的模块对象 | 同一进程再次导入时复用对象 |

**有 `.pyc` 仍要执行模块顶层代码，也不会自动缓存函数计算结果。** 它通常改善加载成本，不能由此断言平方和计算会变快。直接按路径运行的主脚本通常不会为自身写出 import 用的缓存，但它导入的其他模块可能会写出缓存。

默认时间戳模式会比较源码的修改时间和大小；还有基于哈希的校验模式。缓存不可用或没有写权限时，常规源码导入仍可编译并运行。源码存在时，删掉这些缓存通常只会让以后重新生成；**删缓存不能让已经运行的进程忘记 `sys.modules` 里的对象**。

```sh
python3 -B -m studytools 2 3
# 输出：13；-B 禁止这次运行写出 .pyc，但并不禁止读取已有缓存
```

项目通常把 `__pycache__/`、`*.pyc`、`.venv/` 加进 `.gitignore`。不要把缓存当作跨版本发布格式，也不要把“删除所有缓存”当作第一步排错。基础说明见 [Python 的 compiled files 小节](https://docs.python.org/3.12/tutorial/modules.html#compiled-python-files)，显式编译工具见 [py_compile](https://docs.python.org/3.12/library/py_compile.html)。

## 遇到报错时，先定位哪一层

| 现象 | 优先检查 | 原因关系 |
| --- | --- | --- |
| `ModuleNotFoundError` | `sys.executable`、`sys.path`、该解释器的 `pip --version` | 当前解释器可能没有安装它，或启动位置不对 |
| 相对导入没有 parent package | 是按路径启动，还是用 `-m` 启动包内模块 | 文件位置正确不代表包上下文存在 |
| `pip show` 有记录，import 仍失败 | 发行名和导入名是否一致；pip 是否属于同一解释器 | 安装记录和 import 使用不同的名字与查找机制 |
| 导入到了意外代码 | 普通模块的 `__file__`，工作目录里的同名文件 | 自己的 `json.py` 等可能遮蔽标准库 |
| `partially initialized module` | 两个模块是否在顶层互相导入 | 循环导入时一方可能还没定义被另一方请求的名字 |
| 修改源码没有生效 | 新进程还是旧进程；普通安装还是 editable；实际导入来源 | 可能是内存缓存，或运行了另一份代码 |

不要仅凭异常名称就安装或升级包。Traceback 中缺失的也可能是某个已找到包的内部依赖，需要读清楚最早失败的位置。涉及 `.so` / `.pyd` 的动态库加载错误，继续看 [原生扩展与兼容性](native-interop.md)。

## 自己动手检查理解

1. 在项目根目录分别运行 `python3 studytools/cli.py 2 3` 和 `python3 -m studytools.cli 2 3`，解释为什么只有后者成功。
2. 运行 `python3 -c 'import studytools.cli'`，确认没有启动命令行解析。再说明入口保护保护了哪些语句。
3. 导入一次 `studytools.stats`，查看生成的 `__pycache__`；用 `-B` 在一个全新的项目副本中导入，观察是否写出缓存。
4. editable 安装后，从项目目录之外，用该环境解释器运行 `-m studytools 2 3`，确认安装后的搜索路径确实可用。
5. 把 `sum_squares` 的实现改成返回 `42`，比较 editable 安装与普通安装在新进程中的结果，然后恢复源码。

完成这些实验后，应能独立解释：**文件在哪里、模块叫什么、谁启动它、哪个环境提供它，以及缓存发生在哪一层。**
