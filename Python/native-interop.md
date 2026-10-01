# Python 与其他语言互操作

Python 项目不一定只运行 Python 代码。常见结构是：**Python 负责组织调用，C/C++ 或其他语言负责实现某些计算；双方通过一个明确的接口交换数据。** pybind11 是把 C++ 接口暴露给 Python 的工具，不能把任意多语言代码自动“兼容”起来。

先读 [模块、包与运行方式](modules-and-packages.md)，理解 import 和安装的区别。本页以 Python 3.10+、传统启用 GIL 的 CPython 和 C++17 为例；示例计算使用小数值，不承担数值精度或性能基准的验证。

## 多语言程序怎样连接起来

首先区分两条路线：**同一个进程内调用**，或者**不同进程之间传递请求与结果**。前者要处理对象、内存和二进制接口；后者要约定消息格式、错误与进程生命周期。

| 方式 | 适用场景 | 如何交换数据 | 要付出的成本 |
| --- | --- | --- | --- |
| 原生扩展：CPython C API、pybind11、Cython、Rust 的 PyO3 等 | 高频调用、复用本地计算代码 | Python 对象和本地类型之间转换或共享内存 | 构建、ABI、对象生命周期、崩溃会影响整个进程 |
| FFI：`ctypes`、CFFI 等 | 已有库提供 C 接口 | 按指定签名传数值、指针、缓冲区 | 必须准确约定参数、布局、所有权 |
| `subprocess` | 已有独立命令行工具 | 参数、标准输入输出、文件 | 启动进程和数据编码成本 |
| RPC / HTTP 等 | 跨机器，或服务由不同语言实现 | 按协议编码请求与响应 | 网络、序列化、超时与服务管理 |
| embedding，嵌入解释器 | C++ 应用需要运行 Python 脚本 | 宿主创建解释器，再调用 Python | 宿主管理解释器、线程与对象生命周期 |

这张表是按边界选择方式的思路，不是固定的性能排名。批量调用可能让进程通信足够划算，反复传很小的数据也可能让原生扩展的转换成本占主导。这里先做 pybind11 的最小闭环，理解 Python 调用 C++ 的具体过程。

## 原生扩展为什么能被 import

普通 Python 模块可以来自 `.py`；**原生扩展模块**则是实现了 Python 扩展接口的动态加载库。CPython 加载它的二进制代码，调用规定的初始化入口，建立 Python 模块对象。调用者随后仍然使用 `import`、属性访问和函数调用。

```text
native_ops.cpp + 绑定代码
  → C++ 编译器编译、链接
  → native_ops.<解释器与平台标签>.so / .pyd
  → Python import native_ops
  → Python 模块对象：暴露 sum_squares 函数
```

Linux / macOS 常见扩展后缀是 `.so`，Windows 是 `.pyd`；实际名称通常还有解释器与平台信息。macOS 的普通动态库也常见 `.dylib`，但不能把任意 `.dylib` 改名为 `.so` 就拿来 import：它还需要 Python 规定的初始化接口。

pybind11 是一套 C++ 头文件库，协助注册 Python 可见的函数或类、转换参数及返回值。它不是 Python 解释器，也不是 C++ 编译器。首次使用仍需要编译器、Python 开发头文件与构建工具。基础机制见 [pybind11 官方入门](https://pybind11.readthedocs.io/en/stable/basics.html)。

## 一个能构建安装的 pybind11 项目

目标仍是平方和：Python 传入 `[2, 3]`，C++ 计算后返回 `13.0`。在另一个独立目录 `native-demo/` 中创建这三个文件，避免和上一页的项目混在一起。

```text
native-demo/
├── native_ops.cpp        ← 计算逻辑与绑定
├── setup.py              ← 告诉 setuptools 怎样编译扩展
└── pyproject.toml        ← 声明构建工具与发行信息
```

### C++ 计算与绑定

下面保存为 `native_ops.cpp`。这里用 `double` 计算，所以 Python 看到的返回值是浮点数。

```cpp
// 文件：native_ops.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

double sum_squares(const std::vector<double>& nums) {
    double total = 0.0;
    for (double x : nums) {
        total += x * x;
    }
    return total;
}

PYBIND11_MODULE(native_ops, m) {
    m.doc() = "平方和的 C++ 示例";
    m.def("sum_squares", &sum_squares, "计算输入数值的平方和");
}
```

读这段代码时，先分清三件事：

1. `sum_squares` 是普通 C++ 函数，它本身不认识 Python 列表。
2. `pybind11/stl.h` 提供容器转换，让 Python 列表可以转换成 `std::vector<double>`。
3. `PYBIND11_MODULE(native_ops, m)` 定义扩展模块初始化接口，`m.def` 把 C++ 函数指针注册为 Python 可调用的函数。

`native_ops` 是导入名，它必须与扩展的构建名称匹配。函数名前面的字符串是 Python 对外接口名，不必和 C++ 函数名相同；本例保持一致以便理解。

### 构建配置

`setup.py` 仅用于配置 setuptools 的扩展编译。安装命令仍然走 pip 和 `pyproject.toml`，不需要执行旧式的 `python setup.py install`。

```python
# 文件：setup.py
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    ext_modules=[
        Pybind11Extension("native_ops", ["native_ops.cpp"], cxx_std=17),
    ],
    cmdclass={"build_ext": build_ext},
)
```

发行名使用 `native-ops-demo`，导入名使用 `native_ops`，再次体现安装与导入的名字不同。`py-modules = []` 表示没有需要一同打包的纯 Python 单文件模块；扩展由上面的 `ext_modules` 声明。

```toml
# 文件：pyproject.toml
[build-system]
requires = ["setuptools>=64", "pybind11>=2.11"]
build-backend = "setuptools.build_meta"

[project]
name = "native-ops-demo"
version = "0.1.0"
requires-python = ">=3.10"

[tool.setuptools]
py-modules = []
```

pybind11 在本例中是构建依赖：它的头文件参与编译，不要求调用者每次执行 `sum_squares` 都导入 pybind11。复杂项目也可以用 CMake / scikit-build-core 等后端；这里选择 setuptools，让需要理解的文件数量保持最少。构建方式见 [pybind11 官方构建文档](https://pybind11.readthedocs.io/en/stable/compiling.html#modules-with-setuptools)。

### 安装并调用

先确认解释器与编译器，以下编译器诊断命令针对 macOS / Linux；Windows 通常需要相应的 MSVC 开发工具环境。再在 `native-demo/` 内创建环境并安装。

```sh
python3 -c 'import sys; print(sys.executable); print(sys.version)'
c++ --version
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -c 'import native_ops; print(native_ops.sum_squares([2, 3]))'
# 输出：13.0
.venv/bin/python -c 'import native_ops; print(native_ops.__file__)'
# 输出位置应是该环境中安装的 .so 或 .pyd 文件
```

pip 会创建隔离的构建环境并准备 `build-system.requires` 中的依赖，再由后端调用编译器、生成 wheel 并安装。**构建依赖隔离与项目的运行环境隔离是两层不同的事情。** 下载依赖需要网络；编译失败时，应从构建日志里的首个实际错误定位是工具缺失、头文件缺失还是代码问题。

想查看可分发的产物，可以在同一 demo 目录执行：

```sh
.venv/bin/python -m pip wheel . --no-deps --wheel-dir dist
```

它构建当前项目的 wheel，而不是自动生成所有系统都能安装的包。`--no-deps` 跳过运行依赖的 wheel 构建，不取消 `build-system.requires` 的构建依赖准备。

## 这次调用中发生了哪些转换

```text
Python 列表 [2, 3]
  → pybind11 逐元素转换、构造 std::vector<double>{2.0, 3.0}
  → 调用 C++ sum_squares(const std::vector<double>&)
  → 返回 C++ double：13.0
  → 转成 Python float：13.0
```

这个示例中的 `const std::vector<double>&` 避免函数再复制一次已经转换出的 vector，**但不会免掉 Python 列表到 vector 的转换和复制**。对于 `n` 个元素，这一步有 O(n) 的处理成本；每次跨边界调用也有包装与检查开销。不能只看到计算写成了 C++，就认为端到端一定更快。

Python list 保存对象引用，C++ vector 保存连续的 `double` 元素，二者内存布局不同。自动把 vector 转回 list 也会构造 Python 对象，修改转换后的 C++ 容器通常不会直接修改调用者的 Python 列表。依据见 [pybind11 STL 容器转换](https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html)。

如果目标是处理大型数组，通常进一步研究 NumPy buffer / `py::array_t`、类型和连续性检查、批量调用。即便用了数组接口，也要核实是否发生类型转换或隐式复制；CPU 数组接口更不会自动变成对 GPU 显存指针的安全访问。本例只验证互操作，不用于证明加速效果。

## 所有权、异常与 GIL

跨语言最容易出问题的地方，往往是双方对同一个对象的理解不一致。

| 边界 | 必须回答的问题 | 典型错误 |
| --- | --- | --- |
| 类型 | 参数是整数、浮点数、字符串还是数组？大小和布局是什么？ | 把对象指针当成数值缓冲区 |
| 所有权与生命周期 | 谁分配，谁释放；返回的引用能活多久？ | 返回局部变量地址，Python 继续访问已释放内存 |
| 异常 | C++ 错误如何报告给 Python？ | 误以为所有本地错误都会变成可捕获异常 |
| 线程 | 当前是否允许访问 Python 对象？本地共享数据有没有同步？ | 释放 GIL 后继续调用 Python API |

返回数值的本例比较简单；返回指针、引用或共享缓冲区时，需要了解 pybind11 的 `return_value_policy`、`keep_alive` 等所有权机制。`reference` 不会让短命对象自动延长寿命。见 [返回值策略](https://pybind11.readthedocs.io/en/stable/advanced/classes.html#return-value-policies)。

pybind11 能把许多标准 C++ 异常翻译为 Python 异常，例如 `std::invalid_argument` 对应 `ValueError`。越界导致的未定义行为、段错误和进程终止则不是普通 Python 异常；原生代码的内存错误可以直接让整个 Python 进程崩溃。需要从异常与内存安全两个层面检查。

### C++ 函数会自动并行吗

在传统启用 **GIL（Global Interpreter Lock，全局解释器锁）** 的 CPython 中，线程访问 Python 对象通常需要持有 GIL。pybind11 从 Python 进入 C++ 函数时，不会仅仅因为代码是 C++ 就自动释放它。

对于耗时、完全不访问 Python 对象的本地计算，可以在绑定时使用 `py::call_guard<py::gil_scoped_release>()` 等机制，在计算期间释放 GIL。参数转换发生在调用前，返回值转换发生在计算后的适当持锁阶段；本地计算内部若要回调 Python，就需要重新取得 GIL。

释放 GIL 只是允许其他 Python 线程有机会运行，**不会给这个平方和循环自动创建工作线程**。C++ 自己的数据竞争也不会由 GIL 的释放机制解决。free-threaded CPython 是另一种运行配置，其扩展支持要求需要单独确认；本页不把传统 GIL 规则推广到所有解释器。见 [pybind11 的 GIL 说明](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil)。

## 多语言兼容到底兼容哪些东西

“Python 可以调用 C++”只说明存在接口。**源码能编译、二进制能加载、输入输出语义一致，是三个不同的条件。**

| 层次 | 需要匹配的内容 | 例子 |
| --- | --- | --- |
| 源码与构建 | 语言标准、编译器、头文件、构建选项 | C++17 源码需要支持该标准的编译器 |
| API | 函数名、参数与返回值的含义 | 调用者传“秒”，实现却当成“毫秒” |
| ABI，二进制接口 | 符号、调用约定、类型布局、运行时约定 | 扩展针对另一版 CPython 或 C++ 运行库构建 |
| 平台 | 操作系统、CPU 架构、平台标签 | macOS ARM64 的扩展不能直接作为 Linux x86-64 库加载 |
| 外部动态库 | 扩展依赖的库是否存在、版本是否符合约定 | 扩展已安装，但依赖的本地库无法加载 |

ABI 可以理解为“编译后双方如何在二进制层面约定调用与内存表示”。C++ 还有名称修饰、标准库实现等细节；跨编译器直接暴露 C++ 对象较难，因此很多通用库提供边界更稳定的 C 接口，但 C 接口也仍有平台、类型大小和调用约定要求。

wheel 文件名的末尾包含 **Python tag、ABI tag、platform tag**。例如一个带 `cp312-cp312-...` 的扩展 wheel 通常针对 CPython 3.12 及对应 ABI；`py3-none-any` 常见于不含平台二进制的纯 Python wheel。标签只帮助安装工具筛选，不会证明所有运行依赖都可用。Python 的 Stable ABI / `abi3` 有专门限制，不能把普通 pybind11 构建产物重命名就获得它。见 [wheel 平台兼容标签规范](https://packaging.python.org/en/latest/specifications/platform-compatibility-tags/)。

### .pyc 与 .so / .pyd 不能混为一谈

| 文件 | 内容 | 怎样产生 | 怎样使用 |
| --- | --- | --- | --- |
| `.py` | Python 源码 | 人写的源文件 | 解释器编译并执行 |
| `.pyc` | Python 字节码 | CPython 编译源码、缓存 | 由兼容的解释器读取执行 |
| `.so` / `.pyd` 扩展 | 本地机器码与 Python 扩展接口 | 本地编译器编译、链接 | 由解释器动态加载 |
| `.whl` | 代码和发行元数据的安装包 | 构建后端打包 | pip 安装后，解释器导入其中的模块 |

删除 `__pycache__` 不会重新编译 C++ 扩展。改了 `native_ops.cpp` 后，需要重新构建安装；重新加载正在使用的扩展也比普通 Python 模块复杂，实践中先用新进程验证构建后的产物。

## 已有工具或 C 库时，还有哪些路线

### subprocess：调用另一个进程

如果一个工具本来就有可执行程序，不一定要把它改成 Python 扩展。Python 可以启动它，发送输入，再读取结果。下面用当前 Python 解释器扮演外部工具，让例子无需安装其他命令。

```python
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "-c", "print(2 * 2 + 3 * 3)"],
    check=True,
    capture_output=True,
    text=True,
)
print(result.stdout.strip())  # 输出：13
```

真实工具可以用 C++、Rust 或其他语言实现，只要双方约定好输入输出。这里传的是参数列表，不经 shell 拼接；`check=True` 在非零退出码时抛出异常，`text=True` 把捕获结果解码成文本。另一个进程拥有自己的地址空间，不能直接传一个 Python list 的内存地址当作数据。见 [subprocess 官方文档](https://docs.python.org/3.12/library/subprocess.html#subprocess.run)。

### ctypes：已有 C 接口的动态库

`ctypes` 是标准库，可加载动态库并按 C 调用约定调用函数。若库提供 `double sum_squares(const double* values, size_t n)`，调用方需要声明 `argtypes`、`restype`，构造匹配的连续数组，保证数组在调用期间有效。

它不是“把任意 C++ 函数名传进去就能用”：类、模板、名称修饰和对象生命周期都需要专门接口设计，常见方式是在库端提供明确的 `extern "C"` 包装。`ctypes` 默认返回类型等约定也可能与你的库不一致，错误签名可能导致崩溃，而非清晰的 Python 类型错误。见 [ctypes 官方文档](https://docs.python.org/3.12/library/ctypes.html)。

### Rust、Java 与跨机器服务

Rust 可以通过 PyO3 等工具实现 Python 扩展，也可以导出 C 接口，或作为独立进程提供服务；JVM 程序则可以通过专用桥接工具或服务协议协作。学习时先确定边界在哪里，再研究相应工具，不能因为两个语言都能编译就认为其对象可以直接互传。

跨机器的服务通常通过 HTTP / RPC 等协议交换可编码的数据。此时“兼容”主要依赖消息字段、版本、单位和错误约定，同时需要处理超时、进程故障与网络故障。只有明确设计的共享内存或设备缓冲区协议，才可能绕开部分数据复制；它们又增加同步和生命周期要求。

## 最小排错顺序与练习

1. **确认调用来源。** 用 `sys.executable` 与 `native_ops.__file__` 确认解释器和扩展来自预期环境。
2. **确认是否已经构建。** `.cpp` 只是源码，不能直接 import；阅读 pip 日志中的首个构建错误。
3. **确认能否加载。** 已有 `.so` 仍出现 `dlopen`、`undefined symbol`、`wrong architecture` 等错误时，检查架构、Python ABI 与外部动态库。
4. **确认接口语义。** 本例比较 `sum_squares([])`、`sum_squares([2, 3])`、`sum_squares([-2, 3])`，应得到 `0.0`、`13.0`、`13.0`；传无法转换的对象，观察 Python 端错误。
5. **确认修改生效。** 修改 C++ 函数后重新安装，再用新进程执行，观察结果和扩展路径。删除 `.pyc` 无法代替这一步。

理解这些边界后，再去研究更复杂的数组、对象绑定和 GPU 扩展。下一步应能回答：**运行的是哪个进程、跨边界传了什么、谁持有内存、二进制针对什么环境构建。**
