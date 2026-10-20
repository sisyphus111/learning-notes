# 求知录 · Learning Notes

记录学习中的问题、推导与实践。当前内容涵盖算法、编程语言、机器学习与计算系统。

**阅读网站：[sisyphus111.github.io/learning-notes](https://sisyphus111.github.io/learning-notes/)**

- 10 个题型，133 个题目条目；保留「焚诀」、思路推导、复杂度和原有参考链接。
- 同一解法提供 C++ / Python 标签页，恢复 7 个记忆化搜索小节；运行时间伪造片段仍不收录。
- Python 语法参考从基础数据结构开始，包含内置容器、deque、Counter、defaultdict、heapq 及复制/可哈希等易错点；另有生成器表达式与 `all`、`any`、`sum` 等常用函数的用法。
- [类与对象](Python/classes-and-objects.md) 用简短表格和独立的 `assert` 示例讲解 `class`、`self`、`__init__`、实例与类属性、方法、继承 / `super()`、property、特殊方法和 dataclass。
- Python 项目入门包含模块与包、`import`、`python -m`、入口保护、pip / venv、`pyproject.toml`、editable 安装和 `__pycache__`；语言互操作页提供可构建的 pybind11 示例，并解释数据转换、所有权、GIL、ABI 与进程通信。
- 5 个 STL 页面：速查、对象语义、序列容器、关联容器、适配器与算法。
- PyTorch 教学分为三页：[Tensor 与 GPU 内存](PyTorch/tensors.md) 讲 shape、计算、stride、共享存储与异步执行；[神经网络基础](PyTorch/neural-networks.md) 讲 torch.nn、functional、模型与参数、自动求导及训练/推理流程；[常见计算与组件](PyTorch/operations-and-components.md) 展开 Linear、softmax、LayerNorm、RMSNorm、SwiGLU 和 attention 的 Torch 实现。
- CUDA 教学独立成板块：[执行层级与硬件映射](CUDA/execution-hierarchy.md) 讲线程分组、SM / GPC 放置、资源驻留与同步范围；[PTX、SASS 与编译流程](CUDA/compilation-and-instructions.md) 讲三类 kernel 的编译路径、ptxas / PTX Compiler 库和 AOT / JIT。
- GPU 软件栈独立成板块：[驱动、运行时与框架](GPU/driver-runtime-framework.md) 从 H100 设备讲到内核驱动、libcuda / libcudart、PyTorch、容器和版本；[GPU 软件栈与工具](GPU/software-stack.md) 区分计算库、CUTLASS / CuTe、编译工具与推理框架；[计算图、编译与 CUDA Graph](GPU/graphs-and-compilation.md) 解释编译优化与执行重放，并展开 CUDA / PyTorch API、固定虚拟地址、图内分配、private pool 与跨图共享。
- [浮点格式、混合精度与量化](GPU/precision-and-quantization.md) 讲符号/指数/小数部分、FP16 与 BF16、TF32、FP8、E8M0、MXFP8 / NVFP4，以及存储、乘法、累加和 scale 的区别。
- [GEMM 计算与优化](CUDA/gemm.md) 从点积和 K 分段讲起，连接 CTA/register tiling、内存布局、Tensor Core、异步流水线、Split-K、persistent/grouped GEMM、融合与 scale 转置。
- 按题号、题目或 API 搜索（快捷键 `/`），章节目录、代码复制、冰蓝配色、深浅主题切换及手机阅读。首次访问跟随系统主题，手动选择后记住偏好。

## 内容来源

刷题主体基于 Cong Li 的 **Fuck-Leetcode** 笔记，本仓库引入的原始快照为
[`db6f82c`](https://github.com/sisyphus111/learning-notes/commit/db6f82c)。这是他人笔记的双语言改编版：保留原作者的讲解风格与既有出处，不代表原作者提供或验证了补充实现。原文中的 TODO 仍表示未完成的内容。

STL 内容选自本仓库的旧手册。旧 `main` 的完整提交历史保存在独立分支
[`codex/backup-main-2026-10-01`](https://github.com/sisyphus111/learning-notes/tree/codex/backup-main-2026-10-01)，快照提交为 `7f888bd`。

## 维护方式

编辑各题型目录、`STL/`、`Python/`、`PyTorch/`、`CUDA/` 与 `GPU/` 下的 Markdown，然后生成网站。根目录的 HTML、`assets/code.css` 和 `assets/search-index.json` 是构建产物；`assets/style.css` 与 `assets/site.js` 是网页样式和交互源文件。

构建脚本使用 Python，仅用于维护网站；笔记中的算法实现同时提供 C++ 和 Python。在仓库根目录创建独立环境：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/build.py
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
```

访问 [本地预览](http://127.0.0.1:8765/)。使用 HTTP 预览，搜索功能需要读取本地索引。代码高亮在构建时生成；公式通过 MathJax CDN 渲染，需要网络。

GitHub Pages 从 `main` 分支根目录发布。生成网页并提交、推送到 `main` 后，GitHub 会重新部署。

## 示例与验证

C++ 题解省略标准库头文件和 `using namespace std;`，LeetCode 的 `ListNode`、`TreeNode` 等类型由题目提供。STL 页中的 API 片段、反例以及构造函数替换片段需要放在其所在小节的上下文中使用，不能当作独立程序编译。

```sh
.venv/bin/python scripts/check_cpp.py
.venv/bin/python scripts/test_examples.py
.venv/bin/python scripts/check_site.py
python3.10 scripts/check_python.py  # 使用任意 Python 3.10+ 解释器
node --check assets/site.js
git diff --check
```

- `check_cpp.py`：分别编译 197 个完整 C++17 示例，提供标准库与题目节点类型的测试前置定义；明确排除非独立片段和 STL 参考片段。
- `test_examples.py`：直接提取文档代码，运行 155 个实现版本的代表性与边界检查，启用 AddressSanitizer 和 UBSan。其中包含对子数组计数的穷举对照。
- `check_python.py`：检查全部 Python 代码块的语法，运行 164 个题解实现版本的代表性用例，并运行 Python 参考页的独立示例、核对打印结果。以 `# 文件：` 标注的多文件片段仅做语法检查，需要按正文组装项目后运行；该检查不构建 C++ 扩展。需要 Python 3.10+，以支持 `Counter` 的比较语义。
- 验证覆盖编译和上述行为检查，不等于全部题目都经过 LeetCode 在线判题。

模块/包与 pybind11 多文件示例另在 macOS ARM64、CPython 3.12.14、Apple Clang 21.0.0 下按文档组装验证：包与模块的 `-m` 入口、安装生成的命令、入口保护、相对导入失败、普通与 editable 安装的源码修改行为、`.pyc` 生成及 `-B`、扩展的 wheel 构建与安装、空输入/负数/错误类型调用均通过。Windows / Linux 尚未实测，也未进行扩展性能比较。

旧的 `count.py`、`check.py` 已由网页的条目统计和搜索替代。
