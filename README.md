# Learning Notes

算法、编程语言与 ML 系统的学习笔记，包含概念解释、推导、代码示例和参考来源。

**在线阅读：[sisyphus111.github.io/learning-notes](https://sisyphus111.github.io/learning-notes/)**

## 主题索引

| 主题 | 内容与入口 |
| --- | --- |
| 算法与题目 | [数组](content/algorithms/arrays.md)、[链表](content/algorithms/linked-lists.md)、[二叉树](content/algorithms/trees.md)、[图](content/algorithms/graphs.md)、[动态规划](content/algorithms/dynamic-programming.md)、[回溯](content/algorithms/backtracking.md)、[贪心](content/algorithms/greedy.md)、[位运算](content/algorithms/bitwise.md)、[数据结构](content/algorithms/data-structures.md)、[设计](content/algorithms/design.md)；提供 C++17 / Python 题解 |
| C++ STL | [速查](content/languages/cpp-stl/quickref.md)、[对象语义](content/languages/cpp-stl/value-semantics.md)、[容器](content/languages/cpp-stl/sequence-containers.md)、[迭代器与算法](content/languages/cpp-stl/adapters-iterators.md) |
| Python | [数据结构](content/languages/python/data-structures.md)、[常用语法](content/languages/python/iterables-and-functions.md)、[类与对象](content/languages/python/classes-and-objects.md)、[模块与包](content/languages/python/modules-and-packages.md)、[语言互操作](content/languages/python/native-interop.md) |
| PyTorch | [Tensor 与 GPU 内存](content/ml-systems/pytorch/tensors.md)、[神经网络基础](content/ml-systems/pytorch/neural-networks.md)、[常见计算与组件](content/ml-systems/pytorch/operations-and-components.md) |
| CUDA | [执行层级](content/ml-systems/cuda/execution-hierarchy.md)、[编译与指令](content/ml-systems/cuda/compilation-and-instructions.md)、[GEMM](content/ml-systems/cuda/gemm.md)、[性能分析](content/ml-systems/cuda/gemm-performance-analysis.md)、[FlashAttention](content/ml-systems/cuda/flash-attention.md) |
| GPU 软件栈 | [驱动与运行时](content/ml-systems/gpu/driver-runtime-framework.md)、[工具与计算库](content/ml-systems/gpu/software-stack.md)、[CUDA Graph](content/ml-systems/gpu/graphs-and-compilation.md)、[精度与量化](content/ml-systems/gpu/precision-and-quantization.md)、[通信与 AllReduce](content/ml-systems/gpu/communication-and-allreduce.md) |
| Attention 与模型架构 | [Transformer 运行流程](content/ml-systems/attention/transformer-runtime.md)、[线性注意力](content/ml-systems/attention/flash-linear-attention.md)、[模型技术总表](content/ml-systems/attention/model-technology-map.md)、[架构对照](content/ml-systems/attention/model-architectures.md)、[残差与条件记忆](content/ml-systems/attention/residual-experts-memory.md)、[TP 性能案例](content/ml-systems/attention/qwen3-h100-tp-case.md)、[EP 与 TP](content/ml-systems/attention/moe-ep-tp.md) |
| DeepSeek | [架构演进](content/ml-systems/deepseek/evolution.md)、[前向计算与张量形状](content/ml-systems/deepseek/forward-shapes.md) |

网站支持按题号、概念和 API 搜索（快捷键 `/`），以及章节目录、代码复制、C++ / Python 标签切换和深浅色主题。

## 内容来源

算法题解部分改编自 [Cong Li 的 Fuck-Leetcode](https://github.com/CodeLoser404/fuck-leetcode)，保留原有思路、题目出处与参考链接，并补充 C++ / Python 实现。正文中的 TODO 表示尚未完成的内容。

C++ STL 内容整理自本仓库原有手册。Python、PyTorch、CUDA、GPU 软件栈和模型架构笔记的参考文档、论文、源码及适用版本见各页。

## 本地构建

Markdown 源文件位于 `content/`，构建和验证脚本位于 `scripts/`。根目录的 HTML、`assets/search-index.json` 与 `assets/code.css` 为构建产物；网页样式和交互代码位于 `assets/`。

在仓库根目录创建 Python 环境并构建：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/build_v2.py
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
```

访问 [本地预览](http://127.0.0.1:8765/)。通过 HTTP 预览可使用搜索功能；MathJax 公式渲染需要网络。编辑 Markdown 或页面模板后，重新运行构建脚本，同步更新生成的页面和搜索索引。

## 验证

页面链接、代码语法与示例行为的检查入口：

```sh
.venv/bin/python scripts/check_site.py
.venv/bin/python scripts/check_cpp.py
.venv/bin/python scripts/test_examples.py
.venv/bin/python scripts/check_python.py  # 需要 Python 3.10+
node --check assets/site.js
git diff --check
```

C++ 题解采用 C++17，LeetCode 提供题目所需的节点类型；API 片段、反例和多文件示例需结合所在小节使用。验证脚本覆盖页面链接、可独立执行的代码和代表性用例，不代表所有题解都经过在线判题。模型形状、容量与性能推导不等于实机测量，各页标明具体的验证范围。
