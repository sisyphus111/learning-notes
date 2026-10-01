# 算法刷题笔记

按题型整理的算法思路与 C++17 / Python 3.10+ 实现，配套 STL 速查和容器行为参考。

**阅读网站：[sisyphus111.github.io/code-notes](https://sisyphus111.github.io/code-notes/)**

- 10 个题型，133 个题目条目；保留「焚诀」、思路推导、复杂度和原有参考链接。
- 同一解法提供 C++ / Python 标签页，恢复 7 个记忆化搜索小节；运行时间伪造片段仍不收录。
- 5 个 STL 页面：速查、对象语义、序列容器、关联容器、适配器与算法。
- 按题号、题目或 API 搜索（快捷键 `/`），章节目录、代码复制、与个人主页相同的太阳/月亮主题图标及手机阅读。

## 内容来源

刷题主体基于 Cong Li 的 **Fuck-Leetcode** 笔记，本仓库引入的原始快照为
[`db6f82c`](https://github.com/sisyphus111/code-notes/commit/db6f82c)。这是他人笔记的双语言改编版：保留原作者的讲解风格与既有出处，不代表原作者提供或验证了补充实现。原文中的 TODO 仍表示未完成的内容。

STL 内容选自本仓库的旧手册。旧 `main` 的完整提交历史保存在独立分支
[`codex/backup-main-2026-10-01`](https://github.com/sisyphus111/code-notes/tree/codex/backup-main-2026-10-01)，快照提交为 `7f888bd`。

## 维护方式

编辑各题型目录与 `STL/` 下的 Markdown，然后生成网站。根目录的 HTML、`assets/code.css` 和 `assets/search-index.json` 是构建产物；`assets/style.css` 与 `assets/site.js` 是网页样式和交互源文件。

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
- `check_python.py`：检查全部 209 个 Python 块的语法，并运行 164 个实现版本的代表性用例。需要 Python 3.10+，以支持 `Counter` 的比较语义。
- 验证覆盖编译和上述行为检查，不等于全部题目都经过 LeetCode 在线判题。

旧的 `count.py`、`check.py` 已由网页的条目统计和搜索替代。
