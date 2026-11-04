# 笔记仓库重构计划

## 目标
1. 统一目录命名，建立清晰的内容层级
2. 增量构建，减少 git diff 噪音
3. 添加元数据系统，支持更丰富的导航和过滤
4. 保持现有的简洁构建流程

## 新目录结构

```
learning-notes/
├── content/                    # 所有源文件
│   ├── algorithms/             # 算法题（原 数组、链表 等）
│   │   ├── arrays.md
│   │   ├── linked-lists.md
│   │   ├── trees.md
│   │   ├── graphs.md
│   │   ├── dynamic-programming.md
│   │   ├── backtracking.md
│   │   ├── greedy.md
│   │   ├── bitwise.md
│   │   ├── data-structures.md
│   │   └── design.md
│   ├── languages/              # 编程语言
│   │   ├── cpp-stl/            # C++ STL 参考
│   │   │   ├── quickref.md
│   │   │   ├── value-semantics.md
│   │   │   ├── sequence-containers.md
│   │   │   ├── associative-containers.md
│   │   │   └── adapters-iterators.md
│   │   └── python/             # Python 参考
│   │       ├── data-structures.md
│   │       ├── math-operations.md
│   │       ├── iterables-functions.md
│   │       ├── classes-objects.md
│   │       ├── modules-packages.md
│   │       └── native-interop.md
│   └── ml-systems/             # ML 系统
│       ├── pytorch/            # PyTorch 教学
│       │   ├── tensors.md
│       │   ├── neural-networks.md
│       │   └── operations-components.md
│       ├── cuda/               # CUDA 教学
│       │   ├── execution-hierarchy.md
│       │   ├── gemm.md
│       │   ├── gemm-performance.md
│       │   ├── flash-attention.md
│       │   └── compilation-instructions.md
│       ├── gpu/                # GPU 软件栈
│       │   ├── driver-runtime-framework.md
│       │   ├── software-stack.md
│       │   ├── graphs-compilation.md
│       │   ├── precision-quantization.md
│       │   └── communication-allreduce.md
│       ├── attention/          # Attention 与 Transformer
│       │   ├── transformer-runtime.md
│       │   ├── flash-linear-attention.md
│       │   ├── model-technology-map.md
│       │   ├── model-architectures.md
│       │   ├── residual-experts-memory.md
│       │   ├── qwen3-h100-tp-case.md
│       │   └── moe-ep-tp.md
│       └── deepseek/           # DeepSeek
│           ├── evolution.md
│           └── forward-shapes.md
├── scripts/                    # 构建和验证脚本
├── site/                       # 生成的网站（新）
│   ├── index.html
│   ├── algorithms/
│   ├── languages/
│   ├── ml-systems/
│   └── assets/
├── assets/                     # 源资源文件
│   ├── style.css
│   ├── site.js
│   └── fonts/
├── .venv/
├── requirements.txt
└── README.md
```

## 元数据系统

每个 Markdown 文件顶部添加 YAML front matter：

```yaml
---
title: 数组
category: algorithms
tags: [哈希表, 双指针, 滑窗, 前缀和, 二分]
difficulty: mixed
problem_count: 25
updated: 2024-11-03
reading_time: 45
prerequisites: []
related: [linked-lists, trees]
---
```

## 构建优化

1. **增量构建**：只重新生成修改过的 Markdown
2. **内容哈希**：HTML 文件名包含内容哈希，支持长期缓存
3. **生成目录分离**：HTML 输出到 `site/` 目录
4. **源码与产物分离**：main 分支只提交源文件，GitHub Pages 从 site/ 发布

## 迁移步骤

### 第一步：创建新目录结构（不删除旧文件）
- 创建 content/ 和 site/ 目录
- 复制文件到新位置，保留 git 历史

### 第二步：更新构建脚本
- 修改 scripts/build.py 支持新结构
- 实现增量构建逻辑
- 添加元数据解析

### 第三步：验证
- 运行所有验证脚本
- 本地预览网站
- 检查所有链接

### 第四步：删除旧文件
- 删除旧的中文目录
- 删除根目录的 HTML 文件
- 更新 .gitignore

### 第五步：配置 GitHub Pages
- 设置从 site/ 目录发布
- 或配置 gh-pages 分支自动部署

## 兼容性

- 保持 URL 结构尽量不变（通过重定向或别名）
- 外部链接不失效
- 搜索索引更新

## 估计工作量

- 目录迁移：2-3 小时
- 构建脚本重写：4-6 小时
- 测试验证：2-3 小时
- 文档更新：1 小时

总计：**1-2 天**
