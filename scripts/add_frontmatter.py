#!/usr/bin/env python3
"""为所有 Markdown 文件添加 YAML front matter 元数据"""
from pathlib import Path
import re
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

# 文档元数据配置
METADATA = {
    # 算法
    'content/algorithms/arrays.md': {
        'title': '数组',
        'category': 'algorithms',
        'tags': ['哈希表', '双指针', '滑窗', '前缀和', '二分'],
        'description': '哈希表 · 双指针 · 滑窗 · 前缀和 · 二分'
    },
    'content/algorithms/linked-lists.md': {
        'title': '链表',
        'category': 'algorithms',
        'tags': ['基础操作', '快慢指针', '递归'],
        'description': '基础操作 · 快慢指针 · 递归'
    },
    'content/algorithms/trees.md': {
        'title': '二叉树',
        'category': 'algorithms',
        'tags': ['递归', '搜索树', '层序遍历'],
        'description': '递归 · 搜索树 · 层序遍历'
    },
    'content/algorithms/graphs.md': {
        'title': '图',
        'category': 'algorithms',
        'tags': ['搜索', '拓扑排序', '最小生成树', '最短路'],
        'description': '搜索 · 拓扑排序 · 最小生成树 · 最短路'
    },
    'content/algorithms/dynamic-programming.md': {
        'title': '动态规划',
        'category': 'algorithms',
        'tags': ['背包', '线性DP', '区间DP', '树形DP'],
        'description': '背包 · 线性 DP · 区间 DP · 树形 DP'
    },
    'content/algorithms/backtracking.md': {
        'title': '回溯',
        'category': 'algorithms',
        'tags': ['组合', '子集', '排列', '棋盘'],
        'description': '组合 · 子集 · 排列 · 棋盘'
    },
    'content/algorithms/greedy.md': {
        'title': '贪心',
        'category': 'algorithms',
        'tags': ['跳跃游戏', '区间划分'],
        'description': '跳跃游戏 · 区间划分'
    },
    'content/algorithms/bitwise.md': {
        'title': '位运算',
        'category': 'algorithms',
        'tags': ['异或', '公共前缀', 'lowbit'],
        'description': '异或 · 公共前缀 · lowbit'
    },
    'content/algorithms/data-structures.md': {
        'title': '数据结构',
        'category': 'algorithms',
        'tags': ['堆', '单调结构', 'Trie', '树状数组', '线段树'],
        'description': '堆 · 单调结构 · Trie · 树状数组 · 线段树'
    },
    'content/algorithms/design.md': {
        'title': '设计',
        'category': 'algorithms',
        'tags': ['最小栈', 'LRU缓存'],
        'description': '最小栈 · LRU 缓存'
    },

    # C++ STL
    'content/languages/cpp-stl/quickref.md': {
        'title': 'STL 速查',
        'category': 'cpp-stl',
        'tags': ['容器', '算法', 'API'],
        'description': '容器、算法与常用写法'
    },
    'content/languages/cpp-stl/value-semantics.md': {
        'title': '对象语义',
        'category': 'cpp-stl',
        'tags': ['构造', '复制', '移动', '引用', '传参'],
        'description': '构造 · 复制与移动 · 引用 · 传参'
    },
    'content/languages/cpp-stl/sequence-containers.md': {
        'title': '序列容器',
        'category': 'cpp-stl',
        'tags': ['vector', 'string', 'deque', 'list'],
        'description': 'vector · string · deque · list'
    },
    'content/languages/cpp-stl/associative-containers.md': {
        'title': '关联容器',
        'category': 'cpp-stl',
        'tags': ['map', 'set', 'unordered_map'],
        'description': 'map / set · unordered 系列'
    },
    'content/languages/cpp-stl/adapters-iterators.md': {
        'title': '适配器与算法',
        'category': 'cpp-stl',
        'tags': ['栈', '队列', '堆', '迭代器', '标准算法'],
        'description': '栈 · 队列 · 堆 · 迭代器 · 标准算法'
    },

    # Python
    'content/languages/python/data-structures.md': {
        'title': 'Python 数据结构速查',
        'category': 'python',
        'tags': ['语法', '容器', 'API'],
        'description': '语法 · 返回值 · 原地修改 · 常见错误'
    },
    'content/languages/python/math-operations.md': {
        'title': '数学运算速查',
        'category': 'python',
        'tags': ['整除', '取余', '乘方', '取整'],
        'description': '整除与取余 · 乘方 · 取整 · math · 浮点误差'
    },
    'content/languages/python/iterables-and-functions.md': {
        'title': '可迭代对象与常用函数',
        'category': 'python',
        'tags': ['生成器', 'all', 'any', 'sum', 'map', 'filter'],
        'description': '生成器表达式 · all / any · sum · min / max · map / filter'
    },
    'content/languages/python/classes-and-objects.md': {
        'title': '类与对象',
        'category': 'python',
        'tags': ['class', 'self', '__init__', '继承', '特殊方法'],
        'description': 'class · self · __init__ · 属性与方法 · 继承 · 特殊方法'
    },
    'content/languages/python/modules-and-packages.md': {
        'title': '模块、包与运行方式',
        'category': 'python',
        'tags': ['import', 'package', 'pip', 'venv'],
        'description': 'import · package · python -m · pip / venv · __pycache__'
    },
    'content/languages/python/native-interop.md': {
        'title': 'Python 与其他语言互操作',
        'category': 'python',
        'tags': ['pybind11', 'C++扩展', 'GIL', 'ABI'],
        'description': 'pybind11 · C++ 扩展 · 类型转换 · GIL · ABI · 多语言边界'
    },

    # PyTorch
    'content/ml-systems/pytorch/tensors.md': {
        'title': 'Tensor 与 GPU 内存',
        'category': 'pytorch',
        'tags': ['shape', '广播', '矩阵乘法', 'stride', 'view', 'CUDA'],
        'description': 'shape · 广播 · 矩阵乘法 · stride · view · CUDA 执行'
    },
    'content/ml-systems/pytorch/neural-networks.md': {
        'title': '神经网络基础',
        'category': 'pytorch',
        'tags': ['torch.nn', 'Module', '参数', '自动求导'],
        'description': 'torch.nn · functional · Module · 参数 · 自动求导 · 训练与推理'
    },
    'content/ml-systems/pytorch/operations-and-components.md': {
        'title': '常见计算与组件',
        'category': 'pytorch',
        'tags': ['Linear', 'softmax', 'LayerNorm', 'attention'],
        'description': 'Linear · softmax · LayerNorm · RMSNorm · SwiGLU · attention'
    },

    # CUDA
    'content/ml-systems/cuda/execution-hierarchy.md': {
        'title': '执行层级与硬件映射',
        'category': 'cuda',
        'tags': ['thread', 'warp', 'block', 'SM', 'GPC'],
        'description': 'thread · warp · block · cluster · grid · SM / GPC · 同步范围'
    },
    'content/ml-systems/cuda/gemm.md': {
        'title': 'GEMM 计算与优化',
        'category': 'cuda',
        'tags': ['分块', 'Tensor Core', 'TMA', '流水线'],
        'description': '分块复用 · K 循环 · Tensor Core · TMA · 流水线 · Split-K · scale 布局'
    },
    'content/ml-systems/cuda/gemm-performance-analysis.md': {
        'title': 'GEMM 性能分析方法',
        'category': 'cuda',
        'tags': ['并行度', '算术强度', 'roofline'],
        'description': '宏观约束与微观执行 · 并行度 · 分层算术强度 · GEMV / GEMM · TP'
    },
    'content/ml-systems/cuda/compilation-and-instructions.md': {
        'title': 'PTX、SASS 与编译流程',
        'category': 'cuda',
        'tags': ['PTX', 'SASS', 'ptxas', 'cubin'],
        'description': 'CUDA C++ · Triton · CuTe DSL · ptxas · cubin · AOT / JIT'
    },
    'content/ml-systems/cuda/flash-attention.md': {
        'title': 'FlashAttention 1–4',
        'category': 'cuda',
        'tags': ['FlashAttention', 'online softmax', 'TMA'],
        'description': '分块省显存 · warp 分工 · 异步流水线 · A100 / H100 / B200'
    },

    # GPU
    'content/ml-systems/gpu/driver-runtime-framework.md': {
        'title': '驱动、运行时与框架',
        'category': 'gpu',
        'tags': ['驱动', 'libcuda', 'libcudart', 'PyTorch'],
        'description': 'H100 · libcuda / libcudart · 内核模块 · PyTorch · 容器与版本'
    },
    'content/ml-systems/gpu/software-stack.md': {
        'title': 'GPU 软件栈与工具',
        'category': 'gpu',
        'tags': ['cuBLAS', 'cuDNN', 'CUTLASS', 'Triton'],
        'description': 'cuBLAS · cuDNN · CUTLASS · CuTe DSL · Triton · 编译与运行'
    },
    'content/ml-systems/gpu/graphs-and-compilation.md': {
        'title': '计算图、编译与 CUDA Graph',
        'category': 'gpu',
        'tags': ['torch.compile', 'CUDA Graph', 'capture', 'replay'],
        'description': 'torch.compile · capture / replay · 虚拟地址 · 图内分配 · 共享 pool'
    },
    'content/ml-systems/gpu/precision-and-quantization.md': {
        'title': '浮点格式、混合精度与量化',
        'category': 'gpu',
        'tags': ['FP16', 'BF16', 'FP8', '量化'],
        'description': 'FP16 / BF16 · TF32 · FP8 · E8M0 · MXFP8 / NVFP4 · scale'
    },
    'content/ml-systems/gpu/communication-and-allreduce.md': {
        'title': 'MPI 与 AllReduce 算法',
        'category': 'gpu',
        'tags': ['MPI', 'AllReduce', 'Ring', 'Tree'],
        'description': 'Ring / Tree · Recursive Doubling · One-shot / Two-shot · 通信量与依赖'
    },

    # Attention
    'content/ml-systems/attention/transformer-runtime.md': {
        'title': 'Transformer 运行全过程',
        'category': 'attention',
        'tags': ['Transformer', 'Q/K/V', 'prefill', 'decode', 'KV cache'],
        'description': '文本到 token · Q/K/V · 多头与 GQA · FFN · prefill / decode · KV cache · 训练'
    },
    'content/ml-systems/attention/flash-linear-attention.md': {
        'title': 'Flash Linear Attention',
        'category': 'attention',
        'tags': ['线性注意力', 'GLA', 'DeltaNet'],
        'description': '状态递推 · chunkwise · GLA · DeltaNet / Gated DeltaNet · FLA 接口'
    },
    'content/ml-systems/attention/model-technology-map.md': {
        'title': '模型技术总表',
        'category': 'attention',
        'tags': ['DeepSeek', 'Kimi', 'Qwen', 'MLA', 'MoE'],
        'description': '按模型查 attention · FFN / MoE · residual · 条件记忆 · MTP · 训练技术'
    },
    'content/ml-systems/attention/model-architectures.md': {
        'title': 'DeepSeek、Kimi 与 Qwen 架构',
        'category': 'attention',
        'tags': ['MLA', 'GQA', 'DSA', 'CSA', 'MoE'],
        'description': 'MLA / GQA · DSA / CSA / QSA · KDA / GDN · MoE · 残差与缓存'
    },
    'content/ml-systems/attention/residual-experts-memory.md': {
        'title': '残差、LatentMoE 与条件记忆',
        'category': 'attention',
        'tags': ['残差', 'LatentMoE', 'Engram'],
        'description': 'Gated Residual · Block AttnRes · Stable LatentMoE · Engram · n-gram embedding'
    },
    'content/ml-systems/attention/qwen3-h100-tp-case.md': {
        'title': '案例：Qwen3 / H100 的 TP 性能',
        'category': 'attention',
        'tags': ['Qwen3', 'H100', 'TP', 'roofline'],
        'description': 'GEMV / GEMM · 延迟与吞吐 · roofline · CTA / warp / register · 通信成本'
    },
    'content/ml-systems/attention/moe-ep-tp.md': {
        'title': 'MoE FFN：EP 与 TP',
        'category': 'attention',
        'tags': ['MoE', 'EP', 'TP', 'AllReduce'],
        'description': 'expert GEMM · A2A / AllReduce · token 布局 · 负载均衡 · 显存与混合并行'
    },

    # DeepSeek
    'content/ml-systems/deepseek/evolution.md': {
        'title': 'V2–V4 架构演进',
        'category': 'deepseek',
        'tags': ['DeepSeek', 'MLA', 'DSA', 'CSA', 'MoE'],
        'description': 'MLA · DSA · CSA / HCA · MoE · mHC · Muon · 缓存与训练'
    },
    'content/ml-systems/deepseek/forward-shapes.md': {
        'title': '一次前向与张量形状',
        'category': 'deepseek',
        'tags': ['DeepSeek', '张量形状', 'prefill', 'decode'],
        'description': 'V3 / V3.2 / V4-Pro · 权重 · decode / prefill · 压缩状态 · logits'
    },
}


def add_frontmatter(file_path: Path, metadata: dict):
    """为文件添加 front matter"""
    content = file_path.read_text()

    # 检查是否已有 front matter
    if content.startswith('---\n'):
        print(f"跳过 {file_path.relative_to(ROOT)}（已有 front matter）")
        return

    # 构建 front matter
    frontmatter = ['---']
    frontmatter.append(f"title: {metadata['title']}")
    frontmatter.append(f"category: {metadata['category']}")
    frontmatter.append(f"tags: [{', '.join(metadata['tags'])}]")
    frontmatter.append(f"description: {metadata['description']}")
    frontmatter.append(f"updated: {datetime.now().strftime('%Y-%m-%d')}")
    frontmatter.append('---')
    frontmatter.append('')

    # 写入文件
    new_content = '\n'.join(frontmatter) + content
    file_path.write_text(new_content)
    print(f"✓ {file_path.relative_to(ROOT)}")


def main():
    """为所有文档添加 front matter"""
    for rel_path, metadata in METADATA.items():
        file_path = ROOT / rel_path
        if file_path.exists():
            add_frontmatter(file_path, metadata)
        else:
            print(f"⚠ 文件不存在：{rel_path}")

    print(f"\n完成！共处理 {len(METADATA)} 个文件")


if __name__ == '__main__':
    main()
