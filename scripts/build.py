#!/usr/bin/env python3
"""Render the Markdown notes to static HTML and a section-level search index."""
from pathlib import Path
from html import escape, unescape
from hashlib import sha256
import json
import re
import unicodedata
from urllib.parse import quote, unquote, urlsplit
import xml.etree.ElementTree as ET

import markdown
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor
from markdown.util import AtomicString
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import CppLexer, PythonLexer, TextLexer

ROOT = Path(__file__).resolve().parents[1]
STYLE_VERSION = sha256((ROOT / 'assets/style.css').read_bytes()).hexdigest()[:10]
SCRIPT_VERSION = sha256((ROOT / 'assets/site.js').read_bytes()).hexdigest()[:10]
CHAPTERS = [
    ('数组/fuck数组.md', 'arrays.html', '数组', '哈希表 · 双指针 · 滑窗 · 前缀和 · 二分'),
    ('链表/fuck链表.md', 'linked-lists.html', '链表', '基础操作 · 快慢指针 · 递归'),
    ('二叉树/fuck二叉树.md', 'trees.html', '二叉树', '递归 · 搜索树 · 层序遍历'),
    ('图/fuck图.md', 'graphs.html', '图', '搜索 · 拓扑排序 · 最小生成树 · 最短路'),
    ('动态规划/fuck动态规划.md', 'dynamic-programming.html', '动态规划', '背包 · 线性 DP · 区间 DP · 树形 DP'),
    ('回溯/fuck回溯.md', 'backtracking.html', '回溯', '组合 · 子集 · 排列 · 棋盘'),
    ('贪心/fuck贪心.md', 'greedy.html', '贪心', '跳跃游戏 · 区间划分'),
    ('位运算/fuck位运算.md', 'bitwise.html', '位运算', '异或 · 公共前缀 · lowbit'),
    ('数据结构/fuck数据结构.md', 'data-structures.html', '数据结构', '堆 · 单调结构 · Trie · 树状数组 · 线段树'),
    ('设计/fuck设计.md', 'design.html', '设计', '最小栈 · LRU 缓存'),
]
REFERENCES = [
    ('STL/cpp-stl.md', 'cpp-stl.html', 'STL 速查', '容器、算法与常用写法'),
    ('STL/stl-value-semantics.md', 'stl-value-semantics.html', '对象语义', '构造 · 复制与移动 · 引用 · 传参'),
    ('STL/stl-sequence-containers.md', 'stl-sequence-containers.html', '序列容器', 'vector · string · deque · list'),
    ('STL/stl-associative-containers.md', 'stl-associative-containers.html', '关联容器', 'map / set · unordered 系列'),
    ('STL/stl-adapters-iterators.md', 'stl-adapters-iterators.html', '适配器与算法', '栈 · 队列 · 堆 · 迭代器 · 标准算法'),
]
PYTHON_REFERENCES = [
    ('Python/data-structures.md', 'python-data-structures.html', 'Python 数据结构速查',
     '语法 · 返回值 · 原地修改 · 常见错误'),
    ('Python/math-operations.md', 'python-math.html', '数学运算速查',
     '整除与取余 · 乘方 · 取整 · math · 浮点误差'),
    ('Python/iterables-and-functions.md', 'python-iterables-and-functions.html',
     '可迭代对象与常用函数', '生成器表达式 · all / any · sum · min / max · map / filter'),
    ('Python/classes-and-objects.md', 'python-classes-and-objects.html',
     '类与对象', 'class · self · __init__ · 属性与方法 · 继承 · 特殊方法'),
    ('Python/modules-and-packages.md', 'python-modules-and-packages.html',
     '模块、包与运行方式', 'import · package · python -m · pip / venv · __pycache__'),
    ('Python/native-interop.md', 'python-native-interop.html',
     'Python 与其他语言互操作', 'pybind11 · C++ 扩展 · 类型转换 · GIL · ABI · 多语言边界'),
]
PYTORCH_REFERENCES = [
    ('PyTorch/tensors.md', 'pytorch-tensors.html', 'Tensor 与 GPU 内存',
     'shape · 广播 · 矩阵乘法 · stride · view · CUDA 执行'),
    ('PyTorch/neural-networks.md', 'pytorch-neural-networks.html', '神经网络基础',
     'torch.nn · functional · Module · 参数 · 自动求导 · 训练与推理'),
    ('PyTorch/operations-and-components.md', 'pytorch-components.html', '常见计算与组件',
     'Linear · softmax · LayerNorm · RMSNorm · SwiGLU · attention'),
]
CUDA_REFERENCES = [
    ('CUDA/execution-hierarchy.md', 'cuda-execution-hierarchy.html', '执行层级与硬件映射',
     'thread · warp · block · cluster · grid · SM / GPC · 同步范围'),
    ('CUDA/gemm.md', 'cuda-gemm.html', 'GEMM 计算与优化',
     '分块复用 · K 循环 · Tensor Core · TMA · 流水线 · Split-K · scale 布局'),
    ('CUDA/compilation-and-instructions.md', 'cuda-compilation-and-instructions.html', 'PTX、SASS 与编译流程',
     'CUDA C++ · Triton · CuTe DSL · ptxas · cubin · AOT / JIT'),
]
ATTENTION_REFERENCES = [
    ('Attention/transformer-runtime.md', 'attention-transformer-runtime.html', 'Transformer 运行全过程',
     '文本到 token · Q/K/V · 多头与 GQA · FFN · prefill / decode · KV cache · 训练'),
    ('CUDA/flash-attention.md', 'cuda-flash-attention.html', 'FlashAttention 1–4',
     'online softmax · 工作划分 · Ampere / Hopper / Blackwell · TMA / TMEM'),
    ('Attention/flash-linear-attention.md', 'attention-flash-linear-attention.html', 'Flash Linear Attention',
     '状态递推 · chunkwise · GLA · DeltaNet / Gated DeltaNet · FLA 接口'),
    ('Attention/model-architectures.md', 'attention-model-architectures.html', 'DeepSeek、Kimi 与 Qwen 架构',
     'MLA / GQA · DSA / CSA / QSA · KDA / GDN · MoE · 残差与缓存'),
    ('Attention/residual-experts-memory.md', 'attention-residual-experts-memory.html', '残差、LatentMoE 与条件记忆',
     'Gated Residual · Block AttnRes · Stable LatentMoE · Engram · n-gram embedding'),
    ('Attention/qwen3-h100-tp-case.md', 'attention-qwen3-h100-tp-case.html', '案例：Qwen3 / H100 的 TP 性能',
     'GEMV / GEMM · 延迟与吞吐 · roofline · CTA / warp / register · 通信成本'),
    ('Attention/moe-ep-tp.md', 'attention-moe-ep-tp.html', 'MoE FFN：EP 与 TP',
     'expert GEMM · A2A / AllReduce · token 布局 · 负载均衡 · 显存与混合并行'),
]
DEEPSEEK_REFERENCES = [
    ('DeepSeek/evolution.md', 'deepseek-evolution.html', 'V2–V4 架构演进',
     'MLA · DSA · CSA / HCA · MoE · mHC · Muon · 缓存与训练'),
    ('DeepSeek/forward-shapes.md', 'deepseek-forward-shapes.html', '一次前向与张量形状',
     'V3 / V3.2 / V4-Pro · 权重 · decode / prefill · 压缩状态 · logits'),
]
GPU_REFERENCES = [
    ('GPU/driver-runtime-framework.md', 'gpu-driver-runtime-framework.html', '驱动、运行时与框架',
     'H100 · libcuda / libcudart · 内核模块 · PyTorch · 容器与版本'),
    ('GPU/software-stack.md', 'gpu-software-stack.html', 'GPU 软件栈与工具',
     'cuBLAS · cuDNN · CUTLASS · CuTe DSL · Triton · 编译与运行'),
    ('GPU/graphs-and-compilation.md', 'gpu-graphs-and-compilation.html', '计算图、编译与 CUDA Graph',
     'torch.compile · capture / replay · 虚拟地址 · 图内分配 · 共享 pool'),
    ('GPU/precision-and-quantization.md', 'gpu-precision-and-quantization.html', '浮点格式、混合精度与量化',
     'FP16 / BF16 · TF32 · FP8 · E8M0 · MXFP8 / NVFP4 · scale'),
    ('GPU/communication-and-allreduce.md', 'gpu-communication-and-allreduce.html', 'MPI 与 Ring / Tree AllReduce',
     '通信语义 · 逐轮部分和 · ReduceScatter / AllGather · 双树 · 流量与延迟'),
]
PAGES = CHAPTERS + REFERENCES + PYTHON_REFERENCES + PYTORCH_REFERENCES + CUDA_REFERENCES + ATTENTION_REFERENCES + DEEPSEEK_REFERENCES + GPU_REFERENCES
BY_SOURCE = {str((ROOT / source).resolve()): output for source, output, _, _ in PAGES}


def slug(text, separator='-'):
    text = unicodedata.normalize('NFKC', text).lower()
    return re.sub(r'[^\w\u4e00-\u9fff]+', separator, text).strip(separator) or 'section'


class InlineMath(InlineProcessor):
    def handleMatch(self, match, data):
        element = ET.Element('span', {'class': 'math'})
        element.text = AtomicString('\\(' + match.group(1) + '\\)')
        return element, match.start(0), match.end(0)


class MathExtension(Extension):
    def extendMarkdown(self, md):
        md.inlinePatterns.register(InlineMath(r'(?<!\\)\$([^$\n]+?)\$', md), 'math', 185)


def headings(text):
    """Ignore fence contents so code comments cannot become navigation headings."""
    inside = False
    result = []
    for line in text.splitlines():
        if line.startswith('```'):
            inside = not inside
        if inside:
            continue
        match = re.match(r'^(#{1,3})\s+(.+)', line)
        if match:
            result.append((len(match[1]), match[2].replace('==', '').strip()))
    return result


def render(source):
    text = (ROOT / source).read_text()
    examples = []
    formatter = HtmlFormatter(cssclass="codehilite")
    text_formatter = HtmlFormatter(cssclass="codehilite text-diagram")
    def pair(match):
        index = len(examples)
        html = '<div class="code-example"><div class="code-tabs" role="tablist" aria-label="代码语言">'
        for language, label in [('cpp', 'C++'), ('python', 'Python')]:
            selected = language == 'cpp'
            html += (f'<button type="button" role="tab" id="example-{index}-{language}-tab" '
                     f'aria-controls="example-{index}-{language}" aria-selected="{str(selected).lower()}" '
                     f'tabindex="{0 if selected else -1}">{label}</button>')
        html += '</div>'
        for language, code, lexer in [('cpp', match[1], CppLexer()), ('python', match[2], PythonLexer())]:
            hidden = ' hidden' if language == 'python' else ''
            html += (f'<div role="tabpanel" id="example-{index}-{language}" '
                     f'aria-labelledby="example-{index}-{language}-tab" data-language="{language}" tabindex="0"{hidden}>'
                     + highlight(code, lexer, formatter) + '</div>')
        examples.append(html + '</div>')
        return f'\n\n<div>CODEEXAMPLE{index}</div>\n\n'
    text = re.sub(r'^```cpp\n((?:(?!^```)[\s\S])*)^```\s*^```python\n((?:(?!^```)[\s\S])*)^```', pair, text, flags=re.M | re.S)
    def single(match):
        index = len(examples)
        language, code = match[1], match[2]
        lexer = {'cpp': CppLexer, 'python': PythonLexer}.get(language, TextLexer)()
        examples.append(highlight(code, lexer, text_formatter if language in ('text', 'plaintext') else formatter))
        return f'\n\n<div>CODEEXAMPLE{index}</div>\n\n'
    # Protect all code from prose transformations, including == and dollar signs.
    text = re.sub(r'^```(\w*)\n(.*?)^```', single, text, flags=re.M | re.S)
    # Typora's highlight markers are presentation syntax, not part of headings.
    text = re.sub(r'==([^=\n]+)==', r'\1', text)
    displays = []
    def display(match):
        index = len(displays)
        displays.append('<div class="math display-math">\\[' + escape(match[1]) + '\\]</div>')
        return f'\n\n<div>MATHDISPLAY{index}</div>\n\n'
    text = re.sub(r'\$\$(.*?)\$\$', display, text, flags=re.S)
    md = markdown.Markdown(extensions=['extra', 'toc', 'codehilite', MathExtension()],
                           extension_configs={'toc': {'slugify': slug}, 'codehilite': {'guess_lang': False}},
                           tab_length=2)
    html = md.convert(text)
    # Keep native column sizing inside a separate, keyboard-scrollable viewport.
    html = re.sub(r'(<table>.*?</table>)',
                  r'<div class="table-scroll" role="region" aria-label="表格（可横向滚动）" tabindex="0">\1</div>',
                  html, flags=re.S)
    for i, example in enumerate(examples):
        html = html.replace(f'<div>CODEEXAMPLE{i}</div>', example)
    for i, formula in enumerate(displays):
        html = html.replace(f'<div>MATHDISPLAY{i}</div>', formula)
    tokens = []
    def flatten(items):
        for item in items:
            tokens.append(item)
            flatten(item['children'])
    flatten(md.toc_tokens)
    html = re.sub(r'[ \t]+$', '', html, flags=re.M)
    return html, tokens


RENDERED = {source: render(source) for source, _, _, _ in PAGES}


def resolve_anchor(source, fragment):
    fragment = unquote(fragment).replace('==', '')
    tokens = RENDERED[source][1]
    for item in tokens:
        if fragment == item['id'] or slug(fragment) == slug(item['name']):
            return item['id']
    # Original links sometimes omit the problem number or difficulty suffix.
    needle = re.sub(r'[^\w\u4e00-\u9fff]', '', fragment).lower()
    needle = re.sub(r'(简单|中等|困难)$', '', needle)
    for item in tokens:
        name = re.sub(r'[^\w\u4e00-\u9fff]', '', item['name']).lower()
        name = re.sub(r'(简单|中等|困难)$', '', name)
        if name == needle or re.sub(r'^\d+', '', name) == needle:
            return item['id']
    raise ValueError(f'Broken heading link in {source}: #{fragment}')


def links(html, source):
    def replace(match):
        attribute, target = match[1], unquote(match[2]).replace('\\', '/')
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith('//'):
            return match[0]
        if attribute == 'href' and (parsed.path.endswith('.md') or not parsed.path):
            destination = (ROOT / source).parent / parsed.path if parsed.path else ROOT / source
            resolved = str(destination.resolve())
            if resolved not in BY_SOURCE:
                raise ValueError(f'Unknown Markdown link in {source}: {target}')
            dest_source = str(destination.resolve().relative_to(ROOT))
            url = BY_SOURCE[resolved]
            if parsed.fragment:
                url += '#' + resolve_anchor(dest_source, parsed.fragment)
            return f'{attribute}="{escape(url, quote=True)}"'
        # Keep downloadable local evidence usable from root-level HTML pages.
        if attribute == 'href' and parsed.path.endswith('.json'):
            destination = ((ROOT / source).parent / parsed.path).resolve()
            if not destination.is_file():
                raise ValueError(f'Missing JSON reference in {source}: {target}')
            return f'href="{quote(str(destination.relative_to(ROOT)), safe="/")}"'
        if attribute == 'src':
            destination = ((ROOT / source).parent / parsed.path).resolve()
            if not destination.is_file():
                raise ValueError(f'Missing image in {source}: {target}')
            return f'src="{quote(str(destination.relative_to(ROOT)), safe="/")}" loading="lazy"'
        return match[0]
    return re.sub(r'(href|src)="([^"]+)"', replace, html)


def sidebar(current):
    result = '<a class="brand" href="index.html"><img src="assets/favicon.svg" width="36" height="36" alt=""><span>求知录<small>LEARNING NOTES</small></span></a>'
    for label, pages in [('算法与题目', CHAPTERS), ('STL 参考', REFERENCES), ('Python 参考', PYTHON_REFERENCES), ('PyTorch 教学', PYTORCH_REFERENCES), ('CUDA 教学', CUDA_REFERENCES), ('Attention 与 Transformer', ATTENTION_REFERENCES), ('DeepSeek', DEEPSEEK_REFERENCES), ('GPU 软件栈', GPU_REFERENCES)]:
        result += f'<p class="nav-label">{label}</p><nav aria-label="{label}">'
        for _, output, title, _ in pages:
            active = ' aria-current="page"' if current == output else ''
            result += f'<a href="{output}"{active}>{title}</a>'
        result += '</nav>'
    result += '<a class="source-link" href="https://github.com/sisyphus111/learning-notes">GitHub ↗</a>'
    return result


def shell(title, current, content, toc='', subtitle='', reference_kind=''):
    kind = reference_kind or '求知录'
    page_assets = ''
    if current == 'cuda-gemm.html':
        for name in ('gemm.css', 'gemm.js'):
            version = sha256((ROOT / 'assets' / name).read_bytes()).hexdigest()[:10]
            url = f'assets/{name}?v={version}'
            page_assets += (f'<link rel="stylesheet" href="{url}">\n' if name.endswith('.css')
                            else f'<script defer src="{url}"></script>\n')
    if reference_kind == 'Python 参考':
        source_note = 'Python 语法、项目运行与语言互操作，附官方文档链接；多文件示例按正文说明使用。'
    elif reference_kind == 'PyTorch 教学':
        source_note = '以 CUDA Tensor 为主的教学笔记，附官方文档链接；示例结果为推导预期，实机验证范围见正文。'
    elif reference_kind == 'CUDA 教学':
        source_note = 'CUDA 执行、GEMM、attention 与 kernel 编译笔记，附论文、官方文档和源码；示例用于说明语义，未做 CUDA 编译或实机验证。'
    elif reference_kind == 'Attention 与 Transformer':
        source_note = 'Transformer 运行流程与 attention 教学，附论文、官方文档和源码；形状与代码用于说明机制，未执行模型或 CUDA 验证。'
    elif reference_kind == 'DeepSeek':
        source_note = 'DeepSeek 架构与前向教学，附官方配置快照、技术报告和参考代码；形状与容量为静态推导，未运行模型或 GPU 性能验证。'
    elif reference_kind == 'GPU 软件栈':
        source_note = 'GPU 软件工具与图编译教学，附官方文档；示例与图示用于解释机制，未执行 CUDA 或性能验证。'
    elif reference_kind == 'STL 参考':
        source_note = 'STL 内容选自旧手册，现以 Markdown 维护。'
    else:
        source_note = '基于 Cong Li 的 Fuck-Leetcode 笔记改编，保留思路与原有题目来源；提供 C++17 与 Python 版本。'
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title if title == '求知录' else title + ' · 求知录')}</title><meta name="description" content="{escape(subtitle)}">
<link rel="icon" href="favicon.ico" sizes="16x16 32x32 48x48">
<link rel="icon" type="image/svg+xml" href="assets/favicon.svg" sizes="any">
<link rel="stylesheet" href="assets/style.css?v={STYLE_VERSION}"><link rel="stylesheet" href="assets/code.css">
{page_assets}<script>(()=>{{let saved;try{{saved=localStorage.getItem('notes-theme')}}catch(e){{}}const manual=saved==='light'||saved==='dark';document.documentElement.dataset.theme=manual?saved:(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');document.documentElement.dataset.themeSource=manual?'manual':'system'}})();</script>
<script defer src="assets/site.js?v={SCRIPT_VERSION}"></script>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
</head><body>
<a class="skip-link" href="#main">跳到正文</a>
<aside class="sidebar" id="sidebar">{sidebar(current)}</aside>
<div class="page"><header class="topbar"><button class="menu-toggle" aria-label="展开导航" aria-expanded="false" aria-controls="sidebar">☰</button>
<a href="index.html" class="crumb">求知录</a><div class="toolbar"><button class="search-toggle">搜索题号 / 关键词 <kbd>/</kbd></button><button class="theme-toggle" type="button" aria-label="切换到深色模式" aria-pressed="false" title="切换到深色模式"><svg class="moon" aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20.5 14A8.5 8.5 0 0 1 10 3.5 8.5 8.5 0 1 0 20.5 14Z" /></svg><svg class="sun" aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" /></svg></button></div></header>
<div class="reading-layout"><main id="main"><div class="page-intro"><p class="eyebrow">{kind}</p><h1>{escape(title)}</h1><p class="subtitle">{escape(subtitle)}</p></div>
<article>{content}</article><footer>{source_note}</footer></main>{toc}</div>
</div>
<dialog id="search-dialog"><form method="dialog" class="search-heading"><label for="search-input">搜索笔记与语法参考</label><button aria-label="关闭搜索">关闭</button></form>
<input id="search-input" type="search" placeholder="例如：146、滑动窗口、defaultdict" autocomplete="off"><p id="search-status" role="status">输入题号、题目或容器 / API 名称</p><div id="search-results"></div></dialog>
</body></html>'''


index = []
problem_count = 0
for source, output, title, subtitle in PAGES:
    reference_kind = 'DeepSeek' if source.startswith('DeepSeek/') else 'Attention 与 Transformer' if any(source == entry[0] for entry in ATTENTION_REFERENCES) else 'GPU 软件栈' if source.startswith('GPU/') else 'CUDA 教学' if source.startswith('CUDA/') else 'PyTorch 教学' if source.startswith('PyTorch/') else 'Python 参考' if source.startswith('Python/') else 'STL 参考' if source.startswith('STL/') else ''
    body, tokens = RENDERED[source]
    # Each chapter starts with a page title; original headings remain available.
    body = links(body, source)
    references = '<p class="reference-links">写法参考：<a href="cpp-stl.html">STL 速查</a> · <a href="stl-value-semantics.html">对象与传参</a> · <a href="stl-adapters-iterators.html">迭代器与算法</a> · <a href="python-data-structures.html">Python 数据结构</a></p>'
    if reference_kind:
        # Avoid repeating the reference's h1 below the page title.
        body = re.sub(r'^<h1[^>]*>.*?</h1>\s*', '', body, count=1, flags=re.S)
    else:
        body = references + '<details class="code-conventions"><summary>代码示例约定</summary><p>使用 C++ / Python 标签切换同一解法。C++ 采用 C++17，省略标准库头文件与 <code>using namespace std;</code>；Python 采用 3.10+，包含所需标准库导入。LeetCode 提供 <code>ListNode</code>、<code>TreeNode</code> 等题目类型。输入、输出、框架及局部片段按所在小节的上下文使用。</p></details>' + body
    top = [item for item in tokens if item['level'] <= 2]
    if reference_kind and top and top[0]['level'] == 1:
        top = top[1:]
    toc = '<aside class="toc"><p>本页目录</p><nav aria-label="本页目录">' + ''.join(f'<a class="level-{item["level"]}" href="#{item["id"]}">{escape(item["name"])}</a>' for item in top) + '</nav></aside>'
    (ROOT / output).write_text(shell(title, output, body, toc, subtitle, reference_kind))
    index.append({'title': title, 'chapter': title, 'url': output, 'text': subtitle})
    # Search through headings and their following prose, including STL API names.
    parts = re.split(r'(<h[1-3]\b[^>]*>.*?</h[1-3]>)', body, flags=re.S)
    problem_title = ""
    for i in range(1, len(parts), 2):
        heading = parts[i]
        match = re.search(r'id="([^"]+)"', heading)
        if not match:
            continue
        label = re.sub('<[^>]+>', '', heading)
        text = unescape(re.sub('<[^>]+>', ' ', parts[i+1] if i+1 < len(parts) else ''))
        if re.match(r'^\d+\s*\.', label):
            problem_title = label
        elif re.match(r'<h[12]\b', heading):
            problem_title = ''
        search_title = problem_title + ' · ' + label if problem_title and heading.startswith('<h3') else label
        index.append({'title': search_title, 'chapter': title, 'url': output+'#'+match[1], 'text': text})
        if not reference_kind and re.match(r'^\d+\s*\.', label):
            problem_count += 1

cards = lambda pages: '<div class="cards">' + ''.join(f'<a class="card" href="{output}"><h3>{title}<span>↗</span></h3><p>{desc}</p></a>' for _, output, title, desc in pages) + '</div>'
home = f'''<div class="home-lead"><p>把问题想清楚，把理解记下来。</p><div class="home-meta"><span>{problem_count} 个题目条目</span><span>10 个题型</span><span>C++17 / Python 3.10+</span></div></div>
<section class="home-section"><h2>算法与题目</h2><p class="section-desc">先看「焚诀」抓住联想线索，再读题目、思路和实现。</p>{cards(CHAPTERS)}</section>
<section class="home-section"><h2>STL 参考</h2><p class="section-desc">写题时查 API；遇到复制、引用或迭代器问题，进入对应专题。</p>{cards(REFERENCES)}</section>
<section class="home-section"><h2>Python 参考</h2><p class="section-desc">从容器、常用语法与类和对象，走到模块、包、运行环境和 C++ 等语言的互操作。</p>{cards(PYTHON_REFERENCES)}</section>
<section class="home-section"><h2>PyTorch 教学</h2><p class="section-desc">先理解 Tensor 操作与 GPU 内存，再学习网络层、模型与参数，最后展开常见组件的计算实现。</p>{cards(PYTORCH_REFERENCES)}</section>
<section class="home-section"><h2>CUDA 教学</h2><p class="section-desc">从线程与硬件映射开始，理解 GEMM 的分块与流水线，以及 kernel 怎样编译成 GPU 指令。</p>{cards(CUDA_REFERENCES)}</section>
<section class="home-section"><h2>Attention 与 Transformer</h2><p class="section-desc">先走通 Transformer 从文本到生成的全过程，再理解 attention 的数学、变体与高性能实现。</p>{cards(ATTENTION_REFERENCES)}</section>
<section class="home-section"><h2>DeepSeek</h2><p class="section-desc">从 MLA、DSA 到 CSA/HCA，理解残差与训练算法的变化，再按真实模型尺寸追踪一次完整前向。</p>{cards(DEEPSEEK_REFERENCES)}</section>
<section class="home-section"><h2>GPU 软件栈</h2><p class="section-desc">从 H100 的驱动与运行时开始，区分计算库、kernel 开发工具和编译器，再连接算子图优化与 CUDA Graph 重放。</p>{cards(GPU_REFERENCES)}</section>
<section class="reading-note"><h2>怎么用这份笔记</h2><p>按主题阅读，用搜索定位概念、写法或题目。先理解问题和原理，再对照推导与实现；需要时回到参考文档核实细节。算法部分保留原作者的思路与 TODO 标记。</p></section>'''
(ROOT / 'index.html').write_text(shell('求知录', 'index.html', home, subtitle='记录学习中的问题、推导与实践。'))
(ROOT / 'assets/search-index.json').write_text(json.dumps(index, ensure_ascii=False, separators=(',', ':')))
(ROOT / 'assets/code.css').write_text(HtmlFormatter(style='friendly').get_style_defs('.codehilite'))
print(f'Built {len(PAGES)+1} pages; {problem_count} problem entries; {len(index)} searchable sections.')
