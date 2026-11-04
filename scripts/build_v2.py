#!/usr/bin/env python3
"""构建静态网站：从 content/ 生成 HTML 到根目录，支持增量构建和元数据"""
from pathlib import Path
from html import escape
from hashlib import sha256
import json
import re
import unicodedata
from urllib.parse import quote, unquote, urlsplit
import xml.etree.ElementTree as ET
from datetime import datetime
import yaml

import markdown
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor
from markdown.util import AtomicString
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import CppLexer, PythonLexer, TextLexer

ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / 'content'
OUTPUT_DIR = ROOT  # 输出到根目录，保持 GitHub Pages 兼容
ASSETS_DIR = ROOT / 'assets'

# 读取资源版本
STYLE_VERSION = sha256((ASSETS_DIR / 'style.css').read_bytes()).hexdigest()[:10]
SCRIPT_VERSION = sha256((ASSETS_DIR / 'site.js').read_bytes()).hexdigest()[:10]

# 页面配置：新结构
PAGES_CONFIG = {
    'algorithms': {
        'label': '算法与题目',
        'files': [
            ('arrays.md', 'arrays.html'),
            ('linked-lists.md', 'linked-lists.html'),
            ('trees.md', 'trees.html'),
            ('graphs.md', 'graphs.html'),
            ('dynamic-programming.md', 'dynamic-programming.html'),
            ('backtracking.md', 'backtracking.html'),
            ('greedy.md', 'greedy.html'),
            ('bitwise.md', 'bitwise.html'),
            ('data-structures.md', 'data-structures.html'),
            ('design.md', 'design.html'),
        ]
    },
    'languages/cpp-stl': {
        'label': 'STL 参考',
        'files': [
            ('quickref.md', 'cpp-stl.html'),
            ('value-semantics.md', 'stl-value-semantics.html'),
            ('sequence-containers.md', 'stl-sequence-containers.html'),
            ('associative-containers.md', 'stl-associative-containers.html'),
            ('adapters-iterators.md', 'stl-adapters-iterators.html'),
        ]
    },
    'languages/python': {
        'label': 'Python 参考',
        'files': [
            ('data-structures.md', 'python-data-structures.html'),
            ('math-operations.md', 'python-math.html'),
            ('iterables-and-functions.md', 'python-iterables-and-functions.html'),
            ('classes-and-objects.md', 'python-classes-and-objects.html'),
            ('modules-and-packages.md', 'python-modules-and-packages.html'),
            ('native-interop.md', 'python-native-interop.html'),
        ]
    },
    'ml-systems/pytorch': {
        'label': 'PyTorch 教学',
        'files': [
            ('tensors.md', 'pytorch-tensors.html'),
            ('neural-networks.md', 'pytorch-neural-networks.html'),
            ('operations-and-components.md', 'pytorch-components.html'),
        ]
    },
    'ml-systems/cuda': {
        'label': 'CUDA 教学',
        'files': [
            ('execution-hierarchy.md', 'cuda-execution-hierarchy.html'),
            ('gemm.md', 'cuda-gemm.html'),
            ('gemm-performance-analysis.md', 'cuda-gemm-performance-analysis.html'),
            ('compilation-and-instructions.md', 'cuda-compilation-and-instructions.html'),
            ('flash-attention.md', 'cuda-flash-attention.html'),
        ]
    },
    'ml-systems/attention': {
        'label': 'Attention 与 Transformer',
        'files': [
            ('transformer-runtime.md', 'attention-transformer-runtime.html'),
            ('flash-linear-attention.md', 'attention-flash-linear-attention.html'),
            ('model-technology-map.md', 'attention-model-technology-map.html'),
            ('model-architectures.md', 'attention-model-architectures.html'),
            ('residual-experts-memory.md', 'attention-residual-experts-memory.html'),
            ('qwen3-h100-tp-case.md', 'attention-qwen3-h100-tp-case.html'),
            ('moe-ep-tp.md', 'attention-moe-ep-tp.html'),
        ]
    },
    'ml-systems/deepseek': {
        'label': 'DeepSeek',
        'files': [
            ('evolution.md', 'deepseek-evolution.html'),
            ('forward-shapes.md', 'deepseek-forward-shapes.html'),
        ]
    },
    'ml-systems/gpu': {
        'label': 'GPU 软件栈',
        'files': [
            ('driver-runtime-framework.md', 'gpu-driver-runtime-framework.html'),
            ('software-stack.md', 'gpu-software-stack.html'),
            ('graphs-and-compilation.md', 'gpu-graphs-and-compilation.html'),
            ('precision-and-quantization.md', 'gpu-precision-and-quantization.html'),
            ('communication-and-allreduce.md', 'gpu-communication-and-allreduce.html'),
        ]
    },
}


def parse_frontmatter(text):
    """解析 YAML front matter"""
    if not text.startswith('---\n'):
        return {}, text

    parts = text.split('---\n', 2)
    if len(parts) < 3:
        return {}, text

    try:
        metadata = yaml.safe_load(parts[1])
        content = parts[2]
        return metadata or {}, content
    except yaml.YAMLError:
        return {}, text


def slug(text, separator='-'):
    """生成 URL slug"""
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


def render_markdown(source_path):
    """渲染 Markdown 为 HTML"""
    text = source_path.read_text()
    metadata, content = parse_frontmatter(text)

    # 处理代码块
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

    content = re.sub(r'^```cpp\n((?:(?!^```)[\s\S])*)^```\s*^```python\n((?:(?!^```)[\s\S])*)^```',
                     pair, content, flags=re.M | re.S)

    def single(match):
        index = len(examples)
        language, code = match[1], match[2]
        lexer = {'cpp': CppLexer, 'python': PythonLexer}.get(language, TextLexer)()
        examples.append(highlight(code, lexer, text_formatter if language in ('text', 'plaintext') else formatter))
        return f'\n\n<div>CODEEXAMPLE{index}</div>\n\n'

    content = re.sub(r'^```(\w*)\n(.*?)^```', single, content, flags=re.M | re.S)
    content = re.sub(r'==([^=\n]+)==', r'\1', content)

    # 处理数学公式
    displays = []
    def display(match):
        index = len(displays)
        displays.append('<div class="math display-math">\\[' + escape(match[1]) + '\\]</div>')
        return f'\n\n<div>MATHDISPLAY{index}</div>\n\n'

    content = re.sub(r'\$\$(.*?)\$\$', display, content, flags=re.S)

    # Markdown 渲染
    md = markdown.Markdown(
        extensions=['extra', 'toc', 'codehilite', MathExtension()],
        extension_configs={'toc': {'slugify': slug}, 'codehilite': {'guess_lang': False}},
        tab_length=2
    )
    html = md.convert(content)

    # 表格滚动
    html = re.sub(r'(<table>.*?</table>)',
                  r'<div class="table-scroll" role="region" aria-label="表格（可横向滚动）" tabindex="0">\1</div>',
                  html, flags=re.S)

    # 恢复代码和公式
    for i, example in enumerate(examples):
        html = html.replace(f'<div>CODEEXAMPLE{i}</div>', example)
    for i, formula in enumerate(displays):
        html = html.replace(f'<div>MATHDISPLAY{i}</div>', formula)

    # 提取 TOC
    tokens = []
    def flatten(items):
        for item in items:
            tokens.append(item)
            flatten(item['children'])
    flatten(md.toc_tokens)

    html = re.sub(r'[ \t]+$', '', html, flags=re.M)

    return html, tokens, metadata


def generate_sidebar(current_page):
    """生成侧边栏导航"""
    result = '<a class="brand" href="index.html"><img src="assets/favicon.svg" width="36" height="36" alt=""><span>求知录<small>LEARNING NOTES</small></span></a>'

    for category, config in PAGES_CONFIG.items():
        result += f'<p class="nav-label">{config["label"]}</p><nav aria-label="{config["label"]}">'
        for source_file, output_file in config['files']:
            source_path = CONTENT_DIR / category / source_file
            if source_path.exists():
                text = source_path.read_text()
                metadata, _ = parse_frontmatter(text)
                title = metadata.get('title', source_file.replace('.md', ''))
                active = ' aria-current="page"' if current_page == output_file else ''
                result += f'<a href="{output_file}"{active}>{title}</a>'
        result += '</nav>'

    result += '<a class="source-link" href="https://github.com/sisyphus111/learning-notes">GitHub ↗</a>'
    return result


def generate_html(title, current, content, toc='', subtitle='', category=''):
    """生成完整的 HTML 页面"""
    # 页面特定资源
    page_assets = ''
    if current == 'cuda-gemm.html':
        for name in ('gemm.css', 'gemm.js'):
            version = sha256((ASSETS_DIR / name).read_bytes()).hexdigest()[:10]
            url = f'assets/{name}?v={version}'
            page_assets += (f'<link rel="stylesheet" href="{url}">\n' if name.endswith('.css')
                            else f'<script defer src="{url}"></script>\n')
    if current == 'cuda-flash-attention.html':
        for name in ('flash-attention.css', 'flash-attention.js'):
            version = sha256((ASSETS_DIR / name).read_bytes()).hexdigest()[:10]
            url = f'assets/{name}?v={version}'
            page_assets += (f'<link rel="stylesheet" href="{url}">\n' if name.endswith('.css')
                            else f'<script defer src="{url}"></script>\n')

    # 来源说明
    source_notes = {
        'Python 参考': 'Python 语法、项目运行与语言互操作，附官方文档链接；多文件示例按正文说明使用。',
        'PyTorch 教学': '以 CUDA Tensor 为主的教学笔记，附官方文档链接；示例结果为推导预期，实机验证范围见正文。',
        'CUDA 教学': 'CUDA 执行、GEMM、attention 与 kernel 编译笔记，附论文、官方文档和源码；示例用于说明语义，未做 CUDA 编译或实机验证。',
        'Attention 与 Transformer': 'Transformer 运行流程与 attention 教学，附论文、官方文档和源码；形状与代码用于说明机制，未执行模型或 CUDA 验证。',
        'DeepSeek': 'DeepSeek 架构与前向教学，附官方配置快照、技术报告和参考代码；形状与容量为静态推导，未运行模型或 GPU 性能验证。',
        'GPU 软件栈': 'GPU 软件工具与图编译教学，附官方文档；示例与图示用于解释机制，未执行 CUDA 或性能验证。',
        'STL 参考': 'STL 内容选自旧手册，现以 Markdown 维护。',
    }
    source_note = source_notes.get(category, '基于 Cong Li 的 Fuck-Leetcode 笔记改编，保留思路与原有题目来源；提供 C++17 与 Python 版本。')

    html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><meta name="description" content="{subtitle or title}">
<link rel="icon" href="favicon.ico" sizes="16x16 32x32 48x48">
<link rel="icon" type="image/svg+xml" href="assets/favicon.svg" sizes="any">
<link rel="stylesheet" href="assets/style.css?v={STYLE_VERSION}"><link rel="stylesheet" href="assets/code.css">
{page_assets}<script>(()=>{{let saved;try{{saved=localStorage.getItem('notes-theme')}}catch(e){{}}const manual=saved==='light'||saved==='dark';document.documentElement.dataset.theme=manual?saved:(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');document.documentElement.dataset.themeSource=manual?'manual':'system'}})();</script>
<script defer src="assets/site.js?v={SCRIPT_VERSION}"></script>
<script>window.MathJax={{tex:{{inlineMath:[['\\\\(','\\\\)']],displayMath:[['\\\\[','\\\\]']]}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"></script>
</head><body>
<a class="skip-link" href="#main">跳到正文</a>
<aside class="sidebar" id="sidebar">{generate_sidebar(current)}</aside>
<div class="page"><header class="topbar"><button class="menu-toggle" aria-label="展开导航" aria-expanded="false" aria-controls="sidebar">☰</button>
<a href="index.html" class="crumb">求知录</a><div class="toolbar"><button class="search-toggle">搜索题号 / 关键词 <kbd>/</kbd></button><button class="theme-toggle" type="button" aria-label="切换到深色模式" aria-pressed="false" title="切换到深色模式"><svg class="moon" aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M20.5 14A8.5 8.5 0 0 1 10 3.5 8.5 8.5 0 1 0 20.5 14Z" /></svg><svg class="sun" aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="12" cy="12" r="4" /><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5" /></svg></button></div></header>
<div class="reading-layout">{toc}<main id="main"><h1>{title}</h1>{f'<p class="subtitle">{subtitle}</p>' if subtitle else ''}<article>{content}</article><footer>{source_note}</footer></main></div>
</div>
<dialog id="search-dialog"><form method="dialog" class="search-heading"><label for="search-input">搜索笔记与语法参考</label><button aria-label="关闭搜索">关闭</button></form>
<input id="search-input" type="search" placeholder="例如：146、滑动窗口、defaultdict" autocomplete="off"><p id="search-status" role="status">输入题号、题目或容器 / API 名称</p><div id="search-results"></div></dialog>
</body></html>'''

    return html


def build_page(category, source_file, output_file):
    """构建单个页面"""
    source_path = CONTENT_DIR / category / source_file
    output_path = OUTPUT_DIR / output_file

    # 检查是否需要重新构建
    if output_path.exists():
        source_mtime = source_path.stat().st_mtime
        output_mtime = output_path.stat().st_mtime
        if source_mtime <= output_mtime:
            return False  # 跳过，未修改

    # 渲染 Markdown
    html_content, toc_tokens, metadata = render_markdown(source_path)

    # 生成 TOC
    toc_html = ''
    if toc_tokens:
        toc_html = '<aside class="toc"><nav>'
        for token in toc_tokens:
            if token['level'] <= 2:
                toc_html += f'<a href="#{token["id"]}">{token["name"]}</a>'
        toc_html += '</nav></aside>'

    # 查找类别标签
    category_label = ''
    for cat, config in PAGES_CONFIG.items():
        if cat == category:
            category_label = config['label']
            break

    # 生成完整 HTML
    title = metadata.get('title', source_file.replace('.md', ''))
    subtitle = metadata.get('description', '')
    html = generate_html(title, output_file, html_content, toc_html, subtitle, category_label)

    # 写入文件
    output_path.write_text(html)
    print(f'✓ {output_file}')
    return True


def build_all():
    """构建所有页面"""
    built_count = 0
    skipped_count = 0

    for category, config in PAGES_CONFIG.items():
        for source_file, output_file in config['files']:
            source_path = CONTENT_DIR / category / source_file
            if source_path.exists():
                if build_page(category, source_file, output_file):
                    built_count += 1
                else:
                    skipped_count += 1
                    print(f'⊙ {output_file} (未修改)')
            else:
                print(f'⚠ 源文件不存在：{source_path}')

    print(f'\n构建完成：{built_count} 个文件已更新，{skipped_count} 个未修改')


if __name__ == '__main__':
    build_all()
