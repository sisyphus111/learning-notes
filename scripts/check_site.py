#!/usr/bin/env python3
"""Validate every generated local link, fragment, image and search destination."""
import ast
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import json

ROOT = Path(__file__).resolve().parents[1]

class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.duplicates = []
        self.code_blocks = []
        self.current_code = None
        self.controls = []
        self.stack = []
        self.layout_children = []
        self.page_intro = False
        self.toc_label = False
        self.toc_link_levels = []
        self.article_h1_count = 0
        self.sidebar_links = set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = set(attrs.get('class', '').split())
        if self.stack and 'reading-layout' in self.stack[-1][1]:
            self.layout_children.append('toc' if 'toc' in classes else tag)
        if 'page-intro' in classes and any(item[0] == 'main' for item in self.stack):
            self.page_intro = True
        if any('toc' in item[1] for item in self.stack):
            if tag == 'nav' and attrs.get('aria-label') == '本页目录':
                self.toc_label = True
            if tag == 'a':
                self.toc_link_levels.append(bool(classes & {'level-1', 'level-2'}))
        if tag == 'h1' and any(item[0] == 'article' for item in self.stack):
            self.article_h1_count += 1
        if tag == 'a' and any(item[2] == 'sidebar' for item in self.stack):
            self.sidebar_links.add(attrs.get('href', ''))
        if tag not in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}:
            self.stack.append((tag, classes, attrs.get('id')))
        if tag == "pre": self.current_code = []
        self.controls.extend(attrs.get("aria-controls", "").split())
        self.controls.extend(attrs.get("aria-labelledby", "").split())
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                self.duplicates.append(attrs['id'])
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])

    def handle_data(self, data):
        if self.current_code is not None: self.current_code.append(data)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if tag == 'pre' and self.current_code is not None:
            self.code_blocks.append(''.join(self.current_code))
            self.current_code = None


def main():
    pages = {path.name: Page(path.read_text()) for path in ROOT.glob('*.html')}
    errors = []
    count = 0
    def check(source, link):
        nonlocal count
        target = urlsplit(link)
        if target.scheme or link.startswith('//'):
            return
        destination = unquote(target.path) or source
        path = ROOT / destination
        if not path.is_file():
            errors.append(f'{source}: missing file {destination}')
        elif target.fragment and destination.endswith('.html'):
            fragment = unquote(target.fragment)
            if destination not in pages or fragment not in pages[destination].ids:
                errors.append(f'{source}: missing fragment {destination}#{fragment}')
        count += 1
    for filename, page in pages.items():
        errors.extend(f'{filename}: duplicate id {name}' for name in page.duplicates)
        errors.extend(f'{filename}: missing control target {name}' for name in page.controls if name not in page.ids)
        # The desktop grid assigns its wide column to the first child.
        expected_layout = ['main'] if filename == 'index.html' else ['main', 'toc']
        if page.layout_children != expected_layout:
            errors.append(f'{filename}: reading layout must contain {expected_layout} in that order')
        if not page.page_intro:
            errors.append(f'{filename}: missing styled page introduction')
        if filename != 'index.html' and (not page.toc_label or not page.toc_link_levels or not all(page.toc_link_levels)):
            errors.append(f'{filename}: missing accessible, hierarchical table of contents')
        for link in page.links:
            check(filename, link)
    index = json.loads((ROOT / 'assets/search-index.json').read_text())
    for item in index:
        check('index.html', item['url'])
    # Read the current manifest without importing or running the site builder.
    manifest = []
    for node in ast.parse((ROOT / 'scripts/build_v2.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'PAGES_CONFIG' for target in node.targets):
            for category, config in ast.literal_eval(node.value).items():
                manifest.extend((Path('content') / category / source, output, category) for source, output in config['files'])
    if not manifest:
        errors.append('build_v2.py: missing page manifest')
    for source, output, category in manifest:
        if output not in pages:
            errors.append(f'{output}: missing generated page')
            continue
        if category != 'algorithms' and pages[output].article_h1_count:
            errors.append(f'{output}: reference title is repeated in the article')
        expected = ['\n'.join(line.rstrip() for line in match[1].strip().splitlines()) for match in re.finditer(r'^```\w*\n(.*?)^```', (ROOT / source).read_text(), re.M | re.S)]
        actual = ['\n'.join(line.rstrip() for line in code.strip().splitlines()) for code in pages[output].code_blocks]
        if actual != expected: errors.append(f'{output}: rendered code differs from Markdown source')
    if 'index.html' not in pages:
        errors.append('index.html: missing home page')
    else:
        errors.extend(f'index.html: missing sidebar link {output}' for _, output, _ in manifest if output not in pages['index.html'].sidebar_links)
    for error in errors:
        print(error)
    print(f'{len(pages)} pages and {count} local links/search destinations checked; {len(errors)} errors.')
    raise SystemExit(bool(errors))

if __name__ == '__main__':
    main()
