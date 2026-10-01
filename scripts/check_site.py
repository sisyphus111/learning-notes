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
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
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
        for link in page.links:
            check(filename, link)
    index = json.loads((ROOT / 'assets/search-index.json').read_text())
    for item in index:
        check('index.html', item['url'])
    # Read the page manifest without importing build.py (which writes output).
    manifest = []
    for node in ast.parse((ROOT / 'scripts/build.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ('CHAPTERS', 'REFERENCES', 'PYTHON_REFERENCES') for target in node.targets):
            manifest.extend(ast.literal_eval(node.value))
    for source, output, _, _ in manifest:
        expected = ['\n'.join(line.rstrip() for line in match[1].strip().splitlines()) for match in re.finditer(r'^```\w*\n(.*?)^```', (ROOT / source).read_text(), re.M | re.S)]
        actual = ['\n'.join(line.rstrip() for line in code.strip().splitlines()) for code in pages[output].code_blocks]
        if actual != expected: errors.append(f'{output}: rendered code differs from Markdown source')
    for error in errors:
        print(error)
    print(f'{len(pages)} pages and {count} local links/search destinations checked; {len(errors)} errors.')
    raise SystemExit(bool(errors))

if __name__ == '__main__':
    main()
