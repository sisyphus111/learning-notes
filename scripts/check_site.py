#!/usr/bin/env python3
"""Validate every generated local link, fragment, image and search destination."""
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
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            if attrs['id'] in self.ids:
                self.duplicates.append(attrs['id'])
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])


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
        for link in page.links:
            check(filename, link)
    index = json.loads((ROOT / 'assets/search-index.json').read_text())
    for item in index:
        check('index.html', item['url'])
    for error in errors:
        print(error)
    print(f'{len(pages)} pages and {count} local links/search destinations checked; {len(errors)} errors.')
    raise SystemExit(bool(errors))

if __name__ == '__main__':
    main()
