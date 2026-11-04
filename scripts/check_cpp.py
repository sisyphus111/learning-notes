#!/usr/bin/env python3
"""Compile complete note examples independently with the LeetCode type prelude."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HEADERS = """
#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
using namespace std;
struct ListNode {int val; ListNode* next; ListNode(int v=0,ListNode* p=nullptr):val(v),next(p){}};
struct TreeNode {int val; TreeNode *left,*right; TreeNode(int v=0,TreeNode* l=nullptr,TreeNode* r=nullptr):val(v),left(l),right(r){}};
struct Node {int val; Node *next=nullptr,*random=nullptr; explicit Node(int v):val(v){}};
"""

def examples():
    # 搜索 content/ 目录下的所有 Markdown 文件
    for path in sorted((ROOT / 'content').glob('**/*.md')):
        # STL 和 Python 参考文档包含独立片段，跳过
        if 'cpp-stl' in str(path) or 'python' in str(path):
            continue
        text = path.read_text()
        for index, match in enumerate(re.finditer(r'^```cpp\n(.*?)^```', text, re.M | re.S)):
            code = match[1]
            if re.search(r'\bclass\s+\w+|^int (?:zero_one|unbounded)_knapsack\(', code):
                yield path, index, code

def compile_one(example, compiler):
    path, index, code = example
    result = subprocess.run([compiler, '-std=c++17', '-fsyntax-only', '-Werror=return-type', '-x', 'c++', '-'],
                            input=HEADERS + code, capture_output=True, text=True)
    return path, index, result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler', default='clang++')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    items = list(examples())
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda item: compile_one(item, args.compiler), items))
    failed = 0
    for path, index, result in results:
        if result.returncode:
            failed += 1
            print(f'{path.relative_to(ROOT)} block {index}:\n{result.stderr}')
    print(f'{len(items) - failed}/{len(items)} complete C++17 examples compile; fragments are excluded.')
    raise SystemExit(bool(failed))

if __name__ == '__main__':
    main()
