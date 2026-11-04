---
title: 关联容器
category: cpp-stl
tags: [map, set, unordered_map]
description: map / set · unordered 系列
updated: 2026-11-04
---
# STL 03：树容器与哈希容器

`map/set` 与 `unordered_map/unordered_set` 都按 key 组织元素，但一个维护有序树，一个维护哈希桶。选择它们时必须同时考虑顺序需求、复杂度保证、访问副作用和引用稳定性。

## 1. 两个家族的底层模型

| 家族 | 典型实现 | 顺序 | 查找复杂度 |
| --- | --- | --- | --- |
| `map/set` | 平衡二叉搜索树，常见为红黑树 | 按比较器排序 | 最坏 O(log n) |
| `unordered_*` | bucket 数组与哈希组织 | 无稳定遍历顺序 | 平均 O(1)，最坏 O(n) |

标准规定可观察行为和复杂度，不强制某一种内部实现。代码可以依赖“map 有序、操作 O(log n)”，不能依赖具体树颜色或 unordered 的桶链表布局。

## 2. map 元素的真实类型

```cpp
std::map<std::string, int> counts;
// 元素效果上是：pair<const std::string, int>
```

key 是 `const`，因为原地修改 key 会破坏树顺序或哈希桶位置；mapped value 可以修改。

```cpp
for (auto& [key, value] : counts) {
    // key = "new"; // 编译错误
    ++value;         // 合法
}
```

## 3. `operator[]` 是“查找或插入”

```cpp
std::map<std::string, int> counts;
int value = counts["missing"]; // 插入 {"missing", 0}
```

`operator[]` 找不到 key 时会默认构造 mapped value，所以只能用于 map，不能用于 set，也不能用于 const map。

| 需求 | C++17 写法 | 副作用 |
| --- | --- | --- |
| 只判断存在 | `find(key) != end()` 或 `count(key)` | 不插入 |
| 读取且不存在时报错 | `at(key)` | 抛 `out_of_range` |
| 计数/不存在时创建默认值 | `operator[]` | 可能插入 |
| 构造并得知是否新插入 | `emplace/try_emplace` | 按需插入 |

```cpp
auto [it, inserted] = counts.try_emplace("alice", 0);
++it->second;
```

## 4. 访问得到的是节点中 value 的引用

```cpp
int& count = counts["alice"];
count += 1; // 直接修改 map 中的值

auto it = counts.find("alice");
if (it != counts.end()) {
    it->second += 1;
}
```

同样要区分 `auto value = it->second` 的复制与 `auto& value = it->second` 的引用。

## 5. map/set 的引用为什么稳定

树容器通常每个元素一个独立节点。插入新节点不需要搬动旧节点，因此插入不会使已有元素的引用、指针和迭代器失效。删除只让被删除元素的观察者失效。

```cpp
auto it = counts.find("alice");
counts["bob"] = 3; // it 仍有效
counts.erase("alice"); // it 立即失效
```

这种节点稳定性适合保存长期引用，但节点分配和指针跳转会增加内存与缓存成本。

## 6. 比较器必须形成 strict weak ordering

```cpp
struct ByLengthThenText {
    bool operator()(const std::string& a,
                    const std::string& b) const {
        if (a.size() != b.size()) return a.size() < b.size();
        return a < b;
    }
};

std::set<std::string, ByLengthThenText> words;
```

比较器表达“a 是否严格排在 b 前面”，相等时必须返回 false。不要写 `<=`；比较规则不一致会破坏容器不变量，行为可能未定义。

对 set/map 而言，若 `!comp(a,b) && !comp(b,a)`，两个 key 就被视为等价，即使 `operator==` 认为它们不同。

## 7. unordered 容器的 bucket、负载因子与 rehash

```cpp
std::unordered_map<std::string, int> counts;
counts.reserve(1000);       // 为预计元素数准备桶
counts.max_load_factor(0.8f);
```

负载因子大致是 `size / bucket_count`。元素增多超过阈值时会 rehash：重新分配 bucket 并把元素归入新桶。

* rehash 会使所有迭代器失效。
* 对元素的引用和指针通常仍有效，因为节点本身不必搬迁；被 erase 的元素除外。
* 遍历顺序可能在插入或 rehash 后改变，绝不能依赖。

## 8. 哈希函数与相等判断必须一致

若两个 key 按相等谓词相等，它们必须产生相同哈希值。否则查找可能进入错误桶。

```cpp
struct Point {
    int x;
    int y;
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

struct PointHash {
    std::size_t operator()(const Point& p) const {
        std::size_t h1 = std::hash<int>{}(p.x);
        std::size_t h2 = std::hash<int>{}(p.y);
        return h1 ^ (h2 << 1);
    }
};

std::unordered_set<Point, PointHash> points;
```

## 9. 复制、移动与传参

* 复制 map/unordered_map 会复制全部 key-value 元素，通常 O(n)。
* 移动通常接管内部节点和桶，但移动后源容器只保证有效、值未指定。
* 只读查询函数应接受 `const Map&`，不要无意中按值复制整张表。
* 返回新构建的 map 可以按值返回，依赖 copy elision/move。

```cpp
int lookup(const std::unordered_map<std::string, int>& table,
           const std::string& key) {
    auto it = table.find(key);
    if (it == table.end()) return -1;
    return it->second;
}
```

## 10. 如何选择

| 需求 | 选择 |
| --- | --- |
| 需要有序遍历、lower_bound 或稳定最坏复杂度 | `map/set` |
| 只需按 key 查找，追求平均常数时间 | `unordered_map/set` |
| key 范围小且连续 | 优先考虑 vector/array 直接索引 |
| 需要保存指向元素的长期引用 | 两家族节点引用通常都较稳定，但必须避开 erase |
| 输入可能恶意构造哈希冲突 | 考虑 map 或安全自定义哈希 |

**掌握标准：**能够解释 `mp[key]` 为什么会插入、map 的 key 为什么是 const、unordered rehash 到底使什么失效，并根据是否需要顺序和复杂度保证选择容器。
