# STL 04：适配器、迭代器与算法

`stack`、`queue` 和 `priority_queue` 不是新的底层存储结构，而是限制已有容器接口的适配器。算法又不拥有容器，而是通过迭代器操作半开区间。理解这两层关系，才能判断元素是否复制、算法能否使用以及修改后迭代器是否有效。

## 1. 容器适配器是“受限视图”

| 适配器 | 默认底层容器 | 暴露的结构 |
| --- | --- | --- |
| `stack<T>` | `deque<T>` | 后进先出，只看 top |
| `queue<T>` | `deque<T>` | 先进先出，看 front/back |
| `priority_queue<T>` | `vector<T>` | 堆，只保证 top 是最高优先级 |

它们故意不提供迭代器和任意位置访问，因为接口要维持抽象。需要遍历时，通常应直接选择底层容器，而不是试图绕过适配器。

## 2. push 是复制还是移动

```cpp
std::queue<std::string> q;
std::string value = "task";

q.push(value);            // 复制
q.push(std::move(value)); // 移动
q.emplace("task");       // 在底层容器中直接构造
```

`front()`、`back()` 和 `top()` 返回引用；`pop()` 只删除，不返回元素。因此需要先移动/复制，再 pop：

```cpp
std::string next = std::move(q.front());
q.pop();
```

**先取值再删除。**`q.pop()` 后原元素已经析构，之前指向它的引用立即失效。

## 3. priority_queue 的比较器为什么“看起来反着”

```cpp
std::priority_queue<int> max_heap;
std::priority_queue<int,
                    std::vector<int>,
                    std::greater<int>> min_heap;
```

比较器表示“第一个元素的优先级是否低于第二个”。默认 `less` 让最大元素留在 top；`greater` 让最小元素留在 top。

```cpp
struct LaterDeadlineFirst {
    bool operator()(const Task& a, const Task& b) const {
        return a.deadline > b.deadline; // deadline 小的优先
    }
};
```

底层 vector 只维护堆序，不维护全局排序。反复 pop 才能按优先级取出所有元素。

## 4. 迭代器是什么

迭代器是“指向容器中某个位置的抽象对象”。它通常支持解引用 `*it` 获得元素引用，以及递增移动到下一个位置。

```cpp
auto it = values.begin();
int& element = *it;
++it;
```

`begin()` 指向第一个元素，`end()` 指向最后一个元素之后的位置，不能解引用。STL 统一使用半开区间 `[first,last)`，这样空区间可以写成 `first == last`，长度也容易组合。

## 5. 迭代器能力由底层结构决定

| 类别 | 典型容器 | 能力 |
| --- | --- | --- |
| 随机访问 | vector、deque、array | `it+n`、差值、O(1) 跳转 |
| 双向 | list、map、set | `++it` 和 `--it` |
| 前向 | forward_list、unordered 容器 | 只能向前多次遍历 |

`std::sort` 需要随机访问迭代器，因此能排序 vector，不能直接排序 list 或 map。list 提供自己的 `list::sort`；map 的顺序由 key 比较器决定，不能原地重排。

## 6. 算法接收区间，不接收容器

```cpp
std::sort(values.begin(), values.end());
auto it = std::find(values.begin(), values.end(), target);
int sum = std::accumulate(values.begin(), values.end(), 0);
```

算法只看到迭代器，不知道原对象是不是 vector。这样同一个算法能复用于不同容器和子区间：

```cpp
std::sort(values.begin() + left, values.begin() + right); // [left,right)
```

## 7. 算法是否修改元素

| 类型 | 例子 | 行为 |
| --- | --- | --- |
| 只读 | `find`、`count`、`accumulate` | 读取元素，不改容器结构 |
| 重排/修改值 | `sort`、`reverse`、`fill` | 通过迭代器修改现有元素 |
| 逻辑移除 | `remove`、`unique` | 移动保留元素，返回新逻辑结尾，不改变 size |

```cpp
auto new_end = std::remove(values.begin(), values.end(), 0);
values.erase(new_end, values.end()); // 真正缩短 vector
```

`remove` 不能调用容器的 erase，因为算法只有迭代器，没有容器对象。这就是 erase-remove 需要两步的根本原因。

## 8. lambda 捕获也有值和引用

```cpp
int threshold = 10;

auto by_value = [threshold](int x) { return x > threshold; };
auto by_reference = [&threshold](int x) { return x > threshold; };
```

* `[threshold]` 在创建 lambda 时复制值，之后外部变量变化不会影响副本。
* `[&threshold]` 保存引用，调用时读取当前值，但 lambda 不能活得比外部变量更久。
* `[=]` 和 `[&]` 捕获范围太宽时不易看出生命周期，项目代码优先显式列出。

## 9. 比较器的参数要避免复制

```cpp
std::sort(records.begin(), records.end(),
          [](const Record& a, const Record& b) {
              if (a.score != b.score) return a.score > b.score;
              return a.name < b.name;
          });
```

排序会调用比较器 O(n log n) 次。若 Record 较大，按值参数会产生大量复制。比较器必须保持 strict weak ordering，不能写 `a.score >= b.score`。

## 10. 范围 for 背后的引用行为

```cpp
for (auto x : values) {}        // 每轮复制元素
for (auto& x : values) {}       // 每轮引用元素，可修改
for (const auto& x : values) {} // 每轮只读引用
```

范围 for 本质上先保存范围对象，再使用 begin/end 迭代。循环中让容器扩容或删除当前元素，可能使内部迭代器失效，因此不能随意修改容器结构。

## 11. 二分算法要求“区间已按同一规则排序”

```cpp
std::sort(values.begin(), values.end());
auto first = std::lower_bound(values.begin(), values.end(), x);
auto last = std::upper_bound(values.begin(), values.end(), x);
```

如果区间没有排序，或 sort 与 lower_bound 使用了不同比较规则，结果不是“慢一点”，而是没有正确性保证。对 map 使用成员函数 `map.lower_bound(key)`，它能利用树结构 O(log n) 查找；通用 `std::lower_bound` 在双向迭代器上可能需要线性移动。

## 12. 修改结构前先检查失效规则

```cpp
for (auto it = values.begin(); it != values.end(); ) {
    if (bad(*it)) it = values.erase(it);
    else ++it;
}
```

不要写 `values.erase(it); ++it;`，因为 erase 后 it 已失效。不同容器的 erase 返回值都应优先利用；对 unordered 容器还要考虑插入是否触发 rehash。

## 13. 选择结构的最终问题

1. 我需要什么访问模式：下标、两端、按 key、最值还是稳定节点？
2. 这个操作返回值还是引用？我是否无意中复制了大对象？
3. 下一次插入、删除或扩容后，保存的迭代器/指针还能用吗？
4. 算法要求哪类迭代器，输入区间满足排序等前置条件吗？
5. 比较器、哈希与相等规则是否满足容器不变量？

**掌握标准：**知道适配器的底层容器；能判断算法是否修改元素或容器大小；能区分范围 for 和 lambda 中的复制/引用；能在修改结构后正确更新迭代器。
