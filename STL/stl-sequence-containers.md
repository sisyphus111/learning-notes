# STL 02：序列容器的底层行为

序列容器都表示“按位置排列的一组元素”，但存储结构不同：连续数组、分段数组和链表会给访问、插入、缓存局部性与引用稳定性带来完全不同的行为。

## 1. 先看底层结构

| 容器 | 典型存储结构 | 核心特征 |
| --- | --- | --- |
| `array<T,N>` | 对象内部固定长度连续数组 | 尺寸编译期固定，不单独扩容 |
| `vector<T>` | 一块动态连续内存 | 随机访问快，扩容会整体搬迁 |
| `string` | 连续字符序列 | 行为类似专用 vector；实现可用 SSO |
| `deque<T>` | 多块固定大小缓冲区加索引 | 两端增长快，但整体不连续 |
| `list<T>` | 双向链表节点 | 已知位置 O(1) 插删，不能随机访问 |

## 2. `vector` 的三个状态量

```cpp
std::vector<int> values;
values.size();     // 已构造的元素数量
values.capacity(); // 当前内存最多容纳多少元素
values.data();     // 连续存储区首地址
```

`size` 以内是已经存在、可以访问的对象；`[size, capacity)` 只是预留原始空间，不能用下标访问。

```cpp
values.reserve(100); // capacity 至少 100，size 不变
values.resize(100);  // size 变为 100，新元素被构造
```

**`reserve` 不是 `resize`。**执行 `reserve(100)` 后写 `values[50]` 仍是越界。

## 3. 为什么 `push_back` 有时让引用失效

容量足够时，vector 在原内存尾部构造新元素；容量不足时，它会申请更大的连续区域，把所有旧元素移动/复制过去，再释放旧区域。

```cpp
std::vector<int> v = {1, 2, 3};
v.reserve(10);
int* p = &v[0];
v.push_back(4); // 未超过 capacity，p 仍有效

v.reserve(100); // 如果发生重新分配，p、引用和全部迭代器失效
```

`push_back` 的单次最坏成本是 O(n)，但连续多次追加的均摊成本是 O(1)。增长倍数由实现决定，不能依赖具体数值。

## 4. vector 插入删除的失效规则

| 操作 | 没有重新分配 | 发生重新分配 |
| --- | --- | --- |
| `push_back/emplace_back` | 原 `end()` 失效；已有元素引用通常保持 | 全部引用、指针、迭代器失效 |
| 中间 `insert` | 插入点及之后失效 | 全部失效 |
| `erase` | 删除点及之后失效 | 不涉及重新分配 |
| `clear` | 所有元素被销毁，指向元素的观察者失效；capacity 通常保留 | - |

```cpp
for (auto it = v.begin(); it != v.end(); ) {
    if (should_remove(*it)) {
        it = v.erase(it); // 必须使用返回的下一个有效迭代器
    } else {
        ++it;
    }
}
```

## 5. `emplace_back` 不等于永远更快

```cpp
std::vector<std::string> words;
words.push_back(std::string("hello"));
words.emplace_back("hello");
```

`emplace_back(args...)` 在容器目标位置直接用参数构造元素，适合“手里只有构造参数”的场景；如果已经有一个对象，`push_back(object)` 或 `push_back(std::move(object))` 更清楚。两者都可能触发扩容。

## 6. `operator[]` 与 `at`

```cpp
int& a = v[3];    // 不检查边界；越界是未定义行为
int& b = v.at(3); // 越界抛 std::out_of_range
```

两者都返回引用，正常访问都是 O(1)。机试热路径通常用 `[]` 并保证不变量；外部输入形成的索引应先校验，或使用 `at` 建立明确错误边界。

## 7. `string` 是拥有字符的值类型

```cpp
std::string a = "hello";
std::string b = a;          // 独立复制
std::string_view view = a;  // 不拥有字符，只观察 a
```

标准保证现代 `string` 字符连续存储。许多实现对短字符串使用 SSO，把字符直接放在 string 对象内部，但 SSO 大小属于实现细节，不能据此写逻辑。

`string_view` 不复制字符，因此更轻量；但原 string 被销毁、重新分配或修改后，view 可能悬空或内容变化。

```cpp
std::string_view broken() {
    std::string local = "temporary";
    return local; // 错误：返回后 view 悬空
}
```

## 7.1 字符串查找与统计

### 查找字符和子串

```cpp
std::string s = "hello world";

// find：从前往后找第一个
size_t pos = s.find('o');           // 4（第一个 'o' 的位置）
size_t pos2 = s.find('o', 5);       // 7（从位置 5 开始找）
size_t not_found = s.find('x');     // string::npos（没找到）

// rfind：从后往前找
size_t last = s.rfind('o');         // 7（最后一个 'o'）

// 查找子串
size_t sub = s.find("world");       // 6

// 判断是否存在某个字符
if (s.find('o') != std::string::npos) {
    // 找到了
}
```

### 查找任意字符

```cpp
std::string s = "hello world";

// 查找第一个元音字母
size_t vowel = s.find_first_of("aeiou");  // 1（'e' 的位置）

// 查找第一个数字
std::string text = "abc123def";
size_t digit = text.find_first_of("0123456789");  // 3

// 查找第一个不是元音的字符
size_t consonant = s.find_first_not_of("aeiou");  // 0（'h'）
```

### 统计字符个数

```cpp
#include <algorithm>

std::string s = "hello world";

// 方法 1：使用 count
int cnt = std::count(s.begin(), s.end(), 'o');  // 2

// 方法 2：手动遍历
int cnt2 = 0;
for (char c : s) {
    if (c == 'o') cnt2++;
}

// 方法 3：查找所有出现位置
int cnt3 = 0;
size_t pos = 0;
while ((pos = s.find('o', pos)) != std::string::npos) {
    cnt3++;
    pos++;
}
```

### find 函数族

| 函数 | 作用 |
| --- | --- |
| `find(c, pos=0)` | 从 pos 开始查找字符或子串 |
| `rfind(c)` | 反向查找 |
| `find_first_of(chars)` | 查找 chars 中任意字符第一次出现 |
| `find_last_of(chars)` | 反向查找 chars 中任意字符 |
| `find_first_not_of(chars)` | 查找第一个不在 chars 中的字符 |

**重要：**所有 find 函数找不到时返回 `std::string::npos`，必须检查返回值！

## 7.2 子串操作（substr）

```cpp
std::string s = "hello world";

// substr(pos) - 从 pos 到末尾
std::string sub1 = s.substr(6);      // "world"

// substr(pos, len) - 从 pos 开始取 len 个字符
std::string sub2 = s.substr(0, 5);   // "hello"
std::string sub3 = s.substr(6, 3);   // "wor"

// 注意：pos 是整数下标（从 0 开始），不是迭代器
std::string sub4 = s.substr(1);      // "ello world" ✓

// 错误示范：
// std::string sub5 = s.substr(s.begin() + 1);  // 编译错误！
```

| 调用方式 | 含义 | 示例（s = "hello"） | 结果 |
| --- | --- | --- | --- |
| `substr()` | 整个字符串 | `s.substr()` | "hello" |
| `substr(pos)` | 从 pos 到末尾 | `s.substr(2)` | "llo" |
| `substr(pos, len)` | 从 pos 开始取 len 个 | `s.substr(1, 3)` | "ell" |

**常见错误：**`substr` 的参数是**整数下标**，不是迭代器！

## 7.3 字符串修改

```cpp
std::string s = "hello";

// 追加
s += " world";                 // "hello world"
s.append("!");                 // "hello world!"
s.push_back('?');              // "hello world!?"

// 插入
s.insert(5, ",");              // 在位置 5 插入 ","

// 删除
s.erase(0, 4);                 // 删除前 4 个字符
s.erase(5);                    // 删除位置 5 及之后的所有

// 替换
s.replace(0, 5, "goodbye");    // 将 [0, 5) 替换为 "goodbye"

// 清空
s.clear();                     // s = ""
```

## 7.4 字符串转换

### 大小写转换

```cpp
#include <algorithm>
#include <cctype>

std::string s = "Hello World";

// 转大写
std::transform(s.begin(), s.end(), s.begin(), ::toupper);  // "HELLO WORLD"

// 转小写
std::transform(s.begin(), s.end(), s.begin(), ::tolower);  // "hello world"
```

### 去除空白（trim）

```cpp
std::string s = "  hello world  ";

// 去除前导空白
size_t start = s.find_first_not_of(" \t\n\r");
if (start != std::string::npos) {
    s = s.substr(start);
}

// 去除尾部空白
size_t end = s.find_last_not_of(" \t\n\r");
if (end != std::string::npos) {
    s = s.substr(0, end + 1);
}
```

### 字符串与数字互转

```cpp
#include <string>

// 数字 -> 字符串
std::string s = std::to_string(42);

// 字符串 -> 数字
int n = std::stoi("123");
long long ll = std::stoll("1234567890");
double d = std::stod("3.14");
```

## 7.5 字符串分割

```cpp
#include <sstream>
#include <vector>

// 按空白分割
std::string s = "hello world test";
std::vector<std::string> words;
std::istringstream iss(s);
std::string word;
while (iss >> word) {
    words.push_back(word);
}

// 按特定字符分割
std::string csv = "apple,banana,cherry";
std::vector<std::string> parts;
std::istringstream iss2(csv);
std::string part;
while (std::getline(iss2, part, ',')) {
    parts.push_back(part);
}
```

## 8. `deque` 为什么能高效操作两端

deque 通常用一组分段缓冲区，而不是一块连续内存。它支持 O(1) 随机访问和两端均摊 O(1) 插入删除，但：

* 不能把 `&dq[0]` 当作整个容器的连续数组。
* 中间插入删除仍然需要移动大量元素。
* 插入操作的迭代器失效规则比 vector 更复杂，不要长期保存 deque 迭代器跨修改使用。

queue 默认以 deque 为底层，就是因为队列需要尾部插入和头部删除。

## 9. `list` 的节点稳定性与代价

list 每个元素位于独立节点，通过前后指针连接。插入和删除不会移动其他节点，因此指向其他元素的引用和迭代器保持有效；只有被删除元素的观察者失效。

```cpp
auto it = items.begin();
items.insert(it, value); // 已知位置时 O(1)
items.erase(it);         // it 失效，其他节点迭代器仍有效
```

代价是每个节点有额外指针、频繁分配、缓存局部性差，而且寻找第 k 个元素仍是 O(n)。“中间插入 O(1)”只在已经拿到目标迭代器时成立；先遍历到位置仍需 O(n)。

## 10. 容器选择不是只看大 O

| 需求 | 优先选择 | 原因 |
| --- | --- | --- |
| 默认动态序列 | `vector` | 连续、缓存友好、API 完整 |
| 编译期固定数量 | `array` | 无动态分配，可整体复制 |
| 频繁两端插删 | `deque` | 无需整体搬迁 |
| 需要稳定节点且已知插删位置 | `list` | 其他节点引用不失效 |
| 字符文本 | `string` | 专用字符操作和连续存储 |

**默认规则：**没有明确证据需要其他结构时先用 vector。链表的理论 O(1) 插入并不自动战胜 vector 的连续内存和缓存局部性。
