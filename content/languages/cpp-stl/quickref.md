---
title: STL 速查
category: cpp-stl
tags: [容器, 算法, API]
description: 容器、算法与常用写法
updated: 2026-11-04
---
# STL 板块：总览与速查

STL 不只是一组可以背诵的函数。容器的底层结构决定了它怎样分配内存、复制元素、返回引用，以及插入删除后哪些指针和迭代器还有效。先建立统一对象模型，再选择具体容器，代码才不容易“能编译但行为不对”。

**固定学习问题：**每种结构都要回答：元素存在哪里？容器是否拥有元素？复制和移动做了什么？函数参数会不会复制？访问返回值还是引用？修改后引用和迭代器是否失效？

## 板块导航

[STL 01对象语义、构造与传参值语义、所有权、copy/move、引用、const 与返回值。](stl-value-semantics.md)
[STL 02序列容器的底层行为array、vector、string、deque、list 的存储与失效规则。](stl-sequence-containers.md)
[STL 03树容器与哈希容器map/set 与 unordered 系列的查找、插入和引用稳定性。](stl-associative-containers.md)
[STL 04适配器、迭代器与算法stack、queue、priority_queue、范围语义与算法修改行为。](stl-adapters-iterators.md)

## lambda 匿名函数

lambda 用来在使用位置直接定义一小段函数逻辑。机考里常把它作为 `sort` 或 `priority_queue` 的自定义比较器，也可以先用 `auto` 保存，再像普通函数一样调用。

### 定义与调用

基本形式是 `auto 函数变量 = [捕获列表](参数列表) -> 返回类型 { 函数体 };`。返回类型通常可以推导，此时省略 `-> 返回类型`。

```cpp
auto add = [](int a, int b) -> int {
    return a + b;
};
int sum = add(2, 3);                       // 5

// 省略返回类型；下面的 cmp 可以传给 sort
auto cmp = [](int a, int b) {
    return a < b;
};
bool before = cmp(2, 3);                   // true
```

`[]` 是捕获列表，决定函数体怎样使用外部局部变量；`(int a, int b)` 才是每次调用时传入的参数。

| 捕获写法 | 含义 |
| --- | --- |
| `[]` | 不捕获外部局部变量 |
| `[limit]` | 复制 `limit`，保留创建 lambda 时的值 |
| `[&limit]` | 引用 `limit`，调用时访问外部变量 |
| `[=]` / `[&]` | 默认按值 / 按引用捕获函数体所需的外部局部变量 |

```cpp
int limit = 10;
auto by_value = [limit](int x) { return x > limit; };
auto by_reference = [&limit](int x) { return x > limit; };
limit = 20;
bool first = by_value(15);                 // true，比较 15 > 10
bool second = by_reference(15);            // false，比较 15 > 20
```

引用捕获要求外部变量在调用时仍然存活。比较器若捕获外部状态，排序过程中或堆仍有元素时，不要改变会影响比较结果的状态。

## 常用写法速查

### 容器

### vector - 动态数组

```cpp
vector<int> v;              // 空
vector<int> v(n);           // n 个 0
vector<int> v(n, val);      // n 个 val
vector<int> v = {1, 2, 3};  // 初始化列表

v.push_back(x);             // 尾部添加
v.pop_back();               // 尾部删除
v.size();                   // 大小
v.empty();                  // 是否为空
v.clear();                  // 清空
v[i];                       // 访问（不检查边界）
v.at(i);                    // 访问（检查边界）
v.front();                  // 首元素
v.back();                   // 尾元素

// 遍历
for (int x : v) { }
for (int i = 0; i < v.size(); ++i) { }

// 二维
vector<vector<int>> mat(n, vector<int>(m, 0));
```

**注意**：`v.size()` 返回 `size_t`（无符号），倒序遍历要小心：

```cpp
// 错误：size() - 1 在空数组时会溢出
for (int i = v.size() - 1; i >= 0; --i) { }

// 正确
for (int i = (int)v.size() - 1; i >= 0; --i) { }
```

---

### string - 字符串

```cpp
string s;
string s = "hello";
string s(n, 'a');           // n 个 'a'

s.size() / s.length();      // 长度
s.empty();
s.clear();
s.push_back('c');
s.pop_back();
s.substr(pos, len);         // 子串
s.find("sub");              // 查找子串，返回位置或 npos
s.rfind("sub");             // 反向查找

// 拼接
s += "world";
s = s1 + s2;

// 字符串流
stringstream ss(s);
string word;
while (ss >> word) { }      // 按空白分割
```

---

### map / unordered_map - 映射

```cpp
map<string, int> mp;        // 有序
unordered_map<string, int> mp; // 无序，更快

mp[key] = value;            // 插入/修改
mp.count(key);              // 是否存在（返回 0 或 1）
mp.find(key);               // 查找，返回迭代器
mp.erase(key);              // 删除
mp.size();

// 遍历
for (auto [k, v] : mp) { }
for (auto& p : mp) {
    p.first;  // key
    p.second; // value
}
```

**注意**：`mp[key]` 会插入默认值，判断存在用 `count` 或 `find`。

---

### set / unordered_set - 集合

```cpp
set<int> st;                // 有序
unordered_set<int> st;      // 无序，更快

st.insert(x);               // 插入
st.erase(x);                // 删除
st.count(x);                // 是否存在
st.size();
st.empty();

// set 特有（有序）
st.lower_bound(x);          // 第一个 >= x
st.upper_bound(x);          // 第一个 > x
```

---

### stack - 栈

```cpp
stack<int> stk;

stk.push(x);                // 入栈
stk.pop();                  // 出栈
stk.top();                  // 栈顶
stk.size();
stk.empty();
```

---

### queue - 队列

```cpp
queue<int> q;

q.push(x);                  // 入队
q.pop();                    // 出队
q.front();                  // 队首
q.back();                   // 队尾
q.size();
q.empty();
```

---

### deque - 双端队列

```cpp
deque<int> dq;

dq.push_back(x);            // 尾部插入
dq.push_front(x);           // 头部插入
dq.pop_back();              // 尾部删除
dq.pop_front();             // 头部删除
dq.front();
dq.back();
dq[i];                      // 随机访问
```

---

### priority_queue - 优先队列（堆）

```cpp
priority_queue<int> pq;                     // 默认大根堆，top() 最大
priority_queue<int, vector<int>, greater<int>> min_heap; // 小根堆，top() 最小

pq.push(x);                 // 插入
pq.pop();                   // 删除堆顶
pq.top();                   // 堆顶
pq.size();
pq.empty();

// pair 先比较 first；first 相同时再比较 second
priority_queue<pair<int, int>> pairs;
```

**用 lambda 自定义最小堆：**

```cpp
auto cmp = [](int a, int b) { return a > b; };
priority_queue<int, vector<int>, decltype(cmp)> min_heap(cmp);

min_heap.push(5);
min_heap.push(1);
min_heap.push(3);
int first = min_heap.top();                 // 1
min_heap.pop();
int second = min_heap.top();                // 3
```

`priority_queue<元素类型, 底层容器类型, 比较器类型>` 的第三个参数需要一个**类型**，所以用 `decltype(cmp)` 取得这个 lambda 的类型；构造时的 `(cmp)` 再把实际比较器对象传进去。C++17 中这种写法不能省掉构造参数 `(cmp)`。

对堆而言，`cmp(a, b) == true` 表示 `a` 的优先级比 `b` 低。因此 `a > b` 会让较小的值留在堆顶；`pop()` 只删除堆顶，不返回它，取值时先 `top()` 再 `pop()`。

自定义结构体同样可用，例如按截止时间从早到晚取任务，相同时先取编号小的：

```cpp
struct Task {
    int deadline;
    int id;
};
auto cmp = [](const Task& a, const Task& b) {
    if (a.deadline != b.deadline) return a.deadline > b.deadline;
    return a.id > b.id;
};
priority_queue<Task, vector<Task>, decltype(cmp)> tasks(cmp);
tasks.push({5, 1});
tasks.push({2, 3});
tasks.push({2, 1});
Task next = tasks.top();                    // deadline = 2，id = 1
```

这些片段使用 `<queue>`、`<vector>`；`greater<int>` 还需要 `<functional>`。按本站片段约定，省略了 `std::` 前缀。

---

## 算法

### 排序

```cpp
sort(v.begin(), v.end());                   // 升序
sort(v.begin(), v.end(), greater<int>());   // 降序

// 直接传入 lambda：降序
sort(v.begin(), v.end(), [](int a, int b) {
    return a > b;
});

// 也可以先定义，再传给 sort
auto cmp = [](int a, int b) { return a < b; };
sort(v.begin(), v.end(), cmp);              // 升序

// 稳定排序：比较器认为等价的元素保留原来的相对顺序
stable_sort(v.begin(), v.end());
```

**结构体按多个字段排序：**先按 `x` 升序，`x` 相同时按 `y` 升序。

```cpp
struct Node {
    int x, y;
};
vector<Node> nodes = {{2, 1}, {1, 3}, {1, 2}};
sort(nodes.begin(), nodes.end(), [](const Node& a, const Node& b) {
    if (a.x != b.x) return a.x < b.x;
    return a.y < b.y;
});
// 排序结果：{1, 2}, {1, 3}, {2, 1}
```

对 `sort` 而言，`cmp(a, b) == true` 表示 `a` 应排在 `b` 前面。比较器必须满足严格弱序，不能把 `<` / `>` 写成 `<=` / `>=`；相等时应返回 `false`。结构体用 `const Node&` 传参，避免每次比较都复制对象。`sort` 需要 `<algorithm>`。

| 想让较小元素先出现 | 比较器 |
| --- | --- |
| `sort` 升序排列 | `return a < b;` |
| `priority_queue` 最小堆 | `return a > b;` |

---

### 二分查找

```cpp
// 前提：数组已排序
vector<int> v = {1, 2, 2, 3, 4};

// 是否存在
bool found = binary_search(v.begin(), v.end(), 2);

// 第一个 >= x
auto it = lower_bound(v.begin(), v.end(), 2);
int pos = it - v.begin();

// 第一个 > x
auto it = upper_bound(v.begin(), v.end(), 2);

// 自定义比较
lower_bound(v.begin(), v.end(), x, greater<int>());
```

---

### 反转、旋转、去重

```cpp
// 反转
reverse(v.begin(), v.end());

// 旋转（将 [first, middle) 移到末尾）
rotate(v.begin(), v.begin() + k, v.end());

// 去重（必须先排序）
sort(v.begin(), v.end());
v.erase(unique(v.begin(), v.end()), v.end());
```

---

### 最值

```cpp
int max_val = *max_element(v.begin(), v.end());
int min_val = *min_element(v.begin(), v.end());

auto [min_it, max_it] = minmax_element(v.begin(), v.end());

// 最大最小值本身
int a = 3, b = 5;
int mx = max(a, b);
int mn = min(a, b);
int mx3 = max({a, b, c});  // 多个值
```

---

### 累加、填充

```cpp
// 求和
int sum = accumulate(v.begin(), v.end(), 0);
long long sum = accumulate(v.begin(), v.end(), 0LL);

// 填充
fill(v.begin(), v.end(), value);

// 递增序列
iota(v.begin(), v.end(), start); // v = [start, start+1, ...]
```

---

### 全排列

```cpp
vector<int> v = {1, 2, 3};
sort(v.begin(), v.end());

do {
    // 使用当前排列
} while (next_permutation(v.begin(), v.end()));
```

---

## 输入输出

### 标准输入输出

```cpp
int n;
cin >> n;                   // 读一个数
cout << n << "\n";          // 输出

// 读整行
string line;
getline(cin, line);

// 读到 EOF
while (cin >> n) { }
while (getline(cin, line)) { }
```

**注意**：`cin >> x` 后用 `getline` 会读到残留换行符，需要：

```cpp
cin >> n;
cin.ignore();               // 忽略换行符
getline(cin, line);
```

---

### 加速输入输出

```cpp
ios::sync_with_stdio(false);
cin.tie(nullptr);
```

**注意**：使用后不能混用 `scanf/printf` 和 `cin/cout`。

---

### 格式化输出

```cpp
#include <iomanip>

// 精度
cout << fixed << setprecision(2) << x << "\n";

// 宽度
cout << setw(5) << x << "\n";

// 填充
cout << setfill('0') << setw(5) << x << "\n";  // 00123
```

---

### 字符串流

```cpp
#include <sstream>

// 分割字符串
string s = "1 2 3";
stringstream ss(s);
int x;
while (ss >> x) { }

// 拼接
stringstream ss;
ss << "x = " << x << ", y = " << y;
string result = ss.str();
```

---

## 常用函数

### 字符处理

```cpp
#include <cctype>

isdigit(c);                 // 是否数字
isalpha(c);                 // 是否字母
isalnum(c);                 // 是否数字或字母
islower(c);                 // 是否小写
isupper(c);                 // 是否大写
isspace(c);                 // 是否空白

tolower(c);                 // 转小写
toupper(c);                 // 转大写
```

---

### 数学函数

```cpp
#include <cmath>

abs(x);                     // 绝对值
sqrt(x);                    // 平方根
pow(x, y);                  // x^y
ceil(x);                    // 向上取整
floor(x);                   // 向下取整
round(x);                   // 四舍五入

gcd(a, b);                  // 最大公约数（C++17）
lcm(a, b);                  // 最小公倍数（C++17）
```

---

### 位运算内置函数

```cpp
__builtin_popcount(x);      // 二进制中 1 的个数 (int)
__builtin_popcountll(x);    // long long 版本
__builtin_clz(x);           // 前导 0 个数
__builtin_ctz(x);           // 后缀 0 个数
__builtin_ffs(x);           // 最低位 1 的位置（1-indexed）
```

---

## 数值边界：numeric_limits

`std::numeric_limits<T>` 用来查询类型 `T` 的数值范围，需要头文件 `<limits>`。机考中获取 `int` 的最大值和最小值，直接写：

```cpp
#include <limits>

constexpr int int_max = std::numeric_limits<int>::max();
constexpr int int_min = std::numeric_limits<int>::min();
constexpr int int_lowest = std::numeric_limits<int>::lowest(); // 对 int 等于 min()

constexpr long long ll_max = std::numeric_limits<long long>::max();
constexpr long long ll_min = std::numeric_limits<long long>::lowest();
```

在常见的 32 位 `int` 环境中，最大值是 `2147483647`，最小值是 `-2147483648`。使用上面的函数可以避免硬编码位宽；最小值不要写成 `-int_max`，它在该环境中会差 1。

常用来初始化“求最大值 / 最小值”的答案：

```cpp
vector<int> values = {-5, -2, -9};           // 此处保证非空
int best_max = std::numeric_limits<int>::lowest();
int best_min = std::numeric_limits<int>::max();
for (int x : values) {
    best_max = std::max(best_max, x);
    best_min = std::min(best_min, x);
}
// best_max = -2，best_min = -9
```

| 写法 | 含义 |
| --- | --- |
| `numeric_limits<int>::max()` | `int` 的最大值 |
| `numeric_limits<int>::min()` / `lowest()` | `int` 的最小值 |
| `numeric_limits<double>::max()` | `double` 的最大有限值 |
| `numeric_limits<double>::lowest()` | `double` 的最负有限值 |
| `numeric_limits<double>::min()` | `double` 的最小正正规数，不是最负值 |

也可以用 `<climits>` 中的 `INT_MAX`、`INT_MIN`、`LLONG_MAX`、`LLONG_MIN`。这些整数边界是有限值，不是数学上的无穷；不要直接计算 `int_max + 1`、`int_min - 1` 或 `-int_min`，它们会造成有符号整数溢出。最短路或 DP 若用最大值表示“不可达”，先判断是否可达，并确保后续加法不超出类型范围。

## 常见陷阱

1. **无符号类型**：`size()` 返回无符号类型，减法可能溢出
2. **迭代器失效**：遍历时修改容器可能导致迭代器失效
3. **map[] 副作用**：会插入默认值，判断存在用 `count`
4. **比较函数**：必须满足严格弱序，不能写 `<=`
5. **优先队列**：默认大根堆，小根堆要用 `greater<int>`
6. **整数溢出**：`accumulate` 的初始值要匹配类型

## 容器选择

| 需求 | 容器 | 复杂度 |
| --- | --- | --- |
| 动态数组 | `vector` | O(1) 随机访问 |
| 有序键值对 | `map` | O(log n) 查找 |
| 无序键值对 | `unordered_map` | O(1) 平均查找 |
| 有序集合 | `set` | O(log n) 插入/查找 |
| 无序集合 | `unordered_set` | O(1) 平均插入/查找 |
| 栈 | `stack` | O(1) 栈顶操作 |
| 队列 | `queue` | O(1) 队首队尾操作 |
| 双端队列 | `deque` | O(1) 两端操作 |
| 优先队列 | `priority_queue` | O(log n) 插入/删除 |
