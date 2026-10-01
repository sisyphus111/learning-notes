# STL 01：对象语义、构造与传参

理解 STL 的第一步不是记容器 API，而是理解“容器本身也是一个普通 C++ 对象”。它拥有一组元素，遵循构造、析构、复制、移动、引用和 `const` 的统一语言规则。

## 1. 容器默认拥有元素

```cpp
std::vector<std::string> names = {"Ada", "Linus"};
```

`names` 拥有两个独立的 `std::string` 对象，而不是保存外部字符串变量的别名。容器析构时会先析构所有元素，再释放自己管理的存储空间。

```cpp
std::string name = "Ada";
std::vector<std::string> names;
names.push_back(name);      // 复制 name
name = "Grace";            // names[0] 仍然是 "Ada"
names.push_back(std::move(name)); // 移动 name 的资源
```

如果元素类型是指针，容器拥有的是“指针值”，不自动拥有指针指向的对象。需要独占所有权时优先使用 `std::unique_ptr<T>`。

## 2. 构造语法可能只差一对括号

| 代码 | 结果 |
| --- | --- |
| `vector<int> a;` | 空 vector |
| `vector<int> a(3);` | 3 个值初始化的 `int`，即 `{0,0,0}` |
| `vector<int> a(3, 7);` | 3 个 7 |
| `vector<int> a{3};` | 1 个元素，值为 3 |
| `vector<int> a{3, 7};` | 两个元素 `{3,7}` |
| `vector<int> a(first, last);` | 复制半开区间 `[first,last)` |

**高频错误：**`vector<int>(n)` 表示 n 个元素，`vector<int>{n}` 表示一个值为 n 的元素。

## 3. 复制容器是深度逐元素复制

```cpp
std::vector<std::string> a = {"x", "y"};
std::vector<std::string> b = a;
b[0] = "changed";
// a[0] 仍是 "x"
```

复制构造和复制赋值都会产生独立元素。对 n 个元素的容器，通常至少是 O(n)，元素自身的复制成本还要继续计算。

| 表达式 | 发生什么 |
| --- | --- |
| `Container b = a;` | 复制构造：创建新对象并复制 a 的元素 |
| `b = a;` | 复制赋值：替换 b 原有内容，再复制 a |
| `Container b = std::move(a);` | 移动构造：通常接管 a 的内部资源 |
| `b = std::move(a);` | 移动赋值：释放/替换 b 的资源并接管 a |

## 4. 移动后对象仍有效，但值未指定

```cpp
std::vector<int> source = {1, 2, 3};
std::vector<int> target = std::move(source);

source.clear();      // 合法
source.push_back(9); // 合法
// 不应假设 source 一定为空，也不应继续读取旧内容。
```

“valid but unspecified”意味着对象仍满足类型不变量，可以析构、赋值或调用没有前置值要求的操作；但标准没有承诺它保留什么元素。移动通常是 O(1)，但分配器不兼容等场景可能退化为逐元素移动。

## 5. 传参方式决定是否复制

```cpp
void inspect(const std::vector<int>& values); // 不复制，只读
void mutate(std::vector<int>& values);        // 不复制，可修改调用者
void consume(std::vector<int> values);        // 创建函数自己的副本/移动结果
void accept_temporary(std::vector<int>&& values); // 只接右值引用
```

| 需求 | 推荐参数 | 行为 |
| --- | --- | --- |
| 只读现有容器 | `const Container&` | 无 O(n) 复制，不能修改元素 |
| 修改调用者容器 | `Container&` | 修改对调用者可见 |
| 函数需要一份自己的值 | `Container` | 左值调用时复制，右值调用时可移动 |
| 参数很小或本来就要复制 | 按值 | 接口简单，便于在函数内移动 |

```cpp
void store(std::vector<int> values) {
    member_values = std::move(values);
}

store(existing);                  // existing 被复制到 values
store(std::move(existing));       // existing 被移动到 values
store(std::vector<int>{1, 2});    // 临时对象直接用于构造 values
```

## 6. 返回容器通常直接按值返回

```cpp
std::vector<int> make_values() {
    std::vector<int> result = {1, 2, 3};
    return result;
}
```

现代 C++ 会进行 copy elision/NRVO，或者在无法消除时移动返回值。不要为了“避免复制”返回局部变量的引用：

```cpp
const std::vector<int>& broken() {
    std::vector<int> local = {1, 2, 3};
    return local; // 错误：函数结束后引用悬空
}
```

## 7. 访问表达式常常返回引用

```cpp
std::vector<std::string> names = {"Ada"};

auto value = names[0];   // 复制一个 string
auto& ref = names[0];    // 引用容器中的 string
const auto& cref = names[0]; // 只读引用

ref = "Grace";          // names[0] 也变为 "Grace"
```

`operator[]`、`at`、`front`、`back` 和迭代器解引用通常返回元素引用。是否写 `&` 会直接决定你在修改原元素还是修改副本。

```cpp
for (auto item : names) { item += "!"; }   // 修改副本
for (auto& item : names) { item += "!"; } // 修改容器元素
for (const auto& item : names) { /* 只读且不复制 */ }
```

## 8. `const` 会沿访问传播

```cpp
void print(const std::vector<std::string>& names) {
    // names[0] 的类型效果是 const std::string&
    // names[0] = "x"; // 编译错误
}
```

一个 `const` 容器不允许通过自身接口修改元素，即使元素实际存放在堆上。注意：如果元素是裸指针，`const vector<T*>` 只禁止修改指针值，并不自动让 `*pointer` 变成 const。

## 9. 结构化绑定也有复制与引用之分

```cpp
std::map<std::string, int> counts = {{"a", 1}};

for (auto [key, value] : counts) {
    ++value; // 修改副本
}
for (auto& [key, value] : counts) {
    ++value; // 修改 map 中的 value；key 本身仍是 const
}
```

## 10. 引用有效性取决于容器是否移动了元素

```cpp
std::vector<int> values = {1, 2, 3};
int& first = values[0];
values.push_back(4); // 如果触发扩容，first 立即失效
// 使用 first 可能成为未定义行为
```

引用、指针和迭代器只是指向容器内部位置的观察者，不拥有元素。容器重新分配、移动节点或删除元素后，它们可能悬空。具体规则必须结合容器底层结构判断，后续页面会逐类展开。

## 11. 一页决策表

| 写法 | 是否复制元素 | 能否修改原容器 |
| --- | --- | --- |
| `Container b = a` | 是 | 否，b 独立 |
| `Container& b = a` | 否 | 是 |
| `const Container& b = a` | 否 | 否 |
| `auto x = container[i]` | 复制一个元素 | 否 |
| `auto& x = container[i]` | 否 | 是 |
| `auto x = std::move(a)` | 通常接管资源 | a 的旧值不可再假设 |

**掌握标准：**看到一次容器构造、赋值、传参、返回或访问时，能明确指出是否产生 O(n) 复制、得到的是值还是引用、以及对象的所有权和生命周期在哪里。
