# Data Structures

## Segment Tree

### Recursive

+ Implicit segment tree, range query + point update

```cpp
struct Node {
  int lc, rc, p;
};

struct SegTree {
  vector<Node> t = {{}};
  SegTree(int n) { t.reserve(n * 40); }
  int modify(int p, int l, int r, int x, int v) {
    int u = p;
    if (p == 0) {
      t.push_back(t[p]);
      u = (int)t.size() - 1;
    }
    if (r - l == 1) {
      t[u].p = t[p].p + v;
    } else {
      int m = (l + r) / 2;
      if (x < m) {
        t[u].lc = modify(t[p].lc, l, m, x, v);
      } else {
        t[u].rc = modify(t[p].rc, m, r, x, v);
      }
      t[u].p = t[t[u].lc].p + t[t[u].rc].p;
    }
    return u;
  }
  int query(int p, int l, int r, int x, int y) {
    if (x <= l && r <= y) return t[p].p;
    int m = (l + r) / 2, res = 0;
    if (x < m) res += query(t[p].lc, l, m, x, y);
    if (y > m) res += query(t[p].rc, m, r, x, y);
    return res;
  }
};
```

+ Persistent implicit, range query + point update

```cpp
struct Node {
  int lc = 0, rc = 0, p = 0;
};

struct SegTree {
  vector<Node> t = {{}};  // init all
  SegTree() = default;
  SegTree(int n) { t.reserve(n * 20); }
  int modify(int p, int l, int r, int x, int v) {
    // p: original node, update a[x] -> v
    t.push_back(t[p]);
    int u = (int)t.size() - 1;
    if (r - l == 1) {
      t[u].p = v;
    } else {
      int m = (l + r) / 2;
      if (x < m) {
        t[u].lc = modify(t[p].lc, l, m, x, v);
        t[u].rc = t[p].rc;
      } else {
        t[u].lc = t[p].lc;
        t[u].rc = modify(t[p].rc, m, r, x, v);
      }
      t[u].p = t[t[u].lc].p + t[t[u].rc].p;
    }
    return u;
  }
  int query(int p, int l, int r, int x, int y) {
    // query sum a[x]...a[y-1] rooted at p
    // t[p] holds the info of [l, r)
    if (x <= l && r <= y) return t[p].p;
    int m = (l + r) / 2, res = 0;
    if (x < m) res += query(t[p].lc, l, m, x, y);
    if (y > m) res += query(t[p].rc, m, r, x, y);
    return res;
  }
};
```

### Iterating

+ Iterating, range query + point update

```cpp
struct Node {
  ll v = 0, init = 0;
};

Node pull(const Node &a, const Node &b) {
  if (!a.init) return b;
  if (!b.init) return a;
  Node c;
  return c;
}

struct SegTree {
  ll n;
  vector<Node> t;
  SegTree(ll _n) : n(_n), t(2 * n){};
  void modify(ll p, const Node &v) {
    t[p += n] = v;
    for (p /= 2; p; p /= 2) t[p] = pull(t[p * 2], t[p * 2 + 1]);
  }
  Node query(ll l, ll r) {
    Node left, right;
    for (l += n, r += n; l < r; l /= 2, r /= 2) {
      if (l & 1) left = pull(left, t[l++]);
      if (r & 1) right = pull(t[--r], right);
    }
    return pull(left, right);
  }
};
```

+ Iterating, range query + range update

```cpp
struct Node {
  ll v = 0;
};
struct Tag {
  ll v = 0;
};
Node pull(const Node& a, const Node& b) { return {max(a.v, b.v)}; }
Tag pull(const Tag& a, const Tag& b) { return {a.v + b.v}; }
Node apply_tag(const Node& a, const Tag& b) { return {a.v + b.v}; }

struct SegTree {
  ll n, h;
  vector<Node> t;
  vector<Tag> lazy;
  SegTree(ll _n) : n(_n), h((ll)log2(n)), t(2 * _n), lazy(2 * _n) {}
  void apply(ll x, const Tag& tag) {
    t[x] = apply_tag(t[x], tag);
    lazy[x] = pull(lazy[x], tag);
  }
  void build(ll l) {
    for (l = (l + n) / 2; l > 0; l /= 2) {
      if (!lazy[l].v) t[l] = pull(t[l * 2], t[2 * l + 1]);
    }
  }
  void push(ll l) {
    l += n;
    for (ll s = h; s > 0; s--) {
      ll i = l >> s;
      if (lazy[i].v) {
        apply(2 * i, lazy[i]);
        apply(2 * i + 1, lazy[i]);
      }
      lazy[i] = Tag();
    }
  }
  void modify(ll l, ll r, const Tag& v) {
    push(l), push(r - 1);
    ll l0 = l, r0 = r;
    for (l += n, r += n; l < r; l /= 2, r /= 2) {
      if (l & 1) apply(l++, v);
      if (r & 1) apply(--r, v);
    }
    build(l0), build(r0 - 1);
  }
  Node query(ll l, ll r) {
    push(l), push(r - 1);
    Node left, right;
    for (l += n, r += n; l < r; l /= 2, r /= 2) {
      if (l & 1) left = pull(left, t[l++]);
      if (r & 1) right = pull(t[--r], right);
    }
    return pull(left, right);
  }
};
```

+ AtCoder Segment Tree (recursive structure but iterative)

```cpp
template <class T> struct PointSegmentTree {
  int size = 1;
  vector<T> tree;
  PointSegmentTree(int n) : PointSegmentTree(vector<T>(n)) {}
  PointSegmentTree(vector<T>& arr) {
    while(size < (int)arr.size())
      size <<= 1;
    tree = vector<T>(size << 1);
    for(int i = size + arr.size() - 1; i >= 1; i--)
      if(i >= size) tree[i] = arr[i - size];
      else consume(i);
  }
  void set(int i, T val) {
    tree[i += size] = val;
    for(i >>= 1; i >= 1; i >>= 1)
      consume(i);
  }
  T get(int i) { return tree[i + size]; }
  T query(int l, int r) {
    T resl, resr;
    for(l += size, r += size + 1; l < r; l >>= 1, r >>= 1) {
      if(l & 1) resl = resl * tree[l++];
      if(r & 1) resr = tree[--r] * resr;
    }
    return resl * resr;
  }
  T query_all() { return tree[1]; }
  void consume(int i) { tree[i] = tree[i << 1] * tree[i << 1 | 1]; }
};


struct SegInfo {
  ll v;
  SegInfo() : SegInfo(0) {}
  SegInfo(ll val) : v(val) {}
  SegInfo operator*(SegInfo b) {
    return SegInfo(v + b.v);
  }
};
```

## Union Find

```cpp
vector<int> p(n);
iota(p.begin(), p.end(), 0);
function<int(int)> find = [&](int x) { return p[x] == x ? x : (p[x] = find(p[x])); };
auto merge = [&](int x, int y) { p[find(x)] = find(y); };
```

+ Persistent version

```cpp
struct Node {
  int lc, rc, p;
};

struct SegTree {
  vector<Node> t = {{0, 0, -1}};  // init all
  SegTree() = default;
  SegTree(int n) { t.reserve(n * 20); }
  int modify(int p, int l, int r, int x, int v) {
    // p: original node, update a[x] -> v
    t.push_back(t[p]);
    int u = (int)t.size() - 1;
    if (r - l == 1) {
      t[u].p = v;
    } else {
      int m = (l + r) / 2;
      if (x < m) {
        t[u].lc = modify(t[p].lc, l, m, x, v);
        t[u].rc = t[p].rc;
      } else {
        t[u].lc = t[p].lc;
        t[u].rc = modify(t[p].rc, m, r, x, v);
      }
      t[u].p = t[t[u].lc].p + t[t[u].rc].p;
    }
    return u;
  }
  int query(int p, int l, int r, int x, int y) {
    // query sum a[x]...a[y-1] rooted at p
    // t[p] holds the info of [l, r)
    if (x <= l && r <= y) return t[p].p;
    int m = (l + r) / 2, res = 0;
    if (x < m) res += query(t[p].lc, l, m, x, y);
    if (y > m) res += query(t[p].rc, m, r, x, y);
    return res;
  }
};

struct DSU {
  int n;
  SegTree seg;
  DSU(int _n) : n(_n), seg(n) {}
  int get(int p, int x) { return seg.query(p, 0, n, x, x + 1); }
  int set(int p, int x, int v) { return seg.modify(p, 0, n, x, v); }
  int find(int p, int x) {
    int parent = get(p, x);
    if (parent < 0) return x;
    return find(p, parent);
  }
  int is_same(int p, int x, int y) { return find(p, x) == find(p, y); }
  int merge(int p, int x, int y) {
    int rx = find(p, x), ry = find(p, y);
    if (rx == ry) return -1;
    int rank_x = -get(p, rx), rank_y = -get(p, ry);
    if (rank_x < rank_y) {
      p = set(p, rx, ry);
    } else if (rank_x > rank_y) {
      p = set(p, ry, rx);
    } else {
      p = set(p, ry, rx);
      p = set(p, rx, -rx - 1);
    }
    return p;
  }
};
```

## Fenwick Tree

+ askd version

```cpp
template <typename T> struct FenwickTree {
  int size = 1, high_bit = 1;
  vector<T> tree;
  FenwickTree(int _size) : size(_size) {
    tree.resize(size + 1);
    while((high_bit << 1) <= size) high_bit <<= 1;
  }
  FenwickTree(vector<T>& arr) : FenwickTree(arr.size()) {
    for(int i = 0; i < size; i++) update(i, arr[i]);
  }
  int lower_bound(T x) {
    int res = 0; T cur = 0;
    for(int bit = high_bit; bit > 0; bit >>= 1) {
      if((res|bit) <= size && cur + tree[res|bit] < x) {
        res |= bit; cur += tree[res];
      }
    }
    return res;
  }
  T prefix_sum(int i) {
    T ret = 0;
    for(i++; i > 0; i -= (i & -i)) ret += tree[i];
    return ret;
  }
  T range_sum(int l, int r) { return (l > r) ? 0 : prefix_sum(r) - prefix_sum(l - 1); }
  void update(int i, T delta) { for(i++; i <= size; i += (i & -i)) tree[i] += delta; }
};
```

+ Nea1 version

```cpp
template <typename T>
struct Fenwick {
  const int n;
  vector<T> a;
  Fenwick(int n) : n(n), a(n) {}
  void add(int x, T v) {
    for (int i = x + 1; i <= n; i += i & -i) {
      a[i - 1] += v;
    }
  }
  T sum(int x) {
    T ans = 0;
    for (int i = x; i > 0; i -= i & -i) {
      ans += a[i - 1];
    }
    return ans;
  }
  T rangeSum(int l, int r) { return sum(r) - sum(l); }
};
```

## PBDS

```cpp
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
using namespace std;
using namespace __gnu_pbds;
template<typename T>
using ordered_set = tree<T, null_type, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
template<typename T, typename X>
using ordered_map = tree<T, X, less<T>, rb_tree_tag, tree_order_statistics_node_update>;
template<typename T, typename X>
using fast_map = cc_hash_table<T, X>;
template<typename T, typename X>
using ht = gp_hash_table<T, X>;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());

struct splitmix64 {
    size_t operator()(size_t x) const {
        static const size_t fixed = chrono::steady_clock::now().time_since_epoch().count();
        x += 0x9e3779b97f4a7c15 + fixed;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
};
```

## Treap
+ (No rotation version)

```cpp
struct Node {
  Node *l, *r;
  int s, sz;
  // int t = 0, a = 0, g = 0; // for lazy propagation
  ll w;

  Node(int _s) : l(nullptr), r(nullptr), s(_s), sz(1), w(rng()) {}
  void apply(int vt, int vg) {
    // for lazy propagation
    // s -= vt;
    // t += vt, a += vg, g += vg;
  }
  void push() {
    // for lazy propagation
    // if (l != nullptr) l->apply(t, g);
    // if (r != nullptr) r->apply(t, g);
    // t = g = 0;
  }
  void pull() { sz = 1 + (l ? l->sz : 0) + (r ? r->sz : 0); }
};

std::pair<Node *, Node *> split(Node *t, int v) {
  if (t == nullptr) return {nullptr, nullptr};
  t->push();
  if (t->s < v) {
    auto [x, y] = split(t->r, v);
    t->r = x;
    t->pull();
    return {t, y};
  } else {
    auto [x, y] = split(t->l, v);
    t->l = y;
    t->pull();
    return {x, t};
  }
}

Node *merge(Node *p, Node *q) {
  if (p == nullptr) return q;
  if (q == nullptr) return p;
  if (p->w < q->w) swap(p, q);
  auto [x, y] = split(q, p->s + rng() % 2);
  p->push();
  p->l = merge(p->l, x);
  p->r = merge(p->r, y);
  p->pull();
  return p;
}

Node *insert(Node *t, int v) {
  auto [x, y] = split(t, v);
  return merge(merge(x, new Node(v)), y);
}

Node *erase(Node *t, int v) {
  auto [x, y] = split(t, v);
  auto [p, q] = split(y, v + 1);
  return merge(merge(x, merge(p->l, p->r)), q);
}

int get_rank(Node *&t, int v) {
  auto [x, y] = split(t, v);
  int res = (x ? x->sz : 0) + 1;
  t = merge(x, y);
  return res;
}

Node *kth(Node *t, int k) {
  k--;
  while (true) {
    int left_sz = t->l ? t->l->sz : 0;
    if (k < left_sz) {
      t = t->l;
    } else if (k == left_sz) {
      return t;
    } else {
      k -= left_sz + 1, t = t->r;
    }
  }
}

Node *get_prev(Node *&t, int v) {
  auto [x, y] = split(t, v);
  Node *res = kth(x, x->sz);
  t = merge(x, y);
  return res;
}

Node *get_next(Node *&t, int v) {
  auto [x, y] = split(t, v + 1);
  Node *res = kth(y, 1);
  t = merge(x, y);
  return res;
}
```

+ USAGE

```cpp
int main() {
  cin.tie(nullptr)->sync_with_stdio(false);
  int n;
  cin >> n;
  Node *t = nullptr;
  for (int op, x; n--;) {
    cin >> op >> x;
    if (op == 1) {
      t = insert(t, x);
    } else if (op == 2) {
      t = erase(t, x);
    } else if (op == 3) {
      cout << get_rank(t, x) << "\n";
    } else if (op == 4) {
      cout << kth(t, x)->s << "\n";
    } else if (op == 5) {
      cout << get_prev(t, x)->s << "\n";
    } else {
      cout << get_next(t, x)->s << "\n";
    }
  }
}
```

## Implicit treap

+ Split by size

```cpp
struct Node {
  Node *l, *r;
  int s, sz;
  // int lazy = 0;
  ll w;

  Node(int _s) : l(nullptr), r(nullptr), s(_s), sz(1), w(rnd()) {}
  void apply() {
    // for lazy propagation
    // lazy ^= 1;
  }
  void push() {
    // for lazy propagation
    // if (lazy) {
    //   swap(l, r);
    //   if (l != nullptr) l->apply();
    //   if (r != nullptr) r->apply();
    //   lazy = 0;
    // }
  }
  void pull() { sz = 1 + (l ? l->sz : 0) + (r ? r->sz : 0); }
};

std::pair<Node *, Node *> split(Node *t, int v) {
  // first->sz == v
  if (t == nullptr) return {nullptr, nullptr};
  t->push();
  int left_sz = t->l ? t->l->sz : 0;
  if (left_sz < v) {
    auto [x, y] = split(t->r, v - left_sz - 1);
    t->r = x;
    t->pull();
    return {t, y};
  } else {
    auto [x, y] = split(t->l, v);
    t->l = y;
    t->pull();
    return {x, t};
  }
}

Node *merge(Node *p, Node *q) {
  if (p == nullptr) return q;
  if (q == nullptr) return p;
  if (p->w < q->w) {
    p->push();
    p->r = merge(p->r, q);
    p->pull();
    return p;
  } else {
    q->push();
    q->l = merge(p, q->l);
    q->pull();
    return q;
  }
}
```

## Persistent implicit treap

```cpp
pair<Node *, Node *> split(Node *t, int v) {
  // first->sz == v
  if (t == nullptr) return {nullptr, nullptr};
  t->push();
  int left_sz = t->l ? t->l->sz : 0;
  t = new Node(*t);
  if (left_sz < v) {
    auto [x, y] = split(t->r, v - left_sz - 1);
    t->r = x;
    t->pull();
    return {t, y};
  } else {
    auto [x, y] = split(t->l, v);
    t->l = y;
    t->pull();
    return {x, t};
  }
}

Node *merge(Node *p, Node *q) {
  if (p == nullptr) return new Node(*q);
  if (q == nullptr) return new Node(*p);
  if (p->w < q->w) {
    p = new Node(*p);
    p->push();
    p->r = merge(p->r, q);
    p->pull();
    return p;
  } else {
    q = new Node(*q);
    q->push();
    q->l = merge(p, q->l);
    q->pull();
    return q;
  }
}
```

## 2D Sparse Table

+ Sorry that this sucks - askd

```cpp
template <class T, class Compare = less<T>>
struct SparseTable2d {
  int n = 0, m = 0;
  T**** table;
  int* log;
  inline T choose(T x, T y) {
    return Compare()(x, y) ? x : y;
  }
  SparseTable2d(vector<vector<T>>& grid) {
    if(grid.empty() || grid[0].empty()) return;
    n = grid.size(); m = grid[0].size();
    log = new int[max(n, m) + 1];
    log[1] = 0;
    for(int i = 2; i <= max(n, m); i++)
      log[i] = log[i - 1] + ((i ^ (i - 1)) > i);
    table = new T***[n];
    for(int i = n - 1; i >= 0; i--) {
      table[i] = new T**[m];
      for(int j = m - 1; j >= 0; j--) {
        table[i][j] = new T*[log[n - i] + 1];
        for(int k = 0; k <= log[n - i]; k++) {
          table[i][j][k] = new T[log[m - j] + 1];
          if(!k) table[i][j][k][0] = grid[i][j];
          else table[i][j][k][0] = choose(table[i][j][k-1][0], table[i+(1<<(k-1))][j][k-1][0]);
          for(int l = 1; l <= log[m - j]; l++)
            table[i][j][k][l] = choose(table[i][j][k][l-1], table[i][j+(1<<(l-1))][k][l-1]);
        }
      }
    }
  }
  T query(int r1, int r2, int c1, int c2) {
    assert(r1 >= 0 && r2 < n && r1 <= r2);
    assert(c1 >= 0 && c2 < m && c1 <= c2);
    int rl = log[r2 - r1 + 1], cl = log[c2 - c1 + 1];
    T ca1 = choose(table[r1][c1][rl][cl], table[r2-(1<<rl)+1][c1][rl][cl]);
    T ca2 = choose(table[r1][c2-(1<<cl)+1][rl][cl], table[r2-(1<<rl)+1][c2-(1<<cl)+1][rl][cl]);
    return choose(ca1, ca2);
  }
};
```

+ USAGE

```cpp
vector<vector<int>> test =  {
  {1, 2, 3, 4}, {2, 3, 4, 5}, {9, 9, 9, 9}, {-1, -1, -1, -1}
};

SparseTable2d<int> st(test);                // Range min query
SparseTable2d<int,greater<int>> st2(test);  // Range max query
```

## K-D Tree

```cpp
struct Point {
  int x, y;
};
struct Rectangle {
  int lx, rx, ly, ry;
};

bool is_in(const Point &p, const Rectangle &rg) {
  return (p.x >= rg.lx) && (p.x <= rg.rx) && (p.y >= rg.ly) && (p.y <= rg.ry);
}

struct KDTree {
  vector<Point> points;
  struct Node {
    int lc, rc;
    Point point;
    Rectangle range;
    int num;
  };
  vector<Node> nodes;
  int root = -1;
  KDTree(const vector<Point> &points_) {
    points = points_;
    Rectangle range = {-1e9, 1e9, -1e9, 1e9};
    root = tree_construct(0, (int)points.size(), range, 0);
  }
  int tree_construct(int l, int r, Rectangle range, int depth) {
    if (l == r) return -1;
    if (l > r) throw;
    int mid = (l + r) / 2;
    auto comp = (depth % 2) ? [](Point &a, Point &b) { return a.x < b.x; }
                            : [](Point &a, Point &b) { return a.y < b.y; };
    nth_element(points.begin() + l, points.begin() + mid, points.begin() + r, comp);
    Rectangle l_range(range), r_range(range);
    if (depth % 2) {
      l_range.rx = points[mid].x;
      r_range.lx = points[mid].x;
    } else {
      l_range.ry = points[mid].y;
      r_range.ly = points[mid].y;
    }
    Node node = {tree_construct(l, mid, l_range, depth + 1),
                 tree_construct(mid + 1, r, r_range, depth + 1), points[mid], range, r - l};
    nodes.push_back(node);
    return (int)nodes.size() - 1;
  }

  int inner_query(int id, const Rectangle &rec, int depth) {
    if (id == -1) return 0;
    Rectangle rg = nodes[id].range;
    if (rg.lx >= rec.lx && rg.rx <= rec.rx && rg.ly >= rec.ly && rg.ry <= rec.ry) {
      return nodes[id].num;
    }
    int ans = 0;
    if (depth % 2) { // pruning
      if (rec.lx <= nodes[id].point.x) ans += inner_query(nodes[id].lc, rec, depth + 1);
      if (rec.rx >= nodes[id].point.x) ans += inner_query(nodes[id].rc, rec, depth + 1);
    } else {
      if (rec.ly <= nodes[id].point.y) ans += inner_query(nodes[id].lc, rec, depth + 1);
      if (rec.ry >= nodes[id].point.y) ans += inner_query(nodes[id].rc, rec, depth + 1);
    }
    if (is_in(nodes[id].point, rec)) ans += 1;
    return ans;
  }
  int query(const Rectangle &rec) { return inner_query(root, rec, 0); }
};
```

## Link/Cut Tree

```cpp
struct Node {
  Node *ch[2], *p;
  int id;
  bool rev;
  Node(int id) : ch{nullptr, nullptr}, p(nullptr), id(id), rev(false) {}
  friend void reverse(Node *p) {
    if (p != nullptr) {
      swap(p->ch[0], p->ch[1]);
      p->rev ^= 1;
    }
  }
  void push() {
    if (rev) {
      reverse(ch[0]);
      reverse(ch[1]);
      rev = false;
    }
  }
  void pull() {}
  bool is_root() { return p == nullptr || p->ch[0] != this && p->ch[1] != this; }
  bool pos() { return p->ch[1] == this; }
  void rotate() {
    Node *q = p;
    bool x = !pos();
    q->ch[!x] = ch[x];
    if (ch[x] != nullptr) ch[x]->p = q;
    p = q->p;
    if (!q->is_root()) q->p->ch[q->pos()] = this;
    ch[x] = q;
    q->p = this;
    pull();
    q->pull();
  }
  void splay() {
    vector<Node *> s;
    for (Node *i = this; !i->is_root(); i = i->p) s.push_back(i->p);
    while (!s.empty()) s.back()->push(), s.pop_back();
    push();
    while (!is_root()) {
      if (!p->is_root()) {
        if (pos() == p->pos()) {
          p->rotate();
        } else {
          rotate();
        }
      }
      rotate();
    }
    pull();
  }
  void access() {
    for (Node *i = this, *q = nullptr; i != nullptr; q = i, i = i->p) {
      i->splay();
      i->ch[1] = q;
      i->pull();
    }
    splay();
  }
  void makeroot() {
    access();
    reverse(this);
  }
};
void link(Node *x, Node *y) {
  x->makeroot();
  x->p = y;
}
void split(Node *x, Node *y) {
  x->makeroot();
  y->access();
}
void cut(Node *x, Node *y) {
  split(x, y);
  x->p = y->ch[0] = nullptr;
  y->pull();
}
bool connected(Node *p, Node *q) {
    p->access();
    q->access();
    return p->p != nullptr;
}
```
