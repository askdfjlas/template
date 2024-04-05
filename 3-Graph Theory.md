# Graph Theory

## Max Flow

```cpp
struct Edge {
  int from, to, cap, remain;
};

struct Dinic {
  int n;
  vector<Edge> e;
  vector<vector<int>> g;
  vector<int> d, cur;
  Dinic(int _n) : n(_n), g(n), d(n), cur(n) {}
  void add_edge(int u, int v, int c) {
    g[u].push_back((int)e.size());
    e.push_back({u, v, c, c});
    g[v].push_back((int)e.size());
    e.push_back({v, u, 0, 0});
  }
  ll max_flow(int s, int t) {
    int inf = 1e9;
    auto bfs = [&]() {
      fill(d.begin(), d.end(), inf), fill(cur.begin(), cur.end(), 0);
      d[s] = 0;
      vector<int> q{s}, nq;
      for (int step = 1; q.size(); swap(q, nq), nq.clear(), step++) {
        for (auto& node : q) {
          for (auto& edge : g[node]) {
            int ne = e[edge].to;
            if (!e[edge].remain || d[ne] <= step) continue;
            d[ne] = step, nq.push_back(ne);
            if (ne == t) return true;
          }
        }
      }
      return false;
    };
    function<int(int, int)> find = [&](int node, int limit) {
      if (node == t || !limit) return limit;
      int flow = 0;
      for (int i = cur[node]; i < g[node].size(); i++) {
        cur[node] = i;
        int edge = g[node][i], oe = edge ^ 1, ne = e[edge].to;
        if (!e[edge].remain || d[ne] != d[node] + 1) continue;
        if (int temp = find(ne, min(limit - flow, e[edge].remain))) {
          e[edge].remain -= temp, e[oe].remain += temp, flow += temp;
        } else {
          d[ne] = -1;
        }
        if (flow == limit) break;
      }
      return flow;
    };
    ll res = 0;
    while (bfs())
      while (int flow = find(s, inf)) res += flow;
    return res;
  }
};
```

+ USAGE

```cpp
int main() {
  int n, m, s, t;
  cin >> n >> m >> s >> t;
  Dinic dinic(n);
  for (int i = 0, u, v, c; i < m; i++) {
    cin >> u >> v >> c;
    dinic.add_edge(u - 1, v - 1, c);
  }
  cout << dinic.max_flow(s - 1, t - 1) << '\n';
}
```

## PushRelabel Max-Flow (faster)

```cpp
// https://github.com/kth-competitive-programming/kactl/blob/main/content/graph/PushRelabel.h
#define rep(i, a, b) for (int i = a; i < (b); ++i)
#define all(x) begin(x), end(x)
#define sz(x) (int)(x).size()
typedef long long ll;
typedef pair<int, int> pii;
typedef vector<int> vi;

struct PushRelabel {
  struct Edge {
    int dest, back;
    ll f, c;
  };
  vector<vector<Edge>> g;
  vector<ll> ec;
  vector<Edge*> cur;
  vector<vi> hs;
  vi H;
  PushRelabel(int n) : g(n), ec(n), cur(n), hs(2 * n), H(n) {}

  void addEdge(int s, int t, ll cap, ll rcap = 0) {
    if (s == t) return;
    g[s].push_back({t, sz(g[t]), 0, cap});
    g[t].push_back({s, sz(g[s]) - 1, 0, rcap});
  }

  void addFlow(Edge& e, ll f) {
    Edge& back = g[e.dest][e.back];
    if (!ec[e.dest] && f) hs[H[e.dest]].push_back(e.dest);
    e.f += f;
    e.c -= f;
    ec[e.dest] += f;
    back.f -= f;
    back.c += f;
    ec[back.dest] -= f;
  }
  ll calc(int s, int t) {
    int v = sz(g);
    H[s] = v;
    ec[t] = 1;
    vi co(2 * v);
    co[0] = v - 1;
    rep(i, 0, v) cur[i] = g[i].data();
    for (Edge& e : g[s]) addFlow(e, e.c);

    for (int hi = 0;;) {
      while (hs[hi].empty())
        if (!hi--) return -ec[s];
      int u = hs[hi].back();
      hs[hi].pop_back();
      while (ec[u] > 0)  // discharge u
        if (cur[u] == g[u].data() + sz(g[u])) {
          H[u] = 1e9;
          for (Edge& e : g[u])
            if (e.c && H[u] > H[e.dest] + 1) H[u] = H[e.dest] + 1, cur[u] = &e;
          if (++co[H[u]], !--co[hi] && hi < v)
            rep(i, 0, v) if (hi < H[i] && H[i] < v)-- co[H[i]], H[i] = v + 1;
          hi = H[u];
        } else if (cur[u]->c && H[u] == H[cur[u]->dest] + 1)
          addFlow(*cur[u], min(ec[u], cur[u]->c));
        else
          ++cur[u];
    }
  }
  bool leftOfMinCut(int a) { return H[a] >= sz(g); }
};
```

## Min-Cost Max-Flow

```cpp
class MCMF {
public:
  static constexpr int INF = 1e9;
  const int n;
  vector<tuple<int, int, int>> e;
  vector<vector<int>> g;
  vector<int> h, dis, pre;
  bool dijkstra(int s, int t) {
    dis.assign(n, INF);
    pre.assign(n, -1);
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> que;
    dis[s] = 0;
    que.emplace(0, s);
    while (!que.empty()) {
      auto [d, u] = que.top();
      que.pop();
      if (dis[u] != d) continue;
      for (int i : g[u]) {
        auto [v, f, c] = e[i];
        if (c > 0 && dis[v] > d + h[u] - h[v] + f) {
          dis[v] = d + h[u] - h[v] + f;
          pre[v] = i;
          que.emplace(dis[v], v);
        }
      }
    }
    return dis[t] != INF;
  }
  MCMF(int n) : n(n), g(n) {}
  void add_edge(int u, int v, int fee, int c) {
    g[u].push_back(e.size());
    e.emplace_back(v, fee, c);
    g[v].push_back(e.size());
    e.emplace_back(u, -fee, 0);
  }
  pair<ll, ll> max_flow(const int s, const int t) {
    int flow = 0, cost = 0;
    h.assign(n, 0);
    while (dijkstra(s, t)) {
      for (int i = 0; i < n; ++i) h[i] += dis[i];
      for (int i = t; i != s; i = get<0>(e[pre[i] ^ 1])) {
        --get<2>(e[pre[i]]);
        ++get<2>(e[pre[i] ^ 1]);
      }
      ++flow;
      cost += h[t];
    }
    return {flow, cost};
  }
};
```

## Max Cost Feasible Flow

```cpp
struct Edge {
  int from, to, cap, remain, cost;
};

struct MCMF {
  int n;
  vector<Edge> e;
  vector<vector<int>> g;
  vector<ll> d, pre;
  MCMF(int _n) : n(_n), g(n), d(n), pre(n) {}
  void add_edge(int u, int v, int c, int w) {
    g[u].push_back((int)e.size());
    e.push_back({u, v, c, c, w});
    g[v].push_back((int)e.size());
    e.push_back({v, u, 0, 0, -w});
  }
  pair<ll, ll> max_flow(int s, int t) {
    ll inf = 1e18;
    auto spfa = [&]() {
      fill(d.begin(), d.end(), -inf); // important!
      vector<int> f(n), seen(n);
      d[s] = 0, f[s] = 1e9;
      vector<int> q{s}, nq;
      for (; q.size(); swap(q, nq), nq.clear()) {
        for (auto& node : q) {
          seen[node] = false;
          for (auto& edge : g[node]) {
            int ne = e[edge].to, cost = e[edge].cost;
            if (!e[edge].remain || d[ne] >= d[node] + cost) continue;
            d[ne] = d[node] + cost, pre[ne] = edge;
            f[ne] = min(e[edge].remain, f[node]);
            if (!seen[ne]) seen[ne] = true, nq.push_back(ne);
          }
        }
      }
      return f[t];
    };
    ll flow = 0, cost = 0;
    while (int temp = spfa()) {
      if (d[t] < 0) break; // important!
      flow += temp, cost += temp * d[t];
      for (ll i = t; i != s; i = e[pre[i]].from) {
        e[pre[i]].remain -= temp, e[pre[i] ^ 1].remain += temp;
      }
    }
    return {flow, cost};
  }
};
```

## Heavy-Light Decomposition

```cpp
struct HeavyLight {
  int root = 0, n = 0;
  std::vector<int> parent, deep, hson, top, sz, dfn;
  HeavyLight(std::vector<std::vector<int>> &g, int _root)
      : root(_root), n(int(g.size())), parent(n), deep(n), hson(n, -1), top(n), sz(n), dfn(n, -1) {
    int cur = 0;
    std::function<int(int, int, int)> dfs = [&](int node, int fa, int dep) {
      deep[node] = dep, sz[node] = 1, parent[node] = fa;
      for (auto &ne : g[node]) {
        if (ne == fa) continue;
        sz[node] += dfs(ne, node, dep + 1);
        if (hson[node] == -1 || sz[ne] > sz[hson[node]]) hson[node] = ne;
      }
      return sz[node];
    };
    std::function<void(int, int)> dfs2 = [&](int node, int t) {
      top[node] = t, dfn[node] = cur++;
      if (hson[node] == -1) return;
      dfs2(hson[node], t);
      for (auto &ne : g[node]) {
        if (ne == parent[node] || ne == hson[node]) continue;
        dfs2(ne, ne);
      }
    };
    dfs(root, -1, 0), dfs2(root, root);
  }
  
  int lca(int x, int y) const {
    while (top[x] != top[y]) {
      if (deep[top[x]] < deep[top[y]]) swap(x, y);
      x = parent[top[x]];
    }
    return deep[x] < deep[y] ? x : y;
  }
  
  std::vector<std::array<int, 2>> get_dfn_path(int x, int y) const {
    std::array<std::vector<std::array<int, 2>>, 2> path;
    bool front = true;
    while (top[x] != top[y]) {
      if (deep[top[x]] > deep[top[y]]) swap(x, y), front = !front;
      path[front].push_back({dfn[top[y]], dfn[y] + 1});
      y = parent[top[y]];
    }
    if (deep[x] > deep[y]) swap(x, y), front = !front;
  
    path[front].push_back({dfn[x], dfn[y] + 1});
    std::reverse(path[1].begin(), path[1].end());
    for (const auto &[left, right] : path[1]) path[0].push_back({right, left});
    return path[0];
  }
  
  Node query_seg(int u, int v, const SegTree &seg) const {
    auto node = Node();
    for (const auto &[left, right] : get_dfn_path(u, v)) {
      if (left > right) {
        node = pull(node, rev(seg.query(right, left)));
      } else {
        node = pull(node, seg.query(left, right));
      }
    }
    return node;
  }
};
```

+ USAGE: 

```cpp
vector<ll> light(n);
SegTree heavy(n), form_parent(n);
// cin >> x >> y, x--, y--;
int z = lca(x, y);
while (x != z) {
  if (dfn[top[x]] <= dfn[top[z]]) {
    // [dfn[z], dfn[x]), from heavy
    heavy.modify(dfn[z], dfn[x], 1);
    break;
  }
  // x -> top[x];
  heavy.modify(dfn[top[x]], dfn[x], 1);
  light[parent[top[x]]] += a[top[x]];
  x = parent[top[x]];
}
while (y != z) {
  if (dfn[top[y]] <= dfn[top[z]]) {
    // (dfn[z], dfn[y]], from heavy
    form_parent.modify(dfn[z] + 1, dfn[y] + 1, 1);
    break;
  }
  // y -> top[y];
  form_parent.modify(dfn[top[y]], dfn[y] + 1, 1);
  y = parent[top[y]];
}
```

## General Unweight Graph Matching

+ Complexity: $O(n^3)$ (?)

```cpp
struct BlossomMatch {
  int n;
  vector<vector<int>> e;
  BlossomMatch(int _n) : n(_n), e(_n) {}
  void add_edge(int u, int v) { e[u].push_back(v), e[v].push_back(u); }
  vector<int> find_matching() {
    vector<int> match(n, -1), vis(n), link(n), f(n), dep(n);
    function<int(int)> find = [&](int x) { return f[x] == x ? x : (f[x] = find(f[x])); };
    auto lca = [&](int u, int v) {
      u = find(u), v = find(v);
      while (u != v) {
        if (dep[u] < dep[v]) swap(u, v);
        u = find(link[match[u]]);
      }
      return u;
    };
    queue<int> que;
    auto blossom = [&](int u, int v, int p) {
      while (find(u) != p) {
        link[u] = v, v = match[u];
        if (vis[v] == 0) vis[v] = 1, que.push(v);
        f[u] = f[v] = p, u = link[v];
      }
    };
    // find an augmenting path starting from u and augment (if exist)
    auto augment = [&](int node) {
      while (!que.empty()) que.pop();
      iota(f.begin(), f.end(), 0);
      // vis = 0 corresponds to inner vertices, vis = 1 corresponds to outer vertices
      fill(vis.begin(), vis.end(), -1);
      que.push(node);
      vis[node] = 1, dep[node] = 0;
      while (!que.empty()) {
        int u = que.front();
        que.pop();
        for (auto v : e[u]) {
          if (vis[v] == -1) {
            vis[v] = 0, link[v] = u, dep[v] = dep[u] + 1;
            // found an augmenting path
            if (match[v] == -1) {
              for (int x = v, y = u, temp; y != -1; x = temp, y = x == -1 ? -1 : link[x]) {
                temp = match[y], match[x] = y, match[y] = x;
              }
              return;
            }
            vis[match[v]] = 1, dep[match[v]] = dep[u] + 2;
            que.push(match[v]);
          } else if (vis[v] == 1 && find(v) != find(u)) {
            // found a blossom
            int p = lca(u, v);
            blossom(u, v, p), blossom(v, u, p);
          }
        }
      }
    };
    // find a maximal matching greedily (decrease constant)
    auto greedy = [&]() {
      for (int u = 0; u < n; ++u) {
        if (match[u] != -1) continue;
        for (auto v : e[u]) {
          if (match[v] == -1) {
            match[u] = v, match[v] = u;
            break;
          }
        }
      }
    };
    greedy();
    for (int u = 0; u < n; ++u)
      if (match[u] == -1) augment(u);
    return match;
  }
};
```

## Maximum Bipartite Matching

+ Needs dinic, complexity $\approx O(n + m\sqrt{n})$

```cpp
struct BipartiteMatch {
  int l, r;
  Dinic dinic = Dinic(0);
  BipartiteMatch(int _l, int _r) : l(_l), r(_r) {
    dinic = Dinic(l + r + 2);
    for (int i = 1; i <= l; i++) dinic.add_edge(0, i, 1);
    for (int i = 1; i <= r; i++) dinic.add_edge(l + i, l + r + 1, 1);
  }
  void add_edge(int u, int v) { dinic.add_edge(u + 1, l + v + 1, 1); }
  ll max_matching() { return dinic.max_flow(0, l + r + 1); }
};
```

## 2-SAT and Strongly Connected Components

```cpp
void scc(vector<vector<int>>& g, int* idx) {
  int n = g.size(), ct = 0;
  int out[n];
  vector<int> ginv[n];
  memset(out, -1, sizeof out);
  memset(idx, -1, n * sizeof(int));
  function<void(int)> dfs = [&](int cur) {
    out[cur] = INT_MAX;
    for(int v : g[cur]) {
      ginv[v].push_back(cur);
      if(out[v] == -1) dfs(v);
    }
    ct++; out[cur] = ct;
  };
  vector<int> order;
  for(int i = 0; i < n; i++) {
    order.push_back(i);
    if(out[i] == -1) dfs(i);
  }
  sort(order.begin(), order.end(), [&](int& u, int& v) {
    return out[u] > out[v];
  });
  ct = 0;
  stack<int> s;
  auto dfs2 = [&](int start) {
    s.push(start);
    while(!s.empty()) {
      int cur = s.top();
      s.pop();
      idx[cur] = ct;
      for(int v : ginv[cur])
        if(idx[v] == -1) s.push(v);
    }
  };
  for(int v : order) {
    if(idx[v] == -1) {
      dfs2(v);
      ct++;
    }
  }
}

// 0 => impossible, 1 => possible
pair<int,vector<int>> sat2(int n, vector<pair<int,int>>& clauses) {
  vector<int> ans(n);
  vector<vector<int>> g(2*n + 1);
  for(auto [x, y] : clauses) {
    x = x < 0 ? -x + n : x;
    y = y < 0 ? -y + n : y;
    int nx = x <= n ? x + n : x - n;
    int ny = y <= n ? y + n : y - n;
    g[nx].push_back(y);
    g[ny].push_back(x);
  }
  int idx[2*n + 1];
  scc(g, idx);
  for(int i = 1; i <= n; i++) {
    if(idx[i] == idx[i + n]) return {0, {}};
    ans[i - 1] = idx[i + n] < idx[i];
  }
  return {1, ans};
}
```

## Enumerating Triangles

+ Complexity: $O(n + m\sqrt{m})$

```cpp
void enumerate_triangles(vector<pair<int,int>>& edges, function<void(int,int,int)> f) {
  int n = 0;
  for(auto [u, v] : edges) n = max({n, u + 1, v + 1});
  vector<int> deg(n);
  vector<int> g[n];
  for(auto [u, v] : edges) {
    deg[u]++;
    deg[v]++;
  }
  for(auto [u, v] : edges) {
    if(u == v) continue;
    if(deg[u] > deg[v] || (deg[u] == deg[v] && u > v))
      swap(u, v);
    g[u].push_back(v);
  }
  vector<int> flag(n);
  for(int i = 0; i < n; i++) {
    for(int v : g[i]) flag[v] = 1;
    for(int v : g[i]) for(int u : g[v]) {
      if(flag[u]) f(i, v, u);
    }
    for(int v : g[i]) flag[v] = 0;
  }
}
```

## Tarjan

+ shrink all circles into points (2-edge-connected-component)

```cpp
int cnt = 0, now = 0;
vector<ll> dfn(n, -1), low(n), belong(n, -1), stk;
function<void(ll, ll)> tarjan = [&](ll node, ll fa) {
  dfn[node] = low[node] = now++, stk.push_back(node);
  for (auto& ne : g[node]) {
    if (ne == fa) continue;
    if (dfn[ne] == -1) {
      tarjan(ne, node);
      low[node] = min(low[node], low[ne]);
    } else if (belong[ne] == -1) {
      low[node] = min(low[node], dfn[ne]);
    }
  }
  if (dfn[node] == low[node]) {
    while (true) {
      auto v = stk.back();
      belong[v] = cnt;
      stk.pop_back();
      if (v == node) break;
    }
    ++cnt;
  }
};
```

+ 2-vertex-connected-component / Block forest

```cpp
int cnt = 0, now = 0;
vector<vector<ll>> e1(n);
vector<ll> dfn(n, -1), low(n), stk;
function<void(ll)> tarjan = [&](ll node) {
  dfn[node] = low[node] = now++, stk.push_back(node);
  for (auto& ne : g[node]) {
    if (dfn[ne] == -1) {
      tarjan(ne);
      low[node] = min(low[node], low[ne]);
      if (low[ne] == dfn[node]) {
        e1.push_back({});
        while (true) {
          auto x = stk.back();
          stk.pop_back();
          e1[n + cnt].push_back(x);
          // e1[x].push_back(n + cnt); // undirected
          if (x == ne) break;
        }
        e1[node].push_back(n + cnt);
        // e1[n + cnt].push_back(node); // undirected
        cnt++;
      }
    } else {
      low[node] = min(low[node], dfn[ne]);
    }
  }
};
```

## Kruskal reconstruct tree

```cpp
int _n, m;
cin >> _n >> m; // _n: # of node, m: # of edge
int n = 2 * _n - 1; // root: n-1
vector<array<int, 3>> edges(m);
for (auto& [w, u, v] : edges) {
  cin >> u >> v >> w, u--, v--;
}
sort(edges.begin(), edges.end());
vector<int> p(n);
iota(p.begin(), p.end(), 0);
function<int(int)> find = [&](int x) { return p[x] == x ? x : (p[x] = find(p[x])); };
auto merge = [&](int x, int y) { p[find(x)] = find(y); };
vector<vector<int>> g(n);
vector<int> val(m);
val.reserve(n);
for (auto [w, u, v] : edges) {
  u = find(u), v = find(v);
  if (u == v) continue;
  val.push_back(w);
  int node = (int)val.size() - 1;
  g[node].push_back(u), g[node].push_back(v);
  merge(u, node), merge(v, node);
}
```

## centroid decomposition
```cpp
vector<char> res(n), seen(n), sz(n);
function<int(int, int)> get_size = [&](int node, int fa) {
  sz[node] = 1;
  for (auto& ne : g[node]) {
    if (ne == fa || seen[ne]) continue;
    sz[node] += get_size(ne, node);
  }
  return sz[node];
};
function<int(int, int, int)> find_centroid = [&](int node, int fa, int t) {
  for (auto& ne : g[node])
    if (ne != fa && !seen[ne] && sz[ne] > t / 2) return find_centroid(ne, node, t);
  return node;
};
function<void(int, char)> solve = [&](int node, char cur) {
  get_size(node, -1); auto c = find_centroid(node, -1, sz[node]);
  seen[c] = 1, res[c] = cur;
  for (auto& ne : g[c]) {
    if (seen[ne]) continue;
    solve(ne, char(cur + 1)); // we can pass c here to build tree
  }
};
```

## virtual tree
```
map<int, vector<int>> gg; vector<int> stk{0};
auto add = [&](int x, int y) { gg[x].push_back(y), gg[y].push_back(x); };
for (int i = 0; i < k; i++) {
  if (a[i] != 0) {
    int p = lca(a[i], stk.back());
    if (p != stk.back()) {
      while (dfn[p] < dfn[stk[int(stk.size()) - 2]]) {
        add(stk.back(), stk[int(stk.size()) - 2]);
        stk.pop_back();
      }
      add(p, stk.back()), stk.pop_back();
      if (dfn[p] > dfn[stk.back()]) stk.push_back(p);
    }
    stk.push_back(a[i]);
  }
}
while (stk.size() > 1) {
  if (stk.back() != 0) {
    add(stk.back(), stk[int(stk.size()) - 2]);
    stk.pop_back();
  }
}
```
