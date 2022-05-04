# String

## AC Automaton

```cpp
struct AC_automaton {
  int sz = 26;
  vector<vector<int>> e = {vector<int>(sz)};  // vector is faster than unordered_map
  vector<int> fail = {0}, end = {0};
  vector<int> fast = {0};  // closest end

  int insert(string& s) {
    int p = 0;
    for (auto c : s) {
      c -= 'a';
      if (!e[p][c]) {
        e.emplace_back(sz);
        fail.emplace_back();
        end.emplace_back();
        fast.emplace_back();
        e[p][c] = (int)e.size() - 1;
      }
      p = e[p][c];
    }
    end[p] += 1;
    return p;
  }

  void build() {
    queue<int> q;
    for (int i = 0; i < sz; i++)
      if (e[0][i]) q.push(e[0][i]);
    while (!q.empty()) {
      int p = q.front();
      q.pop();
      fast[p] = end[p] ? p : fast[fail[p]];
      for (int i = 0; i < sz; i++) {
        if (e[p][i]) {
          fail[e[p][i]] = e[fail[p]][i];
          q.push(e[p][i]);
        } else {
          e[p][i] = e[fail[p]][i];
        }
      }
    }
  }
};
```

## KMP

+ nex[i]: length of longest common prefix & suffix for pat[0..i]

```cpp
vector<int> get_next(vector<int> &pat) {
  int m = (int)pat.size();
  vector<int> nex(m);
  for (int i = 1, j = 0; i < m; i++) {
    while (j && pat[j] != pat[i]) j = nex[j - 1];
    if (pat[j] == pat[i]) j++;
    nex[i] = j;
  }
  return nex;
}
```

+ kmp match for txt and pat

```cpp
auto nex = get_next(pat);
for (int i = 0, j = 0; i < n; i++) {
  while (j && pat[j] != txt[i]) j = nex[j - 1];
  if (pat[j] == txt[i]) j++;
  if (j == m) {
    // do what you want with the match
    // start index is `i - m + 1`
    j = nex[j - 1];
  }
}
```

## Z function

+ z[i]: length of longest common prefix of s and s[i:]

```cpp
vector<int> z_function(string s) {
  int n = (int)s.size();
  vector<int> z(n);
  for (int i = 1, l = 0, r = 0; i < n; ++i) {
    if (i <= r) z[i] = min(r - i + 1, z[i - l]);
    while (i + z[i] < n && s[z[i]] == s[i + z[i]]) ++z[i];
    if (i + z[i] - 1 > r) l = i, r = i + z[i] - 1;
  }
  return z;
}
```

## General Suffix Automaton

```cpp
constexpr int SZ = 26;

struct GSAM {
  vector<vector<int>> e = {vector<int>(SZ)};  // the labeled edges from node i
  vector<int> parent = {-1};                  // the parent of i
  vector<int> length = {0};                   // the length of the longest string

  GSAM(int n) { e.reserve(2 * n), parent.reserve(2 * n), length.reserve(2 * n); };
  int extend(int c, int p) {  // character, last
    bool f = true;            // if already exist
    int r = 0;                // potential new node
    if (!e[p][c]) {           // only extend when not exist
      f = false;
      e.push_back(vector<int>(SZ));
      parent.push_back(0);
      length.push_back(length[p] + 1);
      r = (int)e.size() - 1;
      for (; ~p && !e[p][c]; p = parent[p]) e[p][c] = r;  // update parents
    }
    if (f || ~p) {
      int q = e[p][c];
      if (length[q] == length[p] + 1) {
        if (f) return q;
        parent[r] = q;
      } else {
        e.push_back(e[q]);
        parent.push_back(parent[q]);
        length.push_back(length[p] + 1);
        int qq = parent[q] = (int)e.size() - 1;
        for (; ~p && e[p][c] == q; p = parent[p]) e[p][c] = qq;
        if (f) return qq;
        parent[r] = qq;
      }
    }
    return r;
  }
};
```

+ Topo sort on GSAM

```cpp
ll sz = gsam.e.size();
vector<int> c(sz + 1);
vector<int> order(sz);
for (int i = 1; i < sz; i++) c[gsam.length[i]]++;
for (int i = 1; i < sz; i++) c[i] += c[i - 1];
for (int i = 1; i < sz; i++) order[c[gsam.length[i]]--] = i;
reverse(order.begin(), order.end()); // reverse so that large len to small
```

+ can be used as an ordinary SAM
+ USAGE (the number of distinct substring)

```cpp
int main() {
  int n, last = 0;
  string s;
  cin >> n;
  auto a = GSAM();
  for (int i = 0; i < n; i++) {
    cin >> s;
    last = 0;  // reset last
    for (auto&& c : s) last = a.extend(c, last);
  }
  ll ans = 0;
  for (int i = 1; i < a.e.size(); i++) {
    ans += a.length[i] - a.length[a.parent[i]];
  }
  cout << ans << endl;
  return 0;
}
```

## Manacher

```cpp
string longest_palindrome(string& s) {
  // init "abc" -> "^$a#b#c$"
  vector<char> t{'^', '#'};
  for (char c : s) t.push_back(c), t.push_back('#');
  t.push_back('$');
  // manacher
  int n = t.size(), r = 0, c = 0;
  vector<int> p(n, 0);
  for (int i = 1; i < n - 1; i++) {
    if (i < r + c) p[i] = min(p[2 * c - i], r + c - i);
    while (t[i + p[i] + 1] == t[i - p[i] - 1]) p[i]++;
    if (i + p[i] > r + c) r = p[i], c = i;
  }
	// s[i] -> p[2 * i + 2] (even), p[2 * i + 2] (odd)
  // output answer
  int index = 0;
  for (int i = 0; i < n; i++)
    if (p[index] < p[i]) index = i;
  return s.substr((index - p[index]) / 2, p[index]);
}
```

## Lyndon

+ def: suf(s) > s

```cpp
void duval(const string &s) {
  int n = (int)s.size();
  for (int i = 0; i < n;) {
    int j = i, k = i + 1;
    for (; j < n && s[j] <= s[k]; j++, k++)
      if (s[j] < s[k]) j = i - 1;

    while (i <= j) {
      // cout << s.substr(i, k - j) << '\n';
      i += k - j;
    }
  }
}

int main() {
  string s;
  cin >> s;
  duval(s);
}
```
