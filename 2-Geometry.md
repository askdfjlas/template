# Geometry

## Basic stuff

```cpp
using ll = long long;
using ld = long double;

constexpr auto eps = 1e-8;
const auto PI = acos(-1);
int sgn(ld x) { return (abs(x) <= eps) ? 0 : (x < 0 ? -1 : 1); }

struct Point {
  ld x = 0, y = 0;
  Point() = default;
  Point(ld _x, ld _y) : x(_x), y(_y) {}
  bool operator<(const Point &p) const { return !sgn(p.x - x) ? sgn(y - p.y) < 0 : x < p.x; }
  bool operator==(const Point &p) const { return !sgn(p.x - x) && !sgn(p.y - y); }
  Point operator+(const Point &p) const { return {x + p.x, y + p.y}; }
  Point operator-(const Point &p) const { return {x - p.x, y - p.y}; }
  Point operator*(ld a) const { return {x * a, y * a}; }
  Point operator/(ld a) const { return {x / a, y / a}; }
  auto operator*(const Point &p) const { return x * p.x + y * p.y; }  // dot
  auto operator^(const Point &p) const { return x * p.y - y * p.x; }  // cross
  friend auto &operator>>(istream &i, Point &p) { return i >> p.x >> p.y; }
  friend auto &operator<<(ostream &o, Point p) { return o << p.x << ' ' << p.y; }
};

struct Line {
  Point s = {0, 0}, e = {0, 0};
  Line() = default;
  Line(Point _s, Point _e) : s(_s), e(_e) {}
  friend auto &operator>>(istream &i, Line &l) { return i >> l.s >> l.e; }  // ((x1, y1), (x2, y2)
};

struct Segment : Line {
  using Line::Line;
};

struct Circle {
  Point o = {0, 0};
  ld r = 0;
  Circle() = default;
  Circle(Point _o, ld _r) : o(_o), r(_r) {}
};
```

```cpp
auto dist2(const Point &a) { return a * a; }
auto dist2(const Point &a, const Point &b) { return dist2(a - b); }
auto dist(const Point &a) { return sqrt(dist2(a)); }
auto dist(const Point &a, const Point &b) { return sqrt(dist2(a - b)); }
auto dist(const Point &a, const Line &l) { return abs((a - l.s) ^ (l.e - l.s)) / dist(l.s, l.e); }
auto dist(const Point &p, const Segment &l) {
  if (l.s == l.e) return dist(p, l.s);
  auto d = dist2(l.s, l.e), t = min(d, max((ld)0, (p - l.s) * (l.e - l.s)));
  return dist((p - l.s) * d, (l.e - l.s) * t) / d;
}
/* Needs is_intersect
auto dist(const Segment &l1, const Segment &l2) {
  if (is_intersect(l1, l2)) return (ld)0;
  return min({dist(l1.s, l2), dist(l1.e, l2), dist(l2.s, l1), dist(l2.e, l1)});
} */

Point perp(const Point &p) { return Point(-p.y, p.x); }

auto rad(const Point &p) { return atan2(p.y, p.x); }
```

## Transformation

```cpp
Point project(const Point &p, const Line &l) {
  return l.s + ((l.e - l.s) * ((l.e - l.s) * (p - l.s))) / dist2(l.e - l.s);
}

Point reflect(const Point &p, const Line &l) {
  return project(p, l) * 2 - p;
}

Point dilate(const Point &p, ld scale_x = 1, ld scale_y = 1) { return Point(p.x * scale_x, p.y * scale_y); }
Line dilate(const Line &l, ld scale_x = 1, ld scale_y = 1) { return Line(dilate(l.s, scale_x, scale_y), dilate(l.e, scale_x, scale_y)); }
Segment dilate(const Segment &l, ld scale_x = 1, ld scale_y = 1) { return Segment(dilate(l.s, scale_x, scale_y), dilate(l.e, scale_x, scale_y)); }
vector<Point> dilate(const vector<Point> &p, ld scale_x = 1, ld scale_y = 1) {
  int n = p.size();
  vector<Point> res(n);
  for (int i = 0; i < n; i++)
    res[i] = dilate(p[i], scale_x, scale_y);
  return res;
}

Point rotate(const Point &p, ld a) { return Point(p.x * cos(a) - p.y * sin(a), p.x * sin(a) + p.y * cos(a)); }
Line rotate(const Line &l, ld a) { return Line(rotate(l.s, a), rotate(l.e, a)); }
Segment rotate(const Segment &l, ld a) { return Segment(rotate(l.s, a), rotate(l.e, a)); }
Circle rotate(const Circle &c, ld a) { return Circle(rotate(c.o, a), c.r); }
vector<Point> rotate(const vector<Point> &p, ld a) {
  int n = p.size();
  vector<Point> res(n);
  for (int i = 0; i < n; i++)
    res[i] = rotate(p[i], a);
  return res;
}

Point translate(const Point &p, ld dx = 0, ld dy = 0) { return Point(p.x + dx, p.y + dy); }
Line translate(const Line &l, ld dx = 0, ld dy = 0) { return Line(translate(l.s, dx, dy), translate(l.e, dx, dy)); }
Segment translate(const Segment &l, ld dx = 0, ld dy = 0) { return Segment(translate(l.s, dx, dy), translate(l.e, dx, dy)); }
Circle translate(const Circle &c, ld dx = 0, ld dy = 0) { return Circle(translate(c.o, dx, dy), c.r); }
vector<Point> translate(const vector<Point> &p, ld dx = 0, ld dy = 0) {
  int n = p.size();
  vector<Point> res(n);
  for (int i = 0; i < n; i++)
    res[i] = translate(p[i], dx, dy);
  return res;
}
```

## Relation

```cpp
enum class Relation { SEPARATE, EX_TOUCH, OVERLAP, IN_TOUCH, INSIDE };
Relation get_relation(const Circle &a, const Circle &b) {
  auto c1c2 = dist(a.o, b.o);
  auto r1r2 = a.r + b.r, diff = abs(a.r - b.r);
  if (sgn(c1c2 - r1r2) > 0) return Relation::SEPARATE;
  if (sgn(c1c2 - r1r2) == 0) return Relation::EX_TOUCH;
  if (sgn(c1c2 - diff) > 0) return Relation::OVERLAP;
  if (sgn(c1c2 - diff) == 0) return Relation::IN_TOUCH;
  return Relation::INSIDE;
}

auto get_cos_from_triangle(ld a, ld b, ld c) { return (a * a + b * b - c * c) / (2.0 * a * b); }

bool on_line(const Line &l, const Point &p) { return !sgn((l.s - p) ^ (l.e - p)); }

bool on_segment(const Segment &l, const Point &p) {
  return !sgn((l.s - p) ^ (l.e - p)) && sgn((l.s - p) * (l.e - p)) <= 0;
}

bool on_segment2(const Segment &l, const Point &p) { // assume p on Line l
  if (l.s == p || l.e == p) return true;
  if (min(l.s, l.e) < p && p < max(l.s, l.e)) return true;
  return false;
}

bool is_parallel(const Line &a, const Line &b) { return !sgn((a.s - a.e) ^ (b.s - b.e)); }
bool is_orthogonal(const Line &a, const Line &b) { return !sgn((a.s - a.e) * (b.s - b.e)); }

int is_intersect(const Segment &a, const Segment &b) {
  auto d1 = sgn((a.e - a.s) ^ (b.s - a.s)), d2 = sgn((a.e - a.s) ^ (b.e - a.s));
  auto d3 = sgn((b.e - b.s) ^ (a.s - b.s)), d4 = sgn((b.e - b.s) ^ (a.e - b.s));
  if (d1 * d2 < 0 && d3 * d4 < 0) return 2;  // intersect at non-end point
  return (d1 == 0 && sgn((b.s - a.s) * (b.s - a.e)) <= 0) ||
         (d2 == 0 && sgn((b.e - a.s) * (b.e - a.e)) <= 0) ||
         (d3 == 0 && sgn((a.s - b.s) * (a.s - b.e)) <= 0) ||
         (d4 == 0 && sgn((a.e - b.s) * (a.e - b.e)) <= 0);
}

int is_intersect(const Line &a, const Segment &b) {
  auto d1 = sgn((a.e - a.s) ^ (b.s - a.s)), d2 = sgn((a.e - a.s) ^ (b.e - a.s));
  if (d1 * d2 < 0) return 2;  // intersect at non-end point
  return d1 == 0 || d2 == 0;
}

Point intersect(const Line &a, const Line &b) {
  auto u = a.e - a.s, v = b.e - b.s;
  auto t = ((b.s - a.s) ^ v) / (u ^ v);
  return a.s + u * t;
}

int is_intersect(const Circle &c, const Line &l) {
  auto d = dist(c.o, l);
  return sgn(d - c.r) < 0 ? 2 : !sgn(d - c.r);
}

vector<Point> intersect(const Circle &a, const Circle &b) {
  auto relation = get_relation(a, b);
  if (relation == Relation::INSIDE || relation == Relation::SEPARATE) return {};
  auto vec = b.o - a.o;
  auto d2 = dist2(vec);
  auto p = (d2 + a.r * a.r - b.r * b.r) / ((long double)2 * d2), h2 = a.r * a.r - p * p * d2;
  auto mid = a.o + vec * p, per = perp(vec) * sqrt(max((long double)0, h2) / d2);
  if (relation == Relation::OVERLAP)
    return {mid + per, mid - per};
  else
    return {mid};
}

vector<Point> intersect(const Circle &c, const Line &l) {
  if (!is_intersect(c, l)) return {};
  auto v = l.e - l.s, t = v / dist(v);
  Point a = l.s + t * ((c.o - l.s) * t);
  auto d = sqrt(max((ld)0, c.r * c.r - dist2(c.o, a)));
  if (!sgn(d)) return {a};
  return {a - t * d, a + t * d};
}

int in_poly(const vector<Point> &p, const Point &a) {
  int cnt = 0, n = (int)p.size();
  for (int i = 0; i < n; i++) {
    auto q = p[(i + 1) % n];
    if (on_segment(Segment(p[i], q), a)) return 1;  // on the edge of the polygon
    cnt ^= ((a.y < p[i].y) - (a.y < q.y)) * ((p[i] - a) ^ (q - a)) > 0;
  }
  return cnt ? 2 : 0;
}

int is_intersect(const vector<Point> &p, const Line &a) {
  // 1: touching, >=2: intersect count
  int cnt = 0, edge_cnt = 0, n = (int)p.size();
  for (int i = 0; i < n; i++) {
    auto q = p[(i + 1) % n];
    if (on_line(a, p[i]) && on_line(a, q)) return -1;  // infinity
    auto t = is_intersect(a, Segment(p[i], q));
    (t == 1) && edge_cnt++, (t == 2) && cnt++;
  }
  return cnt + edge_cnt / 2;
}

vector<Point> tangent(const Circle &c, const Point &p) {
  auto d = dist(c.o, p), l = c.r * c.r / d, h = sqrt(c.r * c.r - l * l);
  auto v = (p - c.o) / d;
  return {c.o + v * l + perp(v) * h, c.o + v * l - perp(v) * h};
}

Circle get_circumscribed(const Point &a, const Point &b, const Point &c) {
  Line u((a + b) / 2, ((a + b) / 2) + perp(b - a));
  Line v((b + c) / 2, ((b + c) / 2) + perp(c - b));
  auto o = intersect(u, v);
  return Circle(o, dist(o, a));
}

Circle get_inscribed(const Point &a, const Point &b, const Point &c) {
  auto l1 = dist(b - c), l2 = dist(c - a), l3 = dist(a - b);
  Point o = (a * l1 + b * l2 + c * l3) / (l1 + l2 + l3);
  return Circle(o, dist(o, Line(a, b)));
}

pair<ld, ld> get_centroid(const vector<Point> &p) {
  int n = (int)p.size();
  ld x = 0, y = 0, sum = 0;
  auto a = p[0], b = p[1];
  for (int i = 2; i < n; i++) {
    auto c = p[i];
    auto s = area({a, b, c});
    sum += s;
    x += s * (a.x + b.x + c.x);
    y += s * (a.y + b.y + c.y);
    swap(b, c);
  }
  return {x / (3 * sum), y / (3 * sum)};
}
```

## Area
```cpp
auto area(const vector<Point> &p) {
  int n = (int)p.size();
  long double area = 0;
  for (int i = 0; i < n; i++) area += p[i] ^ p[(i + 1) % n];
  return area / 2.0;
}

auto area(const Point &a, const Point &b, const Point &c) {
  return ((long double)((b - a) ^ (c - a))) / 2.0;
}

auto area2(const Point &a, const Point &b, const Point &c) { return (b - a) ^ (c - a); }

auto area_intersect(const Circle &c, const vector<Point> &ps) {
  int n = (int)ps.size();
  auto arg = [&](const Point &p, const Point &q) { return atan2(p ^ q, p * q); };
  auto tri = [&](const Point &p, const Point &q) {
    auto r2 = c.r * c.r / (long double)2;
    auto d = q - p;
    auto a = d * p / dist2(d), b = (dist2(p) - c.r * c.r) / dist2(d);
    long double det = a * a - b;
    if (sgn(det) <= 0) return arg(p, q) * r2;
    auto s = max((long double)0, -a - sqrt(det)), t = min((long double)1, -a + sqrt(det));
    if (sgn(t) < 0 || sgn(1 - s) <= 0) return arg(p, q) * r2;
    auto u = p + d * s, v = p + d * t;
    return arg(p, u) * r2 + (u ^ v) / 2 + arg(v, q) * r2;
  };
  long double sum = 0;
  for (int i = 0; i < n; i++) sum += tri(ps[i] - c.o, ps[(i + 1) % n] - c.o);
  return sum;
}

auto adaptive_simpson(ld _l, ld _r, function<ld(ld)> f) {
  auto simpson = [&](ld l, ld r) { return (r - l) * (f(l) + 4 * f((l + r) / 2) + f(r)) / 6; };
  function<ld(ld, ld, ld)> asr = [&](ld l, ld r, ld s) {
    auto mid = (l + r) / 2;
    auto left = simpson(l, mid), right = simpson(mid, r);
    if (!sgn(left + right - s)) return left + right;
    return asr(l, mid, left) + asr(mid, r, right);
  };
  return asr(_l, _r, simpson(_l, _r));
}

vector<Point> half_plane_intersect(vector<Line> &L) {
  int n = (int)L.size(), l = 0, r = 0;  // [left, right]
  sort(L.begin(), L.end(),
       [](const Line &a, const Line &b) { return rad(a.s - a.e) < rad(b.s - b.e); });
  vector<Point> p(n), res;
  vector<Line> q(n);
  q[0] = L[0];
  for (int i = 1; i < n; i++) {
    while (l < r && sgn((L[i].e - L[i].s) ^ (p[r - 1] - L[i].s)) <= 0) r--;
    while (l < r && sgn((L[i].e - L[i].s) ^ (p[l] - L[i].s)) <= 0) l++;
    q[++r] = L[i];
    if (sgn((q[r].e - q[r].s) ^ (q[r - 1].e - q[r - 1].s)) == 0) {
      r--;
      if (sgn((q[r].e - q[r].s) ^ (L[i].s - q[r].s)) > 0) q[r] = L[i];
    }
    if (l < r) p[r - 1] = intersect(q[r - 1], q[r]);
  }
  while (l < r && sgn((q[l].e - q[l].s) ^ (p[r - 1] - q[l].s)) <= 0) r--;
  if (r - l <= 1) return {};
  p[r] = intersect(q[r], q[l]);
  return vector<Point>(p.begin() + l, p.begin() + r + 1);
}
```

## Convex

```cpp
vector<Point> get_convex(vector<Point> &points, bool allow_collinear = false) {
  // strict, no repeat, two pass
  sort(points.begin(), points.end());
  points.erase(unique(points.begin(), points.end()), points.end());
  vector<Point> L, U;
  for (auto &t : points) {
    for (ll sz = L.size(); sz > 1 && (sgn((t - L[sz - 2]) ^ (L[sz - 1] - L[sz - 2])) >= 0);
         L.pop_back(), sz = L.size()) {
    }
    L.push_back(t);
  }
  for (auto &t : points) {
    for (ll sz = U.size(); sz > 1 && (sgn((t - U[sz - 2]) ^ (U[sz - 1] - U[sz - 2])) <= 0);
         U.pop_back(), sz = U.size()) {
    }
    U.push_back(t);
  }
  // contain repeats if all collinear, use a set to remove repeats
  if (allow_collinear) {
    for (int i = (int)U.size() - 2; i >= 1; i--) L.push_back(U[i]);
  } else {
    set<Point> st(L.begin(), L.end());
    for (int i = (int)U.size() - 2; i >= 1; i--) {
      if (st.count(U[i]) == 0) L.push_back(U[i]), st.insert(U[i]);
    }
  }
  return L;
}

vector<Point> get_convex2(vector<Point> &points, bool allow_collinear = false) {  // strict, no repeat, one pass
  nth_element(points.begin(), points.begin(), points.end());
  sort(points.begin() + 1, points.end(), [&](const Point &a, const Point &b) {
    int rad_diff = sgn((a - points[0]) ^ (b - points[0]));
    return !rad_diff ? (dist2(a - points[0]) < dist2(b - points[0])) : (rad_diff > 0);
  });
  if (allow_collinear) {
    int i = (int)points.size() - 1;
    while (i >= 0 && !sgn((points[i] - points[0]) ^ (points[i] - points.back()))) i--;
    reverse(points.begin() + i + 1, points.end());
  }
  vector<Point> hull;
  for (auto &t : points) {
    for (ll sz = hull.size();
         sz > 1 && (sgn((t - hull[sz - 2]) ^ (hull[sz - 1] - hull[sz - 2])) >= allow_collinear);
         hull.pop_back(), sz = hull.size()) {
    }
    hull.push_back(t);
  }
  return hull;
}

vector<Point> get_convex_safe(vector<Point> points, bool allow_collinear = false) {
  return get_convex(points, allow_collinear);
}

vector<Point> get_convex2_safe(vector<Point> points, bool allow_collinear = false) {
  return get_convex2(points, allow_collinear);
}

bool is_convex(const vector<Point> &p, bool allow_collinear = false) {
  int n = p.size();
  int lo = 1, hi = -1;
  for (int i = 0; i < n; i++) {
    int cur = sgn((p[(i + 2) % n] - p[(i + 1) % n]) ^ (p[(i + 1) % n] - p[i]));
    lo = min(lo, cur); hi = max(hi, cur);
  }
  return allow_collinear ? (hi - lo) < 2 : (lo == hi && lo);
}

auto rotating_calipers(const vector<Point> &hull) {
  // use get_convex2
  int n = (int)hull.size();  // return the square of longest dist
  assert(n > 1);
  if (n <= 2) return dist2(hull[0], hull[1]);
  ld res = 0;
  for (int i = 0, j = 2; i < n; i++) {
    auto d = hull[i], e = hull[(i + 1) % n];
    while (area2(d, e, hull[j]) < area2(d, e, hull[(j + 1) % n])) j = (j + 1) % n;
    res = max(res, max(dist2(d, hull[j]), dist2(e, hull[j])));
  }
  return res;
}

// Find polygon cut to the left of l
vector<Point> convex_cut(const vector<Point> &p, const Line &l) {
  int n = p.size();
  vector<Point> cut;
  for (int i = 0; i < n; i++) {
    auto a = p[i], b = p[(i + 1) % n];
    if (sgn((l.e - l.s) ^ (a - l.s)) >= 0)
      cut.push_back(a);
    if (sgn((l.e - l.s) ^ (a - l.s)) * sgn((l.e - l.s) ^ (b - l.s)) == -1)
      cut.push_back(intersect(Line(a, b), l));
  }
  return cut;
}

// Sort by angle in range [0, 2pi)
template <class RandomIt>
void polar_sort(RandomIt first, RandomIt last, Point origin = Point(0, 0)) {
  auto get_quad = [&](const Point& p) {
    Point diff = p - origin;
    if (diff.x > 0 && diff.y >= 0) return 1;
    if (diff.x <= 0 && diff.y > 0) return 2;
    if (diff.x < 0 && diff.y <= 0) return 3;
    return 4;
  };
  auto polar_cmp = [&](const Point& p1, const Point& p2) {
    int q1 = get_quad(p1), q2 = get_quad(p2);
    if (q1 != q2) return q1 < q2;
    return ((p1 - origin) ^ (p2 - origin)) > 0;
  };
  sort(first, last, polar_cmp);
}
```

## Basic 3D
```cpp
using ll = long long;
using ld = long double;

constexpr auto eps = 1e-8;
const auto PI = acos(-1);
int sgn(ld x) { return (abs(x) <= eps) ? 0 : (x < 0 ? -1 : 1); }

struct Point3D {
  ld x = 0, y = 0, z = 0;
  Point3D() = default;
  Point3D(ld _x, ld _y, ld _z) : x(_x), y(_y), z(_z) {}
  bool operator<(const Point3D &p) const { return !sgn(p.x - x) ? (!sgn(p.y - y) ? sgn(p.z - z) < 0 : y < p.y) : x < p.x; }
  bool operator==(const Point3D &p) const { return !sgn(p.x - x) && !sgn(p.y - y) && !sgn(p.z - z); }
  Point3D operator+(const Point3D &p) const { return {x + p.x, y + p.y, z + p.z}; }
  Point3D operator-(const Point3D &p) const { return {x - p.x, y - p.y, z - p.z}; }
  Point3D operator*(ld a) const { return {x * a, y * a, z * a}; }
  Point3D operator/(ld a) const { return {x / a, y / a, z / a}; }
  auto operator*(const Point3D &p) const { return x * p.x + y * p.y + z * p.z; }  // dot
  Point3D operator^(const Point3D &p) const { return {y * p.z - z * p.y, z * p.x - x * p.z, x * p.y - y * p.x}; }  // cross
  friend auto &operator>>(istream &i, Point3D &p) { return i >> p.x >> p.y >> p.z; }
};

struct Line3D {
  Point3D s = {0, 0, 0}, e = {0, 0, 0};
  Line3D() = default;
  Line3D(Point3D _s, Point3D _e) : s(_s), e(_e) {}
};

struct Segment3D : Line3D {
  using Line3D::Line3D;
};

auto dist2(const Point3D &a) { return a * a; }
auto dist2(const Point3D &a, const Point3D &b) { return dist2(a - b); }
auto dist(const Point3D &a) { return sqrt(dist2(a)); }
auto dist(const Point3D &a, const Point3D &b) { return sqrt(dist2(a - b)); }
auto dist(const Point3D &a, const Line3D &l) { return dist((a - l.s) ^ (l.e - l.s)) / dist(l.s, l.e); }
auto dist(const Point3D &p, const Segment3D &l) {
  if (l.s == l.e) return dist(p, l.s);
  auto d = dist2(l.s, l.e), t = min(d, max((ld)0, (p - l.s) * (l.e - l.s)));
  return dist((p - l.s) * d, (l.e - l.s) * t) / d;
}
```

## Miscellaneous

```cpp
tuple<int,int,ld> closest_pair(vector<Point> &p) {
  using Pt = pair<Point,int>;
  int n = p.size();
  assert(n > 1);
  vector<Pt> pts(n), buf;
  for (int i = 0; i < n; i++) pts[i] = {p[i], i};
  sort(pts.begin(), pts.end());
  buf.reserve(n);
  auto cmp_y = [](const Pt& p1, const Pt& p2) { return p1.first.y < p2.first.y; };
  function<tuple<int,int,ld>(int, int)> recurse = [&](int l, int r) -> tuple<int,int,ld> {
    int i = pts[l].second, j = pts[l + 1].second;
    ld d = dist(pts[l].first, pts[l + 1].first);
    if (r - l < 5) {
      for (int a = l; a < r; a++) for (int b = a + 1; b < r; b++) {
        ld cur = dist(pts[a].first, pts[b].first);
        if (cur < d) { i = pts[a].second; j = pts[b].second; d = cur; }
      }
      sort(pts.begin() + l, pts.begin() + r, cmp_y);
    }
    else {
      int mid = (l + r)/2;
      ld x = pts[mid].first.x;
      auto [li, lj, ldist] = recurse(l, mid);
      auto [ri, rj, rdist] = recurse(mid, r);
      if (ldist < rdist) { i = li; j = lj; d = ldist; }
      else { i = ri; j = rj; d = rdist; }
      inplace_merge(pts.begin() + l, pts.begin() + mid, pts.begin() + r, cmp_y);
      buf.clear();
      for (int a = l; a < r; a++) {
        if (abs(x - pts[a].first.x) >= d) continue;
        for (int b = buf.size() - 1; b >= 0; b--) {
          if (pts[a].first.y - buf[b].first.y >= d) break;
          ld cur = dist(pts[a].first, buf[b].first);
          if (cur < d) { i = pts[a].second; j = buf[b].second; d = cur; }
        }
        buf.push_back(pts[a]);
      }
    }
    return {i, j, d};
  };
  return recurse(0, n);
}

Line abc_to_line(ld a, ld b, ld c) {
  assert(!sgn(a) || !sgn(b));
  if(a == 0) return Line(Point(0, -c/b), Point(1, -c/b));
  if(b == 0) return Line(Point(-c/a, 0), Point(-c/a, 1));
  Point s(0, -c/b), e(1, (-c - a)/b), diff = e - s;
  return Line(s, s + diff/dist(diff));
}

tuple<ld,ld,ld> line_to_abc(const Line& l) {
  Point diff = l.e - l.s;
  return {-diff.y, diff.x, -(diff ^ l.s)};
}
```
